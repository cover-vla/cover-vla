"""
Evaluator/Client for the all kinds environment environment.

The model will be served on a websocket server, and the client will connect to it.
The client will send observations to the server, and the server will return actions.
The client will also log the results of the evaluation to a file. The server will do some auxiliary logging to stdout.
"""
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from policy_server_client import websocket_policy_client as _client

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # experiments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))) # src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))) # project_folder

from src.agent.configuration_pipeline import TrainPipelineConfig
from src.utils.monitor import setup_logger
from src.utils.pipeline import set_seed_everywhere

os.environ["WANDB__SERVICE_WAIT"] = "300"

class BaseEvaluator:
    def __init__(self, pipeline_cfg: TrainPipelineConfig):
        '''
        Initializes the evaluator with configuration objects for evaluating a trained model.

            pipeline_cfg: TrainPipelineConfig, a dataclass containing all configurations
                required for the training pipeline, including evaluation and model settings.
        '''
        self.pipeline_cfg = pipeline_cfg
        self.eval_cfg = pipeline_cfg.eval_cfg

        # Policy
        self.action_step = self.eval_cfg.action_step

        # Name of the run
        if pipeline_cfg.name is None:
            self.name = (
                pipeline_cfg.data.train.dataset_mix
                + "_"
                + pipeline_cfg.data.train.split
                + "_tp"
                + str(self.action_step)
            )
        else:
            self.name = pipeline_cfg.name



        # Server/client info
        self.port = pipeline_cfg.eval_cfg.port
        self.host = pipeline_cfg.eval_cfg.host

        # Debugging
        self.debug = pipeline_cfg.debug

        # List of checkpoints that we want to evaluate
        self.gradient_steps = self.eval_cfg.pretrained_model_gradient_step_cnt
        if self.gradient_steps is None:
            self.gradient_steps = [15130]
            self.no_gradient_steps = True
        else:
            self.no_gradient_steps = False

        # Task
        self.simulator_name = self.eval_cfg.simulator_name
        self.task_lists = self.eval_cfg.task_list

        # Seeding
        self.seed = pipeline_cfg.seed
        set_seed_everywhere(self.seed, train=False) # if not in training, then we don't seed tensrorflow because it only handles train/val datasets

        # Devices
        self.gpu_id = pipeline_cfg.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.main_rank = True  # This is kinda moot for eval now.

        # Evaluation parameters
        self.n_eval_episode = self.eval_cfg.n_eval_episode
        self.n_video = self.eval_cfg.n_video
        self.resize_size = pipeline_cfg.env.image_size

        # Model parameters
        self.use_amp = pipeline_cfg.use_amp
        self.dtype = torch.bfloat16 if pipeline_cfg.use_bf16 else torch.float32
        self.use_torch_compile = pipeline_cfg.use_torch_compile

        self.use_wandb = pipeline_cfg.use_wandb
        if self.use_wandb:
            self.run_id = pipeline_cfg.wandb.run_id # most likely None, unless resuming
            wandb.init(
                project=pipeline_cfg.wandb.project,
                name=time.strftime("%Y%m%d-%H%M%S") + "_" + self.name + f'_ta{self.action_step}',
                config=asdict(pipeline_cfg),
                entity=pipeline_cfg.wandb.entity,
                id=self.run_id,
                resume="allow",
            )
            self.wandb_metrics = {k: None for k in self.task_lists}

        self.client = _client.WebsocketPolicyClient(self.host, self.port)
        print(f"Connected to server at {self.host}:{self.port}")

    def evaluate(self):
        '''Run evaluation on all tasks in the task list for all checkpoints'''
        raise NotImplementedError("This method should be implemented in subclasses")

    def evaluate_task(self, task_name: str):
        '''Run evaluation on a specific task'''
        raise NotImplementedError("This method should be implemented in subclasses")

    def _initialze_model_client(self, model_path: str, gradient_step: int):
        '''
        On the client side, initialize the model at specific gradient step
        '''
        response = self.client.switch_model(model_path)

        if response["status"] != "model switched":
            raise RuntimeError(f"Failed to switch to model {model_path} and step {gradient_step}")

        # Logging
        self.log_dir: Path = (
            Path(os.environ["VLA_LOG_DIR"])
            / "eval_online"
            / self.simulator_name
            / self.name
            / f'step_{gradient_step!s}'
            / f'ta_{self.action_step}'
            / str(self.seed)
            / time.strftime("%Y-%m-%d_%H-%M-%S")
        )

        if self.main_rank:
            os.makedirs(self.log_dir, exist_ok=True)

        self.main_logger = setup_logger(
            main_rank=self.main_rank,
            filename=self.log_dir / "eval.log" if not self.debug else None, # log to console when debug is True
            debug=self.debug,
            name='main_logger'
        )

        self.main_logger.info(f"Model path: {model_path!s}. Step: {gradient_step!s}")

    def _preprocess_task_instruction(self, instruction: str):
        '''
        iterate through every word in the instruction, and map it to a new word
        '''
        for key in self.language_mapper.mapping_candidates.keys():
            instruction = instruction.replace(key, self.language_mapper.map(key))
        return instruction

    def _log_summary(self, logger, cnt_episode, eval_time, metrics):
        '''Helper method to log task summary information'''
        logger.info("============ Evaluation Summary ============")
        logger.info(f"Number of episodes: {cnt_episode}")
        logger.info(f"Total Task Eval Time: {eval_time / 60:.3f} minutes")
        # logger.info(f"Peak VRAM usage: {torch.cuda.max_memory_reserved(0) / 1024 ** 3:.2f} GB")
        for metric_name, metric_values in metrics.items():
            logger.info(f"{metric_name}: {metric_values:.2%}")
        logger.info("============================================")

