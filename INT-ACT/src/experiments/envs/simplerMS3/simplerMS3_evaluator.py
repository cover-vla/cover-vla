"""
Main evaluation agent. Using torch.compile and bfloat16 by default.
"""
import collections
import gc
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.visualization.misc import images_to_video
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict
from typing_extensions import override

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agent.configuration_pipeline import TrainPipelineConfig
from src.experiments.env_adapter.language_mapper import PersistentLanguageMapper
from src.experiments.envs.base_evaluator import BaseEvaluator
from src.utils.monitor import setup_logger

os.environ["WANDB__SERVICE_WAIT"] = "300"

class SimplerMS3Evaluator(BaseEvaluator):
    def __init__(self,
                 pipeline_cfg: TrainPipelineConfig):
        '''
        Initializes the evaluator with configuration objects for evaluating a trained model.

            pipeline_cfg: TrainPipelineConfig, a dataclass containing all configurations
                required for the training pipeline, including evaluation and model settings.
        '''
        super().__init__(pipeline_cfg)

        self.language_logic_chain = pipeline_cfg.eval_cfg.language_logic_chain

        if self.language_logic_chain:
            language_mapping_candidates = {
                "carrot": ["the yellow vegetable", "the veggie", "the yellow thing that rabbit likes", "the veggie that rabbit likes"],
                "eggplant": ["the purple vegetable", "the veggie", "the thing that looks like a purple balloon"],
                "spoon": ["the silver spoon", "the thing that people use to eat soup", "the shiny thing"],
                "cube": ["the thing that looks like a box", "lego"],
            }
            self.language_mapper = PersistentLanguageMapper(mapping_candidates=language_mapping_candidates, seed=self.seed)

        # MS3 uses different task names
        self.ms3_translator = {
            "widowx_carrot_on_plate" : "PutCarrotOnPlateInScene-v1",
            "widowx_put_eggplant_in_basket": "PutEggplantInBasketScene-v1",
            "widowx_spoon_on_towel" : "PutSpoonOnTableClothInScene-v1",
            "widowx_stack_cube" : "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
            # TODO: add new tasks here instead of the SimplerEnv_MS3/simpler_env/__init__.py
            "widowx_eggplant_on_carrot": "PutEggplantOnCarrotInScene-v1",
            "widowx_coke_can_on_plate": "PutCokeCanOnPlateInScene-v1"
        }

        self.n_parallel_eval = pipeline_cfg.eval_cfg.n_parallel_eval

        self.log_dir_dict = {}

        # in MS3, I will log wandb metric differently. The key will be the steps instead since wandb doesn't seem to allow the pattern that I want
        # basically, I want to log all steps of scene A, and then log all steps of scene B. But for wandb, steps must always be increasing.
        # at the end of the day, this is due to Maniskill3 memory leak, so I cannot iterate with gradient steps as the outer loop
        del self.wandb_metrics
        if self.use_wandb:
            self.step_metrics_dict: dict[int, dict] = {}

    @override
    def evaluate(self):
        '''Run evaluation on all tasks in the task list'''

        for task_name in self.task_lists:
            # Set up environment
            ms3_task_name = self.ms3_translator.get(task_name, task_name)

            self.env: BaseEnv = gym.make(
            ms3_task_name,
            obs_mode="rgb+segmentation",
            num_envs=self.n_parallel_eval,
            sensor_configs={"shader_pack": "default"},)

            for gradient_step in self.gradient_steps:

                if self.use_wandb:
                    self.current_step_metric_dict: dict[str, float] = self.step_metrics_dict.get(gradient_step, None)
                    if self.current_step_metric_dict is None:
                        self.current_step_metric_dict = {}
                        self.step_metrics_dict[gradient_step] = self.current_step_metric_dict

                self._initialze_model_client(gradient_step)
                self.evaluate_task(task_name)

            self.env.close()
            del self.env

        if self.use_wandb:
            # we upload all logged metrics at the end, so we can log with step as x-axis and in an increasing order
            self._upload_wandb()

        # old loop
        # for gradient_step in self.gradient_steps:
        #     self._initialze_model_client(gradient_step)
        #     for task_name in self.task_lists:
        #         if not self.debug:
        #             self._update_n_eval_episode(task_name) # some tasks have different number of possible episodes
        #         self.evaluate_task(task_name)



    @override
    def evaluate_task(self, task_name):
        '''
        Evaluate a single task

        Args:
            task_name: Name of the task to evaluate

        Returns:
            success_rate: The success rate achieved on this task
        '''
        start_task_time = time.time()
        task_seed = 0
        # Initialize task-specific logging
        task_log_dir = self.log_dir / task_name
        video_dir = task_log_dir / "videos"
        if self.main_rank:
            os.makedirs(video_dir, exist_ok=True)

        task_logger = setup_logger(
            main_rank=self.main_rank,
            filename=task_log_dir / f"{task_name}.log" if not self.debug else None,  # log to console when debug is True
            debug=self.debug,
            name=f'{task_name}_logger'
        )

        task_logger.info(f"Task suite: {task_name}")
        self.main_logger.info(f"Task suite: {task_name}")


        cnt_episode = 0
        eval_metrics = collections.defaultdict(list)

        # Set up receding horizon control
        action_plan = collections.deque()

        while cnt_episode < self.n_eval_episode:
            task_seed = task_seed + cnt_episode
            obs, _ = self.env.reset(seed=[task_seed + i for i in range(self.n_parallel_eval)],
                                    options={"episode_id": torch.tensor([task_seed + i for i in range(self.n_parallel_eval)]),
                                             "reconfigure": True})

            instruction = self.env.unwrapped.get_language_instruction()

            images = []
            predicted_terminated, truncated = False, False
            images.append(get_image_from_maniskill3_obs_dict(self.env, obs).cpu().numpy())
            elapsed_steps = 0

            while not (predicted_terminated or truncated):
                if not action_plan:
                    # action horizon is all executed
                    # Query model to get action
                    element = {
                            "observation.images.top": images[-1],
                            "observation.state": obs['agent']['eef_pos'].cpu().numpy(),
                            "task": instruction
                            }
                    action_chunk = self.client.infer(element)

                    # action chunk is of the size [batch, action_step, action_dim]
                    # but dequeue can only take something like [action_step, batch, action_dim]
                    action_plan.extend(action_chunk[:, :self.action_step, :].transpose(1, 0, 2))

                action = action_plan.popleft()
                obs, reward, terminated, truncated, info = self.env.step(action)
                elapsed_steps += 1
                info = common.to_numpy(info)

                truncated = bool(truncated.any()) # note that all envs truncate and terminate at the same time.
                images.append(get_image_from_maniskill3_obs_dict(self.env, obs).cpu().numpy())

            for k, v in info.items():
                eval_metrics[k].append(v.flatten())

            if self.pipeline_cfg.eval_cfg.recording:
                from concurrent.futures import ThreadPoolExecutor
                def save_video_task(args):
                    imgs, video_dir, video_name, fps, verbose = args
                    return images_to_video(imgs, video_dir, video_name, fps=fps, verbose=verbose)
                task_args = [
                    (
                        [img[i] for img in images],
                        video_dir,
                        f"video_{cnt_episode + i}{'_success' if info['success'][i].item() else ''}",
                        10,
                        False
                    )
                    for i in range(len(images[-1]))
                ]
                with ThreadPoolExecutor() as executor:
                    list(executor.map(save_video_task, task_args))
            # if self.pipeline_cfg.eval_cfg.recording:
            #     for i in range(len(images[-1])):
            #         # save video. The naming is ugly but it's to follow previous naming scheme
            #         success_string = "_success" if info['success'][i].item() else ""
            #         images_to_video([img[i] for img in images], video_dir, f"video_{cnt_episode + i}{success_string}", fps=10, verbose=True)

            cnt_episode += self.n_parallel_eval
            # if self.n_parallel_eval == 1:
            #     print(f"Evaluated episode {cnt_episode}. Seed {self.seed}. Results after {cnt_episode} episodes:")
            # else:
            #     print(f"Evaluated {self.n_parallel_eval} episodes, seeds {self.seed} to {cnt_episode}. Results after {cnt_episode} episodes:")
            # for k, v in eval_metrics.items():
            #     print(f"{k}: {np.mean(v)}")

        mean_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
        success_rate = mean_metrics['success']
        task_eval_time = time.time() - start_task_time

        # log results
        self._log_summary(logger=task_logger,
                               cnt_episode=cnt_episode,
                               eval_time=task_eval_time,
                               success_rate=success_rate)

        self._log_summary(logger=self.main_logger,
                               cnt_episode=cnt_episode,
                               eval_time=task_eval_time,
                               success_rate=success_rate)

        if self.use_wandb:
            self.current_step_metric_dict[task_name] = success_rate

        del images
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def _initialze_model_client(self, gradient_step: int):
        '''
        On the client side, initialize the model at specific gradient step
        '''

        self.model_path = Path(self.eval_cfg.pretrained_model_path) / f"step_{gradient_step!s}"
        response = self.client.switch_model(gradient_step)
        if response["status"] != "model switched":
            raise RuntimeError(f"Failed to switch model to step {gradient_step}")

        # Logging
        self.log_dir = self.log_dir_dict.get(gradient_step, None)

        if self.log_dir is None:
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
            self.log_dir_dict[gradient_step] = self.log_dir

        if self.main_rank:
            os.makedirs(self.log_dir, exist_ok=True)

        self.main_logger = setup_logger(
            main_rank=self.main_rank,
            filename=self.log_dir / "eval.log" if not self.debug else None, # log to console when debug is True
            debug=self.debug,
            name='main_logger'
        )

        self.main_logger.info(f"Model path: {self.model_path!s}. Step: {gradient_step!s}")

    def _upload_wandb(self):
        '''
        Upload all logged metrics to wandb
        '''
        for step in self.gradient_steps:
            current_step_metric_dict = self.step_metrics_dict.get(step, None)
            if current_step_metric_dict:
                wandb.log(current_step_metric_dict, step=int(step), commit=True)
    # def _update_n_eval_episode(self, task_name):
    #     if "google_robot" in task_name:
    #         if 'coke' in task_name:
    #             self.n_eval_episode = 25 * 4 # 25 locations, 4 urdfs, 10 trials each
    #         elif 'move' in task_name:
    #             self.n_eval_episode = 60 * 4 # 60 locations, 4 urdfs, 10 trials each
    #         elif 'drawer' in task_name:
    #             self.n_eval_episode = 3 * 4 * 9 # 3 drawers, 4 urdfs, 9 locations/rgb_overlay_paths, 10 trials each
    #         elif 'apple' in task_name:
    #             self.n_eval_episode = 9 * 4 * 3 # 9 apple locations, 4 urdfs, 3 robot locations/rgb_overlay_paths, 10 trials each
    #         self.n_video = self.n_eval_episode
