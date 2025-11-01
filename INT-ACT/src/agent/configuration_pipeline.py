import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class TrainDataConfig:
    dataset_mix: str = "bridge"
    split: str = "train"
    data_path: Path = Path(os.environ["VLA_DATA_DIR"]) / "resize_224"
    window_size: int = 1
    action_horizon: Optional[int] = None # how many actions to predict in the future, not always how many action executed. We will get this value from the model config
    skip_unlabeled: bool = True
    load_proprio: bool = True
    shuffle_buffer_size: int = 200000
    num_parallel_calls: int = 50
    traj_transform_threads: int = 20
    traj_read_threads: int = 20
    max_action_future: int = 50 # maximum seq of future actions 50 is probably too long?

@dataclass
class ValDataConfig:
    dataset_mix: Optional[str] = None
    split: str = "val"
    data_path: Optional[Path] = None
    window_size: Optional[int] = None
    action_horizon: Optional[int] = None
    skip_unlabeled: Optional[bool] = None
    load_proprio: Optional[bool] = None
    shuffle_buffer_size: int = 10000
    num_parallel_calls: Optional[int] = None
    traj_transform_threads: Optional[int] = None
    traj_read_threads: Optional[int] = None
    max_action_future: Optional[int] = None # maximum seq of future actions

@dataclass
class DataConfig:
    train: TrainDataConfig = field(default_factory=TrainDataConfig)
    val: ValDataConfig = field(default_factory=ValDataConfig)
    dataset_stats: dict[str, dict[str, list]] = field(
        default_factory=lambda: {
            "observation.state": {
                "mean": ([
                    0.30904945731163025,
                    0.03045589290559292,
                    0.06558273732662201,
                    0.00706630339846015,
                    -0.07828629016876221,
                    0.10661222040653229,
                    0.7149746417999268
                ]),
                "std": ([0.06059328466653824,
                    0.09172434359788895,
                    0.05185756832361221,
                    0.1313914805650711,
                    0.1698099821805954,
                    0.573583722114563,
                    0.3517141044139862]),
            },
            "action": {
                "mean": ([0.00021758403454441577,
                    0.00012507825158536434,
                    -0.00017109014152083546,
                    -0.0001617111702216789,
                    -0.0002524859446566552,
                    0.0002515816013328731,
                    0.5879487991333008]),
                "std": ([0.009632210247218609,
                    0.013500974513590336,
                    0.012510341592133045,
                    0.028145477175712585,
                    0.03028254210948944,
                    0.07585873454809189,
                    0.4877150356769562]),
            },
        }
    )

    # the number of episodes used in the training set
    # this will affect the number of gradient steps
    train_episode_count: int = 1550000


@dataclass
class WandBConfig:
    project: str = "INT-ACT"
    entity: str = "ai4ce"
    run_id: str | None = None # if none, then will be auto generated

@dataclass
class EnvConfig:
    '''
    Configuration for the simulation experiment environment
    Can leave this unchanged if doing offline batch eval on the dataset
    '''
    dataset_statistics_path: str | None = None
    image_size: tuple[int, int] = (224, 224)
    action_normalization_type: str = "bound"
    state_normalization_type: str = "bound"

@dataclass
class EvalConfig:
    '''
    Configuration for offline evaluation and simulation experiment
    '''
    simulator_name: str = "simpler" # e.g. "simpler", "libero"

    env_adapter: str | None = "BridgeSimplerAdapter"

    # use this one testing on one specific task
    # if not none, then will only eval on this task

    # use this one for testing on multiple tasks
    task_list: list[str] | None = field(default_factory=lambda: ["widowx_carrot_on_plate",
                                                                 "widowx_put_eggplant_in_basket",
                                                                 "widowx_spoon_on_towel",
                                                                 "widowx_stack_cube"])

    n_eval_episode: int = 24 # how many different configurations in a single random seed
    n_video: int = 24
    n_parallel_eval: int | None = None # how many parallel evals to run. This is only applicable for simplerMS3
    recording: bool = True

    # checkpoint
    pretrained_model_path: str | None = None # this should not include the step, but where all checkpoints throughout training are saved
    pretrained_model_gradient_step_cnt: list[int] | None = None # instead, we use this to determine the step

    # server client
    role: str = "server" # "server" or "client"
    host: str = "0.0.0.0"
    port: int = 8000

    # TODO: verify if doing this allows me to not import lerobot
    action_step: int = 4 # how many actions to actually execute given all the predicted actions

    # language logic chain testing
    language_logic_chain: bool = False # whether to test language logic chain

    unnorm_key: str | None = None # Only used for OpenVLA-like HF models

@dataclass
class TrainPipelineConfig:
    # if paraphrase instruction, see https://huggingface.co/datasets/rail-berkeley/OXE_paraphrases and data/utils/task_augmentation.py
    task_paraphrase: bool = False

    # data
    data: DataConfig = field(default_factory=DataConfig)

    # meta
    name: str | None = None # name of the experiment. If none, then will be auto generated based on time and other parameters

    seed: int = 42
    debug: bool = False # whether to run in debug mode. Affect log level and other things

    # multi-gpu
    n_nodes: int = 1
    multi_gpu: bool = torch.cuda.device_count() > 1 or n_nodes > 1
    mechanism: str = "ddp" # "fsdp"
    if multi_gpu:
        from torch.distributed import destroy_process_group, init_process_group

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend="nccl")

        gpu_id = int(os.environ["LOCAL_RANK"])
    else:
        gpu_id = 0
    gpu_id = gpu_id

    # training
    use_torch_compile: bool = True
    use_amp: bool = True
    use_bf16: bool = True

    global_batch_size: int = 1024
    per_device_batch_size: int = 32
    n_epochs: int = 15
    max_grad_norm: float = 1.0

    # these two will be user-supplied, or set in __post_init__ based on the config
    n_updates: int | None = None # the number of gradient steps to take
    save_model_freq: int | None = None # in gradient steps, how often to save the model

    # logging
    log_freq: int = 4 # log every log_freq gradient steps
    # To log different train metrics, aside from modifyin this, also let your model output the metrics with the same name specified here in output dict
    train_log_metrics: list = field(default_factory=lambda: ["l2_loss"]) # a list of metrics that we hope to keep track of during training

    # To log different eval metrics
    # Note that accuracy for different thresholds are always logged, no need to specify them here
    eval_log_metrics: list = field(default_factory=lambda: ["l1_loss"]) # a list of metrics that we hope to keep track of during eval

    # online eval that happens during training
    eval_thresholds: list = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5])
    eval_freq: int = 250 # eval every eval_freq gradient steps
    eval_size: int = 1024

    # model
    model_cfg: PreTrainedConfig = field(default_factory=lambda: PreTrainedConfig.from_pretrained("test"))
    freeze_lm_head: bool = True # traditional pi0 does not need lm head.
    freeze_vlm: bool = False # freeze VLM
    load_from_checkpoint: str | None = None # the path to the FOLDER that contains the checkpoint
    # TODO: This name is a bit weird. Maybe load_weight_only?
    resume_run: bool = True # if True, load optimizer and wandb to resume the previous run, if False, only load the chkpt weight and make it a new run

    # wandb logging
    use_wandb: bool = True
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # eval
    eval_cfg: EvalConfig | None = None
    env: EnvConfig = field(default_factory=EnvConfig)

    def __post_init__(self):
        self.validate_lm_head()
        self.validate_parallel_eval()

        # convert dataset_stats's list to tensors because you cannot decode tensors from yaml/json
        for _, stats in self.data.dataset_stats.items():
            for k, v in stats.items():
                stats[k] = torch.tensor(v)

        # maintain consistency between dataloader's tp and model's tp
        if self.data.train.action_horizon is None:
            self.data.train.action_horizon = self.model_cfg.chunk_size if hasattr(self.model_cfg, "chunk_size") else 1

        # fill val data config's None values with train data config's values
        for key, value in vars(self.data.train).items():
            if getattr(self.data.val, key) is None:
                setattr(self.data.val, key, value)

        # determine gradient step count
        if self.n_updates is None:
            self.n_updates = self.data.train_episode_count // self.global_batch_size * self.n_epochs

        # save every epoch
        if self.save_model_freq is None:
            self.save_model_freq = self.data.train_episode_count // self.global_batch_size * 1


        if self.eval_cfg is not None:
            # construct the env adapter and evaluator full path
            simulator_name = self.eval_cfg.simulator_name
            if self.eval_cfg.env_adapter is not None:
                self.eval_cfg.env_adapter_path = f"src.experiments.env_adapters.{simulator_name}.{self.eval_cfg.env_adapter}"
            else:
                self.eval_cfg.env_adapter_path = f"src.experiments.env_adapters.{simulator_name}.BridgeSimplerAdapter"

            if simulator_name is not None:
                self.eval_cfg.simulator_path = f"src.experiments.envs.{simulator_name}.{simulator_name}_evaluator.{simulator_name[:1].upper() + simulator_name[1:]}Evaluator"
            else:
                raise ValueError("Simulator name is not specified in the config.")

    def validate_lm_head(self):
        pass

    def validate_parallel_eval(self):
        '''
        Check if parallel eval is set correctly
        '''
        if self.eval_cfg is not None:
            if self.eval_cfg.n_parallel_eval is not None:
                if self.eval_cfg.simulator_name != "simplerMS3":
                    raise ValueError("n_parallel_eval is only applicable for simplerMS3")
                if self.eval_cfg.n_parallel_eval <= 1:
                    raise ValueError("n_parallel_eval should be greater than 1")
                if "Batch" not in self.eval_cfg.env_adapter:
                    raise ValueError("You need to use an env adapter that supports batch eval for n_parallel_eval>1")
            if self.eval_cfg.n_parallel_eval is None and self.eval_cfg.simulator_name == "simplerMS3":
                raise ValueError("n_parallel_eval should be set for simplerMS3")
