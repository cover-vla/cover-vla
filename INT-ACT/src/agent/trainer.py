import os
import sys
import time
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

# from torchao.prototype.low_bit_optim import AdamW8bit
import bitsandbytes as bnb
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import einops
import torch.distributed as dist
from lerobot.common.policies.pretrained import PreTrainedPolicy
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

# TODO: need a better implementation of this importing. As it is not always needed and should not enable every time.
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from src.agent.configuration_pipeline import TrainPipelineConfig
from src.agent.dataset import TorchRLDSInterleavedDataset
from src.utils.metric import get_action_accuracy
from src.utils.monitor import Timer, blockprint, log_allocated_gpu_memory, log_execution_time, setup_logger
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions
from src.utils.pipeline import process_images, set_seed_everywhere

full_state_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

os.environ["WANDB__SERVICE_WAIT"] = "300"

class BaseTrainer:
    def __init__(self,
                 train_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        self.train_cfg = train_cfg
        self.model_cfg = train_cfg.model_cfg
        self.model_class = model_class

        # Setup run name
        if train_cfg.name is None:
            self.name = (train_cfg.data.train.dataset_mix + "_" +
                         train_cfg.data.train.split + "_tp" +
                         str(train_cfg.data.train.action_horizon))
        else:
            self.name = train_cfg.name

        self.wandb_runid = None

        # Seeding
        set_seed_everywhere(train_cfg.seed)

        # Device and multi-GPU settings
        self.gpu_id = train_cfg.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.multi_gpu = train_cfg.multi_gpu
        self.world_size = 1 # single node for now
        self.main_rank = True # every process defults to True
        if self.multi_gpu:
            self.global_rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.group_rank = int(os.environ["GROUP_RANK"])
            self.main_rank = True if self.global_rank == 0 else False

        if not self.main_rank:
            blockprint()

        # checkpoint/log/directory setup
        self.debug = train_cfg.debug
        self.log = setup_logger(main_rank=self.main_rank,
                                filename=None, # log to file. If None then to stdout
                                debug=self.debug) # If debug=True, DEBUG level and up will show, else INFO
        if self.multi_gpu:
            self.log.info(f"GPU local ID: {self.gpu_id}. Global rank: {self.global_rank}. Local rank: {self.local_rank}. \
                Local world size: {self.local_world_size}. World size: {self.world_size}. Group rank: {self.group_rank}"
        )
            for i in range(torch.cuda.device_count()):
                self.log.info(f"Local rank: {self.local_rank}, GPU UUID: {torch.cuda.get_device_properties(i).uuid}")

        self.save_model_freq = int(train_cfg.save_model_freq)
        self.log_freq = train_cfg.log_freq

        if self.main_rank:
            self._dir_setup()

        # Training parameters
        self.n_updates = int(train_cfg.n_updates) # number of gradient updates. != gradient steps due to gradient accumulation
        self.use_amp = train_cfg.use_amp
        self.dtype = torch.bfloat16 if train_cfg.use_bf16 else torch.float32

        self.multi_gpu_mechanism = train_cfg.mechanism

        # Model initialization
        self._initialize_model(train_cfg, model_class)

        #
        if self.model_class.name == "pi0":
            self.model.model.paligemma_with_expert.gemma_expert.lm_head = None # remove action expert lm head
            if train_cfg.freeze_lm_head:
                self.log.info("Freezing lm head")
                # # Remove and freeze parameters
                # self.model.model.paligemma_with_expert.gemma_expert.lm_head = None # remove action expert lm head
                self.model.model.paligemma_with_expert.paligemma.language_model.lm_head = None # remove VLM lm head
                # freeze paligemma's token embeddings
                for param in self.model.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.parameters():
                    param.requires_grad = False
            if train_cfg.freeze_vlm:
                # freeze VLM
                self.log.info("Freezing VLM")
                for param in self.model.model.paligemma_with_expert.paligemma.language_model.model.parameters():
                    param.requires_grad = False

        if self.multi_gpu and self.multi_gpu_mechanism == "fsdp":
            from torch.distributed.fsdp import MixedPrecision
            fsdp_mp_config = MixedPrecision(
                param_dtype = torch.bfloat16,
                reduce_dtype = torch.bfloat16,
                buffer_dtype = torch.bfloat16,
            )
        else:
            if not hasattr(self.model_cfg, "precision"):
                # we only manually convert model dtype if no such precision is specified in model config
                # because models usually have special ways to handle dtype, so whole model conversion is too rough
                self.model.to(self.dtype)
        self.model.to(self.device)
        if train_cfg.use_torch_compile:
            self.model = torch.compile(self.model,
                                       mode="default")

        self.log.info(f"Using cuda device: {self.device}, dtype: {self.dtype}")

        if self.multi_gpu:
            if self.multi_gpu_mechanism == "ddp":
                # import torch.distributed as dist
                # from torch.nn.parallel import DistributedDataParallel as DDP
                self.log.info(f"Using {self.local_world_size} GPUs in each of the {train_cfg.n_nodes} nodes")
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    gradient_as_bucket_view=True,
                    static_graph=False,
                    find_unused_parameters=False if train_cfg.freeze_lm_head else True,
                )
                dist.barrier()
            elif self.multi_gpu_mechanism == "fsdp":

                self.model = FSDP(self.model,
                                  use_orig_params=True,
                                  mixed_precision = fsdp_mp_config,
                                  )
                dist.barrier()
            else:
                raise NotImplementedError("Please specify a supported parallel mechanism.")

        log_allocated_gpu_memory(log=self.log, stage="loading model", device=self.gpu_id)

        self.action_horizon = self.model_cfg.chunk_size

        # Determine gradient accumulation and batch sizes
        self.grad_accumulation_steps = max(
            train_cfg.global_batch_size // train_cfg.per_device_batch_size // self.world_size, 1
        )
        self.actual_global_batch_size = train_cfg.per_device_batch_size * self.grad_accumulation_steps * self.world_size

        # Dataloaders
        self.train_dataloader = DataLoader(
            TorchRLDSInterleavedDataset(train_cfg.data.train, train=True, task_paraphrase=train_cfg.task_paraphrase).dataset,
            batch_size=train_cfg.per_device_batch_size,
            pin_memory=True,
        )
        self.val_dataiterator = iter(
            DataLoader(
                TorchRLDSInterleavedDataset(train_cfg.data.val, train=False).dataset,
                batch_size=train_cfg.per_device_batch_size,
                pin_memory=True,
            )
        )

        # Evaluation parameters
        self.eval_thresholds = train_cfg.eval_thresholds
        self.eval_freq = train_cfg.eval_freq
        self.per_device_num_eval_batch = train_cfg.eval_size // train_cfg.per_device_batch_size // self.world_size

        self.log.info(f"Total number of gradient updates: {self.n_updates}")
        self.log.info(f"Actual global batch size: {self.actual_global_batch_size}")
        self.log.info(f"Per device batch size: {train_cfg.per_device_batch_size}")
        self.log.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")
        self.log.info(f"Global batch_size after gradient accumulation: {train_cfg.global_batch_size}")
        self.log.info(f"Number of training batches: {len(self.train_dataloader)}")

        # Optimizer and scheduler
        self.optimizer = bnb.optim.AdamW8bit(
            self.model.parameters(),
            lr=self.model_cfg.optimizer_lr,
            betas=self.model_cfg.optimizer_betas,
            eps=self.model_cfg.optimizer_eps,
            weight_decay=self.model_cfg.optimizer_weight_decay,
        )
        # # ? if full precision optimizer will be more stable?
        # self.optimizer = optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.model_cfg.optimizer_lr,
        #     betas=self.model_cfg.optimizer_betas,
        #     eps=self.model_cfg.optimizer_eps,
        #     weight_decay=self.model_cfg.optimizer_weight_decay,
        # )

        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=10000000,
            cycle_mult=1.0,
            max_lr=self.model_cfg.optimizer_lr,
            min_lr=1e-8,
            warmup_steps=self.model_cfg.scheduler_warmup_steps,
            gamma=1.0,
        )
        self.log.info(f"Number of trained parameters: {get_num_params_in_billions(self.optimizer):.3f}B")

        # * hard coded with if else for now, to accomodate fusiona.
        # Parameter counting (account for DDP wrapper)
        base_model = self.model.module if self.multi_gpu else self.model
        if self.model_class.name == "pi0":
            vlm_parameters = base_model.model.paligemma_with_expert.paligemma.parameters()
            action_expert_parameters = base_model.model.paligemma_with_expert.gemma_expert.parameters()
            self.vlm_param_count = sum(p.numel() for p in vlm_parameters)
            self.action_expert_param_count = sum(p.numel() for p in action_expert_parameters)
            self.log.info(f"Number of trained parameters (VLM): {self.vlm_param_count/1e9:.3f}B")
            self.log.info(f"Number of trained parameters (action expert): {self.action_expert_param_count/1e9:.3f}B")
        elif self.model_class.name == "fusiona":
            fusion_param_count = sum(p.numel() for p in base_model.model.paligemma_fusion.parameters())
            projector_param_count = sum(p.numel() for p in base_model.model.state_proj.parameters()) + \
                sum(p.numel() for p in base_model.model.action_in_proj.parameters()) + \
                sum(p.numel() for p in base_model.model.action_out_proj.parameters()) + \
                sum(p.numel() for p in base_model.model.action_time_mlp_in.parameters()) + \
                sum(p.numel() for p in base_model.model.action_time_mlp_out.parameters())

            self.log.info(f"Number of trained parameters (fusion): {fusion_param_count/1e9:.3f}B")
            self.log.info(f"Number of trained parameters (projector): {projector_param_count/1e9:.3f}B")

        # Training state
        self.timer = Timer()
        self.cnt_batch = 0 # number of batches processed
        self.cnt_update = 0 # number of gradient updates. Can be smaller than cnt_batch due to gradient accumulation

        if train_cfg.resume_run and train_cfg.load_from_checkpoint is not None:
            self.log.info(f"resume previous run? {train_cfg.resume_run}, loading optimizer states and auxiliary data.")
            self._load_optimizer_and_auxiliary_data(train_cfg.load_from_checkpoint, resume_wandb=True)

        # Training log-related
        # Every metric get its own deque
        self.train_log_metrics = train_cfg.train_log_metrics # a list of metrics that we hope to keep track of
        self.train_log_metrics_dict = {} # holds the temporary metric before reduce
        self.train_log_deque_dict = {} # holds all the metrics after reduce
        for metric in self.train_log_metrics:
            self.train_log_metrics_dict[metric] = 0.0
            self.train_log_deque_dict[metric] = deque(maxlen=self.grad_accumulation_steps)

        # Evaluation log-related
        self.eval_log_metrics = train_cfg.eval_log_metrics # a list of metrics that we hope to keep track of
        self.eval_log_metrics_dict = {}
        self.new_eval_from_last_log = False # this is to check if we need to log the evaluation metrics

        # wandb setup
        if train_cfg.use_wandb and self.main_rank:
            wandb.init(
                project=train_cfg.wandb.project,
                name=time.strftime("%Y%m%d-%H%M%S") + "_" + self.name,
                config=asdict(train_cfg),
                entity=train_cfg.wandb.entity,
                id=self.wandb_runid,
                resume="allow",
            )

    def train(self):
        self.model.train()
        while True:
            for batch in self.train_dataloader:

                inputs = self.preprocess_batch(batch=batch)

                # Gradient accumulation check
                # only sync gradients when finish gradient accumulation
                grad_accumulating = (self.cnt_batch + 1) % self.grad_accumulation_steps != 0 and self.multi_gpu
                sync_context = self.model.no_sync() if grad_accumulating else nullcontext()

                with sync_context, torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.use_amp):
                    loss_train, train_dict = self.model(batch=inputs)

                    self._extract_train_log(train_dict)

                    if self.debug:
                        log_allocated_gpu_memory(log=self.log, stage=f"forward batch {self.cnt_batch}")

                normalized_loss = loss_train / self.grad_accumulation_steps
                normalized_loss.backward()  # accumulate gradients

                if not grad_accumulating:
                    # gradients synced, so step the optimizer
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_cfg.max_grad_norm)
                    self._extract_train_log_add("grad norm", grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    if self.debug:
                        log_allocated_gpu_memory(log=self.log, stage=f"optimizer step batch {self.cnt_batch}")

                    self.optimizer.zero_grad(set_to_none=True)
                    self.cnt_update += 1

                    # save model and auxiliary data at the end of a update
                    if self.cnt_update % self.save_model_freq == 0 or self.cnt_update == self.n_updates:
                        self._save_training() # takes care of main rank in the function

                # Loss process for logging
                self._process_train_log()

                # Validation step
                # Validate once in a while to check overfitting
                if self.cnt_update % self.eval_freq == 0 and not grad_accumulating:
                    self.new_eval_from_last_log = True
                    self.validate()
                    self.model.train() # explicitly set back to train mode

                # Log training metrics
                if self.main_rank and self.cnt_update % self.log_freq == 0 and not grad_accumulating:
                    self._log_training()

                    if self.train_cfg.use_wandb:
                        self._log_wandb()

                self.cnt_batch += 1
                if self.cnt_update >= self.n_updates:
                    return # end training

    def validate(self):
        self.model.eval()
        self._initialize_eval_log()

        if self.main_rank:
            self.log.info(f"Running evaluation for {self.per_device_num_eval_batch} batches...")
        with torch.no_grad():
            for _ in range(self.per_device_num_eval_batch):
                batch_eval = next(self.val_dataiterator)

                inputs = self.preprocess_batch(batch=batch_eval)
                gt_actions = inputs["action"]

                with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.use_amp):
                    if self.multi_gpu and self.multi_gpu_mechanism == "ddp":
                        pred_actions = torch.stack(
                            [self.model.module.select_action(inputs) for _ in range(self.action_horizon)],
                            dim=1
                        )
                    elif self.multi_gpu and self.multi_gpu_mechanism == "fsdp":
                        # summon all parameters for the custom function call, avoid sharding/unsharding
                        with FSDP.summon_full_params(self.model):
                            pred_actions = torch.stack(
                                [self.model.select_action(inputs) for _ in range(self.action_horizon)],
                                dim=1
                            )
                    else:
                        pred_actions = torch.stack(
                            [self.model.select_action(inputs) for _ in range(self.action_horizon)],
                            dim=1
                        )
                self._extract_eval_log(gt_actions, pred_actions)

        self._process_eval_log()

        if self.main_rank:
            self._log_validation()

    def preprocess_batch(self, batch):
        # TODO support multi-image / proprio history
        images = batch["observation"]["image_primary"]
        proprios = batch["observation"]["proprio"].squeeze(1)  # remove the time dimension
        actions = batch["action"].squeeze(1)  # remove the time dimension
        texts = [
            text.decode("utf-8") for text in batch["task"]["language_instruction"]
        ]
        images = einops.rearrange(
            images, "B T H W C -> B (T C) H W"
        )  # remove cond_steps dimension
        if self.debug:
            # print the content of images as a string, not the shape
            self.log.debug(f"raw image content {images!s}")
        # Irving: manually make the images [-1, 1]. LeRobot's normalize layer is a bit difficult to work with images
        images = process_images(
                images,
                rescale_factor=1 / 255.0,
            )

        # add action pad mask from the input,
        # squeeze the action dimension because action padding does not happen here
        # [bsz, action_len, action_dim] -> [bsz, action_len] keep 1 if any action_dim is 1
        action_pad_mask = torch.any(batch["action_pad_mask"], dim=-1)
        # to stay consistent with lerobot's implementation
        action_is_pad = ~action_pad_mask
        # TODO: add the action is pad to the inputs. (juexiao)

        if self.debug:
            self.log.debug(f"processed image content {images}")
            self.log.debug(f"raw image shape {images.shape}")
            self.log.debug(f"raw state shape {proprios.shape}")
            self.log.debug(f"raw action shape {actions.shape}")
        inputs = {
            "observation.state": proprios.to(self.dtype).to(self.device),
            "observation.images.top": images.to(self.dtype).to(self.device),
            "task": texts,
            "action": actions.to(self.dtype).to(self.device),
        }

        return inputs

    ############################### Training log related functions ###############################
    def _extract_train_log(self, train_dict):
        '''
        Extract the metrics from the training dict and store them in the train_log_metric_dict for reduce later
        '''
        for metric in self.train_log_metrics:
            self.train_log_metrics_dict[metric] = train_dict.get(metric, torch.tensor(0.).to(self.device))

    def _extract_train_log_add(self, log_key: str, log_v: float | torch.Tensor):
        '''
        Additionally add new key and value to the train_log metrics.
        ! use conservatively.
        '''
        if log_key not in self.train_log_metrics_dict:
            self.train_log_metrics_dict[log_key] = 0.0
        if isinstance(log_v, torch.Tensor):
            self.train_log_metrics_dict[log_key] = log_v.detach()
        else:
            self.train_log_metrics_dict[log_key] = torch.tensor(log_v).to(self.device)

    def _process_train_log(self):
        '''
        Aggregate the metrics in the train_log_metric_dict and store them in the train_log_deque_dict
        To output different metrics, you need to modify the train_log_metrics in the config, and also let your model output the metrics in a dict
        '''

        if self.multi_gpu:
            import torch.distributed as dist
            for metric in self.train_log_metrics:
                dist.all_reduce(self.train_log_metrics_dict[metric], op=dist.ReduceOp.SUM)
                self.train_log_deque_dict[metric].append(self.train_log_metrics_dict[metric].item() / dist.get_world_size())
        else:
            for metric in self.train_log_metrics:
                self.train_log_deque_dict[metric].append(self.train_log_metrics_dict[metric].item())

    def _log_training(self):
        '''
        log the training metrics to logger.
        Utilize the train_log_deque_dict to get the metrics
        '''

        # we only care about the mean of the last grad_accumulation_steps
        self.wandb_train_log_dict = {k: np.mean(v) for k, v in self.train_log_deque_dict.items()}

        peak_vram = torch.cuda.max_memory_reserved(self.gpu_id) / (1024**3)
        log_msg = (f"Batch {self.cnt_batch} Update {self.cnt_update}: t {self.timer():8.4f} | "
                    f"vram {peak_vram:6.3f} |  lr {self.optimizer.param_groups[0]['lr']:10.8f}")
        for k, v in self.wandb_train_log_dict.items():
            log_msg += f" | {k}: {v:.3f}"
        self.log.info(log_msg)

    def _log_wandb(self):
        '''
        Log all metrics to wandb
        Note that we also log eval metrics here
        '''

        wandb_metrics = {
                            "gradient steps": self.cnt_update,
                            "learning rate": self.optimizer.param_groups[0]["lr"],
                        }
        # log various training loss
        for k, v in self.wandb_train_log_dict.items():
            wandb_metrics[f'{k} - train'] = v
        if self.new_eval_from_last_log:
            wandb_metrics.update(
                        {
                            f"eval acc - thres {threshold}": accuracy.item()
                            for threshold, accuracy in zip(
                                self.eval_thresholds, self.eval_log_metrics_dict['eval_accuracy'], strict=False
                            )
                        }
                    )
            for metric in self.eval_log_metrics:
                wandb_metrics[f"{metric} - eval"] = self.eval_log_metrics_dict[metric].item()
            self.new_eval_from_last_log = False
        wandb.log(wandb_metrics, step=self.cnt_update, commit=True)

    ############################### Evaluation log related functions ###############################
    def _initialize_eval_log(self):
        '''
        This will be called in validate to initialize the logging
        '''
        self.eval_log_metrics_dict['eval_accuracy'] = torch.zeros(len(self.eval_thresholds), device=self.device)
        for metric in self.eval_log_metrics:
            self.eval_log_metrics_dict[metric] = torch.tensor(0.0, device=self.device)

    def _extract_eval_log(self, gt_actions, pred_actions):
        '''
        This will be called in validate to extract the logging

        This function usually needs to be rewritten when inheriting from this class
        Because different loss has different way to calculate
        '''

        self.eval_log_metrics_dict['eval_accuracy'] += get_action_accuracy(gt_actions, pred_actions, self.eval_thresholds)
        self.eval_log_metrics_dict['l1_loss'] += torch.nn.functional.l1_loss(pred_actions, gt_actions)

    def _process_eval_log(self):
        '''
        This will be called in validate to process the eval logging
        Usually, this will not be changed when inheriting from this class
        Because we just need to average and reduce the metrics
        '''
        for k, v in self.eval_log_metrics_dict.items():
            self.eval_log_metrics_dict[k] = v / self.per_device_num_eval_batch
        # aggregate the metrics if multi-gpu
        if self.multi_gpu:
            import torch.distributed as dist
            for k, v in self.eval_log_metrics_dict.items():
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
                self.eval_log_metrics_dict[k] = v / dist.get_world_size()

    def _log_validation(self):
        '''
        Log the validation metrics to logger
        Utilize the eval_log_metrics_dict to get the metrics
        Usually, this will not be changed when inheriting from this class
        Because we just log all the metrics specified in the config
        '''
        log_msg = "Eval | "

        # we don't want to treat accuracy as any other scalar metric
        for metric in self.eval_log_metrics:
            log_msg += f"{metric}: {self.eval_log_metrics_dict[metric].item():.3f} | "

        log_msg += "".join(
                [f"acc thres {threshold}: {accuracy.item():.3f}"
                for threshold, accuracy in zip(self.eval_thresholds, self.eval_log_metrics_dict['eval_accuracy'], strict=False)]
            )
        self.log.info(log_msg)

    ############################### Model saving and loading functions ###############################
    def _dir_setup(self):
        self.log_dir: Path = (
            Path(
            os.environ["VLA_LOG_DIR"])
            / "train"
            / self.name
            / (time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{self.train_cfg.seed}")
        )
        self.checkpoint_dir = self.log_dir / "checkpoint"
        if self.main_rank:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # This cleans up previous runs that are empty, which are likely failed runs
        parent_dir = self.log_dir.parent
        for subdir in parent_dir.iterdir():
            # Skip non-directories and self.log_dir itself.
            if not subdir.is_dir() or subdir == self.log_dir:
                continue
            entries = list(subdir.iterdir())
            # Check if the only entry is a directory named "checkpoint".
            if len(entries) == 1 and entries[0].is_dir() and entries[0].name == "checkpoint":
                checkpoint_dir = entries[0]
                # Remove the directory if the checkpoint folder is empty.
                if not any(checkpoint_dir.iterdir()):
                    checkpoint_dir.rmdir()
                    subdir.rmdir()
                    self.log.info(f"Removed empty run directory: {subdir}")

    def _initialize_model(self, train_cfg, model_class):
        # Model initialization
        if train_cfg.load_from_checkpoint is None:
            self.model = model_class(config=self.model_cfg, dataset_stats=train_cfg.data.dataset_stats)
        else:
            self.model = self._load_model(model_class=model_class, checkpoint_dir=train_cfg.load_from_checkpoint)
            self.log.info(f"Loaded checkpoint from {train_cfg.load_from_checkpoint}.")

    @log_execution_time()
    def _save_training(self):
        if self.multi_gpu_mechanism == "fsdp": # rank is checked inside the functions
            # save the model
            self._save_fsdp_model()
            # save the optimzer and other data, separating the function to potentially save the CPU memory
            # ! Not saving the optimizer, because lack of support for low bit optimizer.
            # self._save_fsdp_optim()
        else:
            # normal, non-fsdp, saving
            if self.main_rank:
                model_save_path = self.checkpoint_dir / f"step_{self.cnt_update}"
                data_save_path = model_save_path / "auxiliary_data.pt"

                # In HF, model_save_path is a path to a folder, which contains a .safetensors file
                if self.multi_gpu and self.multi_gpu_mechanism == "ddp":
                    self.model.module.save_pretrained(model_save_path)
                else:
                    self.model.save_pretrained(model_save_path)

                data = {
                    "cnt_update": self.cnt_update,
                    "cnt_batch": self.cnt_batch,
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "wandb_id": wandb.run.id,
                }
                torch.save(data, data_save_path)
                checkpoint_size_in_gb = os.path.getsize(model_save_path / "model.safetensors") / (1024**3)
                self.log.info(f"Saved model to {model_save_path}, size: {checkpoint_size_in_gb:.3f} GB")

    # for FSDP
    def _save_fsdp_model(self,):
            # model
            with FSDP.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, full_state_save_policy
            ):
                cpu_state = self.model.state_dict()
            if self.main_rank:
                model_save_path = self.checkpoint_dir / f"step_{self.cnt_update}"

                # In HF, model_save_path is a path to a folder, which contains a .safetensors file
                # ? NOTE (juexiao): this implementation will have to reinitialize a model to complie with the save_model api in save tensors
                # ? kind of ugly and inefficient but this makes best use of standards. Is there a better way?
                temp_model_config = deepcopy(self.model_cfg)
                temp_model_config.paligemma_pretrained_path = None # make sure no unnecessary load, just a clean temp model for saving purpose
                temp_model_to_save = self.model_class(config=temp_model_config, dataset_stats=self.train_cfg.data.dataset_stats)
                temp_model_to_save.resize_token_embedding()
                temp_model_to_save.register_special_tokens()
                # handles potential key mismatches between the model and the state dict
                temp_state_dict = temp_model_to_save.state_dict()
                # clean up the potential prefixes in the FSDP full state dict
                cleaned_state_dict = {}
                for fsdp_key, fsdp_value in cpu_state.items():
                    clean_key = fsdp_key.replace("_orig_mod.", "")
                    if clean_key in temp_state_dict:
                        cleaned_state_dict[clean_key] = fsdp_value
                    else:
                        self.log.warning(f"**FSDP** saving issue: Key {fsdp_key} from FSDP model not found in rebular model")
                # then load the cleaned state dict
                temp_model_to_save.load_state_dict(cleaned_state_dict, strict=False)
                # temp_model_to_save.load_state_dict(cpu_state)
                temp_model_to_save.save_pretrained(model_save_path)

                checkpoint_size_in_gb = os.path.getsize(model_save_path / "model.safetensors") / (1024**3)
                self.log.info(f"Saved model to {model_save_path}, size: {checkpoint_size_in_gb:.3f} GB")

    def _save_fsdp_optim(self,):
        # ! this logic is working with full precision optimizer, but NOT for low bits. Therefore current implementation does NOT save the optimizer states.
        # ! left for future suppport.
        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, full_state_save_policy
        ):
            optim_state = FSDP.optim_state_dict(self.model, self.optimizer)

        if self.main_rank:
            model_save_path = self.checkpoint_dir / f"step_{self.cnt_update}"
            data_save_path = model_save_path / "auxiliary_data.pt"

            data = {
                    "cnt_update": self.cnt_update,
                    "cnt_batch": self.cnt_batch,
                    "optimizer": optim_state,
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "wandb_id": wandb.run.id,
                }
            torch.save(data, data_save_path)

    @log_execution_time()
    def _load_model(self,
                    model_class: PreTrainedPolicy,
                    checkpoint_dir: str):
        '''
        Resume training from a checkpoint.
        It will only load the model. Nothing else
        '''
        model = model_class.from_pretrained(
            pretrained_name_or_path=checkpoint_dir,
            config=self.model_cfg,
            strict=False,
        )
        return model

    @log_execution_time()
    def _load_optimizer_and_auxiliary_data(self,
                                          checkpoint_dir: str,
                                          resume_wandb: bool = True):
        '''
        Resume training from a checkpoint.
        It will only load the auxiliary data
        '''
        try:
            from src.utils.optim import optimizer_to
            data = torch.load(f"{checkpoint_dir}/auxiliary_data.pt", map_location="cpu")
            self.cnt_update = data["cnt_update"]
            self.cnt_batch = data["cnt_batch"]
            self.optimizer.load_state_dict(data["optimizer"])
            optimizer_to(self.optimizer, self.device)
            self.lr_scheduler.load_state_dict(data["lr_scheduler"])
            if resume_wandb: # if not needed
                self.wandb_runid = data.get("wandb_id", None) # not all run has wandb_id saved
            self.log.info(f"Resuming training from {checkpoint_dir}")
        except (FileNotFoundError, RuntimeError, KeyError, TypeError) as e:
            self.log.info(f"Failed to load optimizer and auxiliary data from {checkpoint_dir}: {e}")
            self.log.info("Use optimizer and auxiliary data as if it is a new training")
            return

class PI0Trainer(BaseTrainer):
    def __init__(self,
                 train_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        super().__init__(train_cfg, model_class)

class PI0FASTTrainer(BaseTrainer):
    def __init__(self,
                 train_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        super().__init__(train_cfg, model_class)
