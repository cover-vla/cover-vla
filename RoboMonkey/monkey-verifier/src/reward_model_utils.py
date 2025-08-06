import json
import os
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Sequence
import einops

import torch
import transformers
from transformers import AutoTokenizer, set_seed

from lora_utils import print_trainable_parameters, DEFAULT_PAD_TOKEN
from models.reward_model import RewardConfig, RewardModel
from action_processing import ActionTokenizer

from llava import conversation as conversation_lib
from llava.model import *
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.train.train import smart_tokenizer_and_embedding_resize

# Deterministic behavior setup
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

logger = logging.getLogger(__name__)

class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    trust_remote_code: Optional[bool] = field(default=False)
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default=None)
    eval_dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    eval_dataset_name: str = field(default="alpaca_human_preference")
    eval_size: int = field(default=500)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = None
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = None
    reward_prompt_file: Optional[str] = None
    image_to_caption_file: Optional[str] = None

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = None
    remove_unused_columns: bool = False
    freeze_mm_mlp_adapter: bool = False
    model_max_length: int = 512
    query_len: int = None
    response_len: int = None
    label_names: List[str] = field(default_factory=lambda: ["index_0", "index_1", "choice"])
    padding: str = "longest"
    full_finetune: bool = False
    adam8bit: bool = False
    double_quant: bool = True
    quant_type: str = "nf4"
    bits: int = 4
    lora_modules: Optional[List[str]] = None
    lora_r: int = 64
    lora_alpha: float = 16
    lora_dropout: float = 0.0
    report_to: str = "none"
    resume_dir: Optional[str] = None
    output_dir: str = "./output"
    optim: str = "paged_adamw_32bit"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    weight_decay: float = 0.0
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    gradient_checkpointing: bool = True
    do_train: bool = True
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    group_by_length: bool = True
    save_strategy: str = "steps"
    save_steps: int = 250
    save_total_limit: int = 40
    resume_from_training: bool = False

def pad_sequence_from_left(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    )
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence
