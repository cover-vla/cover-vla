import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import timm
import timm.data
import tokenizers
import torch
import torch.nn as nn
import transformers
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from sglang.srt.managers.schedule_batch import ImageInputs
from transformers.models.auto import CONFIG_MAPPING
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.openvla import PrismaticProjector, PrismaticVisionBackbone, PrismaticProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaForCausalLM

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.model": "model",
    "language_model.lm_head": "lm_head",
}

TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
    "clip-vit-l": ["quick_gelu"],
    "clip-vit-l-336px": ["quick_gelu"],
    "dinov2-vit-l": [None],
    "in1k-vit-l": [None],
    "siglip-vit-so400m": [None],
    "siglip-vit-so400m-384px": [None],
    "dinoclip-vit-l-336px": [None, "quick_gelu"],
    "dinosiglip-vit-so-224px": [None, None],
    "dinosiglip-vit-so-384px": [None, None],
}

LLM_BACKBONE_TO_HF_METACLASS = {
    "llama2-7b-pure": "llama",
}

logger = logging.getLogger(__name__)

# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


class OpenVLAConfig(PretrainedConfig):
    model_type: str = "openvla"
    is_composition: bool = False

    def __init__(
        self,
        norm_stats: Optional[
            Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]
        ] = None,
        n_action_bins: int = 256,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "vicuna-v15-7b",
        arch_specifier: str = "no-align+gelu-mlp",
        use_fused_vision_backbone: Optional[bool] = None,
        image_resize_strategy: str = "letterbox",
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins

        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # [Contract] All vision backbone parameters are lists =>> supports fused backbones with different preprocessing
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(
                self.vision_backbone_id.startswith(v)
                for v in ["dinoclip", "dinosiglip"]
            )
        )

        self.timm_model_ids = [
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[self.vision_backbone_id]
        self.image_sizes = [224, 224]
        self.image_resize_strategy = None

        self.hf_llm_id = "meta-llama/Llama-2-7b-hf"
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of

        # [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that specific naming!
        self.text_config = (
            CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]](
                **text_config
            )
            if text_config is not None
            else CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]()
        )

        self.hidden_size = 4096
        self.num_attention_heads = 32
        self.num_hidden_layers = 32
        self.vocab_size = 32064
        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class OpenVLAForActionPrediction(PreTrainedModel):
    config_class: PretrainedConfig = OpenVLAConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True
    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def __init__(
        self,
        config: OpenVLAConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__(config)
        self.embeddings_layer = None
        self.past_key_values = None
        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")
        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone,
            config.image_sizes,
            config.timm_model_ids,
            config.timm_override_act_layers,
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = LlamaForCausalLM(
            config.text_config, quant_config=quant_config
        )

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = (
            self.config.text_config.vocab_size - self.config.pad_to_multiple_of
        )

    def pad_input_ids(self, input_ids: List[int], image_inputs: ImageInputs):
        multiple_of=64
        pad_value = 2
        image_pad_len = ((224 - 1) // multiple_of + 1) * multiple_of
        input_ids = input_ids[:1] + [pad_value] * image_pad_len + input_ids[1:]
        if input_ids[-1] != 29871:
            input_ids.append(29871)
        return input_ids

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = list(weights)
        new_weights = []
        params_dict = dict(self.named_parameters())
        for name, weight in weights:
            if not "language_model" in name:
                param = params_dict[name]
                default_weight_loader(param, weight)
                continue

            new_name = None
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    new_name = name.replace(key_to_modify, new_key)

            if new_name is not None:
                new_weights.append((new_name, weight))
            else:
                new_weights.append((name, weight))

        weights = new_weights

        self.language_model.load_weights(weights)
        self.processor = PrismaticProcessor.from_pretrained(
                "openvla/openvla-7b", trust_remote_code=True
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        # pixel_values: Optional[List[Optional[np.array]]] = None,
        # image_sizes: Optional[List[List[int]]] = None,
        # image_offsets: Optional[List[int]] = None,
    ):
        need_vision = forward_batch.image_inputs is not None and any(
            p is not None for p in forward_batch.image_inputs
        ) and forward_batch.extend_seq_lens_cpu is not None
        # === Handle Unimodal Forward ===
        if not need_vision or len(positions) == 1:
            assert (
                input_ids is not None
            ), "Missing `input_ids` in language-only forward!"
            return self.language_model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=None,
            )

        # === Handle Multimodal Forward ===
        embedding_layer = self.language_model.model.embed_tokens
        input_embeddings = embedding_layer(input_ids)
        pt = 0
        # assert len(forward_batch.image_inputs) == 1, "Only single image inputs supported in OpenVLA"
        for i, image_input in enumerate(forward_batch.image_inputs):
            image_offset = 1
            image_size = 256
            pixel_value = self.processor.process_image(image_input.pixel_values).to(torch.bfloat16).to("cuda")
            patch_features = self.vision_backbone(pixel_value)
            projected_patch_embeddings = self.projector(patch_features)
            input_embeddings[pt + image_offset : pt + image_offset + image_size] = (
                projected_patch_embeddings
            )
            pt += forward_batch.extend_seq_lens_cpu[i]

        image_data = forward_batch.image_inputs[0]
        pixel_value = image_data.pixel_values
        pixel_value = self.processor.process_image(pixel_value).to(torch.bfloat16).to("cuda")
        

        patch_features = self.vision_backbone(pixel_value)
        projected_patch_embeddings = self.projector(patch_features)
        input_embeddings[1 :257] = (
            projected_patch_embeddings
        )
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeddings,
        )

EntryClass = OpenVLAForActionPrediction
