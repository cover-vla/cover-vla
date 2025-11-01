import gc
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from lerobot.common.policies.pretrained import PreTrainedPolicy
from typing_extensions import override

sys.path.append(str(Path(__file__).parent / "../../../")) # experiments
sys.path.append(str(Path(__file__).parent / "../../../..")) # src

from src.agent.configuration_pipeline import TrainPipelineConfig
from src.utils.pipeline import get_class_from_path, set_seed_everywhere


class BasePolicyWrapper:
    def __init__(self,
                 pipeline_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        '''
        Initializes

            pipeline_cfg: TrainPipelineConfig, a dataclass containing all configurations
                required for the training pipeline, including evaluation and model settings.
            model_class: The class of the model to be evaluated. This is not an instance
                of the model, but the class itself, which is used for loading the model.
        '''

        self.pipeline_cfg = pipeline_cfg
        self.eval_cfg = pipeline_cfg.eval_cfg
        self.model_cfg = pipeline_cfg.model_cfg
        self.model_class = model_class

        # Policy
        self.action_step = self.eval_cfg.action_step

        self.port = pipeline_cfg.eval_cfg.port
        self.host = pipeline_cfg.eval_cfg.host

        # Debugging
        self.debug = pipeline_cfg.debug

        # Seeding
        self.seed = pipeline_cfg.seed
        set_seed_everywhere(self.seed, train=False)

        # Devices
        self.gpu_id = pipeline_cfg.gpu_id
        self.device = torch.device("cuda")

        # Model parameters
        self.use_amp = pipeline_cfg.use_amp
        self.dtype = torch.bfloat16 if pipeline_cfg.use_bf16 else torch.float32
        self.use_torch_compile = pipeline_cfg.use_torch_compile

        self.model = None
        self.env_adapter = None

    def switch_model(self, new_model_path):
        """Switch the model to a new checkpoint."""
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self._initialze_model_server(new_model_path)
        self.env_adapter = self._initialize_env_adapter()

    def select_action(self, element: dict):
        """Select action from the observation."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def reset(self):
        """Reset the model and the env adapter during evaluation."""
        if self.model is None:
            raise ValueError("Model is not initialized. Please initialize model first.")
        if self.env_adapter is None:
            raise ValueError("Env adapter is not initialized. Please initialize env adapter first. env adapter is usually initialized when you initialize the model.")

        if hasattr(self.model, "reset") and callable(self.model.reset):
            self.model.reset()

        if hasattr(self.env_adapter, "reset") and callable(self.env_adapter.reset):
            self.env_adapter.reset()

    def _initialze_model_server(self, model_path):
        '''On the server side, initialize the model checkpoint from a path'''
        self.model_path = model_path

        # Load model
        model = self._load_checkpoint(model_class=self.model_class,
                                      freeze_lm_head=self.pipeline_cfg.freeze_lm_head,
                                     )
        model = model.to(self.dtype)
        model = model.to(self.device)

        # Apply torch compile if requested
        if self.use_torch_compile:
            self.model = torch.compile(model, mode="default")
        else:
            self.model = model

        self.model.eval()

    def _initialize_env_adapter(self):
        '''Initialize the environment adapter based on task'''
        adapter_class = get_class_from_path(self.eval_cfg.env_adapter_path)
        return adapter_class(config=self.pipeline_cfg)

    def _load_checkpoint(self, **kwargs) -> torch.nn.Module:

        '''Load model checkpoint from the specified path'''
        raise NotImplementedError("This method should be implemented in subclasses.")

class LeRobotPolicyWrapper(BasePolicyWrapper):
    def __init__(self,
                 pipeline_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        super().__init__(pipeline_cfg=pipeline_cfg, model_class=model_class)

    @override
    def select_action(self, element: dict):
        """Select action from the observation."""
        element = self.env_adapter.preprocess(element)
        element = {
                k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in element.items()
                } # instruction is text, not tensor
        # Lerobot's implementation produces 1 action at a time
        with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.use_amp):
            with torch.inference_mode():
                actions = torch.stack(
                    [self.model.select_action(element) for _ in range(self.action_step)], dim=1
                ) # [batch, action_step, action_dim]

        actions = actions[0] if actions.shape[0] == 1 else actions # if not batch inference, then get rid of batch dim

        env_actions = self.env_adapter.postprocess(actions.float().cpu().numpy())
        return env_actions

    @override
    def _load_checkpoint(self, **kwargs) -> torch.nn.Module:

        '''Load model checkpoint from the specified path'''
        model_class = kwargs.get("model_class", None)
        freeze_lm_head = kwargs.get("freeze_lm_head", False)
        strict = kwargs.get("strict", False)

        # TODO: this ad-hoc loading is not very good. We should do something about it in the future.
        if self.model_cfg.type != "pi0" and self.model_cfg.type != "pi0fast":
            model = model_class(config=self.model_cfg, dataset_stats=self.pipeline_cfg.data.dataset_stats)
            model.resize_token_embedding()
            model.register_special_tokens()
            model = model._load_as_safetensor(model=model,
                                              model_file=self.model_path + "/model.safetensors",
                                              map_location='cpu',
                                              strict=strict) #load to cpu by default
        else:
            model = model_class.from_pretrained(
                pretrained_name_or_path=self.model_path,
                config=self.model_cfg
            )

        if freeze_lm_head:
            if self.model_cfg.type == "pi0":
                # Remove unnecessary language model heads
                model.model.paligemma_with_expert.gemma_expert.lm_head = None
                model.model.paligemma_with_expert.paligemma.language_model.lm_head = None

        return model

class HFPolicyWrapper(BasePolicyWrapper):
    def __init__(self,
                 pipeline_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        super().__init__(pipeline_cfg=pipeline_cfg, model_class=model_class)

    @override
    def _initialze_model_server(self, model_path: str):
        '''On the server side, initialize the model checkpoint from HF AutoModel and also a processor'''
        super()._initialze_model_server(model_path=model_path)

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)


class SpatialVLAPolicyWrapper(HFPolicyWrapper):
    def __init__(self,
                 pipeline_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        super().__init__(pipeline_cfg=pipeline_cfg, model_class=model_class)
        self.unnorm_key = self.pipeline_cfg.eval_cfg.unnorm_key # only used for OpenVLA-like HF models

    @override
    def select_action(self, element: dict):
        """Select action from the observation."""
        element = self.env_adapter.preprocess(element)
        element = {
                k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in element.items()
                } # instruction is text, not tensor

        with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.use_amp):
            with torch.inference_mode():
                inputs = self.processor(images=element['observation.images.top'], text=element['task'], return_tensors="pt", do_normalize=False)
                generation_outputs = self.model.predict_action(inputs)

                actions = self.processor.decode_actions(generation_outputs, unnorm_key=self.unnorm_key)

        actions = actions['actions']
        if isinstance(actions, torch.Tensor):
            actions = actions.float().cpu().numpy()
        actions = actions[0] # remove batch, becoming (action_pred_horizon, action_dim)
        env_actions = self.env_adapter.postprocess(actions)
        return env_actions

    @override
    def _load_checkpoint(self, **kwargs):
        '''Load model checkpoint with AutoModel'''
        from transformers import AutoModel
        model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)

        return model

class MagmaPolicyWrapper(HFPolicyWrapper):
    def __init__(self,
                 pipeline_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        super().__init__(pipeline_cfg=pipeline_cfg, model_class=model_class)

        # specific to Magma for action decoding
        self.n_action_bins = 256
        self.bins = np.linspace(-1, 1, self.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    @override
    def select_action(self, element: dict):
        """Select action from the observation."""
        element = self.env_adapter.preprocess(element)
        prompt = self._get_magma_prompt(element['task'])
        inputs = self.processor(images=element['observation.images.top'], texts=prompt, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
        inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
        inputs = inputs.to(self.device).to(self.dtype)

        self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                temperature=0.7,
                do_sample=False,
                num_beams=1,
                max_new_tokens=1000,
                use_cache=False,
            )
            action_ids = output_ids[0, -8:-1].cpu().tolist()


        predicted_action_ids = np.array(action_ids).astype(np.int64)
        discretized_actions = self.processor.tokenizer.vocab_size - predicted_action_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1,
        )
        normalized_actions = self.bin_centers[discretized_actions]

        env_actions = self.env_adapter.postprocess(normalized_actions) # env_action is unnormalized, without batch dim
        return env_actions

    def _get_magma_prompt(self, prompt: str):
        convs = [
            {"role": "system", "content": "You are agent that can see, talk and act."},
            {"role": "user", "content": f"<image>\nWhat action should the robot take to {prompt}?"},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        if self.model.config.mm_use_image_start_end:
            prompt = prompt.replace("<image>", "<image_start><image><image_end>")

        return prompt

    @override
    def _load_checkpoint(self, **kwargs):
        '''Load model checkpoint with AutoModel'''
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                     trust_remote_code=True,
                                                     device_map=self.device,
                                                     attn_implementation="flash_attention_2",
                                                     torch_dtype=self.dtype,
                                                     low_cpu_mem_usage=True)

        return model


class OctoPolicyWrapper(BasePolicyWrapper):
    def __init__(self,
                 pipeline_cfg: TrainPipelineConfig,
                 model_class: PreTrainedPolicy):
        super().__init__(pipeline_cfg=pipeline_cfg, model_class=model_class)
        import jax
        self.jax = jax
        self.obs_steps = pipeline_cfg.model_cfg.n_obs_steps
        self.image_history = deque(maxlen=self.obs_steps)
        # self.rng = jax.random.PRNGKey(self.seed)
        self.num_image_history = 0
        self.task = None # this is a Jax object of a string

    def reset(self):
        super().reset()
        self.image_history.clear() # octo has observation history, so we need to clear it
        self.task = None # reset task to None, so that we can set it again in select_action

    @override
    def select_action(self, element: dict):
        """Select action from the observation."""
        element = self.env_adapter.preprocess(element)

        ##### Basically more preprocessing. TODO: Maybe in the future these can be moved into the env adapter
        # Image preprocessing
        self._add_image_to_history(element['observation.images.top'])
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]
        input_observation = {"image_primary": images, "timestep_pad_mask": pad_mask}

        # RNG preprocessing
        # according to original authors, need to use a different rng key for each model forward step; this has a large impact on model performance
        # self.rng, key = self.jax.random.split(self.rng)  # each shape [2,]

        # Task language preprocessing
        if self.task is None:
            self.task = self.model.create_tasks(texts=[element['task']])

        raw_actions = self.model.sample_actions(
            input_observation,
            self.task,
            rng=self.jax.random.PRNGKey(self.seed),
        )
        raw_actions = raw_actions[0]  # remove batch, becoming (action_pred_horizon, action_dim)
        env_actions = self.env_adapter.postprocess(raw_actions)
        return env_actions


    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.obs_steps)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=np.float64)  # note: this should be of float type, not a bool type
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0

        return images, pad_mask

    @override
    def _initialze_model_server(self, model_path: str):
        '''On the server side, initialize the model checkpoint from Octo (Jax)'''
        self.model_path = f'hf://{self.eval_cfg.pretrained_model_path}'

        # Note that Octo is a Jax model, so no compile, no torch dtype, no amp etc.
        self.model = self._load_checkpoint()


    @override
    def _load_checkpoint(self, **kwargs):
        '''Load model checkpoint from the specified path'''
        from octo.model.octo_model import OctoModel
        model = OctoModel.load_pretrained(self.model_path)

        return model
