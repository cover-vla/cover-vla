from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("magma")
@dataclass
class MagmaConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 4

    def validate_features(self) -> None:
        pass

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
