#!/usr/bin/env python

##################################################################
# Placeholder for SpatialVLA model
##################################################################

from lerobot.common.policies.pretrained import PreTrainedPolicy

from src.model.spatialvla.configuration_spatialvla import SpatialVLAConfig


class SpatialVLAPolicy(PreTrainedPolicy):
    """Wrapper class around FusionA model to train and run inference within LeRobot."""

    config_class = SpatialVLAConfig
    name = "spatial-vla"

    def __init__(
        self,
        config: SpatialVLAConfig,
        dataset_stats: dict[str, dict] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

