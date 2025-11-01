#!/usr/bin/env python

##################################################################
# Placeholder for Octo model
##################################################################

from lerobot.common.policies.pretrained import PreTrainedPolicy

from src.model.octo.configuration_octo import OctoConfig


class OctoPolicy(PreTrainedPolicy):
    """Wrapper class around Octo model to train and run inference within LeRobot."""

    config_class = OctoConfig
    name = "octo"

    def __init__(
        self,
        config: OctoConfig,
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

