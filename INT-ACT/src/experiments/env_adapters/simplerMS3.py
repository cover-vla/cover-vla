import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.experiments.env_adapter.base import BaseEnvAdapter
from src.utils.geometry import euler2axangle, mat2euler, quat2mat
from src.utils.pipeline import process_images


class SimplerBatchAdapter(BaseEnvAdapter):
    def __init__(
        self,
        config
    ):
        super().__init__()
        env_config = config.env
        self.image_size = tuple(env_config.image_size)
        self.action_normalization_type = env_config.action_normalization_type
        self.state_normalization_type = env_config.state_normalization_type
        assert self.action_normalization_type in ["bound", "gaussian"]
        assert self.state_normalization_type in ["bound", "gaussian"]

        # for normalization
        with open(env_config.dataset_statistics_path, "r") as f:
            self.dataset_statistics = json.load(f)

        self.dtype = torch.bfloat16 if config.use_bf16 else torch.float32

        self.seed = config.seed

    def reset(self):
        pass

    def preprocess(
        self,
        obs: dict,
    ) -> dict:
        """using sxyz convention for euler angles"""

        # no normalization for image before processor
        # always on cpu
        # np_images = obs['observation.images.top']
        # print(f"np_images shape: {np_images.shape}")
        images = torch.as_tensor(obs['observation.images.top'], dtype=torch.uint8)  # [batch, h, w, 3]
        images = images.permute(0, 3, 1, 2)  # [batch, 3, h, w]

        # batched resize
        images = F.interpolate(images,
                               size=self.image_size,
                               mode='bilinear',
                               align_corners=False)

        images = process_images(
            images,
            rescale_factor=1 / 255.0,
            image_mean=torch.tensor([0.5, 0.5, 0.5]),
            image_std=torch.tensor([0.5, 0.5, 0.5]),
        ) # scale to [-1, 1]

        # process proprio depending on the robot
        raw_proprio = self.preprocess_proprio(obs["observation.state"])

        # normalize proprios - gripper opening is normalized
        if self.state_normalization_type == "bound":
            proprio = self.normalize_bound(
                raw_proprio,
                np.array(self.dataset_statistics["proprio"]["p01"]),
                np.array(self.dataset_statistics["proprio"]["p99"]),
                clip_min=-1,
                clip_max=1,
            )
        elif self.state_normalization_type == "gaussian":
            proprio = self.normalize_gaussian(
                raw_proprio,
                np.array(self.dataset_statistics["proprio"]["mean"]),
                np.array(self.dataset_statistics["proprio"]["std"]),
            )

        # convert state to tensor so model can use it
        # [batch, dim]
        state = torch.as_tensor(proprio, dtype=self.dtype)

        return {
            "observation.images.top": images,
            "observation.state": state,
            "task": obs["task"],
        }

    def postprocess(
        self,
        actions: np.array,  # now shape [batch, action_step, action_dim]
    ):

        # Process normalization for all timesteps for actions except the gripper
        if self.action_normalization_type == "bound":
            raw_actions_except_gripper = self.denormalize_bound(
                actions[:, :, :-1],
                np.array(self.dataset_statistics["action"]["p01"])[:-1],
                np.array(self.dataset_statistics["action"]["p99"])[:-1],
                clip_min=-1,
                clip_max=1,
            )
        elif self.action_normalization_type == "gaussian":
            raw_actions_except_gripper = self.denormalize_gaussian(
                actions[:, :, :-1],
                np.array(self.dataset_statistics["action"]["mean"])[:-1],
                np.array(self.dataset_statistics["action"]["std"])[:-1],
            )

        # Process each action step for every batch sample.
        processed_actions = []
        for _, (raw_batch, raw_batch_except) in enumerate(zip(actions, raw_actions_except_gripper, strict=False)):
            batch_processed = []
            for action_full, action_except in zip(raw_batch, raw_batch_except, strict=False):
                # Convert Euler angles (roll, pitch, yaw) to axis-angle representation.
                roll, pitch, yaw = action_except[3:6]
                action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
                rotation_vector = action_rotation_ax * action_rotation_angle
                # Process gripper action using the original raw action.
                action_gripper = self.postprocess_gripper(action_full[-1])
                # Concatenate translation, rotation vector, and gripper.
                processed = np.concatenate([action_except[:3], rotation_vector, [action_gripper]])
                batch_processed.append(processed)
            processed_actions.append(np.stack(batch_processed, axis=0))

        return np.array(processed_actions)

    def preprocess_proprio(self, batch_proprio: np.array) -> np.array:
        raise NotImplementedError

    def postprocess_gripper(self, action: float) -> float:
        raise NotImplementedError


class BridgeSimplerBatchAdapter(SimplerBatchAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203

    def reset(self):
        super().reset()

    def preprocess_proprio(self, batch_proprio: np.array) -> np.array:
        # expect batched eef_pos of shape [batch, 8]
        # batch_proprio = obs["agent"]["eef_pos"]
        processed = []
        for single_prop in batch_proprio:
            pos = single_prop[:3]
            quat = single_prop[3:7]
            rm_bridge = quat2mat(quat)
            # apply default_rot for each sample
            rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
            gripper_openness = single_prop[7]
            processed.append(np.concatenate([pos, rpy_bridge_converted, [gripper_openness]]))
        return np.stack(processed, axis=0)

    def postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 close, 1 open for simpler
        action_gripper = 2.0 * (action > 0.5) - 1.0
        return action_gripper

