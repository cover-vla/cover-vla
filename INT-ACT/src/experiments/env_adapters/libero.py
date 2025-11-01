import json
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.experiments.env_adapter.base import BaseEnvAdapter
from src.utils.geometry import mat2euler, quat2axisangle, quat2euler, quat2mat
from src.utils.pipeline import process_images


class LiberoAdapter(BaseEnvAdapter):
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

    def reset(self):
        pass

    def preprocess(
        self,
        obs: dict,
    ) -> dict:
        '''
        Mainly for resizing images, normalizing proprioception.
        It will also convert proprio from Libero' axis angle to whatever representation the model is using (will be defined by a subclass of this)
        obs: dict, observation from the environment
            - observation.images: image(s) from the environment
            - observation.state: proprioception from the environment
            - task: language instruction
        '''
        image = cv2.resize(
            obs['observation.images.top'],
            self.image_size,
            interpolation=cv2.INTER_LANCZOS4,
        )
        # no normalization for image before processor
        # always on cpu
        images = torch.as_tensor(image, dtype=torch.uint8).permute(2, 0, 1)[
            None
        ]  # [1, 3, H, W]
        images = process_images(
            images,
            rescale_factor=1 / 255.0,
            image_mean=torch.tensor([0.5, 0.5, 0.5]),
            image_std=torch.tensor([0.5, 0.5, 0.5]),
        ) # scale to [-1, 1]

        # process proprio depending on the robot
        raw_proprio = self.preprocess_proprio(obs)

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

        # [batch, dim]
        state = torch.as_tensor(proprio, dtype=self.dtype).unsqueeze(0)

        return {
            "observation.images.top": images,
            "observation.state": state,
            "task": [obs["task"]],
        }

    def postprocess(
        self,
        actions: np.array,
    ) -> np.array:
        '''
        Because of proper training data preprocessing, nothing needs to be done here.
        '''
        return actions

    def preprocess_proprio(self, obs: dict) -> np.array:
        # libero environment spit out wxyz quaternion. but libero data used axis angle
        proprio = obs["observation.state"]
        quat = proprio[3:7]
        # convert to axis angle
        axis_angle = quat2axisangle(quat)
        gripper_openness = proprio[7]
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                axis_angle,
                [gripper_openness],
            ]
        )
        return raw_proprio

    def postprocess_gripper(self, action: float) -> float:
        pass

    def preprocess_proprio_gripper(self, gripper_width) -> str:
        '''
        gripper_width: should be a size 2 tensor sliced from proprio

        libero's gripper state is represneted by two values, one position for each finger of the gripper
        At the fully open stage, this value is about 0.036 to 0.039 for the first and -0.036 to -0.039 for the second
        When the values go below this, it means the gripper is starting to close. Fully close stage should give you values around 0
        In reality, how small the value is depends on the object. From inspection, below 0.015 can be considered closed, but it depends on the size of objects
        '''
        if min(math.abs(gripper_width[0]), math.abs(gripper_width[1])) < 0.015:
            return 'closed'
        else:
            return 'open'

class TacoLiberoAdapter(LiberoAdapter):
    '''
    used when the model is trained on taco_play dataset
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess_proprio(self, obs: dict) -> np.array:
        proprio = obs["observation.state"]
        gripper_width = proprio[-2:] # libero has 2 grippers action at the end

        # in taco_play, -1 is closed, 1 is open
        if self.preprocess_proprio_gripper(gripper_width) == 'closed':
            gripper_closedness = -1
        else:
            gripper_closedness = 1


class BridgeSimplerAdapter():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203

    def reset(self):
        super().reset()

    def preprocess_proprio(self, obs: dict) -> np.array:
        # convert ee rotation to the frame of top-down
        proprio = obs["agent"]["eef_pos"]
        rm_bridge = quat2mat(proprio[3:7])
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7]
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper_openness],
            ]
        )
        return raw_proprio

    def postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 close, 1 open for simpler
        action_gripper = 2.0 * (action > 0.5) - 1.0
        return action_gripper


class EDRSimplerAdapter():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Constants
         # same used in Octo. Note this is for every individual action, not every action chunk. Control freq is 3Hz, so roughly sticky for 5 seconds.
        self.sticky_gripper_num_repeat = 15

    def reset(self):
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.previous_gripper_action = None
        super().reset()

    def preprocess_proprio(self, obs: dict) -> np.array:
        """convert wxyz quat from simpler to xyzw used in fractal"""
        quat_xyzw = np.roll(obs["agent"]["eef_pos"][3:7], -1)
        gripper_width = obs["agent"]["eef_pos"][
            7
        ]  # from simpler, 0 for close, 1 for open
        gripper_closedness = (
            1 - gripper_width
        )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
        raw_proprio = np.concatenate(
            (
                obs["agent"]["eef_pos"][:3],
                quat_xyzw,
                [gripper_closedness],
            )
        )
        return raw_proprio

    def postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L187"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 open, 1 close for simpler

        action = (action * 2) - 1  # [0, 1] -> [-1, 1] -1 close, 1 open

        # without sticky
        relative_gripper_action = -action
        # if self.previous_gripper_action is None:
        #     relative_gripper_action = -1  # open
        # else:
        #     relative_gripper_action = -action
        # self.previous_gripper_action = action

        # switch to sticky closing
        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action

        # sticky closing
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action

        # reaching maximum sticky
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0

        return relative_gripper_action


class EDREulerSimplerAdapter():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Constants
         # same used in Octo. Note this is for every individual action, not every action chunk. Control freq is 3Hz, so roughly sticky for 5 seconds.
        self.sticky_gripper_num_repeat = 15

    def reset(self):
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.previous_gripper_action = None
        super().reset()

    def preprocess_proprio(self, obs: dict) -> np.array:
        """convert wxyz quat from simpler to xyzw used in fractal"""
        proprio = obs["agent"]["eef_pos"]
        euler = quat2euler(proprio[3:7]) # wxyz quat to euler
        gripper_width = obs["agent"]["eef_pos"][
            7
        ]  # from simpler, 0 for close, 1 for open
        gripper_closedness = (
            1 - gripper_width
        )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
        raw_proprio = np.concatenate(
            (
                obs["agent"]["eef_pos"][:3],
                euler,
                [gripper_closedness],
            )
        )
        return raw_proprio

    def postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L187"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 open, 1 close for simpler

        action = (action * 2) - 1  # [0, 1] -> [-1, 1] -1 close, 1 open

        # without sticky
        relative_gripper_action = -action
        # if self.previous_gripper_action is None:
        #     relative_gripper_action = -1  # open
        # else:
        #     relative_gripper_action = -action
        # self.previous_gripper_action = action

        # switch to sticky closing
        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action

        # sticky closing
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action

        # reaching maximum sticky
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0

        return relative_gripper_action
