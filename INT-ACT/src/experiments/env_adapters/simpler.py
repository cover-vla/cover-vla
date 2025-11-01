import json
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image
from typing_extensions import override

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.experiments.env_adapters.base import BaseEnvAdapter
from src.utils.geometry import euler2axangle, mat2euler, quat2euler, quat2mat
from src.utils.pipeline import process_images


class SimplerAdapter(BaseEnvAdapter):
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
        state = torch.as_tensor(proprio, dtype=self.dtype).unsqueeze(0)

        return {
            "observation.images.top": images,
            "observation.state": state,
            "task": [obs["task"]],
        }
        
    def postprocess_verifier(self, actions: np.array) -> np.array:
        raw_actions_except_gripper = self.denormalize_bound(
                actions[:, :-1],
                np.array(self.dataset_statistics["action"]["p01"])[:-1],
                np.array(self.dataset_statistics["action"]["p99"])[:-1],
                clip_min=-1,
                clip_max=1,
            )
        raw_actions = np.concatenate(
            [
                raw_actions_except_gripper,
                actions[:, -1:],
            ],
            axis=1,
        )
        # prepare for simpler env
        actions = np.zeros((len(raw_actions), 7))  # chunk
        for idx, raw_action in enumerate(raw_actions):
            action_gripper = self.postprocess_gripper_verifier(raw_action[-1])
            actions[idx] = np.concatenate(
                [
                    raw_action[:6],
                    [action_gripper],
                ]
            )
        return actions

    def postprocess(
        self,
        actions: np.array,
    ):

        # gripper action is not normalized in training dataset
        if self.action_normalization_type == "bound":
            raw_actions_except_gripper = self.denormalize_bound(
                actions[:, :-1],
                np.array(self.dataset_statistics["action"]["p01"])[:-1],
                np.array(self.dataset_statistics["action"]["p99"])[:-1],
                clip_min=-1,
                clip_max=1,
            )
        elif self.action_normalization_type == "gaussian":
            raw_actions_except_gripper = self.denormalize_gaussian(
                actions[:, :-1],
                np.array(self.dataset_statistics["action"]["mean"])[:-1],
                np.array(self.dataset_statistics["action"]["std"])[:-1],
            )
        raw_actions = np.concatenate(
            [
                raw_actions_except_gripper,
                actions[:, -1:],
            ],
            axis=1,
        )

        # prepare for simpler env
        actions = np.zeros((len(raw_actions), 7))  # chunk
        for idx, raw_action in enumerate(raw_actions):
            roll, pitch, yaw = raw_action[3:6]
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_gripper = self.postprocess_gripper(raw_action[-1])

            actions[idx] = np.concatenate(
                [
                    raw_action[:3],
                    action_rotation_ax * action_rotation_angle,
                    [action_gripper],
                ]
            )

        return actions

    def preprocess_proprio(self, obs: dict) -> np.array:
        raise NotImplementedError

    def postprocess_gripper(self, action: float) -> float:
        raise NotImplementedError
    
    def postprocess_gripper_verifier(self, action: float) -> float:
        raise NotImplementedError

    def postprocess_action(self, actions: np.array) -> np.array:
        """Postprocess the action to fit the simpler env's executation convention."""
        raise NotImplementedError(
            "Postprocess action is not implemented for this adapter. Please implement it in the child class."
        )


class BridgeSimplerAdapter(SimplerAdapter):
    def __init__(self, config):
        super().__init__(config)

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

    def postprocess_gripper(self, action: float, binarize: bool = False) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 close, 1 open for simpler. This is not the case for other simulators like libero
        action_gripper = 2.0 * (action > 0.5) - 1.0

        if binarize:
            action_gripper = np.sign(action_gripper)

        return action_gripper
    
    def postprocess_gripper_verifier(self, action: float) -> float:
        # trained with [0, 1], 0 for close, 1 for open
        # convert to 0 for close, 1 for open for verifier. 
        action_gripper = 0 if action < 0.5 else 1
        return action_gripper


class BridgeSimplerSpatialVLAAdapter(BridgeSimplerAdapter):
    '''
    Adapter for the Bridge environment using the SpatialVLA model.
    No normalization for action because the model itself applies normalization using norm_key
    '''
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config.model_cfg, "action_ensemble_temp"):
            self.ensembler = ActionEnsembler(pred_action_horizon=config.model_cfg.chunk_size,
                                            action_ensemble_temp=config.model_cfg.action_ensemble_temp)

    def reset(self):
        super().reset()
        if hasattr(self, "ensembler"):
            self.ensembler.reset()

    @override
    def preprocess(
        self,
        obs: dict,
    ) -> dict:
        '''
        OpenVLA-like model has no proprio input, only image input.
        '''

        images = cv2.resize(
            obs['observation.images.top'],
            self.image_size,
            interpolation=cv2.INTER_AREA,
        )

        images = [Image.fromarray(images).convert("RGB")] # SpatialVLA requires PIL and in a []

        return {
            "observation.images.top": images,
            "task": obs["task"],
        }

    @override
    def postprocess(
        self,
        actions: np.array,
    ):
        raw_actions = actions.copy()
        ensemble_action = self.ensembler.ensemble_action(raw_actions)[None]
        return self.postprocess_action(ensemble_action)

    @override
    def postprocess_action(
        self,
        actions: np.array,
        gripper_binarize: bool = False
    ) -> list[np.array]:
        """Postprocess the action to fit the simpler env's executation convention."""
        raw_action = {
            "world_vector": np.array(actions[0, :3]),
            "rotation_delta": np.array(actions[0, 3:6]),
            "open_gripper": np.array(
                actions[0, 6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"]
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle

        action["gripper"] = self.postprocess_gripper(action=raw_action["open_gripper"], binarize=gripper_binarize)


        return [np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]]
            )]


class BridgeSimplerMagmaAdapter(BridgeSimplerSpatialVLAAdapter):
    def __init__(self, config):
        super().__init__(config)

    @override
    def preprocess(
        self,
        obs: dict,
    ) -> dict:
        '''
        Magma has no proprio input, only image input.
        '''
        images = Image.fromarray(obs['observation.images.top']) # Magma requires PIL
        images = images.resize(self.image_size)
        return {
            "observation.images.top": images,
            "task": obs["task"],
        }

    @override
    def postprocess(self, normalized_actions: np.array):
        # unnormalize actions
        mask = self.dataset_statistics.get("mask", np.ones_like(self.dataset_statistics["action"]["p01"], dtype=bool))
        action_high, action_low = np.array(self.dataset_statistics["action"]["p99"]), np.array(self.dataset_statistics["action"]["p01"])
        raw_action = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return self.postprocess_action(np.expand_dims(raw_action, axis=0), gripper_binarize=True) # convert to the rotation format used in the simpler envs and add gripper

class BridgeSimplerOctoAdapter(BridgeSimplerSpatialVLAAdapter):
    def __init__(self, config):
        super().__init__(config)

    @override
    def preprocess(
        self,
        obs: dict,
    ) -> dict:
        '''
        Octo has no proprio input, only image input.
        '''

        # I don't see how this is different from cv2 resize, but to follow the original code, using tf
        import tensorflow as tf
        images = tf.image.resize(
            obs['observation.images.top'],
            size=self.image_size,
            method="lanczos3",
            antialias=True,
        )

        images = tf.cast(tf.clip_by_value(tf.round(images), 0, 255), tf.uint8).numpy()

        return {
            "observation.images.top": images,
            "task": obs["task"],
        }

    @override
    def postprocess(
        self,
        actions: np.array,
    ):

        if self.action_normalization_type == "bound":
            raise NotImplementedError("Action normalization type 'bound' not supported for Octo")
        elif self.action_normalization_type == "gaussian":
            raw_actions_except_gripper = self.denormalize_gaussian(
                actions[:, :-1],
                np.array(self.dataset_statistics["action"]["mean"])[:-1],
                np.array(self.dataset_statistics["action"]["std"])[:-1],
            )
        raw_actions = np.concatenate(
            [
                raw_actions_except_gripper,
                actions[:, -1:],
            ],
            axis=1,
        )
        return super().postprocess(raw_actions)


class EDRSimplerAdapter(SimplerAdapter):
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


class EDREulerSimplerAdapter(SimplerAdapter):
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

class ActionEnsembler:
    '''
    Used by OpenVLA-like models to ensemble actions over a prediction horizon.
    '''
    def __init__(self, pred_action_horizon, action_ensemble_temp=-0.8):
        self.pred_action_horizon = pred_action_horizon
        self.action_ensemble_temp = action_ensemble_temp
        self.action_history = deque(maxlen=self.pred_action_horizon)

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action) -> np.array:
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history, strict=False)]
            )
        # if temp > 0, more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.action_ensemble_temp * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action
