"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import pickle


from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


# def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
#     """Saves an MP4 replay of an episode."""
#     rollout_dir = f"./rollouts/{DATE}"
#     os.makedirs(rollout_dir, exist_ok=True)
#     processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
#     mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}.mp4"
#     video_writer = imageio.get_writer(mp4_path, fps=30)
#     for img in rollout_images:
#         video_writer.append_data(img)
#     video_writer.close()
#     print(f"Saved rollout MP4 at path {mp4_path}")
#     if log_file is not None:
#         log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
#     return mp4_path

def save_rollout_video(rollout_images, idx, success, transform_type,
                       task_description, log_file=None, score_list=None, 
                       action_list=None, clip_update_num=None,
                       oracle_scorer=False):
    
    """Saves an MP4 replay of an episode."""
    if oracle_scorer:
        rollout_dir = f"./rollouts_oracle/{transform_type}_{clip_update_num}"
    else:
        rollout_dir = f"./rollouts_hack/{transform_type}_{clip_update_num}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")

    # Calculate mean score
    mean_score = np.nanmean(score_list) if score_list else None

    # Format score string explicitly
    if mean_score is not None and not np.isnan(mean_score):
        # Use :.3f format specifier for 3 decimal places
        score_str = f"{mean_score:.3f}"
    else:
        # Handle None or NaN cases
        score_str = "None" # Or you could use "nan" if mean_score is np.nan

    # Use the formatted string in the filename
    mp4_path = f"{rollout_dir}/episode={idx}--success={success}--score={score_str}--task={processed_task_description}.mp4"
    data_path = f"{rollout_dir}/episode={idx}--success={success}--score={score_str}--task={processed_task_description}.pkl"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    if score_list is not None and action_list is not None:
        data = {
            "score_list": score_list,
            "action_list": action_list,
        }
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved data at path {data_path}")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
