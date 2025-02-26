"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256, seed=0):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(model_family, transform_type, seed, task_seed, rollout_images, idx, success, task_description):
    date = DATE
    date_time = DATE_TIME
    processed_transformed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
    rollout_dir = f"/data/xilun/rollouts/{model_family}/{transform_type}/videos"
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = f"{rollout_dir}/{date_time}--episode={idx}--task_seed={task_seed}--seed={seed}--success={success}--task={processed_transformed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    
    return mp4_path


def save_data(model_family, transform_type, seed, task_seed, image_data, action_data, action_prob_data, text_embedding, idx, success, task_description):
    date = DATE
    date_time = DATE_TIME
    
    processed_transformed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
    image_dir = f"/data/xilun/rollouts/{model_family}/{transform_type}/images"
    os.makedirs(image_dir, exist_ok=True)
    npz_path = f"{image_dir}/{date_time}--episode={idx}--task_seed={task_seed}--seed={seed}--success={success}--task={processed_transformed_task_description}.npy"
    save_image_data = {f"episode={idx}": image_data}
    np.save(npz_path, save_image_data, allow_pickle=True)
    print(f"Saved image data at path {npz_path}")
    
    action_dir = f"/data/xilun/rollouts/{model_family}/{transform_type}/actions"
    os.makedirs(action_dir, exist_ok=True)
    npz_path = f"{action_dir}/{date_time}--episode={idx}--task_seed={task_seed}--seed={seed}--success={success}--task={processed_transformed_task_description}.npy"
    save_action_data = {f"episode={idx}": action_data}
    np.save(npz_path, save_action_data,allow_pickle=True)
    print(f"Saved action data at path {npz_path}")
    
    action_prob_dir = f"/data/xilun/rollouts/{model_family}/{transform_type}/action_probs"
    os.makedirs(action_prob_dir, exist_ok=True)
    npz_path = f"{action_prob_dir}/{date_time}--episode={idx}--task_seed={task_seed}--seed={seed}--success={success}--task={processed_transformed_task_description}.npy"
    save_action_prob_data = {f"episode={idx}": action_prob_data}
    np.save(npz_path, save_action_prob_data, allow_pickle=True)
    print(f"Saved action probability data at path {npz_path}")
    
    text_embedding_dir = f"/data/xilun/rollouts/{model_family}/{transform_type}/text_embeddings"
    os.makedirs(text_embedding_dir, exist_ok=True)
    npz_path = f"{text_embedding_dir}/{date_time}--episode={idx}--task_seed={task_seed}--seed={seed}--success={success}--task={processed_transformed_task_description}.npy"
    save_text_embedding_data = {f"episode={idx}": text_embedding}
    np.save(npz_path, save_text_embedding_data, allow_pickle=True)
    print(f"Saved text embedding data at path {npz_path}")
            
        

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
