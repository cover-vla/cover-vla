"""
Evaluation Utilities for SIMPLER Benchmark

Consolidated utility functions for running evaluations with PI0 and RoboMonkey verifier.
This module contains all helper functions for environment setup, action processing,
image processing, data loading, and logging.

For paper: "RoboMonkey: Improving Robot Manipulation through Language Instruction Verification"
"""

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Union

import imageio
import numpy as np
import simpler_env
import tensorflow as tf
import torch
from PIL import Image
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# Add INT-ACT to path for BridgeSimplerAdapter (only importing one class, no full install needed)
_vla_clip_root = Path(__file__).resolve().parents[5]  # Go up to vla-clip root
_int_act_path = _vla_clip_root / "INT-ACT"
if str(_int_act_path) not in sys.path:
    sys.path.append(str(_int_act_path))

from src.experiments.env_adapters.simpler import BridgeSimplerAdapter


# =========================================================================================
# Random Seed and Configuration
# =========================================================================================

def get_image_resize_size(cfg):
    """Get image resize size based on model family.
    
    Args:
        cfg: Configuration object
        
    Returns:
        int: Image size (224 for PI0 models)
    """
    return 224


def set_seed_everywhere(seed):
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =========================================================================================
# Environment Setup
# =========================================================================================

def get_simpler_env(task, model_family):
    """Initialize and return the SIMPLER environment.
    
    Args:
        task: Task object from SIMPLER benchmark
        model_family: Model family name (e.g., 'openvla')
        
    Returns:
        SIMPLER environment instance
    """
    env = simpler_env.make(task)
    return env


def get_simpler_dummy_action(model_family: str):
    """Get dummy action for environment stabilization.
    
    Args:
        model_family: Model family name
        
    Returns:
        np.ndarray: Dummy action vector
    """
    if model_family == "octo":
        # Octo uses action chunks
        return np.tile(np.array([0, 0, 0, 0, 0, 0, -1])[None], (4, 1))
    else:
        return np.array([0, 0, 0, 0, 0, 0, -1])


# =========================================================================================
# Action Processing
# =========================================================================================

def create_bridge_adapter_wrapper(action_ensemble_temp=-0.8):
    """Create a BridgeSimplerAdapter wrapper for action post-processing.
    
    This integrates the INT-ACT BridgeSimplerAdapter for consistent action handling.
    
    Args:
        action_ensemble_temp: Temperature for action ensembling 
                             (negative = more recent actions get more weight)
        
    Returns:
        BridgeSimplerAdapter instance
    """
    class EnvConfig:
        def __init__(self):
            # Use dynamic path relative to this file
            vla_clip_root = Path(__file__).resolve().parents[5]
            self.dataset_statistics_path = str(vla_clip_root / "INT-ACT" / "config" / "dataset" / "bridge_statistics.json")
            self.image_size = (224, 224)
            self.action_normalization_type = "bound"
            self.state_normalization_type = "bound"
    
    class ModelConfig:
        def __init__(self):
            self.chunk_size = 4
            self.action_ensemble_temp = action_ensemble_temp
    
    class Config:
        def __init__(self):
            self.env = EnvConfig()
            self.use_bf16 = False
            self.seed = 42
            self.model_cfg = ModelConfig()
    
    config = Config()
    adapter = BridgeSimplerAdapter(config)
    return adapter


def convert_maniskill_with_bridge_adapter(action, verifier_action=False, action_ensemble_temp=-0.8):
    """Use BridgeSimplerAdapter for proper action post-processing.
    
    This ensures consistency with INT-ACT's action processing pipeline.
    
    Args:
        action: Raw action from model
        verifier_action: Whether to use verifier-specific processing
        action_ensemble_temp: Temperature for action ensembling
        
    Returns:
        np.ndarray: Post-processed action
    """
    # Create adapter with unique key based on temperature (singleton pattern)
    adapter_key = f'_adapter_temp_{action_ensemble_temp}'
    if not hasattr(convert_maniskill_with_bridge_adapter, adapter_key):
        setattr(convert_maniskill_with_bridge_adapter, adapter_key, 
                create_bridge_adapter_wrapper(action_ensemble_temp))
    
    adapter = getattr(convert_maniskill_with_bridge_adapter, adapter_key)
    action_batch = action.reshape(1, -1)
    
    if verifier_action:
        verified_action = adapter.postprocess_verifier(action_batch)
        return verified_action[0]
    else:
        execution_action = adapter.postprocess(action_batch)
        return execution_action[0]


def process_inputs(batch_size, predefined_action_queue, verifier_action=False, 
                   action_history=[], cfg=None):
    """Process action queue into format suitable for verifier model.
    
    Converts raw actions from policy into trajectories that include both past actions 
    (from action_history) and future actions (from predefined_action_queue).
    
    Args:
        batch_size: Number of samples in batch
        predefined_action_queue: Queue of future actions from policy
        verifier_action: Whether to use verifier-specific action preprocessing
        action_history: List of past executed actions
        cfg: Configuration object containing n_action_steps
        
    Returns:
        list: List of action trajectories, one per batch item (batch_size, timesteps, 7)
    """
    processed_future_actions_batch = []
    
    # Process each future timestep
    for i in range(cfg.n_action_steps):
        single_action = predefined_action_queue[i].cpu().numpy()  # (batch_size, 7)
        processed_execution_actions_for_step = []
        
        for batch_idx in range(batch_size):
            sample_1x7 = single_action[batch_idx:batch_idx+1]  # (1, 7)
            processed_execution_1x7 = convert_maniskill_with_bridge_adapter(
                sample_1x7, verifier_action=verifier_action, action_ensemble_temp=-0.8
            )
            processed_execution_1x7 = np.asarray(processed_execution_1x7)
            processed_execution_actions_for_step.append(processed_execution_1x7)

        processed_batch = np.vstack(processed_execution_actions_for_step)
        processed_batch = processed_batch.reshape(batch_size, 7)
        processed_future_actions_batch.append(processed_batch)
    
    # Combine past and future actions
    num_past = min(len(action_history), 6)
    future_actions = np.stack(processed_future_actions_batch)  # (n_action_steps, batch_size, 7)
    future_actions_transposed = future_actions.transpose(1, 0, 2)  # (batch_size, n_action_steps, 7)
    
    if num_past > 0:
        past_actions = np.stack(action_history[-num_past:])
        past_actions = np.expand_dims(past_actions, axis=0).repeat(batch_size, axis=0)
        processed_full_trajectory = np.concatenate([past_actions, future_actions_transposed], axis=1)
    else:
        processed_full_trajectory = future_actions_transposed
    
    action_histories_list = [processed_full_trajectory[i] for i in range(batch_size)]
    return action_histories_list


# =========================================================================================
# Image Processing
# =========================================================================================

def process_raw_image_to_jpg(image: Union[str, Path, np.ndarray, tf.Tensor], 
                             max_res: int = 256) -> np.ndarray:
    """Process a raw image into JPG format following rlds_dataset_mod format.
    
    This function:
    1. Loads the image (if path is provided)
    2. Resizes it to max_res x max_res (default 256x256)
    3. Converts to uint8 format
    4. Returns the processed image as numpy array
    
    Args:
        image: Input image as file path (str/Path), numpy array, or TensorFlow tensor
        max_res: Maximum resolution for both width and height (default: 256)
    
    Returns:
        np.ndarray: Processed image (uint8, shape: [max_res, max_res, 3])
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        image = tf.io.read_file(str(image))
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
    
    # Convert numpy array to tensor
    if isinstance(image, np.ndarray):
        image = tf.convert_to_tensor(image)
    
    if not isinstance(image, tf.Tensor):
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Ensure 3D tensor [height, width, channels]
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)
        image = tf.repeat(image, 3, axis=-1)
    elif len(image.shape) != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape: {image.shape}")
    
    # Ensure 3 channels (RGB)
    if image.shape[-1] == 1:
        image = tf.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:  # RGBA to RGB
        image = image[..., :3]
    elif image.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got: {image.shape[-1]}")
    
    # Resize to max_res x max_res
    size = (max_res, max_res)
    image_resized = tf.image.resize(
        image,
        size,
        method=tf.image.ResizeMethod.BILINEAR,
        preserve_aspect_ratio=False,
        antialias=True,
    )
    
    # Cast to uint8
    image_resized = tf.cast(image_resized, tf.uint8)
    image_np = image_resized.numpy()
    
    return image_np


# =========================================================================================
# Data Loading
# =========================================================================================

def load_rephrases(task_suite_name: str):
    """Load pre-generated language rephrases for the task suite.
    
    Args:
        task_suite_name: Name of the task suite (e.g., 'simpler_widowx')
        
    Returns:
        dict: Dictionary mapping task descriptions to rephrased instructions
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'simpler_rephrased_final_eval_vlm.json')
    
    with open(json_path, 'r') as f:
        all_rephrases = json.load(f)
    
    return all_rephrases.get("instructions", {})


# =========================================================================================
# Logging and Saving
# =========================================================================================

def save_rollout_video_openpi(rollout_images, idx, success, task_description, transformation_type, 
                               model_name, lang_rephrase_num, policy_batch_inference_size, log_file=None):
    """Save an MP4 replay video of an episode.
    
    Args:
        rollout_images: List of RGB images from the episode
        idx: Episode index
        success: Whether the episode succeeded
        task_description: Task description string
        transformation_type: Type of language transformation applied
        model_name: Name of the model checkpoint
        lang_rephrase_num: Number of language rephrases used
        policy_batch_inference_size: Batch size for policy inference
        log_file: Optional file handle for logging
        
    Returns:
        str: Path to saved video file
    """
    if model_name == "juexzz/INTACT-pi0-finetune-rephrase-bridge":
        rollout_dir = f"./rollouts_openpi_rephrase/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    elif model_name == "juexzz/INTACT-pi0-finetune-bridge":
        rollout_dir = f"./rollouts_openpi_original/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
    
    mp4_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_episode_data_openpi(episode_data, idx, success, task_description, transformation_type, 
                              model_name, lang_rephrase_num, policy_batch_inference_size, log_file=None):
    """Save episode data (verifier scores, instructions, actions) to a pickle file.
    
    Args:
        episode_data: Dictionary containing episode information
        idx: Episode index
        success: Whether the episode succeeded
        task_description: Task description string
        transformation_type: Type of language transformation applied
        model_name: Name of the model checkpoint
        lang_rephrase_num: Number of language rephrases used
        policy_batch_inference_size: Batch size for policy inference
        log_file: Optional file handle for logging
        
    Returns:
        str: Path to saved pickle file
    """
    if model_name == "juexzz/INTACT-pi0-finetune-rephrase-bridge":
        rollout_dir = f"./rollouts_openpi_rephrase/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    elif model_name == "juexzz/INTACT-pi0-finetune-bridge":
        rollout_dir = f"./rollouts_openpi_original/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
    
    pkl_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}.pkl"
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(episode_data, f)
    
    print(f"Saved episode data at path {pkl_path}")
    if log_file is not None:
        log_file.write(f"Saved episode data at path {pkl_path}\n")
    return pkl_path

