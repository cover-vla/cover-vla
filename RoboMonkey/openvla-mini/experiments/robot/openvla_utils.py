"""Utils for evaluating the OpenVLA policy."""

import math
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoProcessor
import imageio

from prismatic.models.load import load_vla

import requests
import json_numpy as json

import sys
sys.path.append("../")

from experiments.robot.token_action_converter import TokenActionConverter


def preprocess_actions(output_ids, action):
    # Convert arrays to numpy arrays if they aren't already
    output_ids = np.array(output_ids)
    output_ids = np.where(output_ids == 31745, 31744, output_ids)
    action = np.array(action)
    
    # Apply the range filter
    range_mask = np.all((output_ids >= 31744) & (output_ids <= 32000), axis=1)
    output_ids = output_ids[range_mask]
    action = action[range_mask]
    
    return output_ids, action

def get_unique_actions(output_ids, action):
    output_ids = np.array(output_ids)
    action = np.array(action)
    
    # Get unique rows and their indices
    unique_rows, indices = np.unique(output_ids, axis=0, return_index=True)
    
    # Sort indices to maintain original order
    indices = sorted(indices)
    
    return output_ids[indices], action[indices]

def get_rewards(instruction, image_path, actions, cfg):
    # Initialize rewards list
    all_rewards = []
    
    # Get action rewards in batches of 2, so the reward model fits in a RTX4090 with 24GB memory size
    # Change the `batch_size` accordingly if you are using a different GPU
    batch_size = 2
    num_batches = math.ceil(len(actions) / batch_size)
    
    for i in range(num_batches):
        # Get the current batch of actions
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(actions))
        action_batch = actions[start_idx:end_idx]
        
        payload = {
            "instruction": instruction,
            "image_path": image_path,
            "action": action_batch
        }
        
        response = requests.post(f"http://127.0.0.1:{cfg.reward_server_port}/process", data=json.dumps(payload))
        response_data = json.loads(response.text)
        
        all_rewards.extend(response_data["rewards"])
    
    return all_rewards

def get_batch_actions(instruction: str, image_path: str, batch_size: int = 4, temperature: float = 1.0, cfg = None):
    """
    Get multiple predictions by making individual requests to the processing server.
    """
    # Verify image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    payload = {
        "instruction": instruction,
        "image_path": image_path,
        "batch_size": 1,  # Always set to 1 for individual requests
        "temperature": temperature
    }
    
    all_output_ids = []
    all_actions = []
    
    # Make batch_size number of individual requests
    for _ in range(batch_size):
        # Send request to server
        response = requests.post(
            f"http://127.0.0.1:{cfg.action_server_port}/batch",
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            raise Exception(f"Error from server: {response.text}")
        
        response_data = json.loads(response.text)
        all_output_ids.extend(response_data["output_ids"])
        all_actions.extend(response_data["actions"])
    
    return np.array(all_output_ids), np.array(all_actions)

def generate_augmented_samples_from_batch(batch_actions, num_samples=32):
    """
    Generate augmented samples based on the mean and variance of a batch of actions.
    """
    # Calculate mean and variance for each dimension
    mean_values = np.mean(batch_actions, axis=0)
    var_values = np.var(batch_actions, axis=0)
    
    # Define valid ranges for the action dimensions
    min_values = np.array([-0.02872725307941437,
                         -0.04170349963009357,
                         -0.026093858778476715,
                         -0.08092105075716972,
                         -0.09288699507713317,
                         -0.20718276381492615,
                         0.0])
    max_values = np.array([0.028309678435325586,
                         0.040855254605412394,
                         0.040161586627364146,
                         0.08192047759890528,
                         0.07792850524187081,
                         0.20382574498653397,
                         1.0])
    converter = TokenActionConverter()
        
    # Generate all samples at once
    augmented_array = np.random.normal(
        mean_values, np.sqrt(var_values), 
        size=(num_samples, 7)
    )
    
    # For the 7th dimension (binary), use probability based on mean
    augmented_array[:, -1] = (mean_values[-1] >= 0.5).astype(float)
    
    # Clip values to valid range
    augmented_array[:, :-1] = np.clip(
        augmented_array[:, :-1], 
        min_values[:-1], 
        max_values[:-1]
    )
    
    augmented_ids = np.zeros((num_samples, 7), dtype=np.int64)
    for i in range(num_samples):
        augmented_ids[i] = converter.action_to_token(augmented_array[i])
    
    return augmented_ids, augmented_array

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path
    
def get_prismatic_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Prepare for model loading.
    print(f"[*] Initializing Generation Playground with `{cfg.model_family}`")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    # set_seed(cfg.seed)
    # Load VLA checkpoint.
    print(f"Loading VLM from checkpoint: {cfg.pretrained_checkpoint}")
    vla = load_vla(
        cfg.pretrained_checkpoint,
        hf_token=hf_token,
        load_for_training=False,
    )
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"
    # Cast to half precision.
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(DEVICE)
    return vla


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    return None


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def apply_center_crop(im, t_h, t_w):
    """
    Source: https://github.com/ARISE-Initiative/robomimic/blob/5dee58f9cc1235010d0877142b54d0e82dd23986/robomimic/utils/obs_utils.py#L268

    Takes a center crop of an image.

    Args:
        im (np.array or torch.Tensor): image of shape (..., height, width, channel)
        t_h (int): height of crop
        t_w (int): width of crop

    Returns:
        im (np.array or torch.Tensor): center cropped image
    """
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    assert im.shape[-1] in [1, 3, 6]
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]

#
def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, cfg=None):
    """Generates an action with the VLA policy."""

    # only supports 1 image
    if isinstance(obs["full_image"], list):
        obs["full_image"] = obs["full_image"][0]

    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

        # Save processed image and path for inference
        transfer_dir = f"./transfer_images/"
        os.makedirs(transfer_dir, exist_ok=True)
        image_path = f"{transfer_dir}/vla_processed_img.jpg"
        image.save(image_path)
    
    # Get initial action samples from VLA Serving Engine
    instruction = task_label.lower()
    image_path = str(Path("./transfer_images/vla_processed_img.jpg").absolute())
    output_ids, actions = get_batch_actions(
        instruction=instruction,
        image_path=image_path,
        batch_size=cfg.initial_samples,
        temperature=1,
        cfg=cfg
    )

    # Preprocess initial actions
    if cfg.initial_samples == 1 and cfg.augmented_samples == 1:
        return actions[0] 
    output_ids, actions = preprocess_actions(output_ids, actions)
    _, unique = get_unique_actions(output_ids, actions)
    if len(unique)==1:
        return unique[0]

    # Generate augmented samples based on the mean and variance of a batch of actions.
    output_ids, actions = generate_augmented_samples_from_batch(
        batch_actions=actions,
        num_samples=cfg.augmented_samples
    )

    # Score each action with robomonkey verifier
    output_ids, actions = get_unique_actions(output_ids, actions)
    reward_img = str(Path("./transfer_images/reward_img.jpg").absolute())
    rewards = get_rewards(instruction, reward_img, output_ids, cfg=cfg)

    selected_index = np.argmax(rewards)

    return actions[selected_index]


def get_prismatic_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, **kwargs):
    """Generates an action with the VLA policy."""

    if not isinstance(obs["full_image"], list):
        obs["full_image"] = [obs["full_image"]]

    processed_images = []

    for img in obs["full_image"]:
        image = Image.fromarray(img)
        image = image.convert("RGB")

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), we must multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if center_crop:
            temp_image = np.array(image)  # (H, W, C)
            crop_scale = 0.9
            sqrt_crop_scale = math.sqrt(crop_scale)
            temp_image_cropped = apply_center_crop(
                temp_image,
                t_h=int(sqrt_crop_scale * temp_image.shape[0]),
                t_w=int(sqrt_crop_scale * temp_image.shape[1]),
            )
            temp_image = Image.fromarray(temp_image_cropped)
            temp_image = temp_image.resize(
                image.size, Image.Resampling.BILINEAR
            )  # IMPORTANT: dlimp uses BILINEAR resize
            image = temp_image

        processed_images.append(image)

    # extract for single image
    if len(processed_images) == 1:
        processed_images = processed_images[0]

    action = vla.predict_action(processed_images, task_label, unnorm_key=unnorm_key, **kwargs)
    return action
