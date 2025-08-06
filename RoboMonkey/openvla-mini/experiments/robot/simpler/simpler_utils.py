import numpy as np
import simpler_env
import tensorflow as tf
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transforms3d.euler import euler2axangle

from experiments.robot.robot_utils import normalize_gripper_action
from PIL import Image
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from pathlib import Path

def process_image(image_path, output_dir="./transfer_images/", crop_scale=0.9, target_size=(224, 224), batch_size=1):
    """
    Process an image by center-cropping and resizing using TensorFlow.
    """
    def crop_and_resize(image, crop_scale, batch_size, target_size):
        """
        Center-crops an image and resizes it back to target size.
        """
        # Handle input dimensions
        if image.shape.ndims == 3:
            image = tf.expand_dims(image, axis=0)
            expanded_dims = True
        else:
            expanded_dims = False

        # Calculate crop dimensions
        new_scale = tf.reshape(
            tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), 
            shape=(batch_size,)
        )
        
        # Calculate bounding box
        offsets = (1 - new_scale) / 2
        bounding_boxes = tf.stack(
            [
                offsets,          # height offset
                offsets,          # width offset
                offsets + new_scale,  # height + offset
                offsets + new_scale   # width + offset
            ],
            axis=1
        )

        # Perform crop and resize
        image = tf.image.crop_and_resize(
            image, 
            bounding_boxes, 
            tf.range(batch_size), 
            target_size
        )

        # Remove batch dimension if input was 3D
        if expanded_dims:
            image = image[0]

        return image

    try:
        # Load and convert image to tensor
        image = Image.open(image_path)
        image = image.convert("RGB")

        current_size = image.size  # Returns (width, height)
        
        # Check if current size matches target size
        if current_size == (target_size[1], target_size[0]):
            return image_path
            
        image = tf.convert_to_tensor(np.array(image))
        
        # Store original dtype
        original_dtype = image.dtype

        # Convert to float32 [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Apply transformations
        image = crop_and_resize(image, crop_scale, batch_size, target_size)

        # Convert back to original dtype
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, original_dtype, saturate=True)

        # Convert to PIL Image and save
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

        image.save(image_path)

        return None

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def save_reward_img(image):
    # Encode as JPEG, as done in RLDS dataset builder
    image = tf.image.encode_jpeg(image)  
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8) 
    image = tf.image.resize(
        image, (256, 256), method="lanczos3", antialias=True
    )
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    image = tf.io.encode_jpeg(image, quality=95)

    # Encode again as done in dataset builder
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(
        image, (256, 256), method="lanczos3", antialias=True
    )
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    image = image.numpy()

    transfer_root = str(Path("./transfer_images/").absolute())
    os.makedirs(transfer_root, exist_ok=True)
    Image.fromarray(image).save(f"{transfer_root}/reward_img.jpg")

    # Resize down to 224x224
    process_image(
        f"{transfer_root}/reward_img.jpg",
        output_dir="./output/",
        crop_scale=0.9,
        target_size=(224, 224),
        batch_size=1
    )

def get_simpler_img(env, obs, resize_size):
    """
    Takes in environment and observation and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, int)
    image = get_image_from_maniskill2_obs_dict(env, obs)
    save_reward_img(image) 

    # Preprocess the image the exact same way that the Berkeley Bridge folks did it
    # to minimize distribution shift.
    # NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
    # resized up to a different resolution by some models. This is just so that we're in-distribution
    # w.r.t. the original preprocessing at train time.
    IMAGE_BASE_PREPROCESS_SIZE = 128
    # Resize to image size expected by model
    image = tf.image.encode_jpeg(image)  # Encode as JPEG, as done in RLDS dataset builder
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    image = tf.image.resize(
        image, (IMAGE_BASE_PREPROCESS_SIZE, IMAGE_BASE_PREPROCESS_SIZE), method="lanczos3", antialias=True
    )
    image = tf.image.resize(image, (resize_size, resize_size), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image.numpy()


def get_simpler_env(task, model_family):
    """Initializes and returns the Simpler environment along with the task description."""
    env = simpler_env.make(task)
    return env


def get_simpler_dummy_action(model_family: str):
    if model_family == "octo":
        # TODO: don't hardcode the action horizon for Octo
        return np.tile(np.array([0, 0, 0, 0, 0, 0, -1])[None], (4, 1))
    else:
        return np.array([0, 0, 0, 0, 0, 0, -1])


def convert_maniskill(action):
    """
    Applies transforms to raw VLA action that Maniskill simpler_env env expects.
    Converts rotation to axis_angle.
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1] and binarizes.
    """
    assert action.shape[0] == 7

    # Change rotation to axis-angle
    action = action.copy()
    roll, pitch, yaw = action[3], action[4], action[5]
    action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
    action[3:6] = action_rotation_ax * action_rotation_angle

    # Binarize final gripper dimension & map to [-1...1]
    return normalize_gripper_action(action)
