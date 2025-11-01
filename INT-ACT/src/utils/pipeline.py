import importlib
import os
import random

import numpy as np
import torch

IMAGENET_STANDARD_MEAN = torch.tensor([0.5, 0.5, 0.5])
IMAGENET_STANDARD_STD = torch.tensor([0.5, 0.5, 0.5])

def set_seed_everywhere(seed: int, train: bool = True):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if train:
        import tensorflow as tf
        tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_class_from_path(class_path: str):
    '''
    class_path: str
        The full path to the class, including the module name and class name.
        For example: "my_module.MyClass"
    '''
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def rescale(
    image: torch.LongTensor,
    scale: float,
) -> torch.FloatTensor:
    rescaled_image = image * scale
    return rescaled_image


def normalize(
    image: torch.LongTensor,
    mean: torch.FloatTensor,
    std: torch.FloatTensor,
) -> torch.FloatTensor:
    assert image.ndim == 4, f"Expected 4D tensor, got {image.ndim}D tensor."
    assert (
        image.shape[1] == 3
    ), f"Expected 3 channels at axis 1, got {image.shape[1]} channels."
    mean = mean[None, :, None, None]  # add batch and spatial dimensions
    std = std[None, :, None, None]
    image = (image - mean) / std
    return image


def process_images(
    images: torch.LongTensor,
    rescale_factor: float,
    image_mean: torch.FloatTensor = IMAGENET_STANDARD_MEAN,
    image_std: torch.FloatTensor = IMAGENET_STANDARD_STD,
) -> torch.FloatTensor:
    # Rescale the pixel values to be in the range [0, 1]
    images = rescale(images, scale=rescale_factor)

    # Normalize the images to have mean 0 and standard deviation 1
    images = normalize(images, mean=image_mean, std=image_std)

    return images

def revert_processed_images(
    processed_image: torch.FloatTensor,
    image_mean: torch.FloatTensor = IMAGENET_STANDARD_MEAN,
    image_std: torch.FloatTensor = IMAGENET_STANDARD_STD,
    rescale_factor: float = 1/255.0
) -> torch.LongTensor:
    rescale_factor = torch.tensor(rescale_factor).to(processed_image.device)
    # Undo normalization
    mean = image_mean[None, :, None, None]  # Add batch and spatial dimensions
    mean = mean.to(processed_image.device)
    std = image_std[None, :, None, None]
    std = std.to(processed_image.device)
    image = processed_image * std + mean  # Convert back to [0,1]

    # Undo rescaling
    image = image / rescale_factor  # Convert back to [0,255]

    # Clip values to valid range and convert to integer
    image = torch.clamp(image, 0, 255).to(torch.uint8)

    return image
