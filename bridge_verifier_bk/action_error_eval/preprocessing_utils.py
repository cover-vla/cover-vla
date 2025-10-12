"""
Preprocessing utilities for image processing.
Contains functions for creating 224x224 preprocessed images in different formats.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
from pathlib import Path


def process_image_robomonkey(image_path, output_path, crop_scale=0.9, target_size=(224, 224), batch_size=1):
    """
    Process an image by center-cropping and resizing using TensorFlow.
    Based on process_image function from simpler_utils.py
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save processed image
        crop_scale (float): Scale factor for center crop
        target_size (tuple): Target size (height, width)
        batch_size (int): Batch size for processing
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
            # If size matches, just copy the image to output path
            image.save(output_path)
            return output_path
            
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

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)

        return output_path

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def process_image_openvla(image_path, output_path, resize_size=224):
    """
    Process image using OpenVLA preprocessing approach.
    Based on get_simpler_img function from simpler_utils.py
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save processed image
        resize_size (int): Target size for the output image
    """
    try:
        # Load image
        image = Image.open(image_path)
        image = image.convert("RGB")
        
        # Convert to tensor
        image = tf.convert_to_tensor(np.array(image))
        
        # Preprocess the image the exact same way that the Berkeley Bridge folks did it
        # to minimize distribution shift.
        IMAGE_BASE_PREPROCESS_SIZE = 128
        
        # Resize to image size expected by model
        image = tf.image.encode_jpeg(image)  # Encode as JPEG, as done in RLDS dataset builder
        image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
        image = tf.image.resize(
            image, (IMAGE_BASE_PREPROCESS_SIZE, IMAGE_BASE_PREPROCESS_SIZE), method="lanczos3", antialias=True
        )
        image = tf.image.resize(image, (resize_size, resize_size), method="lanczos3", antialias=True)
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
        
        # Convert back to PIL and save
        image_np = image.numpy()
        image_pil = Image.fromarray(image_np)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_pil.save(output_path)
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error processing image with OpenVLA method: {str(e)}")


def preprocess_all_images(image_folder, output_folder):
    """
    Process all images in a folder to create both OpenVLA and RoboMonkey versions.
    
    Args:
        image_folder (str): Path to folder containing original images
        output_folder (str): Path to folder where processed images will be saved
    """
    # Create output directories
    openvla_folder = os.path.join(output_folder, "openvla")
    robomonkey_folder = os.path.join(output_folder, "robomonkey")
    
    os.makedirs(openvla_folder, exist_ok=True)
    os.makedirs(robomonkey_folder, exist_ok=True)
    
    # Process all jpg files in the image folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            # Extract base number from filename (e.g., "1_clip.jpg" -> "1")
            base_name = filename.split('_')[0]
            
            input_path = os.path.join(image_folder, filename)
            
            # Create output paths
            openvla_output = os.path.join(openvla_folder, f"{base_name}_openvla.jpg")
            robomonkey_output = os.path.join(robomonkey_folder, f"{base_name}_robomonkey.jpg")
            
            try:
                # Process with OpenVLA method
                process_image_openvla(input_path, openvla_output)
                print(f"Processed {filename} -> {base_name}_openvla.jpg")
                
                # Process with RoboMonkey method
                process_image_robomonkey(input_path, robomonkey_output)
                print(f"Processed {filename} -> {base_name}_robomonkey.jpg")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess images using OpenVLA and RoboMonkey methods')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to folder containing original images')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to folder where processed images will be saved')
    
    args = parser.parse_args()
    
    preprocess_all_images(args.image_folder, args.output_folder)
