"""
Extract 100 datapoints from bridge dataset with state, original instruction, 
last 9 history actions, and current ground truth action.
Uses diverse episode sampling across the full 20,000 episode dataset.
Ensures all instructions are unique and valid by skipping episodes with 
duplicate or invalid/garbage instructions (e.g., "bmbfbbfgjjg").
"""

import argparse
import os
import numpy as np
import json
import random
from tqdm import tqdm
from PIL import Image
import tensorflow_datasets as tfds
from preprocessing_utils import preprocess_all_images
import re


def is_valid_instruction(instruction):
    """
    Check if an instruction appears to be valid and not garbage text.
    
    Args:
        instruction (str): The instruction text to validate
        
    Returns:
        bool: True if instruction appears valid, False otherwise
    """
    if not instruction or not instruction.strip():
        return False
    
    instruction = instruction.strip()
    
    # Check minimum length
    if len(instruction) < 3:
        return False
    
    # Check if it's mostly alphabetic characters (allow some punctuation and numbers)
    alpha_count = sum(1 for c in instruction if c.isalpha())
    total_chars = len(instruction.replace(' ', ''))  # Exclude spaces from count
    
    if total_chars == 0:
        return False
    
    # Require at least 60% alphabetic characters
    alpha_ratio = alpha_count / total_chars
    if alpha_ratio < 0.6:
        return False
    
    # Check for obvious random character sequences
    # Look for sequences of 4+ consecutive consonants or random chars
    consonant_pattern = r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}'
    if re.search(consonant_pattern, instruction):
        return False
    
    # Check for repeated character patterns that suggest corruption
    repeated_char_pattern = r'(.)\1{3,}'  # Same character repeated 4+ times
    if re.search(repeated_char_pattern, instruction):
        return False
    
    # Require at least one space (multi-word instruction) or common single words
    common_single_words = {'go', 'stop', 'pick', 'put', 'move', 'turn', 'open', 'close', 'push', 'pull'}
    if ' ' not in instruction and instruction.lower() not in common_single_words:
        # Allow single words that are at least 3 chars and mostly vowels/consonants in reasonable pattern
        if len(instruction) < 3:
            return False
    
    return True


def extract_bridge_samples(builder_dir, num_samples=100, output_json="bridge_samples.json", 
                          images_folder="bridge_images", processed_images_folder="processed_images", seed=42):
    """
    Extract samples from Bridge V2 dataset - one sample per episode from a random timestep.
    Uses shuffled episode order for diverse sampling across the full dataset.
    Ensures all extracted samples have unique and valid instructions by skipping 
    duplicate instructions and invalid/garbage text.
    
    Args:
        builder_dir (str): Path to the Bridge V2 dataset directory
        num_samples (int): Number of samples to extract (diverse episodes with unique, valid instructions)
        output_json (str): Path to save the JSON file
        images_folder (str): Path to folder where original images will be saved
        processed_images_folder (str): Path to folder where processed images will be saved
        seed (int): Random seed for reproducible episode shuffling and timestep selection
    """
    
    # Set random seed for reproducible results
    random.seed(seed)
    np.random.seed(seed)
    
    # Create directories
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(processed_images_folder, exist_ok=True)
    
    print(f"Loading Bridge V2 dataset from {builder_dir}...")
    builder = tfds.builder_from_directory(builder_dir=builder_dir)
    
    samples = []
    image_counter = 0
    episodes_processed = 0
    unique_instructions = set()  # Track unique instructions
    skipped_duplicate_instructions = 0
    skipped_invalid_instructions = 0
    
    # Get dataset info
    info = builder.info
    total_episodes = info.splits['train'].num_examples
    print(f"Total episodes available: {total_episodes}")
    print(f"Extracting one sample per episode from random timesteps (seed={seed})")
    
    # Create a shuffled list of episode IDs for diverse sampling
    episode_ids = list(range(total_episodes))
    random.shuffle(episode_ids)
    print(f"Shuffled episode order for diverse sampling")
    
    with tqdm(total=num_samples, desc="Extracting samples") as pbar:
        episode_idx = 0
        
        while len(samples) < num_samples and episode_idx < len(episode_ids):
            episode_id = episode_ids[episode_idx]
            try:
                # Load single episode
                ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
                episode = next(iter(ds))
                
                # Extract language instruction and steps
                steps = list(episode["steps"])
                if not steps:
                    episode_idx += 1
                    continue
                
                # Get instruction (assumed same for all steps in episode)
                language_instruction = steps[0]["language_instruction"].numpy().decode()
                
                # Skip episodes with empty instructions
                if not language_instruction or not language_instruction.strip():
                    episode_idx += 1
                    continue
                
                # Check if instruction is valid (not garbage text)
                if not is_valid_instruction(language_instruction):
                    skipped_invalid_instructions += 1
                    episode_idx += 1
                    continue
                
                # Check if instruction is unique
                instruction_normalized = language_instruction.strip().lower()
                if instruction_normalized in unique_instructions:
                    skipped_duplicate_instructions += 1
                    episode_idx += 1
                    continue
                
                # Add instruction to unique set
                unique_instructions.add(instruction_normalized)
                
                # Extract actions and observations for all steps
                actions_list = []
                observations_list = []
                
                for step in steps:
                    action = step["action"].numpy()
                    observation = step["observation"]
                    
                    # Extract agent view image
                    agent_view_image = None
                    for img_key in ["image_0", "rgb", "image", "camera_0"]:
                        if img_key in observation:
                            agent_view_image = observation[img_key].numpy()
                            break
                    
                    if agent_view_image is None:
                        break
                    
                    # Process image
                    if agent_view_image.dtype != np.uint8:
                        if agent_view_image.max() <= 1.0:
                            agent_view_image = (agent_view_image * 255).astype(np.uint8)
                        else:
                            agent_view_image = agent_view_image.astype(np.uint8)
                    
                    # Handle RGBD (4 channels) by taking only RGB
                    if agent_view_image.shape[-1] == 4:
                        agent_view_image = agent_view_image[..., :3]
                    
                    actions_list.append(action)
                    observations_list.append({"agent_view_image": agent_view_image})
                
                if len(actions_list) == 0:
                    episode_idx += 1
                    continue
                
                # Convert to numpy array
                actions = np.array(actions_list)
                T = len(actions_list)
                
                # Select a random timestep from this episode
                t = random.randint(0, T - 1)
                
                # Get current ground truth action
                current_action = actions[t]
                
                # Get last 9 history actions (padding with zeros if needed)
                history_length = 9
                if t == 0:
                    # At first timestep, all history is padding
                    history_actions = np.zeros((history_length, actions.shape[1]), dtype=actions.dtype)
                else:
                    # Get available history
                    available_history = min(t, history_length)
                    start_idx = max(0, t - history_length)
                    
                    # Create history with padding if needed
                    history_actions = np.zeros((history_length, actions.shape[1]), dtype=actions.dtype)
                    actual_history = actions[start_idx:t]
                    
                    # Place actual history at the end of the padded array
                    history_actions[-available_history:] = actual_history
                
                # Save original image as {image_counter}_clip.jpg
                image_filename = f"{image_counter}_clip.jpg"
                image_path = os.path.join(images_folder, image_filename)
                Image.fromarray(observations_list[t]["agent_view_image"]).save(image_path, "JPEG", quality=95)
                
                # Get state (current observation - we'll use the image path as state reference)
                state = {
                    "agent_view_image_file": image_filename,
                    "timestep": t,
                    "episode_id": episode_id
                }
                
                # Create sample
                sample = {
                    "sample_id": len(samples),
                    "state": state,
                    "original_instruction": language_instruction.strip(),
                    "last_9_history_actions": history_actions.tolist(),
                    "current_ground_truth_action": current_action.tolist(),
                    "episode_id": episode_id,
                    "timestep": t
                }
                
                samples.append(sample)
                image_counter += 1
                pbar.update(1)
                
                # Update progress
                if len(samples) % 10 == 0:
                    pbar.set_postfix({
                        'Episodes': episodes_processed,
                        'Images': image_counter,
                        'Skipped_Dup': skipped_duplicate_instructions,
                        'Skipped_Invalid': skipped_invalid_instructions
                    })
                
                episodes_processed += 1
                episode_idx += 1
                
            except Exception as e:
                print(f"Error processing episode {episode_id}: {e}")
                episode_idx += 1
                continue
    
    print(f"\nExtracted {len(samples)} samples from {episodes_processed} episodes")
    print(f"Saved {image_counter} images to {images_folder}")
    print(f"Skipped {skipped_duplicate_instructions} episodes due to duplicate instructions")
    print(f"Skipped {skipped_invalid_instructions} episodes due to invalid/garbage instructions")
    print(f"All {len(samples)} instructions are unique and valid")
    
    # Save JSON data
    dataset = {
        "samples": samples,
        "metadata": {
            "num_samples": len(samples),
            "num_episodes_processed": episodes_processed,
            "num_images": image_counter,
            "images_folder": images_folder,
            "processed_images_folder": processed_images_folder,
            "history_length": 9,
            "action_dim": len(samples[0]["current_ground_truth_action"]) if samples else 0,
            "sampling_strategy": "diverse_episodes_random_timestep_unique_valid_instructions",
            "random_seed": seed,
            "skipped_duplicate_instructions": skipped_duplicate_instructions,
            "skipped_invalid_instructions": skipped_invalid_instructions,
            "unique_instructions_enforced": True,
            "instruction_validation_enforced": True
        }
    }
    
    print(f"\nSaving dataset to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print("Dataset saved successfully!")
    
    # Now process all images to create OpenVLA and RoboMonkey versions
    print(f"\nProcessing images for OpenVLA and RoboMonkey formats...")
    try:
        preprocess_all_images(images_folder, processed_images_folder)
        print(f"Processed images saved to {processed_images_folder}")
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract samples from Bridge V2 dataset')
    parser.add_argument('--builder_dir', type=str, default='../../bridge_dataset/1.0.0',
                        help='Path to the Bridge V2 dataset directory')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to extract')
    parser.add_argument('--output_json', type=str, default='json_results/bridge_samples.json',
                        help='Path to save the JSON file')
    parser.add_argument('--images_folder', type=str, default='bridge_images',
                        help='Path to folder where original images will be saved')
    parser.add_argument('--processed_images_folder', type=str, default='processed_images',
                        help='Path to folder where processed images will be saved')
    parser.add_argument('--seed', type=int, default=30,
                        help='Random seed for reproducible timestep selection')
    
    args = parser.parse_args()
    
    # Extract samples
    dataset = extract_bridge_samples(
        builder_dir=args.builder_dir,
        num_samples=args.num_samples,
        output_json=args.output_json,
        images_folder=args.images_folder,
        processed_images_folder=args.processed_images_folder,
        seed=args.seed
    )
    
    print("\nExtraction complete!")
    print(f"JSON file: {args.output_json}")
    print(f"Original images: {args.images_folder}")
    print(f"Processed images: {args.processed_images_folder}")
