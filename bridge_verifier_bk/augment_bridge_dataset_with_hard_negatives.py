import argparse
import os
import numpy as np
from tqdm import tqdm
# Remove pickle import, add PIL and ijson for streaming JSON
from PIL import Image
import ijson
from collections import defaultdict
import warnings
import json
import re
import hashlib
import tensorflow as tf
import tensorflow_datasets as tfds

# Define padding value consistently
ACTION_PADDING_VALUE = -5.0

def normalize_instruction(instr):
    """
    Normalize instruction by removing trailing punctuation and converting to lowercase.
    Returns None if instruction is empty after normalization.
    """
    if not instr or not isinstance(instr, str):
        return None
    
    instr = instr.strip().lower()
    # Remove any trailing punctuation (., !, ?)
    instr = re.sub(r'[.?!]+$', '', instr).strip()
    
    # Return None if instruction is empty after normalization
    if not instr:
        return None
    
    return instr

def hash_action_history(action_hist):
    """
    Create a hash for an action history sequence for deduplication
    """
    # Convert to bytes for hashing
    action_bytes = np.array(action_hist).tobytes()
    return hashlib.md5(action_bytes).hexdigest()

def get_or_create_action_history_id(action_hist, action_histories, action_history_hash_to_id, next_action_id):
    """
    Get existing action history ID or create new one if not exists
    """
    action_hash = hash_action_history(action_hist)
    
    if action_hash in action_history_hash_to_id:
        return action_history_hash_to_id[action_hash], next_action_id
    else:
        # Create new entry
        action_id = f"action_{next_action_id}"
        action_histories[action_id] = action_hist.tolist()
        action_history_hash_to_id[action_hash] = action_id
        return action_id, next_action_id + 1

def get_or_create_instruction_id(instruction, instructions, instruction_to_id, next_instruction_id):
    """
    Get existing instruction ID or create new one if not exists.
    Returns None for instruction_id if instruction is empty or None.
    """
    if not instruction or not isinstance(instruction, str) or not instruction.strip():
        return None, next_instruction_id
        
    if instruction in instruction_to_id:
        return instruction_to_id[instruction], next_instruction_id
    else:
        # Create new entry
        instruction_id = f"instr_{next_instruction_id}"
        instructions[instruction_id] = instruction
        instruction_to_id[instruction] = instruction_id
        return instruction_id, next_instruction_id + 1

def load_hard_negatives_data(json_path):
    """
    Load hard negatives data from JSON file
    
    Args:
        json_path (str): Path to the curated hard negatives JSON file
        
    Returns:
        dict: Dictionary mapping sample_id to hard negatives data
    """
    hard_negatives_dict = {}
    
    with open(json_path, 'r') as f:
        all_data = json.load(f)
    
    for sample_data in all_data:
        sample_id = sample_data['sample_id']
        hard_negatives_dict[sample_id] = sample_data
    
    print(f"Loaded hard negatives for {len(hard_negatives_dict)} samples")
    return hard_negatives_dict

def extract_bridge_dataset_with_hard_negatives(builder_dir, episode_ids, output_path, history_length=10, 
                                               hard_negatives_json_path=None, max_episodes=None, images_folder=None):
    """
    Extract Bridge V2 dataset with action history, agent_view images, language instructions, and hard negatives.
    Includes samples from the start of trajectories, padding histories with ACTION_PADDING_VALUE (-5.0).
    
    Args:
        builder_dir (str): Path to the Bridge V2 dataset directory
        episode_ids (list): List of episode IDs to process
        output_path (str): Path to save the extracted dataset (JSON file)
        history_length (int): Number of past action steps to include in the history (H)
        hard_negatives_json_path (str): Path to the JSON file containing hard negatives data
        max_episodes (int): Maximum number of episodes to process (for debugging)
        images_folder (str): Path to folder where agent_view images will be saved as JPG files
    """
    
    # Create images folder if not specified
    if images_folder is None:
        images_folder = os.path.splitext(output_path)[0] + "_images"
    
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        print(f"Created images folder: {images_folder}")
    
    # Initialize dataset builder
    print(f"Loading Bridge V2 dataset from {builder_dir}...")
    builder = tfds.builder_from_directory(builder_dir=builder_dir)
    
    # Load hard negatives data if provided
    hard_negatives_dict = None
    max_sample_id = None
    if hard_negatives_json_path is not None:
        print(f"Loading hard negatives from {hard_negatives_json_path}...")
        hard_negatives_dict = load_hard_negatives_data(hard_negatives_json_path)
        max_sample_id = max(hard_negatives_dict.keys()) if hard_negatives_dict else None
        print(f"Maximum sample_id in hard negatives: {max_sample_id}")
    
    # Memory-optimized data structures
    action_histories = {}  # {hash_id: action_history_list}
    instructions = {}      # {instruction_id: instruction_text}
    samples = []          # [{action_history_id, image_file, instruction_id, episode_id, timestep, sample_id, positives, hard_negatives}]
    
    # Lookup tables for deduplication
    action_history_hash_to_id = {}  # {hash: id}
    instruction_to_id = {}          # {instruction: id}
    
    # Legacy structure for processing
    raw_episode_data_by_instruction = defaultdict(list)
    action_dim = None
    total_episodes_processed = 0
    image_counter = 0  # Global counter for image naming
    next_action_id = 0
    next_instruction_id = 0
    sample_id_counter = 0  # Global sample ID counter
    
    # Determine episode range
    if episode_ids is None:
        # Get dataset info to determine total number of episodes
        info = builder.info
        total_episodes = info.splits['train'].num_examples
        if max_episodes:
            total_episodes = min(total_episodes, max_episodes)
        episode_ids = list(range(total_episodes))
    else:
        if max_episodes:
            episode_ids = episode_ids[:max_episodes]
    
    print(f"Processing {len(episode_ids)} episodes...")
    
    # Process episodes
    for episode_id in tqdm(episode_ids, desc="Processing episodes"):
        try:
            # Load single episode
            ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
            episode = next(iter(ds))
            
            # Extract language instruction (assume same for all steps in episode)
            steps = list(episode["steps"])
            if not steps:
                continue
                
            language_instruction = steps[0]["language_instruction"].numpy().decode()
            original_instruction = normalize_instruction(language_instruction)
            
            # Skip episodes with empty or invalid instructions
            if original_instruction is None:
                print(f"Warning: Empty instruction in episode {episode_id}, skipping...")
                continue
            
            # Extract actions and observations for all steps
            actions_list = []
            observations_list = []
            
            for step in steps:
                action = step["action"].numpy()
                observation = step["observation"]
                
                # Extract agent view image (Over-the-shoulder RGBD - main view)
                # Try different possible image keys
                agent_view_image = None
                for img_key in ["image_0", "rgb", "image", "camera_0"]:
                    if img_key in observation:
                        agent_view_image = observation[img_key].numpy()
                        break
                
                if agent_view_image is None:
                    print(f"Warning: No agent view image found in episode {episode_id}, skipping...")
                    break
                
                # Save agent_view_image as JPG file
                if agent_view_image.dtype != np.uint8:
                    # Normalize to 0-255 range if needed
                    if agent_view_image.max() <= 1.0:
                        agent_view_image = (agent_view_image * 255).astype(np.uint8)
                    else:
                        agent_view_image = agent_view_image.astype(np.uint8)
                
                # Handle RGBD (4 channels) by taking only RGB
                if agent_view_image.shape[-1] == 4:
                    agent_view_image = agent_view_image[..., :3]
                
                # Save image as JPG
                image_filename = f"{image_counter}.jpg"
                image_path = os.path.join(images_folder, image_filename)
                Image.fromarray(agent_view_image).save(image_path, "JPEG", quality=95)
                    
                actions_list.append(action)
                observations_list.append({"agent_view_image_file": image_filename})
                image_counter += 1
            
            if len(actions_list) == 0:
                continue
                
            # Convert to numpy arrays
            actions = np.array(actions_list)
            
            # Determine action dimension
            if action_dim is None:
                action_dim = actions.shape[1]
                if action_dim <= 0:
                    raise ValueError("Invalid action dimension")
            elif actions.shape[1] != action_dim:
                print(f"Warning: Inconsistent action dim in episode {episode_id}. Skipping episode.")
                continue
            
            # Store raw episode data
            T = len(actions_list)
            raw_episode_data_by_instruction[original_instruction].append({
                'actions': actions.tolist(),  # Convert to list for JSON serialization
                'observations': observations_list,
                'len': T,
                'episode_id': episode_id
            })
            
            total_episodes_processed += 1
            
        except Exception as e:
            print(f"Error processing episode {episode_id}: {e}")
            continue
    
    if action_dim is None:
        raise ValueError("Could not determine action dimension from any valid episode.")
    if total_episodes_processed == 0:
        raise ValueError("No valid episodes found in the specified dataset.")
    
    print(f"\nProcessed {total_episodes_processed} episodes successfully.")
    print(f"Action dimension: {action_dim}")
    print(f"Saved {image_counter} images to {images_folder}")
    
    # Generate normalized dataset with deduplicated action histories
    print("Generating memory-optimized samples with deduplicated action histories...")
    padding_array_template = np.full((1, action_dim), ACTION_PADDING_VALUE, dtype=np.float32)
    
    # Process all samples and create them with sample_id
    for instruction in tqdm(raw_episode_data_by_instruction.keys(), desc="Processing Instructions"):
        norm_instruction = normalize_instruction(instruction)
        
        # Skip instructions that become empty after normalization
        if norm_instruction is None:
            print(f"Warning: Instruction '{instruction}' became empty after normalization, skipping...")
            continue
        
        for episode_data in raw_episode_data_by_instruction[instruction]:
            actions = np.array(episode_data['actions'])  # Convert back to numpy for processing
            observations = episode_data['observations']
            T = episode_data['len']
            episode_id = episode_data['episode_id']
            
            # Process all timesteps
            for t in range(0, T):  # Start from t=0
                # Check if we've reached the limit from hard negatives data
                if max_sample_id is not None and sample_id_counter > max_sample_id:
                    print(f"Reached maximum sample_id {max_sample_id}, stopping sample generation...")
                    break
                
                # Generate padded action history
                available_hist_len = t + 1
                num_padding = max(0, history_length - available_hist_len)
                start_idx = 0
                end_idx = t + 1
                
                if num_padding > 0:
                    actual_actions = actions[start_idx:end_idx]
                    padding = np.repeat(padding_array_template, num_padding, axis=0)
                    action_hist = np.concatenate((padding, actual_actions), axis=0).astype(actions.dtype)
                else:
                    start_idx = t - history_length + 1
                    action_hist = actions[start_idx:end_idx]
                
                # Skip all-padded samples
                if np.all(action_hist == ACTION_PADDING_VALUE):
                    continue
                
                # Final check on history length
                if action_hist.shape[0] != history_length:
                    warnings.warn(f"Generated history length mismatch for '{instruction}' t={t}. Skipping.")
                    continue
                
                # Get or create action history ID (deduplication happens here)
                action_history_id, next_action_id = get_or_create_action_history_id(
                    action_hist, action_histories, action_history_hash_to_id, next_action_id
                )
                
                # Get hard negatives data for this sample_id if available
                positives = []
                hard_negatives = []
                
                if hard_negatives_dict and sample_id_counter in hard_negatives_dict:
                    sample_hn_data = hard_negatives_dict[sample_id_counter]
                    
                    # Extract positive instructions and their hard negatives
                    for pos_instr, pos_data in sample_hn_data['positive_instructions'].items():
                        # Get or create instruction ID for positive
                        pos_instruction_id, next_instruction_id = get_or_create_instruction_id(
                            pos_instr, instructions, instruction_to_id, next_instruction_id
                        )
                        
                        if pos_instruction_id is not None:
                            positives.append(pos_instruction_id)
                            
                            # Extract hard negatives for this positive
                            for neg_instr, neg_data in pos_data['negative_instructions'].items():
                                neg_instruction_id, next_instruction_id = get_or_create_instruction_id(
                                    neg_instr, instructions, instruction_to_id, next_instruction_id
                                )
                                
                                if neg_instruction_id is not None:
                                    hard_negatives.append({
                                        'instruction_id': neg_instruction_id,
                                        'positive_instruction_id': pos_instruction_id,
                                        'similarity': neg_data['similarity'],
                                        'error': neg_data['error']
                                    })
                else:
                    # If no hard negatives data, just use the original instruction as positive
                    original_instruction_id, next_instruction_id = get_or_create_instruction_id(
                        norm_instruction, instructions, instruction_to_id, next_instruction_id
                    )
                    
                    if original_instruction_id is not None:
                        positives.append(original_instruction_id)
                
                # Create normalized sample
                sample = {
                    'sample_id': sample_id_counter,
                    'action_history_id': action_history_id,
                    'agent_view_image_file': observations[t]['agent_view_image_file'],
                    'positives': positives,
                    'hard_negatives': hard_negatives,
                    'episode_id': episode_id,
                    'timestep': t
                }
                
                samples.append(sample)
                sample_id_counter += 1
            
            # Break if we've reached the limit
            if max_sample_id is not None and sample_id_counter > max_sample_id:
                break
        
        # Break if we've reached the limit
        if max_sample_id is not None and sample_id_counter > max_sample_id:
            break
    
    # Create optimized final dataset structure
    total_instructions = len(instructions)
    total_samples = len(samples)
    total_unique_action_histories = len(action_histories)
    
    print(f"\nMemory optimization results:")
    print(f"Total unique action histories: {total_unique_action_histories}")
    print(f"Total samples: {total_samples}")
    print(f"Total instructions: {total_instructions}")
    print(f"Compression ratio: {total_samples / total_unique_action_histories:.2f}x samples per unique action history")
    
    # Count samples with positives and hard negatives
    samples_with_positives = sum(1 for s in samples if s['positives'])
    samples_with_hard_negatives = sum(1 for s in samples if s['hard_negatives'])
    total_hard_negatives = sum(len(s['hard_negatives']) for s in samples)
    
    print(f"Samples with positives: {samples_with_positives}")
    print(f"Samples with hard negatives: {samples_with_hard_negatives}")
    print(f"Total hard negative instances: {total_hard_negatives}")
    
    # Build the final optimized dataset
    final_dataset = {
        'action_histories': action_histories,
        'instructions': instructions,
        'samples': samples,
        '_metadata': {
            'action_dim': action_dim,
            'history_length': history_length,
            'total_images': image_counter,
            'images_folder': images_folder,
            'total_instructions': total_instructions,
            'total_samples': total_samples,
            'total_unique_action_histories': total_unique_action_histories,
            'compression_ratio': total_samples / total_unique_action_histories if total_unique_action_histories > 0 else 0,
            'format_version': '3.0_with_hard_negatives',
            'padding_value': ACTION_PADDING_VALUE,
            'samples_with_positives': samples_with_positives,
            'samples_with_hard_negatives': samples_with_hard_negatives,
            'total_hard_negatives': total_hard_negatives,
            'skip_first_n_timesteps': 0  # We don't skip any timesteps
        }
    }
    
    print(f"\nDataset extraction complete!")
    print(f"History length: {history_length}")
    print(f"Padding Value: {ACTION_PADDING_VALUE}")
    print(f"Total instructions: {total_instructions}")
    print(f"Total samples: {total_samples}")
    print(f"Total unique action histories: {total_unique_action_histories}")
    print(f"Memory efficiency: {total_samples / total_unique_action_histories:.2f}x reuse of action histories")
    print(f"Processed all timesteps including the first ones")
    
    # Save dataset as JSON
    print(f"\nSaving optimized dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(final_dataset, f, indent=2)
    print("Done!")
    
    return final_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Bridge V2 dataset with action histories, agent view images, language instructions, and hard negatives.')
    parser.add_argument('--builder_dir', type=str, default='/root/bridge_dataset/1.0.0',
                        help='Path to the Bridge V2 dataset directory')
    parser.add_argument('--episode_ids', nargs='+', type=int, default=None,
                        help='Specific episode IDs to process (if not specified, processes all)')
    parser.add_argument('--output_path', type=str, default='bridge_dataset_with_hard_negatives.json',
                        help='Path to save the extracted dataset (JSON file)')
    parser.add_argument('--history_length', type=int, default=10,
                        help='Number of past action steps to include in the history (H)')
    parser.add_argument('--hard_negatives_json', type=str, 
                        default='curated_hard_negatives.json',
                        help='Path to the JSON file containing curated hard negatives data')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum number of episodes to process (for debugging/testing)')
    parser.add_argument('--images_folder', type=str, default=None,
                        help='Path to folder where agent_view images will be saved as JPG files (default: output_path_images)')
    
    args = parser.parse_args()
    
    # Extract dataset
    final_dataset = extract_bridge_dataset_with_hard_negatives(
        builder_dir=args.builder_dir,
        episode_ids=args.episode_ids,
        output_path=args.output_path,
        history_length=args.history_length,
        hard_negatives_json_path=args.hard_negatives_json,
        max_episodes=args.max_episodes,
        images_folder=args.images_folder
    ) 
