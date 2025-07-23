import argparse
import h5py
import os
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
import warnings
import json
import re
import random

# Define padding value consistently
ACTION_PADDING_VALUE = -5.0

def normalize_instruction(instr):
    instr = instr.strip().lower()
    # Remove trailing ' demo' if present
    if instr.endswith(' demo'):
        instr = instr[:-5]
    instr = instr.strip()
    # Remove any trailing punctuation (., !, ?)
    instr = re.sub(r'[.?!]+$', '', instr).strip()
    return instr

def generate_negative_action(pos_action_hist, noise_range=0.1, flip_prob=0.5):
    """
    Generate negative actions by adding noise to the first 6 dimensions and 
    potentially flipping the sign of the last dimension.
    
    Args:
        pos_action_hist: Positive action history (H, D) where D is action dimension
        noise_range: Value to sample from (-noise_range or +noise_range)
        flip_prob: Probability of flipping the sign of the last dimension
    
    Returns:
        neg_action_hist: Negative action history with same shape as pos_action_hist
    """
    neg_action_hist = pos_action_hist.copy()
    
    # Add random noise to the first 6 dimensions (sample from -noise_range or +noise_range)
    if neg_action_hist.shape[1] >= 6:
        # Only add noise to non-padded values
        non_padded_mask = neg_action_hist[:, :6] != ACTION_PADDING_VALUE
        if np.any(non_padded_mask):
            # Randomly choose between -noise_range and +noise_range for each element
            noise_signs = np.random.choice([-1, 1], size=(neg_action_hist.shape[0], 6))
            noise = noise_signs * noise_range
            modify_index = np.random.randint(1, neg_action_hist.shape[1])
            # Only apply noise where the original value is not padded
            neg_action_hist[:, :modify_index] = np.where(non_padded_mask[:, :modify_index], 
                                            neg_action_hist[:, :modify_index] + noise[:, :modify_index], 
                                            neg_action_hist[:, :modify_index])
    
    # Randomly flip the sign of the last dimension with probability flip_prob
    if neg_action_hist.shape[1] > 0:
        flip_mask = np.random.random(neg_action_hist.shape[0]) < flip_prob
        neg_action_hist[flip_mask, -1] *= -1
    
    return neg_action_hist

def augment_dataset(dataset_path, dataset_folders, output_path, history_length=10, rephrases_json_path=None, suite_name=None, noise_range=0.1, flip_prob=0.5):
    """
    Creates a dataset mapping original instructions to samples containing
    (image, action_history).
    Includes samples from the start of trajectories, padding histories with
    ACTION_PADDING_VALUE (-5.0) to the specified history_length.
    Rotates images by 180 degrees upon loading.
    Optionally, adds language rephrases for each instruction from a JSON file.
    For negative rephrases, generates negative actions by adding random noise
    (sampling from -noise_range or +noise_range) to the first 6 dimensions and 
    potentially flipping the sign of the last dimension.

    Args:
        dataset_path: Base path to the dataset
        dataset_folders: List of dataset folders to process
        output_path: Path to save the augmented dataset
        history_length: Number of past action steps to include in the history (H).
        rephrases_json_path: Path to the JSON file containing instruction rephrases
        suite_name: Suite name (e.g., libero_spatial) for rephrases JSON
        noise_range: Value to sample from for noise in negative actions (-noise_range or +noise_range)
        flip_prob: Probability of flipping the sign of the last action dimension in negative actions
    """
    # --- Data Structures ---
    histories_by_instruction = defaultdict(list) # Still collect full histories for stats
    final_dataset = {}
    raw_demo_data_by_instruction = defaultdict(list)
    action_dim = None

    total_demos_processed = 0
    for folder in dataset_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Dataset folder {folder_path} does not exist. Skipping.")
            continue

        for task in tqdm(os.listdir(folder_path), desc=f"Processing dataset {folder}"):
            if not task.endswith('.hdf5'): continue

            original_instruction = task.replace('.hdf5', '').replace('_', ' ')
            original_instruction = ''.join(char for char in original_instruction if not char.isupper() and not char.isdigit())
            while original_instruction and original_instruction[0].isspace(): original_instruction = original_instruction[1:]
            if not original_instruction: continue

            task_path = os.path.join(folder_path, task)
            task_has_valid_demo = False
            with h5py.File(task_path, 'r') as f:
                if 'data' not in f: continue
                for demo_key in f['data'].keys():
                    demo_data = f['data'][demo_key]
                    if not all(k in demo_data for k in ['actions', 'obs']): continue

                    actions = demo_data['actions'][()]
                    obs_group = demo_data['obs']
                    # Collect all image keys
                    image_keys = [k for k in obs_group.keys() if obs_group[k].ndim >= 3]
                    obs_data_dict = {k: obs_group[k][()] for k in image_keys}

                    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0: continue

                    if action_dim is None:
                            action_dim = actions.shape[1]
                            if action_dim <= 0: raise ValueError("Invalid action dimension")
                    elif actions.shape[1] != action_dim:
                            print(f"Warning: Inconsistent action dim in {task}, demo {demo_key}. Skipping demo.")
                            continue

                    # Rotate all images by 180 degrees
                    rotated_obs_data_dict = {k: np.array([np.rot90(img, k=2, axes=(0, 1)) for img in v]) for k, v in obs_data_dict.items()}
                    T = actions.shape[0]
                    for k, v in rotated_obs_data_dict.items():
                        if v.shape[0] != T:
                            print(f"Warning: Action/Image length mismatch for key {k} after rotation in {task}, demo {demo_key}. Skipping demo.")
                            break
                    else:
                        # Store raw data for Pass 2 (needed for images and actions)
                        raw_demo_data_by_instruction[original_instruction].append(
                            {'actions': actions, 'images': rotated_obs_data_dict, 'len': T}
                        )
                        task_has_valid_demo = True
                        total_demos_processed += 1

                        # --- Collect ONLY FULL positive histories for statistics ---
                        if T >= history_length:
                            for t in range(history_length - 1, T):
                                pos_action_hist = actions[t - history_length + 1 : t + 1]
                                histories_by_instruction[original_instruction].append(pos_action_hist)
                        # --------------------------------------------------------


    if action_dim is None:
        raise ValueError("Could not determine action dimension from any valid demo.")
    if total_demos_processed == 0:
         raise ValueError("No valid demonstrations found in the specified dataset folders.")

    print(f"\n--- Pass 2: Generating Padded Histories and Final Dataset ---")
    final_dataset = {}
    padding_array_template = np.full((1, action_dim), ACTION_PADDING_VALUE, dtype=np.float32) # Template for padding rows

    # --- Load rephrases if provided ---
    rephrases_dict = None
    negative_rephrases_dict = None
    if rephrases_json_path is not None and suite_name is not None:
        rephrases_dict = {}
        negative_rephrases_dict = {}
        # Support both a single path or a list of paths
        rephrases_files = rephrases_json_path if isinstance(rephrases_json_path, list) else [rephrases_json_path]
        for rephrases_file in rephrases_files:
            with open(rephrases_file, 'r') as f:
                all_rephrases = json.load(f)
            suite_rephrases = all_rephrases[suite_name]
            for task_id, entry in suite_rephrases.items():
                orig = normalize_instruction(entry["original"])
                
                # Handle positive rephrases
                if "rephrases" in entry:
                    rephs = [r.strip() for r in entry["rephrases"]]
                    if orig in rephrases_dict:
                        # Merge, avoiding duplicates
                        rephrases_dict[orig].extend([r for r in rephs if r not in rephrases_dict[orig]])
                    else:
                        rephrases_dict[orig] = rephs
                
                # Handle negative rephrases
                if "negative_rephrases" in entry:
                    neg_rephs = [r.strip() for r in entry["negative_rephrases"]]
                    if orig in negative_rephrases_dict:
                        # Merge, avoiding duplicates
                        negative_rephrases_dict[orig].extend([r for r in neg_rephs if r not in negative_rephrases_dict[orig]])
                    else:
                        negative_rephrases_dict[orig] = neg_rephs

    # Iterate through all instructions that had raw data
    for instruction in tqdm(raw_demo_data_by_instruction.keys(), desc="Generating Samples"):
        norm_instruction = normalize_instruction(instruction)
        final_dataset[norm_instruction] = {'samples': []}

        for demo_data in raw_demo_data_by_instruction[instruction]:
            original_actions = demo_data['actions']
            images_dict = demo_data['images'] # Dict of all image keys
            T = demo_data['len']
            D = action_dim

            for t in range(T):
                # --- Handle Positive History Padding ---
                available_hist_len = t + 1
                num_padding = max(0, history_length - available_hist_len)
                start_idx = 0 # Start index in original_actions
                end_idx = t + 1 # End index (exclusive)

                if num_padding > 0:
                    actual_pos_actions = original_actions[start_idx:end_idx]
                    padding = np.repeat(padding_array_template, num_padding, axis=0)
                    pos_action_hist = np.concatenate((padding, actual_pos_actions), axis=0).astype(original_actions.dtype)
                else:
                    start_idx = t - history_length + 1
                    pos_action_hist = original_actions[start_idx:end_idx]
                #----------------------------------------

                # Skip all-padded samples
                if np.all(pos_action_hist == ACTION_PADDING_VALUE):
                    continue

                # Collect all images for this timestep
                images_at_t = {k: v[t] for k, v in images_dict.items()}

                # Final check on history length
                if pos_action_hist.shape[0] != history_length:
                     warnings.warn(f"Generated history length mismatch for '{instruction}' t={t}. Skipping.")
                     continue

                sample_data = {
                    'images': images_at_t,
                    'pos_action_hist': pos_action_hist
                }
                final_dataset[norm_instruction]['samples'].append(sample_data)

        # --- Add rephrases as additional keys, if available ---
        if rephrases_dict is not None:
            instr_norm = norm_instruction
            matched = None
            for orig, rephs in rephrases_dict.items():
                if orig == instr_norm:
                    matched = rephs
                    break
            if matched is not None:
                for reph in matched:
                    norm_reph = normalize_instruction(reph)
                    if norm_reph not in final_dataset:
                        final_dataset[norm_reph] = {'samples': list(final_dataset[norm_instruction]['samples'])}

        # --- Add negative rephrases with negative actions, if available ---
        if negative_rephrases_dict is not None:
            instr_norm = norm_instruction
            matched = None
            for orig, neg_rephs in negative_rephrases_dict.items():
                if orig == instr_norm:
                    matched = neg_rephs
                    break
            if matched is not None:
                for neg_reph in matched:
                    norm_neg_reph = normalize_instruction(neg_reph)
                    if norm_neg_reph not in final_dataset:
                        # Create negative samples with modified actions (only for non-padded samples)
                        negative_samples = []
                        for sample in final_dataset[norm_instruction]['samples']:
                            pos_action_hist = sample['pos_action_hist']
                            
                            # Check if this is a padded sample (all values are -5.0)
                            is_padded = np.all(pos_action_hist == ACTION_PADDING_VALUE)
                            
                            if is_padded:
                                # For padded samples, keep the same padded values (no noise needed)
                                neg_action_hist = pos_action_hist.copy()
                            else:
                                # For non-padded samples, generate negative action history
                                neg_action_hist = generate_negative_action(pos_action_hist, noise_range=noise_range, flip_prob=flip_prob)
                            
                            negative_sample = {
                                'images': sample['images'],
                                'pos_action_hist': neg_action_hist  # Store negative actions in pos_action_hist field
                            }
                            negative_samples.append(negative_sample)
                        final_dataset[norm_neg_reph] = {'samples': negative_samples}

    # --- Final Cleanup ---
    keys_to_remove = [k for k, v in final_dataset.items() if not v.get('samples')]
    if keys_to_remove:
        print(f"Removing {len(keys_to_remove)} instruction entries with no samples after Pass 2.")
        for k in keys_to_remove: del final_dataset[k]

    # --- Print statistics ---
    total_instructions = len(final_dataset)
    total_samples = sum(len(v.get('samples', [])) for v in final_dataset.values())
    
    # Count positive vs negative instructions more accurately
    positive_instructions = 0
    negative_instructions = 0
    
    # Track which instructions came from positive vs negative rephrases
    positive_instruction_set = set()
    negative_instruction_set = set()
    
    # Add original instructions to positive set
    for instruction in raw_demo_data_by_instruction.keys():
        positive_instruction_set.add(normalize_instruction(instruction))
    
    # Add positive rephrases to positive set
    if rephrases_dict is not None:
        for orig, rephs in rephrases_dict.items():
            for reph in rephs:
                positive_instruction_set.add(normalize_instruction(reph))
    
    # Add negative rephrases to negative set
    if negative_rephrases_dict is not None:
        for orig, neg_rephs in negative_rephrases_dict.items():
            for neg_reph in neg_rephs:
                negative_instruction_set.add(normalize_instruction(neg_reph))
    
    # Count based on the sets
    for instruction in final_dataset.keys():
        if instruction in negative_instruction_set:
            negative_instructions += 1
        else:
            positive_instructions += 1

    print(f"\nDataset creation complete!")
    print(f"History length: {history_length}")
    print(f"Padding Value: {ACTION_PADDING_VALUE}")
    print(f"Total instructions included: {total_instructions}")
    print(f"  - Positive instructions: {positive_instructions}")
    print(f"  - Negative instructions: {negative_instructions}")
    print(f"Total number of (images, action_hist) samples (incl. padded): {total_samples}")
    
    
    # --- Save dataset ---
    print(f"\nSaving dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
    print("Done!")
    return final_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset with potentially padded pos/neg action histories based on global stats.')
    parser.add_argument('--dataset_path', type=str, default='/root/LIBERO/libero/datasets',
                        help='Path to the dataset')
    parser.add_argument('--dataset_folders', nargs='+', default=['libero_spatial_no_noops'],
                        help='Dataset folders to process')
    parser.add_argument('--output_path', type=str, default='libero_spatial_pos_rephrase_neg_negation.pkl',
                        help='Path to save the augmented dataset')
    parser.add_argument('--history_length', type=int, default=10,
                        help='Number of past action steps to include in the history (H)')
    # parser.add_argument('--rephrases_json', type=str, default=['/root/vla-clip/openvla/experiments/robot/libero/libero_rephrase_hard_new.json', '/root/vla-clip/openvla/experiments/robot/libero/libero_rephrase_hard.json', '/root/vla-clip/openvla/experiments/robot/libero/libero_rephrases.json'],
    #                     help='Path to the JSON file containing instruction rephrases')
    parser.add_argument('--rephrases_json', type=str, default=['/root/vla-clip/openvla/experiments/robot/libero/libero_rephrase_pos_rephrase_neg_negation.json'],
                        help='Path to the JSON file containing instruction rephrases')
    parser.add_argument('--suite_name', type=str, default='libero_spatial',
                        help='Suite name (e.g., libero_spatial) for rephrases JSON')
    parser.add_argument('--noise_range', type=float, default=0.1,
                        help='Value to sample from for noise in negative actions (-noise_range or +noise_range)')
    parser.add_argument('--flip_prob', type=float, default=0.5,
                        help='Probability of flipping the sign of the last action dimension in negative actions')

    args = parser.parse_args()

    final_dataset = augment_dataset(args.dataset_path, args.dataset_folders, args.output_path,
                history_length=args.history_length,
                rephrases_json_path=args.rephrases_json,
                suite_name=args.suite_name,
                noise_range=args.noise_range,
                flip_prob=args.flip_prob)
    # final_dataset = pickle.load(open(args.output_path, 'rb'))
    # for instruction, data in final_dataset.items():
    #     print ("instruction", instruction)
    #     # print ("data", data['samples'].keys())
    #     input()
    
