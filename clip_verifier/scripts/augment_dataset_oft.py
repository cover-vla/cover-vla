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

def augment_dataset(dataset_path, dataset_folders, output_path, history_length=10, rephrases_json_path=None, suite_name=None):
    """
    Creates a dataset mapping original instructions to samples containing
    (image, positive_action_history).
    Includes samples from the start of trajectories, padding histories with
    ACTION_PADDING_VALUE (-5.0) to the specified history_length.
    Rotates images by 180 degrees upon loading.
    Optionally, adds language rephrases for each instruction from a JSON file.

    Args:
        dataset_path: Base path to the dataset
        dataset_folders: List of dataset folders to process
        output_path: Path to save the augmented dataset
        history_length: Number of past action steps to include in the history (H).
        rephrases_json_path: Path to the JSON file containing instruction rephrases
        suite_name: Suite name (e.g., libero_spatial) for rephrases JSON
    """
    # --- Data Structures ---
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
                        total_demos_processed += 1

    if action_dim is None:
        raise ValueError("Could not determine action dimension from any valid demo.")
    if total_demos_processed == 0:
         raise ValueError("No valid demonstrations found in the specified dataset folders.")

    print(f"\n--- Pass 2: Generating Padded Histories and Final Dataset ---")
    final_dataset = {}
    padding_array_template = np.full((1, action_dim), ACTION_PADDING_VALUE, dtype=np.float32) # Template for padding rows

    # --- Load rephrases if provided ---
    rephrases_dict = None
    if rephrases_json_path is not None and suite_name is not None:
        rephrases_dict = {}
        # Support both a single path or a list of paths
        rephrases_files = rephrases_json_path if isinstance(rephrases_json_path, list) else [rephrases_json_path]
        for rephrases_file in rephrases_files:
            with open(rephrases_file, 'r') as f:
                all_rephrases = json.load(f)
            suite_rephrases = all_rephrases[suite_name]
            for task_id, entry in suite_rephrases.items():
                orig = normalize_instruction(entry["original"])
                rephs = [r.strip() for r in entry["rephrases"]]
                if orig in rephrases_dict:
                    # Merge, avoiding duplicates
                    rephrases_dict[orig].extend([r for r in rephs if r not in rephrases_dict[orig]])
                else:
                    rephrases_dict[orig] = rephs

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
                # --- Handle Future Action Padding ---
                future_actions = original_actions[t : t + history_length]
                num_padding = max(0, history_length - future_actions.shape[0])
                if num_padding > 0:
                    padding = np.repeat(padding_array_template, num_padding, axis=0)
                    future_action_chunk = np.concatenate((future_actions, padding), axis=0).astype(original_actions.dtype)
                else:
                    future_action_chunk = future_actions
                #-------------------------------------

                # Collect all images for this timestep
                images_at_t = {k: v[t] for k, v in images_dict.items()}

                # Final check on chunk length
                if future_action_chunk.shape[0] != history_length:
                     warnings.warn(f"Generated future chunk length mismatch for '{instruction}' t={t}. Skipping.")
                     continue

                sample_data = {
                    'images': images_at_t,
                    'pos_action_hist': future_action_chunk
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

    # --- Final Cleanup ---
    keys_to_remove = [k for k, v in final_dataset.items() if not v.get('samples')]
    if keys_to_remove:
        print(f"Removing {len(keys_to_remove)} instruction entries with no samples after Pass 2.")
        for k in keys_to_remove: del final_dataset[k]

    # --- Print statistics ---
    total_instructions = len(final_dataset)
    total_samples = sum(len(v.get('samples', [])) for v in final_dataset.values())

    print(f"\nDataset creation complete!")
    print(f"History length: {history_length}")
    print(f"Padding Value: {ACTION_PADDING_VALUE}")
    print(f"Total instructions included: {total_instructions}")
    print(f"Total number of (images, pos_hist) samples (incl. padded): {total_samples}")

    # --- Save dataset ---
    print(f"\nSaving dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
    print("Done!")
    return final_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset with potentially padded pos/neg action histories based on global stats.')
    parser.add_argument('--dataset_path', type=str, default='../../../LIBERO/libero/datasets',
                        help='Path to the dataset')
    parser.add_argument('--dataset_folders', nargs='+', default=['libero_spatial_no_noops'],
                        help='Dataset folders to process')
    parser.add_argument('--output_path', type=str, default='libero_spatial_50_all.pkl',
                        help='Path to save the augmented dataset')
    parser.add_argument('--history_length', type=int, default=50,
                        help='Number of past action steps to include in the history (H)')
    parser.add_argument('--rephrases_json', type=str, default=['../../openvla/experiments/robot/libero/libero_rephrase_hard.json', '../../openvla/experiments/robot/libero/libero_rephrases.json'],
                        help='Path to the JSON file containing instruction rephrases')
    parser.add_argument('--suite_name', type=str, default='libero_spatial',
                        help='Suite name (e.g., libero_spatial) for rephrases JSON')

    args = parser.parse_args()

    fianl_dataset = augment_dataset(args.dataset_path, args.dataset_folders, args.output_path,
                history_length=args.history_length,
                rephrases_json_path=args.rephrases_json,
                suite_name=args.suite_name)
    
    # final_dataset = pickle.load(open("/home/xilun/vla-clip/clip_verifier/augmented_datasets/libero_spatial_oft_all.pkl", 'rb'))
    
# Assume final_dataset is already loaded

    # print("Total language instructions:", len(final_dataset))
    # print("="*60)

    # for instruction, data in final_dataset.items():
    #     print(f"Instruction: {instruction}")
    #     num_samples = len(data['samples'])
    #     print(f"  Number of action samples: {num_samples}")

    #     # Print the shape of one action sample and the shape of images
    #     if num_samples > 0:
    #         first_sample = data['samples'][0]
    #         action_shape = first_sample['pos_action_hist'].shape
    #         print(f"  Shape of one action sample: {action_shape}")
    #         for img_key, img_val in first_sample['images'].items():
    #             print(f"    Image key '{img_key}': shape {img_val.shape}, dtype {img_val.dtype}")

    #     # Count total images (across all samples, all image keys)
    #     total_images = 0
    #     image_keys_set = set()
    #     for sample in data['samples']:
    #         # Each sample['images'] is a dict of image_key: image_array
    #         image_keys = sample['images'].keys()
    #         image_keys_set.update(image_keys)
    #         total_images += len(image_keys)
    #     print(f"  Number of unique image keys per sample: {sorted(image_keys_set)}")
    #     print(f"  Total images (all samples, all keys): {total_images}")

    #     # Optionally, print a breakdown of how many images per key
    #     image_key_counts = {k: 0 for k in image_keys_set}
    #     for sample in data['samples']:
    #         for k in sample['images']:
    #             image_key_counts[k] += 1
    #     print(f"  Images per key: {image_key_counts}")

    #     print("-"*40)
