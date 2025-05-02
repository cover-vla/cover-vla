import argparse
import h5py
import os
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
import warnings

# Define padding value consistently
ACTION_PADDING_VALUE = -5.0

def augment_dataset(dataset_path, dataset_folders, output_path, history_length=10):
    """
    Creates a dataset mapping original instructions to samples containing
    (image, positive_action_history, negative_action_history).
    Includes samples from the start of trajectories, padding histories with
    ACTION_PADDING_VALUE (-5.0) to the specified history_length.
    Rotates images by 180 degrees upon loading.
    Generates negative action histories based on the global mean and std dev
    of positive histories for each instruction.

    Args:
        dataset_path: Base path to the dataset
        dataset_folders: List of dataset folders to process
        output_path: Path to save the augmented dataset
        history_length: Number of past action steps to include in the history (H).
    """
    # --- Data Structures ---
    histories_by_instruction = defaultdict(list) # Still collect full histories for stats
    final_dataset = {}
    raw_demo_data_by_instruction = defaultdict(list)
    action_dim = None

    print(f"--- Pass 1: Collecting Positive Histories & Raw Data (H={history_length}) ---")
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
            try:
                with h5py.File(task_path, 'r') as f:
                    if 'data' not in f: continue
                    for demo_key in f['data'].keys():
                        demo_data = f['data'][demo_key]
                        if not all(k in demo_data for k in ['actions', 'obs']) or \
                           'agentview_rgb' not in demo_data['obs']: continue

                        actions = demo_data['actions'][()]
                        obs_data = demo_data['obs']['agentview_rgb'][()]

                        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0: continue
                        # --- No longer skipping short demos entirely ---
                        # if actions.shape[0] < history_length: continue

                        if action_dim is None:
                             action_dim = actions.shape[1]
                             if action_dim <= 0: raise ValueError("Invalid action dimension")
                        elif actions.shape[1] != action_dim:
                             print(f"Warning: Inconsistent action dim in {task}, demo {demo_key}. Skipping demo.")
                             continue

                        rotated_obs_data = np.array([np.rot90(img, k=2, axes=(0, 1)) for img in obs_data])
                        if rotated_obs_data.shape[0] != actions.shape[0]:
                             print(f"Warning: Action/Image length mismatch after rotation in {task}, demo {demo_key}. Skipping demo.")
                             continue

                        T = actions.shape[0]
                        # Store raw data for Pass 2 (needed for images and actions)
                        raw_demo_data_by_instruction[original_instruction].append(
                            {'actions': actions, 'images': rotated_obs_data, 'len': T}
                        )
                        task_has_valid_demo = True
                        total_demos_processed += 1

                        # --- Collect ONLY FULL positive histories for statistics ---
                        if T >= history_length:
                            for t in range(history_length - 1, T):
                                pos_action_hist = actions[t - history_length + 1 : t + 1]
                                histories_by_instruction[original_instruction].append(pos_action_hist)
                        # --------------------------------------------------------

            except Exception as e:
                print(f"Error processing file {task_path} in Pass 1: {e}. Skipping task.")
                continue
            # if not task_has_valid_demo:
            #     print(f"Note: No valid demos found for task '{original_instruction}' in Pass 1.")


    if action_dim is None:
        raise ValueError("Could not determine action dimension from any valid demo.")
    if total_demos_processed == 0:
         raise ValueError("No valid demonstrations found in the specified dataset folders.")

    print(f"\n--- Calculating Statistics for {len(histories_by_instruction)} Instructions (using full histories only) ---")
    instruction_stats = {}
    instructions_to_process = list(histories_by_instruction.keys()) # Use keys from collected full histories
    for instruction in tqdm(instructions_to_process, desc="Calculating Stats"):
        all_hists = histories_by_instruction[instruction]
        if len(all_hists) < 2:
            print(f"Warning: Instruction '{instruction}' has < 2 *full* valid histories ({len(all_hists)}). Cannot compute reliable stats. Skipping this instruction for negative generation.")
            # Don't delete from raw_demo_data yet, might still generate padded positives
            continue # Skip stat calculation

        try:
            all_hists_np = np.stack(all_hists, axis=0)
            mean_hist = np.mean(all_hists_np, axis=0)
            std_hist = np.std(all_hists_np, axis=0)
            std_hist = np.where(std_hist < 1e-6, 1e-6, std_hist)
            instruction_stats[instruction] = {'mean': mean_hist, 'std': std_hist}
        except Exception as e:
            print(f"Error calculating stats for instruction '{instruction}': {e}. Skipping stats.")
            # Don't delete from raw_demo_data

    print(f"\n--- Pass 2: Generating Padded Histories and Final Dataset ---")
    final_dataset = {}
    padding_array_template = np.full((1, action_dim), ACTION_PADDING_VALUE, dtype=np.float32) # Template for padding rows

    # Iterate through all instructions that had raw data, even if stats failed
    for instruction in tqdm(raw_demo_data_by_instruction.keys(), desc="Generating Samples"):
        final_dataset[instruction] = {'samples': []}
        stats = instruction_stats.get(instruction) # Get stats if available

        # Generate a single default negative history based on stats if possible
        # This avoids regenerating it inside the inner loop
        default_neg_hist = None
        if stats:
            mean_hist = stats['mean']
            std_hist = stats['std']
            signs = np.random.choice([-1, 1], size=mean_hist.shape)
            default_neg_hist_full = mean_hist + signs * std_hist
            default_neg_hist_full = np.clip(default_neg_hist_full, -1.0, 1.0)
            # Handle last dim special case if needed (assuming it was done before)
            default_neg_hist_full[:, -1] = np.random.choice([-1, 1], size=default_neg_hist_full[:, -1].shape)
            default_neg_hist = default_neg_hist_full.astype(np.float32) # Use consistent type


        for demo_data in raw_demo_data_by_instruction[instruction]:
            original_actions = demo_data['actions']
            images = demo_data['images'] # Already rotated
            T = demo_data['len']
            D = action_dim

            # Now loop through ALL timesteps to include padded examples
            for t in range(T):
                image_t = images[t]

                # --- Handle Positive History Padding ---
                available_hist_len = t + 1
                num_padding = max(0, history_length - available_hist_len)
                start_idx = 0 # Start index in original_actions
                end_idx = t + 1 # End index (exclusive)

                if num_padding > 0:
                    # Get available actions
                    actual_pos_actions = original_actions[start_idx:end_idx]
                    # Create padding
                    padding = np.repeat(padding_array_template, num_padding, axis=0)
                    # Concatenate
                    pos_action_hist = np.concatenate((padding, actual_pos_actions), axis=0).astype(original_actions.dtype)
                else: # Full history available
                    start_idx = t - history_length + 1
                    pos_action_hist = original_actions[start_idx:end_idx]
                #----------------------------------------

                # --- Handle Negative History Padding ---
                if default_neg_hist is not None:
                     if num_padding > 0:
                         # Take the *end* part of the default negative history
                         actual_neg_actions = default_neg_hist[num_padding:] # Get the last 'available_hist_len' actions
                         # Create padding
                         padding = np.repeat(padding_array_template, num_padding, axis=0)
                         # Concatenate
                         neg_action_hist = np.concatenate((padding, actual_neg_actions), axis=0).astype(original_actions.dtype)
                     else: # Full history
                         neg_action_hist = default_neg_hist # Use the pre-generated full history
                else:
                     # Fallback: If stats couldn't be calculated, just use padded positive hist as negative
                     # This is not ideal, but prevents crashing. Could also just skip the sample.
                     warnings.warn(f"No stats for instruction '{instruction}', using padded positive history as negative fallback for t={t}.")
                     neg_action_hist = pos_action_hist.copy()
                # ---------------------------------------

                # Final check on history length (should always be correct now)
                if pos_action_hist.shape[0] != history_length or neg_action_hist.shape[0] != history_length:
                     warnings.warn(f"Generated history length mismatch for '{instruction}' t={t}. Skipping.")
                     continue

                sample_data = {
                    'image': image_t,
                    'pos_action_hist': pos_action_hist,
                    'neg_action_hist': neg_action_hist
                }
                final_dataset[instruction]['samples'].append(sample_data)

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
    print(f"Negative Augmentation Method: Global Mean +/- Std Dev")
    print(f"Padding Value: {ACTION_PADDING_VALUE}")
    print(f"Total instructions included: {total_instructions}")
    print(f"Total number of (image, pos_hist, neg_hist) samples (incl. padded): {total_samples}")

    # --- Save dataset ---
    print(f"\nSaving dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
    print("Done!")
    return final_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset with potentially padded pos/neg action histories based on global stats.')
    parser.add_argument('--dataset_path', type=str, default='/home/xilun/LIBERO/libero/datasets',
                        help='Path to the dataset')
    parser.add_argument('--dataset_folders', nargs='+', default=['libero_spatial', 'libero_90', 'libero_object', 'libeero_goal'],
                        help='Dataset folders to process')
    # Updated default name
    parser.add_argument('--output_path', type=str, default='libero_all_pos_neg_hist_globalstd_padded.pkl',
                        help='Path to save the augmented dataset')
    parser.add_argument('--history_length', type=int, default=10,
                        help='Number of past action steps to include in the history (H)')

    args = parser.parse_args()

    # Construct output path with history length
    if args.output_path == 'libero_all_pos_neg_hist_globalstd_padded.pkl': # Check if default name is used
        args.output_path = f'libero_all_pos_neg_globalstd_h{args.history_length}_padded.pkl'
        print(f"Using default output path format: {args.output_path}")

    augment_dataset(args.dataset_path, args.dataset_folders, args.output_path,
                history_length=args.history_length)
