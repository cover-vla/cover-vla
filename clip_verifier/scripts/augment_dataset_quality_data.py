import argparse
import h5py
import os
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
import warnings
from scipy.interpolate import interp1d # Needed for resampling

# Define minimum number of demos required per instruction for statistical filtering
MIN_DEMOS_FOR_STATS = 5
# Define a fixed length to resample trajectories to, or use None to calculate median per instruction
# Using a fixed length can be simpler if trajectory lengths vary wildly.
# Using median adapts better but requires recalculation for each instruction.
# Let's start with a fixed length for simplicity. Adjust as needed.
RESAMPLE_LENGTH = 100 # Example fixed length
# Standard deviation threshold for filtering
STD_DEV_THRESHOLD = 1.0
# History Augmentation
ACTION_PADDING_VALUE = -5.0
DEFAULT_HISTORY_LENGTH = 10

def resample_trajectory(trajectory, target_length):
    """Resamples a trajectory (T, D) to a new length (target_length, D) using linear interpolation."""
    if trajectory.shape[0] == target_length:
        return trajectory
    if trajectory.shape[0] < 2: # Cannot interpolate with less than 2 points
        warnings.warn(f"Trajectory too short ({trajectory.shape[0]} points) to resample to {target_length}. Skipping resampling.")
        return None # Indicate failure

    original_length = trajectory.shape[0]
    dims = trajectory.shape[1]
    original_timepoints = np.linspace(0, 1, original_length)
    target_timepoints = np.linspace(0, 1, target_length)
    resampled = np.zeros((target_length, dims), dtype=trajectory.dtype)

    try:
        for i in range(dims):
            interpolator = interp1d(original_timepoints, trajectory[:, i], bounds_error=False, fill_value="extrapolate") # Handle potential edge cases
            resampled[:, i] = interpolator(target_timepoints)
    except ValueError as e:
         warnings.warn(f"Interpolation failed for trajectory shape {trajectory.shape}: {e}. Skipping resampling.")
         return None

    return resampled

def filter_and_augment_dataset(
    dataset_path, dataset_folders, output_path,
    resample_len=RESAMPLE_LENGTH, std_dev_thresh=STD_DEV_THRESHOLD, min_demos=MIN_DEMOS_FOR_STATS,
    history_length=DEFAULT_HISTORY_LENGTH
):
    """
    Filters dataset trajectories based on quality, then creates an augmented
    dataset with padded positive/negative action histories from the filtered data.

    Args:
        dataset_path: Base path to the dataset.
        dataset_folders: List of dataset folders to process.
        output_path: Path to save the final augmented dataset.
        resample_len: Target length for trajectory resampling during filtering.
        std_dev_thresh: Std dev threshold for outlier trajectory filtering.
        min_demos: Min demos per instruction required for filtering.
        history_length: Number of past action steps for history augmentation (H).
    """
    # --- Pass 1: Collect Raw Trajectories and Images ---
    raw_demo_data_by_instruction = defaultdict(list)
    action_dim = None
    total_demos_collected = 0

    print("--- Pass 1: Collecting Raw Trajectories and Images ---")
    for folder in dataset_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Dataset folder {folder_path} does not exist. Skipping.")
            continue

        for task_filename in tqdm(os.listdir(folder_path), desc=f"Processing dataset {folder}"):
            if not task_filename.endswith('.hdf5'): continue

            # Clean instruction (same logic as before)
            instruction = task_filename.replace('.hdf5', '').replace('_', ' ')
            instruction = ''.join(char for char in instruction if not char.isupper() and not char.isdigit())
            instruction = instruction.strip()
            if not instruction: continue

            task_path = os.path.join(folder_path, task_filename)
            try:
                with h5py.File(task_path, 'r') as f:
                    if 'data' not in f: continue
                    for demo_key in f['data'].keys():
                        demo_data = f['data'][demo_key]
                        # Ensure required keys exist
                        if not all(k in demo_data for k in ['actions', 'obs']) or \
                           'agentview_rgb' not in demo_data['obs']: continue

                        actions = demo_data['actions'][()]
                        obs_data = demo_data['obs']['agentview_rgb'][()] # Rotated later if kept

                        # Basic validation
                        if actions.ndim != 2 or actions.shape[0] < 2 or actions.shape[1] == 0: # Need at least 2 points for resampling
                            warnings.warn(f"Skipping demo {demo_key} in {task_filename}: Invalid actions shape {actions.shape} or too short.")
                            continue
                        if obs_data.shape[0] != actions.shape[0]:
                             warnings.warn(f"Skipping demo {demo_key} in {task_filename}: Action/Image length mismatch ({actions.shape[0]} vs {obs_data.shape[0]})")
                             continue

                        # Check action dimension consistency
                        if action_dim is None:
                            action_dim = actions.shape[1]
                            if action_dim <= 0: raise ValueError("Invalid action dimension")
                        elif actions.shape[1] != action_dim:
                            warnings.warn(f"Skipping demo {demo_key} in {task_filename}: Inconsistent action dim ({actions.shape[1]} vs {action_dim}).")
                            continue

                        # Store raw data (images will be rotated later only if kept)
                        raw_demo_data_by_instruction[instruction].append({
                            'actions': actions,
                            'images': obs_data, # Store original images for now
                            'demo_key': demo_key, # Keep track for debugging/info
                            'task_file': task_filename
                        })
                        total_demos_collected += 1

            except Exception as e:
                print(f"Error processing file {task_path} in Pass 1: {e}. Skipping task.")
                continue

    if action_dim is None:
        raise ValueError("Could not determine action dimension from any valid demo.")
    if total_demos_collected == 0:
         raise ValueError("No valid demonstrations collected from the specified dataset folders.")

    # --- Pass 2: Filter Trajectories ---
    print(f"\n--- Pass 2: Filtering Trajectories (Resample Length={resample_len}, Std Threshold={std_dev_thresh}) ---")
    filtered_data = defaultdict(list) # Store {'instruction': [{'actions': ..., 'images': ...}, ...]} for kept demos
    total_kept_demos = 0
    total_rejected_demos = 0

    for instruction, demos in tqdm(raw_demo_data_by_instruction.items(), desc="Filtering Instructions"):
        if len(demos) < min_demos:
            print(f"Instruction '{instruction}': Skipping filtering, only {len(demos)} demos (min is {min_demos}). Keeping all.")
            # Keep all demos, just rotate images
            for demo_data in demos:
                rotated_obs_data = np.array([np.rot90(img, k=2, axes=(0, 1)) for img in demo_data['images']])
                filtered_data[instruction].append({
                    'actions': demo_data['actions'],
                    'images': rotated_obs_data # Apply rotation now
                })
                total_kept_demos += 1
            continue

        resampled_trajectories = []
        original_indices = []
        for idx, demo_data in enumerate(demos):
            resampled = resample_trajectory(demo_data['actions'], resample_len)
            if resampled is not None:
                resampled_trajectories.append(resampled)
                original_indices.append(idx)
            # else: # Warning already printed in resample_trajectory
            #      warnings.warn(f"Could not resample demo {demo_data['demo_key']} for '{instruction}'. Excluding.")

        if len(resampled_trajectories) < min_demos:
             print(f"Instruction '{instruction}': Not enough successfully resampled demos ({len(resampled_trajectories)}) for filtering (min is {min_demos}). Keeping all successfully resampled demos.")
             for original_idx in original_indices:
                 demo_data = demos[original_idx]
                 rotated_obs_data = np.array([np.rot90(img, k=2, axes=(0, 1)) for img in demo_data['images']])
                 filtered_data[instruction].append({
                     'actions': demo_data['actions'],
                     'images': rotated_obs_data
                 })
                 total_kept_demos += 1
             continue

        resampled_stack = np.stack(resampled_trajectories, axis=0)
        mean_traj = np.mean(resampled_stack, axis=0)
        avg_distances = []
        for resampled_traj in resampled_trajectories:
            distances_t = np.linalg.norm(resampled_traj - mean_traj, axis=1)
            avg_distances.append(np.mean(distances_t))
        avg_distances = np.array(avg_distances)

        mean_of_avg_dists = np.mean(avg_distances)
        std_of_avg_dists = np.std(avg_distances)
        # Ensure std dev is not too small to avoid division by zero or overly sensitive filtering
        std_of_avg_dists = max(std_of_avg_dists, 1e-6)
        distance_threshold = mean_of_avg_dists + std_dev_thresh * std_of_avg_dists

        instruction_rejected_count = 0
        for i, avg_dist in enumerate(avg_distances):
            original_idx = original_indices[i]
            demo_data = demos[original_idx]
            if avg_dist <= distance_threshold:
                rotated_obs_data = np.array([np.rot90(img, k=2, axes=(0, 1)) for img in demo_data['images']])
                filtered_data[instruction].append({
                    'actions': demo_data['actions'],
                    'images': rotated_obs_data
                })
                total_kept_demos += 1
            else:
                total_rejected_demos += 1
                instruction_rejected_count += 1

        if instruction_rejected_count > 0:
             print(f"Instruction '{instruction}': Rejected {instruction_rejected_count}/{len(resampled_trajectories)} demos based on distance threshold.")
        if not filtered_data.get(instruction):
             print(f"Warning: All demos for instruction '{instruction}' were rejected or failed resampling/filtering.")


    # --- Pass 3: Generate Histories from Filtered Data ---
    print(f"\n--- Pass 3: Generating Padded Histories (H={history_length}) from Filtered Data ---")
    histories_by_instruction_filtered = defaultdict(list) # Collect full histories *from filtered data* for stats
    final_augmented_dataset = {}
    padding_array_template = np.full((1, action_dim), ACTION_PADDING_VALUE, dtype=np.float32)

    # --- Pass 3a: Collect Histories for Stats (from filtered data) ---
    print("Collecting histories for statistics from filtered demos...")
    instructions_with_filtered_data = list(filtered_data.keys())
    for instruction in instructions_with_filtered_data:
        for demo_data in filtered_data[instruction]:
            actions = demo_data['actions']
            T = actions.shape[0]
            if T >= history_length:
                for t in range(history_length - 1, T):
                    pos_action_hist = actions[t - history_length + 1 : t + 1]
                    histories_by_instruction_filtered[instruction].append(pos_action_hist)

    # --- Pass 3b: Calculate Stats (from filtered data) ---
    print("Calculating statistics based on filtered histories...")
    instruction_stats_filtered = {}
    instructions_for_stats = list(histories_by_instruction_filtered.keys())
    for instruction in tqdm(instructions_for_stats, desc="Calculating Filtered Stats"):
        all_hists = histories_by_instruction_filtered[instruction]
        # Use a lower threshold for stats calc as filtering might reduce demo count
        min_histories_for_stats = max(2, min_demos // 2) # Heuristic: require at least 2, or half the filtering min
        if len(all_hists) < min_histories_for_stats:
            print(f"Warning: Filtered instruction '{instruction}' has < {min_histories_for_stats} full valid histories ({len(all_hists)}). Cannot compute reliable stats. Negative samples might be less diverse.")
            continue # Skip stat calculation, will use fallback for negatives

        try:
            all_hists_np = np.stack(all_hists, axis=0)
            mean_hist = np.mean(all_hists_np, axis=0)
            std_hist = np.std(all_hists_np, axis=0)
            std_hist = np.where(std_hist < 1e-6, 1e-6, std_hist) # Prevent zero std dev
            instruction_stats_filtered[instruction] = {'mean': mean_hist, 'std': std_hist}
        except Exception as e:
            print(f"Error calculating filtered stats for instruction '{instruction}': {e}. Skipping stats.")


    # --- Pass 3c: Generate Augmented Samples (from filtered data) ---
    print("Generating final augmented samples...")
    final_augmented_dataset = {}
    total_augmented_samples = 0

    for instruction in tqdm(instructions_with_filtered_data, desc="Generating Augmented Samples"):
        final_augmented_dataset[instruction] = {'samples': []}
        stats = instruction_stats_filtered.get(instruction)

        # Pre-generate default negative history if stats are available
        default_neg_hist = None
        if stats:
            mean_hist = stats['mean']
            std_hist = stats['std']
            signs = np.random.choice([-1, 1], size=mean_hist.shape)
            default_neg_hist_full = mean_hist + signs * std_hist
            default_neg_hist_full = np.clip(default_neg_hist_full, -1.0, 1.0)
            # Handle last dim special case if necessary (e.g., gripper)
            if default_neg_hist_full.shape[1] > 0: # Check if action_dim > 0
                 default_neg_hist_full[:, -1] = np.random.choice([-1, 1], size=default_neg_hist_full.shape[0])

            default_neg_hist = default_neg_hist_full.astype(np.float32)

        # Iterate through the *filtered* demos for this instruction
        for demo_data in filtered_data[instruction]:
            original_actions = demo_data['actions'] # Actions from the KEPT demo
            images = demo_data['images'] # Already rotated images from the KEPT demo
            T = original_actions.shape[0]

            for t in range(T):
                image_t = images[t]

                # Positive History Padding
                available_hist_len = t + 1
                num_padding = max(0, history_length - available_hist_len)
                if num_padding > 0:
                    actual_pos_actions = original_actions[0 : t + 1]
                    padding = np.repeat(padding_array_template, num_padding, axis=0)
                    pos_action_hist = np.concatenate((padding, actual_pos_actions), axis=0).astype(original_actions.dtype)
                else:
                    start_idx = t - history_length + 1
                    pos_action_hist = original_actions[start_idx : t + 1]

                # Negative History Padding
                if default_neg_hist is not None:
                    if num_padding > 0:
                        actual_neg_actions = default_neg_hist[num_padding:]
                        padding = np.repeat(padding_array_template, num_padding, axis=0)
                        neg_action_hist = np.concatenate((padding, actual_neg_actions), axis=0).astype(original_actions.dtype)
                    else:
                        neg_action_hist = default_neg_hist
                else:
                    # Fallback if no stats
                    warnings.warn(f"No stats for filtered instruction '{instruction}', using padded positive history as negative fallback for t={t}.", RuntimeWarning) # Use RuntimeWarning to avoid flooding
                    neg_action_hist = pos_action_hist.copy()


                if pos_action_hist.shape[0] != history_length or neg_action_hist.shape[0] != history_length:
                    warnings.warn(f"Generated history length mismatch for '{instruction}' t={t}. Skipping sample.")
                    continue

                sample_data = {
                    'image': image_t,
                    'pos_action_hist': pos_action_hist,
                    'neg_action_hist': neg_action_hist
                }
                final_augmented_dataset[instruction]['samples'].append(sample_data)
                total_augmented_samples += 1

    # --- Final Cleanup & Stats ---
    keys_to_remove = [k for k, v in final_augmented_dataset.items() if not v.get('samples')]
    if keys_to_remove:
        print(f"\nRemoving {len(keys_to_remove)} instruction entries with no augmented samples generated.")
        for k in keys_to_remove: del final_augmented_dataset[k]

    total_instructions_final = len(final_augmented_dataset)

    print(f"\nDataset Filtering and Augmentation complete!")
    print(f"--- Filtering Stats ---")
    print(f"Filter Method: Resample to {resample_len}, reject if avg dist > mean + {std_dev_thresh}*std")
    print(f"Min demos for filtering: {min_demos}")
    print(f"Total demos collected: {total_demos_collected}")
    print(f"Total demos kept after filtering: {total_kept_demos}")
    print(f"Total demos rejected by filtering: {total_rejected_demos}")
    print(f"--- Augmentation Stats ---")
    print(f"History length (H): {history_length}")
    print(f"Padding Value: {ACTION_PADDING_VALUE}")
    print(f"Total instructions in final augmented dataset: {total_instructions_final}")
    print(f"Total (image, pos_hist, neg_hist) samples generated: {total_augmented_samples}")


    # --- Save dataset ---
    print(f"\nSaving final augmented dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(final_augmented_dataset, f)
    print("Done!")
    return final_augmented_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter expert trajectories and create augmented dataset with padded histories.')
    # Dataset Args
    parser.add_argument('--dataset_path', type=str, default='/home/xilun/LIBERO/libero/datasets',
                        help='Path to the original dataset')
    parser.add_argument('--dataset_folders', nargs='+', default=['libero_spatial'],
                        help='Dataset folders to process')
    # Filtering Args
    parser.add_argument('--resample_length', type=int, default=RESAMPLE_LENGTH,
                        help=f'Length to resample trajectories to for filtering comparison (default: {RESAMPLE_LENGTH})')
    parser.add_argument('--std_dev_threshold', type=float, default=STD_DEV_THRESHOLD,
                        help=f'Standard deviation threshold for filtering outlier rejection (default: {STD_DEV_THRESHOLD})')
    parser.add_argument('--min_demos', type=int, default=MIN_DEMOS_FOR_STATS,
                         help=f'Minimum demos per instruction required to apply filtering (default: {MIN_DEMOS_FOR_STATS})')
    # Augmentation Args
    parser.add_argument('--history_length', type=int, default=DEFAULT_HISTORY_LENGTH,
                        help=f'Number of past action steps for history augmentation (H) (default: {DEFAULT_HISTORY_LENGTH})')
    # Output Arg
    parser.add_argument('--output_path', type=str, default=f'libero_spatial_filtered_augmented_h{DEFAULT_HISTORY_LENGTH}.pkl',
                        help='Path to save the final filtered and augmented dataset')

    args = parser.parse_args()

    # Optional: Construct a more descriptive default output path
    if args.output_path == f'libero_spatial_filtered_augmented_h{DEFAULT_HISTORY_LENGTH}.pkl': # Check if default
         args.output_path = f'libero_spatial_filtered_L{args.resample_length}_std{args.std_dev_threshold}_mind{args.min_demos}_augmented_h{args.history_length}.pkl'
         print(f"Using default output path format: {args.output_path}")
    else:
        print(f"Using specified output path: {args.output_path}")


    filter_and_augment_dataset(args.dataset_path, args.dataset_folders, args.output_path,
                                resample_len=args.resample_length,
                                std_dev_thresh=args.std_dev_threshold,
                                min_demos=args.min_demos,
                                history_length=args.history_length)
