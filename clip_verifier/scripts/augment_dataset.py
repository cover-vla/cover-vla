import argparse
import h5py
import os
import json
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
from lang_transform import LangTransform

def augment_dataset(dataset_path, dataset_folders, output_path):
    """
    Augment the dataset by applying language transformations.
    Generates negative action samples based on the distribution of actions 
    at each specific timestep across all demonstrations for a task, clipped to [-1, 1].
    
    Args:
        dataset_path: Base path to the dataset
        dataset_folders: List of dataset folders to process
        output_path: Path to save the augmented dataset
    """
    lang_transform = LangTransform()
    
    transformations = [
        'synonym', 'antonym', 'negation', 
        'out_set', 'rephrase'
        # Add others like 'verb_noun_shuffle', 'random_shuffle' if needed
    ]
    positive_transforms = ['synonym', 'out_set', 'rephrase']
    negative_transforms = [t for t in transformations if t not in positive_transforms]
    
    augmented_dataset = {}
    
    print("Loading and augmenting dataset...")
    for folder in dataset_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Dataset folder {folder_path} does not exist. Skipping.")
            continue
        
        for task in tqdm(os.listdir(folder_path), desc=f"Processing dataset {folder}"):
            if not task.endswith('.hdf5'):
                continue
                
            # --- Clean instruction ---
            original_instruction = task.replace('.hdf5', '').replace('_', ' ')
            original_instruction = ''.join(char for char in original_instruction if not char.isupper() and not char.isdigit())
            while original_instruction and original_instruction[0].isspace():
                original_instruction = original_instruction[1:]
            if not original_instruction:
                 print(f"Warning: Instruction became empty for file {task}. Skipping.")
                 continue

            # === Pass 1: Load all demos and compute timestep statistics ===
            task_demos_data = []
            actions_by_timestep = defaultdict(list) # Store actions grouped by timestep index
            max_len = 0
            action_dim = None # Initialize action_dim here
            task_path = os.path.join(folder_path, task)
            
            try:
                with h5py.File(task_path, 'r') as f:
                    if 'data' not in f:
                        print(f"Warning: 'data' group not found in {task_path}. Skipping task.")
                        continue
                    for demo_key in f['data'].keys():
                        demo_data = f['data'][demo_key]
                        if not all(k in demo_data for k in ['actions', 'obs']) or \
                           'agentview_rgb' not in demo_data['obs']:
                            print(f"Warning: Missing data in demo {demo_key} of {task_path}. Skipping demo.")
                            continue
                            
                        actions = demo_data['actions'][()]
                        obs_data = demo_data['obs']['agentview_rgb'][()]
                        
                        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
                             print(f"Warning: Invalid actions shape {actions.shape} in demo {demo_key} of {task_path}. Skipping demo.")
                             continue
                        
                        if action_dim is None:
                             action_dim = actions.shape[1] # Determine action dim
                             if action_dim <= 0:
                                  print(f"Error: Invalid action dimension {action_dim} detected in {task_path}, demo {demo_key}. Skipping task.")
                                  raise ValueError("Invalid action dimension") # Raise error to break out and skip task
                                  
                        elif actions.shape[1] != action_dim:
                             print(f"Warning: Inconsistent action dimension ({actions.shape[1]} vs {action_dim}) in demo {demo_key} of {task_path}. Skipping demo.")
                             continue # Skip this demo

                        T = actions.shape[0]
                        task_demos_data.append({'actions': actions, 'images': obs_data, 'demo_key': demo_key, 'len': T})
                        max_len = max(max_len, T)
                        
                        for t in range(T):
                            actions_by_timestep[t].append(actions[t]) 

            except Exception as e:
                print(f"Error processing file {task_path}: {e}. Skipping task.")
                continue 

            if not task_demos_data or action_dim is None: # Check action_dim too
                print(f"Warning: No valid demonstrations or action dim could not be determined for task '{original_instruction}'. Skipping task.")
                continue

            # --- Calculate timestep statistics ---
            timestep_stats = {}
            for t in range(max_len):
                if t in actions_by_timestep and len(actions_by_timestep[t]) > 1: 
                    actions_t = np.stack(actions_by_timestep[t], axis=0) 
                    mean_t = np.mean(actions_t, axis=0)
                    std_t = np.std(actions_t, axis=0)
                    std_t = np.where(std_t < 1e-6, 1e-6, std_t) 
                    timestep_stats[t] = {'mean': mean_t, 'std': std_t}
                elif t in actions_by_timestep: 
                     mean_t = actions_by_timestep[t][0] 
                     std_t = np.full_like(mean_t, 1e-6)
                     timestep_stats[t] = {'mean': mean_t, 'std': std_t}

            # === Pass 2: Generate transformations and populate dataset ===

            # --- Setup dataset entries ---
            if original_instruction not in augmented_dataset:
                augmented_dataset[original_instruction] = {'actions': [], 'images': [], 'is_original': True}
            else:
                 print(f"Warning: Original instruction '{original_instruction}' collision. Appending data.")
                 augmented_dataset[original_instruction]['is_original'] = True # Ensure marked as original

            generated_transforms = {} 
            for transform_type in transformations:
                try:
                    transformed_text = lang_transform.transform(original_instruction, transform_type)
                    if transformed_text == original_instruction: continue
                    is_positive = transform_type in positive_transforms
                    
                    # Handle collisions
                    if transformed_text in augmented_dataset and not augmented_dataset[transformed_text].get('is_original', False):
                        print(f"Warning: Transformed text '{transformed_text}' collision for task '{original_instruction}'. Skipping transform '{transform_type}'.")
                        continue
                        
                    if transformed_text not in augmented_dataset or augmented_dataset[transformed_text].get('is_original', False):
                        augmented_dataset[transformed_text] = {
                            'actions': [], 'images': [], 
                            'original_instruction': original_instruction,
                            'transform_type': transform_type, 'is_positive': is_positive
                        }
                        generated_transforms[transform_type] = transformed_text
                except Exception as e:
                    print(f"Error applying {transform_type} to '{original_instruction}': {e}")

            # --- Populate entries with demo data and generate negative actions ---
            fallback_mean = np.zeros(action_dim) 
            fallback_std = np.ones(action_dim) * 1e-6 

            for demo_data in task_demos_data:
                original_actions = demo_data['actions'] # (T, D)
                images = demo_data['images']            # (T, H, W, C)
                T, D = original_actions.shape # D is the original action_dim
                demo_key = demo_data['demo_key']

                # Generate the negative action sequence for this demo
                negative_actions = np.zeros_like(original_actions)
                for t in range(T):
                    if t in timestep_stats:
                        mean_t = timestep_stats[t]['mean'] # Shape (D,)
                        std_t = timestep_stats[t]['std']   # Shape (D,)
                    else:
                        mean_t = fallback_mean
                        std_t = fallback_std
                    
                    # --- Generate first D-1 dimensions based on stats ---
                    if D > 1:
                        mean_t_prefix = mean_t[:-1]
                        std_t_prefix = std_t[:-1]
                        z_prefix = np.random.randn(D - 1) 
                        signs_prefix = np.random.choice([-1, 1], size=D - 1)
                        neg_action_prefix = mean_t_prefix + (signs_prefix * 2 + z_prefix) * std_t_prefix
                    else: # Handle case D=1 separately
                        neg_action_prefix = np.array([]) # Empty prefix

                    # --- Generate last dimension randomly as -1 or 1 ---
                    last_dim_value = np.random.choice([-1.0, 1.0])

                    # --- Combine dimensions ---
                    # Use np.concatenate which handles the D=1 case correctly
                    neg_action_t = np.concatenate((neg_action_prefix, [last_dim_value]))

                    # --- Clip the final action vector (primarily affects the first D-1 dims) ---
                    neg_action_t = np.clip(neg_action_t, -1.0, 1.0)
                    
                    # Assign to the sequence
                    negative_actions[t] = neg_action_t
                
                # Ensure correct dtype at the end of sequence generation
                negative_actions = negative_actions.astype(original_actions.dtype)

                # Add data to original instruction entry
                augmented_dataset[original_instruction]['images'].append(images)
                augmented_dataset[original_instruction]['actions'].append(original_actions)

                # Add data to transformed instruction entries
                for transform_type, transformed_text in generated_transforms.items():
                    augmented_dataset[transformed_text]['images'].append(images)
                    if transform_type in positive_transforms:
                        augmented_dataset[transformed_text]['actions'].append(original_actions)
                    elif transform_type in negative_transforms:
                        # Assign the pre-generated negative sequence with modified last dim
                        augmented_dataset[transformed_text]['actions'].append(negative_actions) 
    
    # --- Final Cleanup ---
    keys_to_remove = [k for k, v in augmented_dataset.items() if not v['actions'] or not v['images']]
    if keys_to_remove:
        print(f"Removing {len(keys_to_remove)} entries with no valid demo data.")
        for k in keys_to_remove: del augmented_dataset[k]

    # --- Print statistics ---
    total_original = sum(1 for v in augmented_dataset.values() if v.get('is_original', False))
    total_transformed = len(augmented_dataset) - total_original
    total_positive = sum(1 for v in augmented_dataset.values() 
                         if not v.get('is_original', False) and v.get('is_positive') is True) 
    total_negative = sum(1 for v in augmented_dataset.values() 
                         if not v.get('is_original', False) and v.get('is_positive') is False) 

    print(f"Dataset augmentation complete!")
    print(f"Original instructions: {total_original}")
    print(f"Transformed instructions: {total_transformed}")
    print(f"  - Positive examples (transformed): {total_positive}")
    print(f"  - Negative examples (transformed): {total_negative}")
    print(f"Total dataset size (instructions): {len(augmented_dataset)}")

    # --- Save dataset ---
    print(f"Saving augmented dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(augmented_dataset, f)
    print("Done!")
    return augmented_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment dataset with language transformations')
    parser.add_argument('--dataset_path', type=str, default='/home/xilun/LIBERO/libero/datasets',
                        help='Path to the dataset')
    parser.add_argument('--dataset_folders', nargs='+', default=['libero_spatial'],
                        help='Dataset folders to process')
    # Update default output name maybe
    parser.add_argument('--output_path', type=str, default='libero_spatial_all.pkl', # Updated default name
                        help='Path to save the augmented dataset') 
    
    args = parser.parse_args()
    
    augment_dataset(args.dataset_path, args.dataset_folders, args.output_path) 