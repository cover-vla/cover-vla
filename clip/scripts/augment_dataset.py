import argparse
import h5py
import os
import json
import numpy as np
from tqdm import tqdm
import pickle
from lang_transform import LangTransform

def augment_dataset(dataset_path, dataset_folders, output_path):
    """
    Augment the dataset by applying all language transformations to each instruction.
    
    Args:
        dataset_path: Base path to the dataset
        dataset_folders: List of dataset folders to process
        output_path: Path to save the augmented dataset
    """
    # Initialize language transformation
    lang_transform = LangTransform()
    
    ######## Select the transformations to use ########
    transformations = [
        'synonym', 
        # 'antonym', 
        # 'negation', 
        # 'verb_noun_shuffle', 
        # 'random_shuffle'
    ]
    
    # Dictionary to track which transformations create positive vs negative examples
    positive_transforms = ['synonym']
    negative_transforms = ['negation','antonym','verb_noun_shuffle','random_shuffle']
    
    # Create the augmented dataset dictionary
    augmented_dataset = {}
    
    print("Loading and augmenting dataset...")
    # Process each dataset folder
    for folder in dataset_folders:
        folder_path = os.path.join(dataset_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Dataset folder {folder_path} does not exist. Skipping.")
            continue
        
        # Process each task file in the folder
        for task in tqdm(os.listdir(folder_path), desc=f"Processing dataset {folder}"):
            if not task.endswith('.hdf5'):
                continue
                
            # Extract task name without .hdf5 extension as the language instruction
            original_instruction = task.replace('.hdf5', '').replace('_', ' ')
            # Add original instruction to dataset
            augmented_dataset[original_instruction] = {
                'actions': [],
                'images': [],
                'is_original': True
            }
            
            # Generate transformed instructions
            transformed_instructions = {}
            for transform_type in transformations:
                try:
                    transformed_text = lang_transform.transform(original_instruction, transform_type)
                    transformed_instructions[transform_type] = transformed_text
                    
                    # Add transformed instruction to dataset
                    augmented_dataset[transformed_text] = {
                        'actions': [],
                        'images': [],
                        'original_instruction': original_instruction,
                        'transform_type': transform_type,
                        'is_positive': transform_type in positive_transforms
                    }
                except Exception as e:
                    print(f"Error applying {transform_type} to '{original_instruction}': {e}")
            
            # Load the actual data
            task_path = os.path.join(folder_path, task)
            with h5py.File(task_path, 'r') as f:
                for demo_key in f['data'].keys():
                    demo_data = f['data'][demo_key]
                    
                    # Get actions data
                    actions = demo_data['actions'][()]
                    
                    # Get observation data
                    obs_group = demo_data['obs']
                    obs_data = obs_group['agentview_rgb'][()]
                    
                    # Add to original instruction
                    augmented_dataset[original_instruction]['actions'].append(actions)
                    augmented_dataset[original_instruction]['images'].append(obs_data)
                    
                    # Add to transformed instructions
                    for transform_type, transformed_text in transformed_instructions.items():
                        if transform_type in positive_transforms:
                            # For positive transforms, keep the same actions
                            augmented_dataset[transformed_text]['actions'].append(actions)
                        else:
                            # For negative transforms, use zero actions
                            zero_actions = np.zeros_like(actions)
                            augmented_dataset[transformed_text]['actions'].append(zero_actions)
                        
                        # Always use the same images
                        augmented_dataset[transformed_text]['images'].append(obs_data)
    
    # Print statistics
    total_original = sum(1 for k, v in augmented_dataset.items() if v.get('is_original', False))
    total_transformed = len(augmented_dataset) - total_original
    total_positive = sum(1 for k, v in augmented_dataset.items() 
                         if not v.get('is_original', False) and v.get('is_positive', False))
    total_negative = total_transformed - total_positive
    
    print(f"Dataset augmentation complete!")
    print(f"Original instructions: {total_original}")
    print(f"Transformed instructions: {total_transformed}")
    print(f"  - Positive transformations: {total_positive}")
    print(f"  - Negative transformations: {total_negative}")
    print(f"Total dataset size: {len(augmented_dataset)}")
    
    # Save the augmented dataset
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
    parser.add_argument('--output_path', type=str, default='augmented_dataset_positive_only.pkl',
                        help='Path to save the augmented dataset')
    
    args = parser.parse_args()
    
    augment_dataset(args.dataset_path, args.dataset_folders, args.output_path) 