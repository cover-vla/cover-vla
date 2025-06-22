#!/usr/bin/env python3
"""
Script to explore and print examples from the augmented dataset.
Shows language instructions, action sequences, and image information.
"""

import argparse
import pickle
import numpy as np
from collections import Counter
import os

def print_action_stats(actions, name="Actions"):
    """Print statistics about action sequences"""
    print(f"\n{name} Statistics:")
    print(f"  Shape: {actions.shape}")
    print(f"  Min: {np.min(actions, axis=0)}")
    print(f"  Max: {np.max(actions, axis=0)}")
    print(f"  Mean: {np.mean(actions, axis=0)}")
    print(f"  Std: {np.std(actions, axis=0)}")
    
    # Check for padding values
    padding_mask = (actions == -5.0)
    if np.any(padding_mask):
        padding_count = np.sum(padding_mask)
        total_elements = actions.size
        print(f"  Padding (-5.0): {padding_count}/{total_elements} elements ({100*padding_count/total_elements:.1f}%)")

def explore_dataset(dataset_path, max_instructions=10, max_samples_per_instruction=3):
    """
    Load and explore the augmented dataset
    
    Args:
        dataset_path: Path to the pickled dataset file
        max_instructions: Maximum number of instructions to show
        max_samples_per_instruction: Maximum samples to show per instruction
    """
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Dataset loaded successfully!")
    
    # Overall statistics
    total_instructions = len(dataset)
    total_samples = sum(len(data.get('samples', [])) for data in dataset.values())
    
    print(f"\n{'='*60}")
    print(f"DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"Total instructions: {total_instructions}")
    print(f"Total samples: {total_samples}")
    print(f"Average samples per instruction: {total_samples/total_instructions:.1f}")
    
    # Sample length distribution
    sample_counts = [len(data.get('samples', [])) for data in dataset.values()]
    print(f"Sample count distribution:")
    print(f"  Min: {min(sample_counts)}")
    print(f"  Max: {max(sample_counts)}")
    print(f"  Mean: {np.mean(sample_counts):.1f}")
    print(f"  Median: {np.median(sample_counts):.1f}")
    
    # Get action dimensions from first sample
    first_instruction = next(iter(dataset.keys()))
    first_sample = dataset[first_instruction]['samples'][0]
    action_hist = first_sample['pos_action_hist']
    history_length, action_dim = action_hist.shape
    
    print(f"\nAction History Configuration:")
    print(f"  History length: {history_length}")
    print(f"  Action dimension: {action_dim}")
    
    # Image information
    image_keys = list(first_sample['images'].keys())
    print(f"\nImage Views Available: {image_keys}")
    for key in image_keys:
        img_shape = first_sample['images'][key].shape
        print(f"  {key}: {img_shape}")
    
    print(f"\n{'='*60}")
    print(f"INSTRUCTION EXAMPLES")
    print(f"{'='*60}")
    
    # Show examples
    instruction_count = 0
    for instruction, data in dataset.items():
        if instruction_count >= max_instructions:
            break
            
        samples = data.get('samples', [])
        if not samples:
            continue
            
        print(f"\n[{instruction_count + 1}] Instruction: \"{instruction}\"")
        print(f"    Number of samples: {len(samples)}")
        
        # Show a few samples for this instruction
        sample_count = 0
        for i, sample in enumerate(samples):
            if sample_count >= max_samples_per_instruction:
                break
                
            print(f"\n    Sample {i+1}:")
            
            # Action history
            action_hist = sample['pos_action_hist']
            print(f"      Action History Shape: {action_hist.shape}")
            
            # Check for padding
            padding_mask = (action_hist == -5.0)
            if np.any(padding_mask):
                non_padded_steps = np.sum(~np.all(padding_mask, axis=1))
                print(f"      Non-padded steps: {non_padded_steps}/{history_length}")
            else:
                print(f"      All steps are non-padded")
            
            # Show first few action steps
            print(f"      First 3 action steps:")
            for step in range(min(3, action_hist.shape[0])):
                action = action_hist[step]
                if np.all(action == -5.0):
                    print(f"        Step {step}: [PADDED]")
                else:
                    action_str = ", ".join([f"{x:.3f}" for x in action])
                    print(f"        Step {step}: [{action_str}]")
            
            # Image info
            images = sample['images']
            print(f"      Images:")
            for img_key, img_data in images.items():
                print(f"        {img_key}: {img_data.shape}, dtype={img_data.dtype}")
                print(f"          Range: [{np.min(img_data):.1f}, {np.max(img_data):.1f}]")
            
            sample_count += 1
        
        instruction_count += 1
    
    # Action statistics across all samples
    print(f"\n{'='*60}")
    print(f"GLOBAL ACTION STATISTICS")
    print(f"{'='*60}")
    
    all_actions = []
    all_non_padded_actions = []
    
    for instruction, data in dataset.items():
        for sample in data.get('samples', []):
            action_hist = sample['pos_action_hist']
            all_actions.append(action_hist)
            
            # Collect non-padded actions
            padding_mask = (action_hist == -5.0)
            non_padded_mask = ~np.all(padding_mask, axis=1)
            if np.any(non_padded_mask):
                non_padded_actions = action_hist[non_padded_mask]
                all_non_padded_actions.append(non_padded_actions)
    
    if all_actions:
        all_actions_array = np.concatenate(all_actions, axis=0)
        print_action_stats(all_actions_array, "All Actions (including padding)")
    
    if all_non_padded_actions:
        all_non_padded_array = np.concatenate(all_non_padded_actions, axis=0)
        print_action_stats(all_non_padded_array, "Non-padded Actions Only")
    
    # Instruction length statistics
    print(f"\n{'='*60}")
    print(f"INSTRUCTION ANALYSIS")
    print(f"{'='*60}")
    
    instruction_lengths = [len(instr.split()) for instr in dataset.keys()]
    print(f"Instruction word count statistics:")
    print(f"  Min words: {min(instruction_lengths)}")
    print(f"  Max words: {max(instruction_lengths)}")
    print(f"  Mean words: {np.mean(instruction_lengths):.1f}")
    print(f"  Median words: {np.median(instruction_lengths):.1f}")
    
    # Most common words
    all_words = []
    for instr in dataset.keys():
        all_words.extend(instr.lower().split())
    
    word_counts = Counter(all_words)
    print(f"\nMost common words:")
    for word, count in word_counts.most_common(10):
        print(f"  '{word}': {count}")
    
    # Show some example instructions by length
    short_instructions = [instr for instr in dataset.keys() if len(instr.split()) <= 3]
    long_instructions = [instr for instr in dataset.keys() if len(instr.split()) >= 6]
    
    if short_instructions:
        print(f"\nExample short instructions:")
        for instr in short_instructions[:5]:
            print(f"  \"{instr}\"")
    
    if long_instructions:
        print(f"\nExample long instructions:")
        for instr in long_instructions[:5]:
            print(f"  \"{instr}\"")

def main():
    parser = argparse.ArgumentParser(description='Explore augmented dataset and print examples')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the pickled dataset file')
    parser.add_argument('--max_instructions', type=int, default=10,
                        help='Maximum number of instructions to show examples for')
    parser.add_argument('--max_samples', type=int, default=3,
                        help='Maximum samples to show per instruction')
    
    args = parser.parse_args()
    
    explore_dataset(args.dataset_path, args.max_instructions, args.max_samples)

if __name__ == "__main__":
    main() 