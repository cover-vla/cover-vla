#!/usr/bin/env python3
"""
Script to trim rephrases in augmented Bridge dataset JSON file.
Ensures each JPG image has at most one rephrase (original + 1 rephrase max).
"""

import json
import argparse
from collections import defaultdict
from tqdm import tqdm


def trim_rephrases_per_jpg(input_json_path, output_json_path, max_rephrases_per_jpg=1):
    """
    Trim the dataset to have at most max_rephrases_per_jpg per JPG image.
    
    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to output JSON file
        max_rephrases_per_jpg: Maximum number of rephrases per JPG (default: 1)
                               This means original + max_rephrases_per_jpg total per JPG
    """
    print(f"Loading dataset from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        dataset = json.load(f)
    
    # Extract components
    action_histories = dataset.get('action_histories', {})
    instructions = dataset.get('instructions', {})
    samples = dataset.get('samples', [])
    metadata = dataset.get('_metadata', {})
    
    print(f"Original dataset:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Total instructions: {len(instructions)}")
    print(f"  Total unique action histories: {len(action_histories)}")
    
    # Group samples by JPG filename
    samples_by_jpg = defaultdict(list)
    for sample in samples:
        jpg_file = sample['agent_view_image_file']
        samples_by_jpg[jpg_file].append(sample)
    
    print(f"\nGrouping samples by JPG...")
    print(f"  Total unique JPG files: {len(samples_by_jpg)}")
    
    # Find samples that appear to be from rephrases vs originals
    # We'll identify this by looking at which samples share the same
    # (agent_view_image_file, action_history_id, episode_id, timestep)
    # but have different instruction_ids
    
    kept_samples = []
    total_removed = 0
    
    print(f"\nTrimming rephrases (keeping at most {max_rephrases_per_jpg} rephrase per JPG)...")
    
    for jpg_file, jpg_samples in tqdm(samples_by_jpg.items(), desc="Processing JPGs"):
        # Group by (action_history_id, episode_id, timestep) to find duplicates
        context_groups = defaultdict(list)
        for sample in jpg_samples:
            context_key = (
                sample['action_history_id'],
                sample['episode_id'],
                sample['timestep']
            )
            context_groups[context_key].append(sample)
        
        # For each context group, keep original + max_rephrases_per_jpg rephrases
        for context_key, context_samples in context_groups.items():
            if len(context_samples) <= max_rephrases_per_jpg + 1:
                # Keep all if we're at or under the limit
                # (+1 because we want original + max_rephrases_per_jpg rephrases)
                kept_samples.extend(context_samples)
            else:
                # Keep only the first (max_rephrases_per_jpg + 1) samples
                # This assumes the first one is the original, followed by rephrases
                kept_count = max_rephrases_per_jpg + 1
                kept_samples.extend(context_samples[:kept_count])
                total_removed += len(context_samples) - kept_count
    
    print(f"\nRemoved {total_removed} samples (rephrases)")
    print(f"Kept {len(kept_samples)} samples")
    
    # Clean up unused instructions and action histories
    print("\nCleaning up unused instructions and action histories...")
    
    used_instruction_ids = set(sample['instruction_id'] for sample in kept_samples)
    used_action_history_ids = set(sample['action_history_id'] for sample in kept_samples)
    
    # Filter instructions
    cleaned_instructions = {
        instr_id: instr_text 
        for instr_id, instr_text in instructions.items() 
        if instr_id in used_instruction_ids
    }
    
    # Filter action histories
    cleaned_action_histories = {
        action_id: action_hist 
        for action_id, action_hist in action_histories.items() 
        if action_id in used_action_history_ids
    }
    
    removed_instructions = len(instructions) - len(cleaned_instructions)
    removed_action_histories = len(action_histories) - len(cleaned_action_histories)
    
    print(f"Removed {removed_instructions} unused instructions")
    print(f"Removed {removed_action_histories} unused action histories")
    
    # Update metadata
    updated_metadata = metadata.copy()
    updated_metadata['total_samples'] = len(kept_samples)
    updated_metadata['total_instructions'] = len(cleaned_instructions)
    updated_metadata['total_unique_action_histories'] = len(cleaned_action_histories)
    
    if len(cleaned_action_histories) > 0:
        updated_metadata['compression_ratio'] = len(kept_samples) / len(cleaned_action_histories)
    
    # Add trimming info to metadata
    updated_metadata['trimmed'] = True
    updated_metadata['max_rephrases_per_jpg'] = max_rephrases_per_jpg
    updated_metadata['samples_removed'] = total_removed
    updated_metadata['instructions_removed'] = removed_instructions
    updated_metadata['action_histories_removed'] = removed_action_histories
    
    # Build trimmed dataset
    trimmed_dataset = {
        'action_histories': cleaned_action_histories,
        'instructions': cleaned_instructions,
        'samples': kept_samples,
        '_metadata': updated_metadata
    }
    
    # Save trimmed dataset
    print(f"\nSaving trimmed dataset to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(trimmed_dataset, f, indent=2)
    
    print("\n✓ Trimming complete!")
    print(f"\nFinal dataset:")
    print(f"  Total samples: {len(kept_samples)}")
    print(f"  Total instructions: {len(cleaned_instructions)}")
    print(f"  Total unique action histories: {len(cleaned_action_histories)}")
    print(f"  Removed samples: {total_removed}")
    print(f"  Removed instructions: {removed_instructions}")
    print(f"  Removed action histories: {removed_action_histories}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trim rephrases in augmented Bridge dataset to have at most N rephrases per JPG'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file (e.g., augment_bridge64.json)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file (e.g., augment_bridge64_trimmed.json)')
    parser.add_argument('--max_rephrases', type=int, default=1,
                        help='Maximum number of rephrases per JPG (default: 1, meaning original + 1 rephrase)')
    
    args = parser.parse_args()
    
    trim_rephrases_per_jpg(
        input_json_path=args.input,
        output_json_path=args.output,
        max_rephrases_per_jpg=args.max_rephrases
    )

