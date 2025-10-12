#!/usr/bin/env python3
"""
Script to rephrase unique instructions from bridge_samples.json using LangTransform
Generates 128 rephrases for each unique instruction using multithreading for speed
"""

import os
import sys
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from lang_transform_openai import LangTransform

def clean_rephrase(rephrase):
    """
    Clean a rephrase to ensure it's lowercase and has no trailing punctuation
    
    Args:
        rephrase (str): The original rephrase
        
    Returns:
        str: Cleaned rephrase (lowercase, no trailing punctuation)
    """
    if not rephrase or not isinstance(rephrase, str):
        return ""
    
    # Convert to lowercase
    cleaned = rephrase.strip().lower()
    
    # Remove trailing punctuation (., !, ?, :, ;)
    cleaned = re.sub(r'[.!?:;]+$', '', cleaned).strip()
    
    return cleaned

def get_unique_instructions_from_bridge_data(json_file):
    """Extract unique instructions from bridge_samples.json"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    instructions = [sample['original_instruction'] for sample in data['samples']]
    unique_instructions = list(set(instructions))
    
    print(f"Found {len(instructions)} total samples with {len(unique_instructions)} unique instructions")
    return unique_instructions

def process_single_instruction(instruction, thread_id):
    """
    Process a single instruction to generate rephrases
    
    Args:
        instruction (str): The instruction to rephrase
        thread_id (int): Thread identifier for debugging
        
    Returns:
        tuple: (instruction, result_dict)
    """
    try:
        # Create a new LangTransform instance for this thread
        lang_transformer = LangTransform()
        
        # Use the rephrase transform to generate 128 variations
        raw_rephrases = lang_transformer.transform(
            curr_instruction=instruction,
            transform_type='rephrase',  # Use the rephrase transform type
            batch_number=128  # Generate 128 rephrases at once
        )
        
        # Clean the rephrases (lowercase, no trailing punctuation)
        cleaned_rephrases = []
        for rephrase in raw_rephrases:
            cleaned = clean_rephrase(rephrase)
            if cleaned:  # Only add non-empty cleaned rephrases
                cleaned_rephrases.append(cleaned)
        
        result = {
            'original': instruction,
            'rephrases': cleaned_rephrases,
            'count': len(cleaned_rephrases),
            'raw_count': len(raw_rephrases),
            'thread_id': thread_id
        }
        
        return (instruction, result)
        
    except Exception as e:
        result = {
            'original': instruction,
            'rephrases': [],
            'count': 0,
            'error': str(e),
            'thread_id': thread_id
        }
        return (instruction, result)

def main():
    # Set the OpenAI API key
    os.environ['OPENAI_API_KEY'] = 'TODO'
    
    # Get unique instructions from bridge dataset
    bridge_json_file = 'json_results/bridge_samples.json'
    if not os.path.exists(bridge_json_file):
        print(f"Error: {bridge_json_file} not found!")
        return
    
    unique_instructions = get_unique_instructions_from_bridge_data(bridge_json_file)
    
    # Dictionary to store all rephrases
    all_rephrases = {}
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(10, len(unique_instructions))  # Limit to 8 threads to avoid API rate limits
    print(f"\nUsing {max_workers} threads to process {len(unique_instructions)} unique instructions...")
    print("Generating 128 rephrases for each instruction...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_instruction = {
            executor.submit(process_single_instruction, instruction, i % max_workers): instruction 
            for i, instruction in enumerate(unique_instructions)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(unique_instructions), desc="Processing instructions") as pbar:
            for future in as_completed(future_to_instruction):
                instruction, result = future.result()
                all_rephrases[instruction] = result
                
                # Update progress
                if result['count'] > 0:
                    status = f"✓ {result['count']} rephrases"
                else:
                    status = f"✗ Failed: {result.get('error', 'Unknown error')[:30]}..."
                
                pbar.set_postfix_str(status)
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"json_results/bridge_instruction_rephrases_{timestamp}.json"
    
    print(f"\nSaving results to {output_file}...")
    
    # Prepare summary data
    summary_data = {
        'timestamp': timestamp,
        'source_file': bridge_json_file,
        'total_original_instructions': len(unique_instructions),
        'target_rephrases_per_instruction': 128,
        'total_rephrases_generated': sum(data['count'] for data in all_rephrases.values()),
        'processing_time_seconds': elapsed_time,
        'max_workers': max_workers,
        'instructions': all_rephrases
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"Results successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
        return
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_rephrases = 0
    successful_instructions = 0
    failed_instructions = 0
    
    for instruction, data in all_rephrases.items():
        count = data['count']
        total_rephrases += count
        if count > 0:
            successful_instructions += 1
        else:
            failed_instructions += 1
    
    print(f"Total unique instructions processed: {len(unique_instructions)}")
    print(f"Successfully rephrased: {successful_instructions}")
    print(f"Failed to rephrase: {failed_instructions}")
    print(f"Total rephrases generated: {total_rephrases}")
    print(f"Target was: {len(unique_instructions) * 128} rephrases")
    print(f"Average rephrases per instruction: {total_rephrases / len(unique_instructions):.1f}")
    print(f"Processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Average time per instruction: {elapsed_time/len(unique_instructions):.1f} seconds")
    print(f"Results saved to: {output_file}")
    
    # Show some statistics about failed instructions if any
    if failed_instructions > 0:
        print(f"\nFailed instructions ({failed_instructions}):")
        for instruction, data in all_rephrases.items():
            if data['count'] == 0:
                error_msg = data.get('error', 'Unknown error')
                print(f"  - '{instruction[:50]}...' (Error: {error_msg[:50]}...)")
    
    # Show preview of successful rephrases
    if successful_instructions > 0:
        print(f"\nPreview of rephrases for first successful instruction:")
        for instruction, data in all_rephrases.items():
            if data['count'] > 0:
                print(f"Original: '{instruction}'")
                print("Sample rephrases:")
                for i, rephrase in enumerate(data['rephrases'][:3], 1):
                    print(f"  {i}. {rephrase}")
                if len(data['rephrases']) > 3:
                    print(f"  ... and {len(data['rephrases']) - 3} more")
                break

if __name__ == "__main__":
    main()
