#!/usr/bin/env python3
"""
Generate monkey-verifier scores for 12,900 datapoints (128 rephrased instructions × 100 datapoints).
Uses the monkey-verifier API to score the combination of:
- Bridge dataset images (_robomonkey.jpg format)
- Rephrased instructions 
- OpenVLA predicted actions
"""

import requests
import numpy as np
import json
import os
from tqdm import tqdm
from datetime import datetime

def load_data_files():
    """Load all required data files"""
    print("Loading data files...")
    
    # Load bridge samples
    with open('json_results/bridge_samples.json', 'r') as f:
        bridge_data = json.load(f)
    print(f"Loaded {len(bridge_data['samples'])} bridge samples")
    
    # Find the most recent rephrase file
    rephrase_files = [f for f in os.listdir('json_results/') 
                     if f.startswith('bridge_instruction_rephrases_') and f.endswith('.json')]
    if not rephrase_files:
        raise FileNotFoundError("No rephrase file found!")
    rephrase_file = sorted(rephrase_files)[-1]  # Get the most recent one
    
    with open(f'json_results/{rephrase_file}', 'r') as f:
        rephrase_data = json.load(f)
    print(f"Loaded rephrased instructions from {rephrase_file}")
    print(f"Found rephrases for {len(rephrase_data['instructions'])} original instructions")
    
    # Find the most recent OpenVLA actions file
    openvla_files = [f for f in os.listdir('json_results/') 
                    if f.startswith('bridge_openvla_actions_') and f.endswith('.json')]
    if not openvla_files:
        raise FileNotFoundError("No OpenVLA actions file found!")
    openvla_file = sorted(openvla_files)[-1]  # Get the most recent one
    
    with open(f'json_results/{openvla_file}', 'r') as f:
        openvla_data = json.load(f)
    print(f"Loaded OpenVLA actions from {openvla_file}")
    print(f"Found {len(openvla_data['results'])} OpenVLA action predictions")
    
    return bridge_data, rephrase_data, openvla_data, rephrase_file, openvla_file

def get_monkey_verifier_score(instruction, image_path, action_tokens):
    """
    Get score from monkey-verifier API
    
    Args:
        instruction (str): The instruction text
        image_path (str): Path to the preprocessed robomonkey image
        action_tokens (list): Tokenized action from OpenVLA output_ids
    
    Returns:
        float: Reward score from the verifier
    """
    url = "http://127.0.0.1:3100/process"
    
    payload = {
        "instruction": instruction,
        "image_path": image_path,
        "action": action_tokens
    }
    
    try:
        response = requests.post(url, data=json.dumps(payload), 
                               headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        result = json.loads(response.text)
        return result['rewards']
    except Exception as e:
        print(f"Error calling monkey-verifier API: {e}")
        return None

def generate_monkey_verifier_scores():
    """Generate monkey-verifier scores for all datapoints"""
    
    # Load data
    bridge_data, rephrase_data, openvla_data, rephrase_file, openvla_file = load_data_files()
    
    # Create mapping from OpenVLA results for fast lookup
    openvla_lookup = {}
    for result in openvla_data['results']:
        key = (result['sample_index'], result['instruction_index'])
        openvla_lookup[key] = result
    
    print(f"Created lookup table for {len(openvla_lookup)} OpenVLA predictions")
    
    # Prepare results
    results = []
    samples = bridge_data['samples']
    
    # Calculate expected total
    total_instructions_per_sample = 0
    for sample in samples:
        original_instruction = sample['original_instruction']
        if original_instruction in rephrase_data['instructions']:
            instruction_data = rephrase_data['instructions'][original_instruction]
            num_variants = 1 + len(instruction_data['rephrases'])  # original + rephrases
            total_instructions_per_sample += num_variants
    
    print(f"Expected total datapoints: {total_instructions_per_sample}")
    print(f"Processing {len(samples)} samples with monkey-verifier...")
    
    # Process each sample
    for sample in tqdm(samples, desc="Processing samples"):
        sample_id = sample['sample_id']
        sample_index = sample_id  # Assuming sample_id matches the index used in OpenVLA
        original_instruction = sample['original_instruction']
        
        # Get RoboMonkey image path (already preprocessed to 224x224)
        image_filename = sample['state']['agent_view_image_file']
        base_name = image_filename.split('_')[0]  # Extract number from "N_clip.jpg"
        robomonkey_image_path = f"processed_images/robomonkey/{base_name}_robomonkey.jpg"
        
        if not os.path.exists(robomonkey_image_path):
            print(f"Warning: RoboMonkey image not found: {robomonkey_image_path}")
            continue
        
        # Get rephrased instructions for this original instruction
        if original_instruction not in rephrase_data['instructions']:
            print(f"Warning: No rephrases found for instruction: {original_instruction}")
            continue
        
        instruction_data = rephrase_data['instructions'][original_instruction]
        all_instructions = [instruction_data['original']] + instruction_data['rephrases']
        
        # Process each instruction variant (original + rephrases)
        for instruction_idx, instruction in enumerate(all_instructions):
            # Get OpenVLA prediction for this sample-instruction combination
            openvla_key = (sample_index, instruction_idx)
            if openvla_key not in openvla_lookup:
                print(f"Warning: No OpenVLA prediction found for sample {sample_index}, instruction {instruction_idx}")
                continue
            
            openvla_result = openvla_lookup[openvla_key]
            predicted_action = openvla_result['generated_action']
            output_ids = openvla_result['output_ids']  # Use pre-computed tokenized action
            
            # Use the output_ids as tokenized action (wrap in list as expected by API)
            action_input = [output_ids]
            
            # Get monkey-verifier score
            try:
                verifier_score = get_monkey_verifier_score(
                    instruction=instruction,
                    image_path=robomonkey_image_path,
                    action_tokens=action_input
                )
                
                # Extract single score if it's a list
                if isinstance(verifier_score, list) and len(verifier_score) > 0:
                    verifier_score = verifier_score[0]
                
            except Exception as e:
                print(f"Error computing verifier score for sample {sample_index}, instruction {instruction_idx}: {e}")
                verifier_score = None
            
            # Store result
            result = {
                'sample_id': sample_id,
                'sample_index': sample_index,
                'instruction_index': instruction_idx,
                'is_original': instruction_idx == 0,
                'instruction': instruction,
                'original_instruction': original_instruction,
                'monkey_verifier_score': verifier_score,
                'predicted_action': predicted_action,
                'output_ids': output_ids,
                'ground_truth_action': sample['current_ground_truth_action'],
                'openvla_nrmse': openvla_result['nrmse'],
                'image_filename': image_filename,
                'robomonkey_image_path': robomonkey_image_path,
                'episode_id': sample['episode_id'],
                'timestep': sample['timestep']
            }
            
            results.append(result)
    
    print(f"Generated {len(results)} monkey-verifier scores")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'json_results/bridge_monkey_verifier_scores_{timestamp}.json'
    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_scores_generated': len(results),
            'expected_total': total_instructions_per_sample,
            'source_bridge_file': 'bridge_samples.json',
            'source_rephrase_file': rephrase_file,
            'source_openvla_file': openvla_file,
            'verifier_api_endpoint': 'http://127.0.0.1:3100/process',
            'image_format': '_robomonkey.jpg',
            'image_size': [224, 224],
            'action_tokenization': 'pre_computed_output_ids_from_openvla',
            'description': 'Monkey-verifier scores for bridge dataset with rephrased instructions and OpenVLA predictions'
        },
        'results': results
    }
    
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Monkey-verifier scoring completed successfully!")
    
    # Print summary statistics
    if results:
        scores = [r['monkey_verifier_score'] for r in results if r['monkey_verifier_score'] is not None]
        if scores:
            print(f"\nSummary Statistics:")
            print(f"Valid scores: {len(scores)}/{len(results)}")
            print(f"Mean verifier score: {np.mean(scores):.4f}")
            print(f"Std verifier score: {np.std(scores):.4f}")
            print(f"Min verifier score: {np.min(scores):.4f}")
            print(f"Max verifier score: {np.max(scores):.4f}")
        else:
            print("No valid scores generated!")
    
    return output_data

if __name__ == "__main__":
    # Check if required data files exist
    required_dirs = [
        'bridge_images',
        'processed_images/robomonkey'
    ]
    
    required_files = [
        'json_results/bridge_samples.json'
    ]
    
    missing_items = []
    for item in required_dirs + required_files:
        if not os.path.exists(item):
            missing_items.append(item)
    
    if missing_items:
        print("Error: Missing required files/directories:")
        for item in missing_items:
            print(f"  - {item}")
        exit(1)
    
    # Check for rephrase and OpenVLA files
    rephrase_files = [f for f in os.listdir('json_results/') 
                     if f.startswith('bridge_instruction_rephrases_') and f.endswith('.json')]
    openvla_files = [f for f in os.listdir('json_results/') 
                    if f.startswith('bridge_openvla_actions_') and f.endswith('.json')]
    
    if not rephrase_files:
        print("Error: No rephrase files found (bridge_instruction_rephrases_*.json)")
        exit(1)
    
    if not openvla_files:
        print("Error: No OpenVLA action files found (bridge_openvla_actions_*.json)")
        exit(1)
    
    print(f"Found {len(rephrase_files)} rephrase file(s) and {len(openvla_files)} OpenVLA file(s)")
    
    # Run scoring
    generate_monkey_verifier_scores()
