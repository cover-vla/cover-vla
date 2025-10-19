#!/usr/bin/env python3
"""
Pre-tokenize all instruction texts from the dataset and save as cache files.
This eliminates the expensive clip.tokenize() calls from the training loop.
"""

import argparse
import os
import json
import torch
import clip
from tqdm import tqdm
import pickle
import hashlib
import ijson

def load_dataset_with_streaming(json_path):
    """Load dataset from JSON file with streaming support"""
    print(f"Loading dataset from {json_path} with streaming...")
    try:
        with open(json_path, 'rb') as f:
            try:
                items = ijson.items(f, '', use_float=True)
                dataset = next(items)
                print("Successfully loaded with ijson streaming")
                return dataset
            except:
                f.seek(0)
                items = ijson.items(f, '')
                dataset = next(items)
                print("Successfully loaded with ijson streaming (fallback)")
                return dataset
    except Exception as e:
        print(f"Warning: Streaming failed ({e}), falling back to regular JSON loading...")
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        return dataset

def create_cache_filename(dataset_path):
    """Create a unique cache filename based on the dataset path and its modification time"""
    # Get dataset file stats
    stat = os.stat(dataset_path)
    dataset_size = stat.st_size
    dataset_mtime = stat.st_mtime
    
    # Create a hash from the path and metadata
    hash_input = f"{dataset_path}_{dataset_size}_{dataset_mtime}".encode('utf-8')
    dataset_hash = hashlib.md5(hash_input).hexdigest()[:8]
    
    # Create cache filename
    dataset_basename = os.path.splitext(os.path.basename(dataset_path))[0]
    cache_filename = f"instruction_tokens_cache_{dataset_basename}_{dataset_hash}.pkl"
    
    return cache_filename

def precompute_instruction_tokens(dataset_path, cache_dir="~", device="cuda"):
    """
    Pre-tokenize all instruction texts from the dataset and save as cache.
    
    Args:
        dataset_path: Path to the dataset JSON file
        cache_dir: Directory to save the cache file (default: home directory)
        device: Device to load CLIP model on
    """
    
    # Expand cache directory path
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename
    cache_filename = create_cache_filename(dataset_path)
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if cache already exists
    if os.path.exists(cache_path):
        print(f"Cache file already exists at {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"Successfully loaded existing cache with {len(cache_data['instruction_tokens'])} tokenized instructions")
            return cache_path, cache_data
        except Exception as e:
            print(f"Error loading existing cache: {e}. Regenerating...")
    
    print(f"Pre-tokenizing instructions from {dataset_path}...")
    
    # Load CLIP model for tokenization
    print(f"Loading CLIP model on {device}...")
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    
    # Load dataset
    dataset_dict = load_dataset_with_streaming(dataset_path)
    
    # Check dataset format
    metadata = dataset_dict.get('_metadata', {})
    format_version = metadata.get('format_version', '1.0_legacy')
    print(f"Dataset format version: {format_version}")
    
    # Collect all unique instruction texts
    unique_instructions = set()
    
    if format_version == '3.0_with_hard_negatives':
        print("Processing hard negatives format...")
        
        # Collect from instructions dictionary
        instructions_dict = dataset_dict.get('instructions', {})
        for instr_id, instr_text in instructions_dict.items():
            unique_instructions.add(instr_text)
        
        print(f"Found {len(unique_instructions)} unique instructions from instructions dictionary")
        
    elif format_version == '2.0_normalized':
        print("Processing normalized format...")
        
        # Collect from instructions dictionary
        instructions_dict = dataset_dict.get('instructions', {})
        for instr_id, instr_text in instructions_dict.items():
            unique_instructions.add(instr_text)
        
        print(f"Found {len(unique_instructions)} unique instructions from instructions dictionary")
        
    else:
        print("Processing legacy format...")
        
        # Collect from legacy format
        for instruction_key, data in dataset_dict.items():
            if instruction_key == '_metadata':
                continue
            
            # Add the instruction key itself (it's the instruction text in legacy format)
            unique_instructions.add(instruction_key)
            
            # Also check samples for language_instruction field
            samples = data.get('samples', [])
            for sample in samples:
                lang_instr = sample.get('language_instruction')
                if lang_instr:
                    unique_instructions.add(lang_instr)
        
        print(f"Found {len(unique_instructions)} unique instructions from legacy format")
    
    # Convert to sorted list for consistent ordering
    instruction_list = sorted(list(unique_instructions))
    print(f"Total unique instructions to tokenize: {len(instruction_list)}")
    
    # Tokenize all instructions
    print("Tokenizing instructions...")
    instruction_tokens = {}
    batch_size = 1000  # Process in batches to avoid memory issues
    
    with torch.no_grad():
        for i in tqdm(range(0, len(instruction_list), batch_size), desc="Tokenizing batches"):
            batch_instructions = instruction_list[i:i + batch_size]
            
            try:
                # Tokenize batch
                tokenized_batch = clip.tokenize(batch_instructions, truncate=True)
                
                # Store individual tokens
                for j, instruction in enumerate(batch_instructions):
                    instruction_tokens[instruction] = tokenized_batch[j].cpu()
                    
            except Exception as e:
                print(f"Error tokenizing batch starting at index {i}: {e}")
                # Fallback to individual tokenization
                for instruction in batch_instructions:
                    try:
                        tokens = clip.tokenize([instruction], truncate=True)[0].cpu()
                        instruction_tokens[instruction] = tokens
                    except Exception as e2:
                        print(f"Error tokenizing instruction '{instruction[:50]}...': {e2}")
                        # Use empty tokens as fallback
                        instruction_tokens[instruction] = torch.zeros(77, dtype=torch.long)
    
    # Create cache data
    cache_data = {
        'instruction_tokens': instruction_tokens,
        'dataset_path': dataset_path,
        'dataset_format_version': format_version,
        'total_instructions': len(instruction_tokens),
        'clip_model': 'ViT-B/32',
        'max_token_length': 77,  # CLIP's standard token length
        'creation_timestamp': torch.tensor(torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0).item() if hasattr(torch.utils.data, 'get_worker_info') else 0
    }
    
    # Save cache
    print(f"Saving tokenized instructions to {cache_path}...")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved cache with {len(instruction_tokens)} tokenized instructions")
        
        # Print cache file size
        cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"Cache file size: {cache_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"Error saving cache: {e}")
        return None, None
    
    return cache_path, cache_data

def load_instruction_tokens_cache(cache_path):
    """Load pre-tokenized instruction cache"""
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"Loaded instruction tokens cache with {len(cache_data['instruction_tokens'])} instructions")
        return cache_data
    except Exception as e:
        print(f"Error loading instruction tokens cache: {e}")
        return None

def find_cache_file(dataset_path, cache_dir="~"):
    """Find existing cache file for a dataset"""
    cache_dir = os.path.expanduser(cache_dir)
    cache_filename = create_cache_filename(dataset_path)
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_path):
        return cache_path
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-tokenize instruction texts from dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--cache_dir', type=str, default='~', help='Directory to save cache file (default: home directory)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load CLIP model on')
    parser.add_argument('--force_regenerate', action='store_true', help='Force regeneration even if cache exists')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found at {args.dataset}")
        exit(1)
    
    # Remove existing cache if force regenerate
    if args.force_regenerate:
        cache_path = find_cache_file(args.dataset, args.cache_dir)
        if cache_path and os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Removed existing cache file: {cache_path}")
    
    # Pre-tokenize instructions
    cache_path, cache_data = precompute_instruction_tokens(
        dataset_path=args.dataset,
        cache_dir=args.cache_dir,
        device=args.device
    )
    
    if cache_path and cache_data:
        print(f"\nSuccess! Cache saved to: {cache_path}")
        print(f"Total instructions tokenized: {cache_data['total_instructions']}")
        print(f"Dataset format: {cache_data['dataset_format_version']}")
        print(f"CLIP model: {cache_data['clip_model']}")
        print(f"Max token length: {cache_data['max_token_length']}")
        print("\nYou can now use this cache file in your training script to eliminate tokenization overhead.")
    else:
        print("Failed to create instruction tokens cache.")
        exit(1)
