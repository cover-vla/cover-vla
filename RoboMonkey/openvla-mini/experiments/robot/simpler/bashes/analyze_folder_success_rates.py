#!/usr/bin/env python3
"""
Script to calculate average success rate for each task in a rollout folder.
Groups episodes by seed (50 episodes per seed) and calculates mean and std across 3 seeds.
"""

import os
import re
import argparse
import numpy as np
from collections import defaultdict

def extract_success_from_filename(filename):
    """Extract success status from video filename."""
    match = re.search(r'--success=(True|False)--', filename)
    if match:
        return match.group(1) == 'True'
    return None

def extract_task_from_filename(filename):
    """Extract task name from filename."""
    match = re.search(r'--task=([^\.]+)', filename)
    if match:
        return match.group(1)
    return None

def extract_episode_number(filename):
    """Extract episode number from filename."""
    match = re.search(r'episode=(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def get_seed_from_episode(episode_num):
    """Determine seed number from episode number. Each seed has 50 episodes."""
    if episode_num is None:
        return None
    # Seed 1: episodes 1-50, Seed 2: episodes 51-100, Seed 3: episodes 101-150
    if 1 <= episode_num <= 50:
        return 1
    elif 51 <= episode_num <= 100:
        return 2
    elif 101 <= episode_num <= 150:
        return 3
    return None  # Outside the range we care about

def analyze_folder(folder_path):
    """
    Analyze all rollouts in a folder and calculate success rates per task per seed.
    Episodes are normalized per task (first 50 = seed 1, next 50 = seed 2, next 50 = seed 3).
    Then compute mean and std across 3 seeds (150 total episodes per task).
    
    Args:
        folder_path: Path to the rollouts folder
    
    Returns:
        Dictionary with task -> mean and std success rate data across seeds
    """
    # First pass: collect all episodes per task
    # Structure: task_name -> list of (episode_num, success)
    task_episodes = defaultdict(list)
    
    # Walk through the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith('.mp4'):
                continue
            
            success = extract_success_from_filename(file)
            task_name = extract_task_from_filename(file)
            episode_num = extract_episode_number(file)
            
            if success is not None and task_name is not None and episode_num is not None:
                task_episodes[task_name].append((episode_num, success))
    
    # Second pass: normalize episodes per task and group into seeds
    # Structure: task_name -> seed -> {successes, total}
    results = defaultdict(lambda: defaultdict(lambda: {'successes': 0, 'total': 0}))
    
    for task_name, episodes in task_episodes.items():
        # Sort episodes by episode number
        sorted_episodes = sorted(episodes, key=lambda x: x[0])
        
        # Only process first 150 episodes per task (3 seeds × 50 episodes)
        for idx, (episode_num, success) in enumerate(sorted_episodes[:150]):
            # Determine seed based on position (0-indexed)
            # Episodes 0-49 = seed 1, 50-99 = seed 2, 100-149 = seed 3
            if idx < 50:
                seed = 1
            elif idx < 100:
                seed = 2
            elif idx < 150:
                seed = 3
            else:
                continue  # Skip beyond 150
            
            results[task_name][seed]['successes'] += int(success)
            results[task_name][seed]['total'] += 1
    
    # Calculate success rates per seed, then mean and std across seeds
    task_success_rates = {}
    for task_name, seed_data in results.items():
        seed_rates = []
        seed_details = {}
        
        # Calculate success rate for each seed
        for seed in [1, 2, 3]:
            if seed in seed_data:
                seed_total = seed_data[seed]['total']
                seed_successes = seed_data[seed]['successes']
                if seed_total > 0:
                    seed_rate = seed_successes / seed_total
                    seed_rates.append(seed_rate)
                    seed_details[seed] = {
                        'success_rate': seed_rate,
                        'successes': seed_successes,
                        'total': seed_total
                    }
        
        # Include tasks with any number of seeds (at least 1)
        if len(seed_rates) > 0:
            mean_rate = np.mean(seed_rates)
            std_rate = np.std(seed_rates) if len(seed_rates) > 1 else 0.0
            
            task_success_rates[task_name] = {
                'mean': mean_rate,
                'std': std_rate,
                'seeds': seed_details,
                'num_seeds': len(seed_rates)
            }
    
    return task_success_rates

def print_results(task_success_rates):
    """Print results in a formatted table with mean and std across seeds."""
    print("\n" + "="*120)
    print("SUCCESS RATE ANALYSIS (Mean ± Std across seeds, 50 episodes per seed)")
    print("="*120)
    print(f"{'Task Name':<50} {'Mean ± Std':<20} {'Seed 1':<12} {'Seed 2':<12} {'Seed 3':<12}")
    print("-"*120)
    
    # Sort by task name for consistent output
    sorted_tasks = sorted(task_success_rates.items())
    
    all_seed_rates = []
    
    for task_name, data in sorted_tasks:
        mean_rate = data['mean']
        std_rate = data['std']
        seeds = data['seeds']
        num_seeds = data['num_seeds']
        
        # Format task name (replace underscores with spaces, capitalize)
        formatted_task = task_name.replace('_', ' ').title()
        
        # Format mean ± std
        mean_std_str = f"{mean_rate:.2%} ± {std_rate:.2%}"
        
        # Get per-seed rates
        seed1_str = f"{seeds[1]['success_rate']:.2%}" if 1 in seeds else "N/A"
        seed2_str = f"{seeds[2]['success_rate']:.2%}" if 2 in seeds else "N/A"
        seed3_str = f"{seeds[3]['success_rate']:.2%}" if 3 in seeds else "N/A"
        
        print(f"{formatted_task:<50} {mean_std_str:<20} {seed1_str:<12} {seed2_str:<12} {seed3_str:<12}")
        
        # Collect all seed rates for overall calculation
        for seed_data in seeds.values():
            all_seed_rates.append(seed_data['success_rate'])
    
    print("-"*120)
    if len(all_seed_rates) > 0:
        overall_mean = np.mean(all_seed_rates)
        overall_std = np.std(all_seed_rates) if len(all_seed_rates) > 1 else 0.0
        overall_str = f"{overall_mean:.2%} ± {overall_std:.2%}"
        num_tasks = len(sorted_tasks)
        total_evals = sum(sum(seed_data['total'] for seed_data in data['seeds'].values()) for _, data in sorted_tasks)
        print(f"{'OVERALL':<50} {overall_str:<20} ({num_tasks} tasks, {total_evals} total evals)")
    print("="*120 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Calculate success rates for each task in a rollout folder')
    parser.add_argument('folder_path', type=str, help='Path to the rollout folder to analyze')
    args = parser.parse_args()
    
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder {args.folder_path} does not exist")
        return
    
    print(f"Analyzing folder: {args.folder_path}")
    task_success_rates = analyze_folder(args.folder_path)
    
    if not task_success_rates:
        print("No valid episode files found in the folder")
        return
    
    print_results(task_success_rates)

if __name__ == "__main__":
    main()

