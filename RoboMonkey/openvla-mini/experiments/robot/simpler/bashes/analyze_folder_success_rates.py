#!/usr/bin/env python3
"""
Script to calculate average success rate for each task in a rollout folder.
Similar to analyze_success_rates.py but simpler - just calculates overall success rates per task.
"""

import os
import re
import argparse
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

def analyze_folder(folder_path):
    """
    Analyze all rollouts in a folder and calculate success rates per task.
    
    Args:
        folder_path: Path to the rollouts folder
    
    Returns:
        Dictionary with task -> success rate data
    """
    results = defaultdict(lambda: {'successes': 0, 'total': 0, 'episodes': []})
    
    # Walk through the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith('.mp4'):
                continue
            
            success = extract_success_from_filename(file)
            task_name = extract_task_from_filename(file)
            
            if success is not None and task_name is not None:
                results[task_name]['successes'] += int(success)
                results[task_name]['total'] += 1
                results[task_name]['episodes'].append((file, success))
    
    # Calculate success rates
    task_success_rates = {}
    for task_name, data in results.items():
        if data['total'] > 0:
            success_rate = data['successes'] / data['total']
            task_success_rates[task_name] = {
                'success_rate': success_rate,
                'successes': data['successes'],
                'total': data['total'],
                'failures': data['total'] - data['successes']
            }
    
    return task_success_rates

def print_results(task_success_rates):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("SUCCESS RATE ANALYSIS")
    print("="*80)
    print(f"{'Task Name':<50} {'Success Rate':<15} {'Successes':<12} {'Total':<10}")
    print("-"*80)
    
    # Sort by task name for consistent output
    sorted_tasks = sorted(task_success_rates.items())
    
    total_successes = 0
    total_episodes = 0
    
    for task_name, data in sorted_tasks:
        success_rate = data['success_rate']
        successes = data['successes']
        total = data['total']
        
        # Format task name (replace underscores with spaces, capitalize)
        formatted_task = task_name.replace('_', ' ').title()
        
        print(f"{formatted_task:<50} {success_rate:<15.2%} {successes:<12} {total:<10}")
        
        total_successes += successes
        total_episodes += total
    
    print("-"*80)
    if total_episodes > 0:
        overall_rate = total_successes / total_episodes
        print(f"{'OVERALL':<50} {overall_rate:<15.2%} {total_successes:<12} {total_episodes:<10}")
    print("="*80 + "\n")

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

