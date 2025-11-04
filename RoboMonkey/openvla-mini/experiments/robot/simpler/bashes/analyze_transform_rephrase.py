#!/usr/bin/env python3
"""
Analyze success rates from transform_rephrase rollouts.
Extracts success rates from episode pickle filenames and computes statistics.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

def parse_episode_filename(filename):
    """Parse episode filename to extract episode number, success status, and task name."""
    # Pattern: episode={idx}--success={True/False}--task={task_name}.pkl
    pattern = r'episode=(\d+)--success=(True|False)--task=(.+)\.pkl'
    match = re.match(pattern, filename)
    if match:
        episode_idx = int(match.group(1))
        success = match.group(2) == 'True'
        task_name = match.group(3)
        return episode_idx, success, task_name
    return None, None, None

def analyze_subfolder(subfolder_path):
    """Analyze all episodes in a subfolder and return success rates per task."""
    task_results = defaultdict(list)  # task_name -> list of success booleans
    
    if not os.path.isdir(subfolder_path):
        return task_results
    
    for filename in os.listdir(subfolder_path):
        if filename.endswith('.pkl'):
            episode_idx, success, task_name = parse_episode_filename(filename)
            if success is not None:
                task_results[task_name].append(success)
    
    return task_results

def calculate_success_rate_per_task(task_results):
    """Calculate success rate for each task."""
    task_success_rates = {}
    for task_name, successes in task_results.items():
        if len(successes) > 0:
            success_rate = sum(successes) / len(successes)
            task_success_rates[task_name] = success_rate
    return task_success_rates

def main():
    base_path = Path("/root/vla-clip/RoboMonkey/openvla-mini/experiments/robot/simpler/bashes/rollouts_openpi_original/transform_rephrase")
    
    # Get all subfolders
    subfolders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('lang_')])
    
    # Store results: subfolder -> list of task success rates
    subfolder_results = {}
    
    for subfolder in subfolders:
        subfolder_path = base_path / subfolder
        print(f"Analyzing {subfolder}...")
        
        # Get task results for this subfolder
        task_results = analyze_subfolder(subfolder_path)
        
        # Calculate success rate per task
        task_success_rates = calculate_success_rate_per_task(task_results)
        
        # Store list of success rates (one per task)
        if task_success_rates:
            subfolder_results[subfolder] = {
                'task_success_rates': list(task_success_rates.values()),
                'num_evaluations': sum(len(successes) for successes in task_results.values()),
                'num_tasks': len(task_results)
            }
    
    def print_table(title, subfolder_list):
        """Print a formatted table for given subfolders."""
        print("\n" + "="*80)
        print(title)
        print("="*80)
        print(f"{'Subfolder':<25} {'Success Rate':<20} {'# Evaluations':<15} {'# Tasks':<10}")
        print("-"*80)
        
        all_success_rates = []
        
        for subfolder in subfolder_list:
            if subfolder in subfolder_results:
                data = subfolder_results[subfolder]
                success_rates = data['task_success_rates']
                
                if len(success_rates) > 0:
                    mean_rate = np.mean(success_rates)
                    std_rate = np.std(success_rates)
                    all_success_rates.extend(success_rates)
                    
                    eval_str = str(data['num_evaluations']) if data['num_evaluations'] > 0 else ""
                    tasks_str = str(data['num_tasks']) if data['num_tasks'] > 0 else ""
                    print(f"{subfolder:<25} {mean_rate:.4f} ± {std_rate:.4f}    {eval_str:<15} {tasks_str:<10}")
        
        # Print overall
        if all_success_rates:
            overall_mean = np.mean(all_success_rates)
            overall_std = np.std(all_success_rates)
            print("-"*80)
            print(f"{'OVERALL':<25} {overall_mean:.4f} ± {overall_std:.4f}")
        
        print("="*80)
    
    # Rephrase Ablation: varies lang_rephrase_num (keeping sample size mostly constant)
    # lang_1_sample_1, lang_2_sample_5, lang_4_sample_5, lang_8_sample_5
    rephrase_ablation_folders = ['lang_1_sample_1', 'lang_2_sample_5', 'lang_4_sample_5', 'lang_8_sample_5']
    print_table("Rephrase Ablation", rephrase_ablation_folders)
    
    # Sample Ablation: varies sample size (keeping lang_rephrase_num constant at lang_8, except baseline)
    # lang_1_sample_1, lang_8_sample_1, lang_8_sample_3, lang_8_sample_5
    sample_ablation_folders = ['lang_1_sample_1', 'lang_8_sample_1', 'lang_8_sample_3', 'lang_8_sample_5']
    print_table("Sample Ablation", sample_ablation_folders)

if __name__ == "__main__":
    main()

