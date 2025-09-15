#!/usr/bin/env python3
"""
Script to plot success rates for each rephrase index across all tasks.
Each task gets its own plot showing success rate vs rephrase index.
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

def extract_episode_info(filename):
    """Extract episode number, success status, and rephrase index from filename."""
    # Pattern: episode=X--success=Y--rephrase_index=Z.mp4 or .pkl
    pattern = r'episode=(\d+)--success=(True|False)--rephrase_index=(\d+)\.(mp4|pkl)'
    match = re.match(pattern, filename)
    if match:
        episode = int(match.group(1))
        success = match.group(2) == 'True'
        rephrase_index = int(match.group(3))
        return episode, success, rephrase_index
    return None, None, None

def extract_baseline_info(filename):
    """Extract episode number, success status, and task name from baseline filename."""
    # Pattern: episode=X--success=Y--score=Z--task=TASK_NAME.mp4 or .pkl
    pattern = r'episode=(\d+)--success=(True|False)--score=([\d.-]+)--task=([^.]+)\.(mp4|pkl)'
    match = re.match(pattern, filename)
    if match:
        episode = int(match.group(1))
        success = match.group(2) == 'True'
        task_name = match.group(4)  # Changed from group(3) to group(4)
        return episode, success, task_name
    return None, None, None

def analyze_task_directory(task_dir):
    """Analyze a single task directory and return success rates per rephrase index."""
    print(f"Analyzing task: {os.path.basename(task_dir)}")
    
    # Dictionary to store results: {rephrase_index: [success_values]}
    rephrase_results = defaultdict(list)
    
    # Get all pickle files in the directory
    pkl_files = [f for f in os.listdir(task_dir) if f.endswith('.pkl')]
    
    for pkl_file in pkl_files:
        episode, success, rephrase_index = extract_episode_info(pkl_file)
        if episode is not None and success is not None and rephrase_index is not None:
            rephrase_results[rephrase_index].append(success)
    
    # Calculate success rates for each rephrase index
    success_rates = {}
    for rephrase_index, success_list in rephrase_results.items():
        if success_list:  # Only if we have data for this rephrase index
            success_rate = sum(success_list) / len(success_list)
            success_rates[rephrase_index] = success_rate
            print(f"  Rephrase {rephrase_index}: {success_rate:.3f} ({sum(success_list)}/{len(success_list)})")
    
    return success_rates

def analyze_baseline_data(baseline_dir):
    """Analyze baseline data from no_transform_1 folder and return success rates per task."""
    print("Analyzing baseline data...")
    
    # Dictionary to store results: {task_name: [success_values]}
    task_results = defaultdict(list)
    
    # Get all pickle files in the baseline directory
    pkl_files = [f for f in os.listdir(baseline_dir) if f.endswith('.pkl')]
    
    for pkl_file in pkl_files:
        episode, success, task_name = extract_baseline_info(pkl_file)
        if episode is not None and success is not None and task_name is not None:
            task_results[task_name].append(success)
    
    # Calculate success rates for each task
    baseline_success_rates = {}
    for task_name, success_list in task_results.items():
        if success_list:  # Only if we have data for this task
            success_rate = sum(success_list) / len(success_list)
            baseline_success_rates[task_name] = success_rate
            print(f"  {task_name}: {success_rate:.3f} ({sum(success_list)}/{len(success_list)})")
    
    return baseline_success_rates

def plot_task_success_rates(task_name, success_rates, baseline_rate, output_dir):
    """Create a plot for a single task showing success rate vs rephrase index."""
    if not success_rates:
        print(f"No data found for task: {task_name}")
        return
    
    # Sort rephrase indices
    rephrase_indices = sorted(success_rates.keys())
    success_rate_values = [success_rates[idx] for idx in rephrase_indices]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(rephrase_indices, success_rate_values, 'o-', linewidth=2, markersize=8, label='Rephrase Results')
    
    # Add baseline line if available
    if baseline_rate is not None:
        plt.axhline(y=baseline_rate, color='red', linestyle='--', linewidth=2, 
                   label=f'Baseline (No Transform): {baseline_rate:.3f}')
    
    plt.xlabel('Rephrase Index', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title(f'Success Rate vs Rephrase Index\n{task_name.replace("_", " ").title()}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.legend()
    
    # Add value labels on points
    for i, (idx, rate) in enumerate(zip(rephrase_indices, success_rate_values)):
        plt.annotate(f'{rate:.2f}', (idx, rate), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Save the plot
    safe_task_name = task_name.replace(" ", "_").replace("/", "_")
    plot_filename = f"{safe_task_name}_success_rates.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path}")

def main():
    """Main function to analyze all tasks and create plots."""
    # Base directory containing all task folders
    base_dir = "/root/vla-clip/RoboMonkey/openvla-mini/experiments/robot/simpler/rollouts_clip_rephrase_selection_rollout"
    
    # Create output directory for plots
    output_dir = os.path.join(base_dir, "success_rate_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze baseline data first
    baseline_dir = os.path.join(base_dir, "no_transform_1")
    baseline_success_rates = {}
    if os.path.exists(baseline_dir):
        baseline_success_rates = analyze_baseline_data(baseline_dir)
    else:
        print("Warning: no_transform_1 directory not found. Baseline data will not be included.")
    
    # Get all task directories (excluding no_transform_1 and success_rate_plots)
    task_dirs = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d)) and d not in ["success_rate_plots", "no_transform_1"]]
    
    print(f"Found {len(task_dirs)} task directories")
    
    # Analyze each task
    all_results = {}
    for task_dir in task_dirs:
        task_path = os.path.join(base_dir, task_dir)
        success_rates = analyze_task_directory(task_path)
        all_results[task_dir] = success_rates
        
        # Get baseline rate for this task
        baseline_rate = baseline_success_rates.get(task_dir, None)
        
        # Create individual plot for this task
        plot_task_success_rates(task_dir, success_rates, baseline_rate, output_dir)
    
    # Create a summary plot with all tasks
    create_summary_plot(all_results, baseline_success_rates, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")

def create_summary_plot(all_results, baseline_success_rates, output_dir):
    """Create a summary plot showing all tasks together."""
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for i, (task_name, success_rates) in enumerate(all_results.items()):
        if not success_rates:
            continue
            
        rephrase_indices = sorted(success_rates.keys())
        success_rate_values = [success_rates[idx] for idx in rephrase_indices]
        
        # Clean up task name for display
        display_name = task_name.replace("_", " ").title()
        plt.plot(rephrase_indices, success_rate_values, 'o-', 
                linewidth=2, markersize=6, label=display_name, color=colors[i])
        
        # Add baseline line for this task if available
        baseline_rate = baseline_success_rates.get(task_name, None)
        if baseline_rate is not None:
            plt.axhline(y=baseline_rate, color=colors[i], linestyle='--', alpha=0.7, 
                       label=f'{display_name} Baseline')
    
    plt.xlabel('Rephrase Index', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Success Rate vs Rephrase Index - All Tasks', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the summary plot
    summary_path = os.path.join(output_dir, "all_tasks_summary.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot: {summary_path}")

if __name__ == "__main__":
    main()
