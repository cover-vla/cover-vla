#!/usr/bin/env python3
"""
Script to analyze success rates from OpenPI rollouts.
Creates bar plots showing success rates and variances for each task across three seeds.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import argparse

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Global store for RoboMonkey episode IDs per task (filled by analyze_robomonkey_folder)
ROBOMONKEY_IDS = {}

def extract_success_from_filename(filename):
    """Extract success status from video filename."""
    match = re.search(r'--success=(True|False)--', filename)
    if match:
        return match.group(1) == 'True'
    return None

def analyze_rollout_folder(folder_path, ood_indicator=False):
    """
    Analyze all rollouts in a folder structure.
    
    Args:
        folder_path: Path to the rollouts folder
        ood_indicator: Whether this is OOD data
    
    Returns:
        Dictionary with task success data
    """
    results = defaultdict(lambda: defaultdict(list))
    
    seeds_found = False
    # First, try the original seed-based directory structure
    for root, dirs, files in os.walk(folder_path):
        if 'seed_' in root:
            seed_match = re.search(r'seed_(\d+)', root)
            if seed_match:
                seeds_found = True
                seed = int(seed_match.group(1))
                
                # Find task folder
                task_path = root
                while task_path != folder_path:
                    if os.path.basename(task_path) not in ['seed_0', 'seed_1', 'seed_2']:
                        task_name = os.path.basename(task_path)
                        break
                    task_path = os.path.dirname(task_path)
                else:
                    continue
                
                # Analyze video files in this task folder
                for video_file in files:
                    if not video_file.endswith('.mp4'):
                        continue
                    success = extract_success_from_filename(video_file)
                    if success is not None:
                        results[task_name][seed].append(success)
    
    # If no seeds found, fall back to a flat recursive scan and use a pseudo-seed 0
    if not seeds_found:
        for root, dirs, files in os.walk(folder_path):
            for video_file in files:
                if not video_file.endswith('.mp4'):
                    continue
                success = extract_success_from_filename(video_file)
                if success is None:
                    continue
                # Try to extract task name from filename first
                task_match = re.search(r'--task=([^\.]+)', video_file)
                if task_match:
                    task_name = task_match.group(1)
                else:
                    # Fallback: use the immediate directory name as task
                    task_name = os.path.basename(root)
                results[task_name][0].append(success)
    
    return results

def analyze_rollouts_normal_rephrase(base_path):
    """
    Analyze rollouts_*_normal_rephrase-style directory (legacy support).
    
    Args:
        base_path: Path to the rollouts_*_normal_rephrase directory
    
    Returns:
        Dictionary with rollouts experiment data
    """
    results = {}
    
    if not os.path.exists(base_path):
        print(f"Warning: Rollouts path {base_path} does not exist")
        return results
    
    # Analyze the rollouts data
    experiment_results = defaultdict(list)
    
    # Walk through the folder structure
    for root, dirs, files in os.walk(base_path):
        # Check if we're in a seed folder
        if 'seed_' in root:
            seed_match = re.search(r'seed_(\d+)', root)
            if seed_match:
                seed = int(seed_match.group(1))
                
                # Find task folder
                task_path = root
                while task_path != base_path:
                    if os.path.basename(task_path) not in ['seed_0', 'seed_1', 'seed_2']:
                        task_name = os.path.basename(task_path)
                        break
                    task_path = os.path.dirname(task_path)
                else:
                    continue
                
                # Analyze video files in this task folder
                video_files = [f for f in files if f.endswith('.mp4')]
                
                for video_file in video_files:
                    success = extract_success_from_filename(video_file)
                    if success is not None:
                        experiment_results[task_name].append(success)
    
    # Store results with the specified label (renamed to OpenPI)
    results["openpi_instruct_aug_Original"] = experiment_results
    
    return results

def analyze_robomonkey_folder(base_path):
    """
    Analyze RoboMonkey folder with video files.
    
    Args:
        base_path: Path to the robomonkey directory
    
    Returns:
        Dictionary with RoboMonkey experiment data
    """
    results = {}
    
    if not os.path.exists(base_path):
        print(f"Warning: RoboMonkey path {base_path} does not exist")
        return results
    
    # Analyze the RoboMonkey data
    experiment_results = defaultdict(list)
    global ROBOMONKEY_IDS
    ROBOMONKEY_IDS = defaultdict(list)
    
    # Get all video files in this folder
    for file in os.listdir(base_path):
        if file.endswith('.mp4'):
            success = extract_success_from_filename(file)
            if success is not None:
                # Extract task name from filename
                task_match = re.search(r'--task=([^\.]+)', file)
                if task_match:
                    task_name = task_match.group(1)
                    experiment_results[task_name].append(success)
                    # Extract episode id if present
                    ep_match = re.search(r'episode=(\d+)', file)
                    if ep_match:
                        ROBOMONKEY_IDS[task_name].append(int(ep_match.group(1)))
    
    # Store results with the specified label
    results["RoboMonkey"] = experiment_results
    
    return results

def analyze_robomonkey_id_folder(base_path):
    """
    Analyze robomonkey_id folder with video files (same format as RoboMonkey),
    label results under experiment name 'robomonkey_id'.
    """
    results = {}
    if not os.path.exists(base_path):
        print(f"Warning: robomonkey_id path {base_path} does not exist")
        return results
    experiment_results = defaultdict(list)
    for file in os.listdir(base_path):
        if file.endswith('.mp4'):
            success = extract_success_from_filename(file)
            if success is not None:
                task_match = re.search(r'--task=([^\.]+)', file)
                if task_match:
                    task_name = task_match.group(1)
                    experiment_results[task_name].append(success)
    results["robomonkey_id"] = experiment_results
    return results

def analyze_rephrase_folders(base_path):
    """
    Analyze rephrase_* and no_transform_* folders in the rollouts_clip_gaussian_consistency directory.
    
    Args:
        base_path: Path to the rollouts_clip_gaussian_consistency directory
    
    Returns:
        Dictionary with rephrase experiment data
    """
    results = {}
    
    # Find all experiment folders
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if not os.path.isdir(item_path):
            continue

        experiment_name = None
        # Support legacy naming: rephrase_* and no_transform_*
        if item.startswith('rephrase_'):
            parts = item.split('_')
            if len(parts) >= 3:
                rephrase_num = parts[1]
                consistency = parts[3] if len(parts) > 3 else "Unknown"
                experiment_name = f"Rephrase_{rephrase_num}_Consistency_{consistency}"
            else:
                experiment_name = item
        elif item.startswith('no_transform_'):
            experiment_name = "Original_Instruction"
        # Support new naming: transform_no_transform and transform_rephrase
        elif item == 'transform_no_transform':
            experiment_name = "openpi_rephrase_no_transform"
        elif item == 'transform_rephrase':
            experiment_name = "openpi_rephrase_rephrase"
        else:
            # Skip unrelated directories
            continue

        # Analyze this folder recursively
        experiment_results = defaultdict(list)
        for root, dirs, files in os.walk(item_path):
            for file in files:
                if not file.endswith('.mp4'):
                    continue
                success = extract_success_from_filename(file)
                if success is None:
                    continue
                task_match = re.search(r'--task=([^\.]+)', file)
                if task_match:
                    task_name = task_match.group(1)
                else:
                    task_name = os.path.basename(root)
                experiment_results[task_name].append(success)

        results[experiment_name] = experiment_results
    
    return results

def calculate_statistics(results):
    """Calculate success rates and statistics for each task."""
    stats = {}
    
    for task_name, seed_data in results.items():
        task_stats = {
            'seeds': [],
            'success_rates': [],
            'episode_counts': [],
            'total_successes': [],
            'total_episodes': []
        }
        
        for seed in sorted(seed_data.keys()):
            successes = seed_data[seed]
            success_rate = np.mean(successes) if successes else 0.0
            episode_count = len(successes)
            total_successes = sum(successes)
            
            task_stats['seeds'].append(seed)
            task_stats['success_rates'].append(success_rate)
            task_stats['episode_counts'].append(episode_count)
            task_stats['total_successes'].append(total_successes)
            task_stats['total_episodes'].append(episode_count)
        
        # Calculate overall statistics
        all_success_rates = task_stats['success_rates']
        task_stats['mean_success_rate'] = np.mean(all_success_rates)
        task_stats['std_success_rate'] = np.std(all_success_rates)
        task_stats['overall_total_episodes'] = sum(task_stats['total_episodes'])
        task_stats['overall_total_successes'] = sum(task_stats['total_successes'])
        
        stats[task_name] = task_stats
    
    return stats

def calculate_rephrase_statistics(rephrase_results):
    """Calculate success rates and statistics for rephrase experiments."""
    stats = {}
    
    for experiment_name, task_data in rephrase_results.items():
        experiment_stats = {}
        
        for task_name, successes in task_data.items():
            if successes:  # Only process if there are results
                success_rate = np.mean(successes)
                episode_count = len(successes)
                total_successes = sum(successes)
                
                experiment_stats[task_name] = {
                    'success_rate': success_rate,
                    'episode_count': episode_count,
                    'total_successes': total_successes,
                    'total_episodes': episode_count
                }
        
        stats[experiment_name] = experiment_stats
    
    return stats

def create_bar_plots(stats_in_dist, stats_ood, rephrase_stats, robomonkey_stats, robomonkey_id_stats=None, output_dir='./analysis_plots', filter_insufficient=True):
    """Create bar plots for success rates including rephrase experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame for plotting
    plot_data = []
    
    # Add OpenPI original/no_transform data
    for task_name in stats_in_dist:
        stats = stats_in_dist[task_name]
        for i, seed in enumerate(stats['seeds']):
            plot_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'Experiment': 'openpi_original_no_transform',
                'Success Rate': stats['success_rates'][i],
                'Episode Count': stats['episode_counts'][i],
                'Data Type': 'Original'
            })
    
    # Add original/rephrase data (if exists)
    for task_name in stats_ood:
        stats = stats_ood[task_name]
        for i, seed in enumerate(stats['seeds']):
            plot_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'Experiment': 'openpi_original_rephrase',
                'Success Rate': stats['success_rates'][i],
                'Episode Count': stats['episode_counts'][i],
                'Data Type': 'Original'
            })
    
    # Add rephrase data
    for experiment_name, task_data in rephrase_stats.items():
        for task_name, task_stats in task_data.items():
            plot_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'Experiment': experiment_name,
                'Success Rate': task_stats['success_rate'],
                'Episode Count': task_stats['episode_count'],
                'Data Type': 'Rephrase'
            })
    
    # Add RoboMonkey data
    for experiment_name, task_data in robomonkey_stats.items():
        for task_name, task_stats in task_data.items():
            plot_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'Experiment': experiment_name,
                'Success Rate': task_stats['success_rate'],
                'Episode Count': task_stats['episode_count'],
                'Data Type': 'RoboMonkey'
            })

    # Add RoboMonkey_ID data
    robomonkey_id_stats = robomonkey_id_stats or {}
    for experiment_name, task_data in robomonkey_id_stats.items():
        for task_name, task_stats in task_data.items():
            plot_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'Experiment': experiment_name,
                'Success Rate': task_stats['success_rate'],
                'Episode Count': task_stats['episode_count'],
                'Data Type': 'RoboMonkey_ID'
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create color palette for different experiments
    experiment_colors = {
        'openpi_original_no_transform': '#2E86AB',   # Blue
        'openpi_original_rephrase': '#A23B72',       # Pink
        'openpi_rephrase_no_transform': '#FF6B35',   # Orange
        'openpi_rephrase_rephrase': '#8B5CF6',       # Purple
        'openpi_instruct_aug_Original': '#8B5CF6',   # Legacy support
        'RoboMonkey': '#E74C3C',                     # Red
        'robomonkey_id': '#16A085',                  # Teal
    }
    
    # Add colors for all other experiments (rephrase, etc.)
    other_experiments = [exp for exp in df['Experiment'].unique() 
                        if exp not in ['openpi_original_no_transform', 'openpi_original_rephrase', 'openpi_rephrase_no_transform', 'openpi_rephrase_rephrase', 'openpi_instruct_aug_Original', 'RoboMonkey', 'robomonkey_id']]
    other_colors = plt.cm.Set3(np.linspace(0, 1, len(other_experiments)))
    for i, exp in enumerate(other_experiments):
        experiment_colors[exp] = other_colors[i]
    
    # Create the main comparison plot
    plt.figure(figsize=(24, 12))
    
    # Filter tasks based on the filter_insufficient parameter
    if filter_insufficient:
        # Only include tasks with multiple experiment types
        task_experiment_counts = df.groupby('Task')['Experiment'].nunique()
        tasks_with_multiple_experiments = task_experiment_counts[task_experiment_counts > 1].index.tolist()
        all_tasks = sorted(tasks_with_multiple_experiments)
        
        print(f"Filtering tasks: {len(tasks_with_multiple_experiments)} tasks have multiple experiment types")
        print(f"Tasks with multiple experiments: {all_tasks}")
        
        # Filter dataframe to only include tasks with multiple experiments
        df_filtered = df[df['Task'].isin(all_tasks)]
    else:
        # Include all tasks
        all_tasks = sorted(df['Task'].unique())
        df_filtered = df
        
        print(f"Including all tasks: {len(all_tasks)} total tasks")
        print(f"All tasks: {all_tasks}")
    
    # Create grouped bar plot with better spacing
    n_experiments = len(df_filtered['Experiment'].unique())
    x_pos = np.arange(len(all_tasks))
    width = 0.7 / n_experiments  # Width of bars
    spacing = 0.1  # Space between task groups
    
    # Adjust x positions to add spacing between task groups
    x_positions = []
    for i, task in enumerate(all_tasks):
        task_x = i * (1 + spacing)  # Add spacing between tasks
        x_positions.append(task_x)
    
    x_positions = np.array(x_positions)
    
    # Plot bars for each experiment
    for i, experiment in enumerate(df_filtered['Experiment'].unique()):
        exp_data = df_filtered[df_filtered['Experiment'] == experiment]
        
        # Calculate mean success rate for each task in this experiment
        task_means = []
        task_stds = []
        for task in all_tasks:
            task_exp_data = exp_data[exp_data['Task'] == task]
            if not task_exp_data.empty:
                task_means.append(task_exp_data['Success Rate'].mean())
                task_stds.append(task_exp_data['Success Rate'].std() if len(task_exp_data) > 1 else 0)
            else:
                task_means.append(0)
                task_stds.append(0)
        
        # Plot bars
        bars = plt.bar(x_positions + i * width, task_means, width, 
                      yerr=task_stds, capsize=3, alpha=0.8,
                      color=experiment_colors[experiment], 
                      edgecolor='black', linewidth=0.8,
                      label=experiment)
        
        # Add value labels on bars
        for j, (bar, mean_val, std_val) in enumerate(zip(bars, task_means, task_stds)):
            if mean_val > 0:  # Only label non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.02,
                        f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # If this is RoboMonkey, also annotate episode IDs used per task
        if experiment == 'RoboMonkey':
            for j, task in enumerate(all_tasks):
                ids = ROBOMONKEY_IDS.get(task.replace(' ', '_').lower(), [])
                if ids:
                    ids_text = ','.join(str(x) for x in ids[:5])
                    plt.text(x_positions[j] + i * width, max(task_means[j], 0.02) + 0.05,
                             f'IDs: {ids_text}'+('…' if len(ids) > 5 else ''),
                             ha='center', va='bottom', fontsize=8, rotation=90, color='#333333')
    
    # Add vertical lines to separate task groups
    for i in range(len(all_tasks) - 1):
        x_line = x_positions[i] + width * n_experiments + spacing/2
        plt.axvline(x=x_line, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.xlabel('Tasks', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate', fontsize=14, fontweight='bold')
    plt.title('Success Rates Comparison Across All Experiments', fontsize=16, fontweight='bold')
    plt.xticks(x_positions + width * (n_experiments - 1) / 2, all_tasks, rotation=45, ha='right', fontsize=12)
    plt.ylim(0, 1.1)  # Add some space at top for labels
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Add task group labels
    for i, task in enumerate(all_tasks):
        plt.text(x_positions[i] + width * n_experiments / 2, -0.1, 
                f'Task {i+1}', ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rates_all_experiments.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create separate plots for original vs rephrase data
    original_data = df_filtered[df_filtered['Data Type'] == 'Original']
    rephrase_data = df_filtered[df_filtered['Data Type'] == 'Rephrase']
    robomonkey_data = df_filtered[df_filtered['Data Type'] == 'RoboMonkey']
    
    if not original_data.empty:
        # Original data plot
        plt.figure(figsize=(16, 8))
        
        # Group by task and experiment for original data
        original_summary = original_data.groupby(['Task', 'Experiment']).agg({
            'Success Rate': ['mean', 'std', 'count']
        }).round(3)
        original_summary.columns = ['_'.join(col).strip() for col in original_summary.columns]
        original_summary = original_summary.reset_index()
        
        # Plot original data
        for experiment in original_data['Experiment'].unique():
            exp_data = original_summary[original_summary['Experiment'] == experiment]
            if not exp_data.empty:
                x_pos = np.arange(len(exp_data))
                bars = plt.bar(x_pos, exp_data['Success Rate_mean'], 
                              yerr=exp_data['Success Rate_std'], 
                              capsize=5, alpha=0.8, 
                              color=experiment_colors[experiment],
                              edgecolor='black', linewidth=1,
                              label=experiment)
                
                # Add value labels
                for i, (bar, mean_val, std_val) in enumerate(zip(bars, exp_data['Success Rate_mean'], exp_data['Success Rate_std'])):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                            f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Tasks', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.title('Original OpenPI Results', fontsize=14, fontweight='bold')
        plt.xticks(range(len(original_summary['Task'].unique())), 
                  original_summary['Task'].unique(), rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'success_rates_original.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    if not rephrase_data.empty:
        # Rephrase data plot
        plt.figure(figsize=(16, 10))
        
        # Get all rephrase experiments
        rephrase_experiments = sorted(rephrase_data['Experiment'].unique())
        
        # Create grouped bar plot for rephrase experiments
        all_tasks_rephrase = sorted(rephrase_data['Task'].unique())
        x_pos = np.arange(len(all_tasks_rephrase))
        width = 0.8 / len(rephrase_experiments)
        
        for i, experiment in enumerate(rephrase_experiments):
            exp_data = rephrase_data[rephrase_data['Experiment'] == experiment]
            
            # Calculate means for each task
            task_means = []
            for task in all_tasks_rephrase:
                task_data = exp_data[exp_data['Task'] == task]
                if not task_data.empty:
                    task_means.append(task_data['Success Rate'].iloc[0])  # Single value per experiment
                else:
                    task_means.append(0)
            
            bars = plt.bar(x_pos + i * width, task_means, width, 
                          alpha=0.8, color=experiment_colors[experiment],
                          edgecolor='black', linewidth=0.5,
                          label=experiment)
            
            # Add value labels
            for j, (bar, mean_val) in enumerate(zip(bars, task_means)):
                if mean_val > 0:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Tasks', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.title('Rephrase Experiments - Success Rates Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width * (len(rephrase_experiments) - 1) / 2, 
                  all_tasks_rephrase, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'success_rates_rephrase.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    if not robomonkey_data.empty:
        # RoboMonkey data plot
        plt.figure(figsize=(16, 8))
        
        # Get all RoboMonkey experiments
        robomonkey_experiments = sorted(robomonkey_data['Experiment'].unique())
        
        # Create grouped bar plot for RoboMonkey experiments
        all_tasks_robomonkey = sorted(robomonkey_data['Task'].unique())
        x_pos = np.arange(len(all_tasks_robomonkey))
        width = 0.8 / len(robomonkey_experiments)
        
        for i, experiment in enumerate(robomonkey_experiments):
            exp_data = robomonkey_data[robomonkey_data['Experiment'] == experiment]
            
            # Calculate means for each task
            task_means = []
            for task in all_tasks_robomonkey:
                task_data = exp_data[exp_data['Task'] == task]
                if not task_data.empty:
                    task_means.append(task_data['Success Rate'].iloc[0])  # Single value per experiment
                else:
                    task_means.append(0)
            
            bars = plt.bar(x_pos + i * width, task_means, width, 
                          alpha=0.8, color=experiment_colors[experiment],
                          edgecolor='black', linewidth=0.5,
                          label=experiment)
            
            # Add value labels
            for j, (bar, mean_val) in enumerate(zip(bars, task_means)):
                if mean_val > 0:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Tasks', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.title('RoboMonkey Experiments - Success Rates Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width * (len(robomonkey_experiments) - 1) / 2, 
                  all_tasks_robomonkey, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'success_rates_robomonkey.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    return df

def print_summary_statistics(stats_in_dist, stats_ood, rephrase_stats, robomonkey_stats):
    """Print detailed summary statistics."""
    print("="*80)
    print("OPENPI - SUCCESS RATE ANALYSIS")
    print("="*80)
    
    print("\nOPENPI INSTRUCTION AUGMENTED TASKS:")
    print("-" * 50)
    for task_name, stats in stats_in_dist.items():
        print(f"\nTask: {task_name.replace('_', ' ').title()}")
        print(f"  Overall Success Rate: {stats['mean_success_rate']:.3f} ± {stats['std_success_rate']:.3f}")
        print(f"  Total Episodes: {stats['overall_total_episodes']}")
        print(f"  Total Successes: {stats['overall_total_successes']}")
        print("  Per Seed:")
        for i, seed in enumerate(stats['seeds']):
            print(f"    Seed {seed}: {stats['success_rates'][i]:.3f} ({stats['total_successes'][i]}/{stats['episode_counts'][i]})")
    
    if stats_ood:
        print("\nOPENPI OUT-OF-DISTRIBUTION TASKS:")
        print("-" * 50)
        for task_name, stats in stats_ood.items():
            print(f"\nTask: {task_name.replace('_', ' ').title()}")
            print(f"  Overall Success Rate: {stats['mean_success_rate']:.3f} ± {stats['std_success_rate']:.3f}")
            print(f"  Total Episodes: {stats['overall_total_episodes']}")
            print(f"  Total Successes: {stats['overall_total_successes']}")
            print("  Per Seed:")
            for i, seed in enumerate(stats['seeds']):
                print(f"    Seed {seed}: {stats['success_rates'][i]:.3f} ({stats['total_successes'][i]}/{stats['episode_counts'][i]})")
    
    print("\nREPHRASE EXPERIMENTS:")
    print("-" * 50)
    for experiment_name, task_data in rephrase_stats.items():
        print(f"\nExperiment: {experiment_name}")
        for task_name, task_stats in task_data.items():
            print(f"  Task: {task_name.replace('_', ' ').title()}")
            print(f"    Success Rate: {task_stats['success_rate']:.3f}")
            print(f"    Episodes: {task_stats['episode_count']}")
            print(f"    Successes: {task_stats['total_successes']}")
    
    print("\nROBOMONKEY EXPERIMENTS:")
    print("-" * 50)
    for experiment_name, task_data in robomonkey_stats.items():
        print(f"\nExperiment: {experiment_name}")
        for task_name, task_stats in task_data.items():
            print(f"  Task: {task_name.replace('_', ' ').title()}")
            print(f"    Success Rate: {task_stats['success_rate']:.3f}")
            print(f"    Episodes: {task_stats['episode_count']}")
            print(f"    Successes: {task_stats['total_successes']}")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY:")
    print("="*80)
    
    all_in_dist_rates = [stats['mean_success_rate'] for stats in stats_in_dist.values()]
    all_ood_rates = [stats['mean_success_rate'] for stats in stats_ood.values()]
    all_rephrase_rates = []
    all_robomonkey_rates = []
    
    for experiment_data in rephrase_stats.values():
        for task_stats in experiment_data.values():
            all_rephrase_rates.append(task_stats['success_rate'])
    
    for experiment_data in robomonkey_stats.values():
        for task_stats in experiment_data.values():
            all_robomonkey_rates.append(task_stats['success_rate'])
    
    if all_in_dist_rates:
        print(f"OpenPI Instruction Augmented Average Success Rate: {np.mean(all_in_dist_rates):.3f} ± {np.std(all_in_dist_rates):.3f}")
        print(f"OpenPI Instruction Augmented Tasks: {len(stats_in_dist)}")
    
    if all_ood_rates:
        print(f"OpenPI OOD Average Success Rate: {np.mean(all_ood_rates):.3f} ± {np.std(all_ood_rates):.3f}")
        print(f"OpenPI OOD Tasks: {len(stats_ood)}")
    
    if all_rephrase_rates:
        print(f"Rephrase Experiments Average Success Rate: {np.mean(all_rephrase_rates):.3f} ± {np.std(all_rephrase_rates):.3f}")
        print(f"Rephrase Experiments: {len(rephrase_stats)}")
        print(f"Total Rephrase Task-Experiment Combinations: {len(all_rephrase_rates)}")
    
    if all_robomonkey_rates:
        print(f"RoboMonkey Experiments Average Success Rate: {np.mean(all_robomonkey_rates):.3f} ± {np.std(all_robomonkey_rates):.3f}")
        print(f"RoboMonkey Experiments: {len(robomonkey_stats)}")
        print(f"Total RoboMonkey Task-Experiment Combinations: {len(all_robomonkey_rates)}")

def main():
    """Main analysis function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze OpenPI success rates with optional filtering')
    parser.add_argument('--include-insufficient', action='store_true', 
                       help='Include tasks with only one experiment type (default: filter them out)')
    parser.add_argument('--output-dir', default='./analysis_plots',
                       help='Output directory for plots (default: ./analysis_plots)')
    args = parser.parse_args()
    
    # Define paths
    # Updated paths to reflect new folder naming/layout
    in_dist_path = "./rollouts_openpi_original/transform_no_transform"
    ood_path = "./rollouts_openpi_original/transform_rephrase"
    rephrase_path = "./rollouts_openpi_rephrase"
    rollouts_normal_rephrase_path = "./rollouts_openpi_original"  # still parsed via analyze_rephrase_folders for transform_* subdirs
    robomonkey_path = "./robomonkey"
    robomonkey_id_path = "./robomonkey_id"
    
    print("Analyzing OpenPI Rollouts...")
    print(f"In-Distribution path: {in_dist_path}")
    print(f"Out-of-Distribution path: {ood_path}")
    print(f"Rephrase experiments path: {rephrase_path}")
    print(f"Rollouts normal rephrase path: {rollouts_normal_rephrase_path}")
    print(f"RoboMonkey path: {robomonkey_path}")
    print(f"RoboMonkey_ID path: {robomonkey_id_path}")
    print(f"Filter insufficient results: {not args.include_insufficient}")
    
    # Analyze in-distribution data
    print("\nAnalyzing in-distribution rollouts...")
    results_in_dist = analyze_rollout_folder(in_dist_path, ood_indicator=False)
    stats_in_dist = calculate_statistics(results_in_dist)
    
    # Analyze OOD data
    print("Analyzing out-of-distribution rollouts...")
    results_ood = analyze_rollout_folder(ood_path, ood_indicator=True)
    stats_ood = calculate_statistics(results_ood)
    
    # Analyze rephrase experiments
    print("Analyzing rephrase experiments...")
    results_rephrase = analyze_rephrase_folders(rephrase_path)
    stats_rephrase = calculate_rephrase_statistics(results_rephrase)
    
    # Analyze rollouts normal rephrase experiments
    print("Analyzing rollouts normal rephrase experiments...")
    results_rollouts_normal = analyze_rollouts_normal_rephrase(rollouts_normal_rephrase_path)
    stats_rollouts_normal = calculate_rephrase_statistics(results_rollouts_normal)
    
    # Analyze RoboMonkey experiments
    print("Analyzing RoboMonkey experiments...")
    results_robomonkey = analyze_robomonkey_folder(robomonkey_path)
    stats_robomonkey = calculate_rephrase_statistics(results_robomonkey)

    # Analyze RoboMonkey_ID experiments
    print("Analyzing robomonkey_id experiments...")
    results_robomonkey_id = analyze_robomonkey_id_folder(robomonkey_id_path)
    stats_robomonkey_id = calculate_rephrase_statistics(results_robomonkey_id)
    
    # Merge rollouts normal rephrase data with rephrase data
    stats_rephrase.update(stats_rollouts_normal)
    
    # Create plots
    print("\nCreating plots...")
    filter_insufficient = not args.include_insufficient
    summary_stats = create_bar_plots(stats_in_dist, stats_ood, stats_rephrase, stats_robomonkey,
                                   robomonkey_id_stats=stats_robomonkey_id,
                                   output_dir=args.output_dir, filter_insufficient=filter_insufficient)
    
    # Print summary
    print_summary_statistics(stats_in_dist, stats_ood, stats_rephrase, stats_robomonkey)
    
    print(f"\nAnalysis complete! Plots saved to '{args.output_dir}/'")
    print("Files generated:")
    print("  - success_rates_all_experiments.png")
    print("  - success_rates_original.png")
    print("  - success_rates_rephrase.png")
    print("  - success_rates_robomonkey.png")

if __name__ == "__main__":
    main()
