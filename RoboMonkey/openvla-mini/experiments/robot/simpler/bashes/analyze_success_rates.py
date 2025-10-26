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
import pickle
import glob

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

def extract_episode_number(filename):
    """Extract episode number from video filename."""
    match = re.search(r'episode=(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def group_episodes_by_evaluation(episode_data, episodes_per_evaluation=50, min_episodes_threshold=30):
    """
    Group episodes into complete evaluations with flexible handling of incomplete periods.
    
    Args:
        episode_data: List of (episode_number, success) tuples
        episodes_per_evaluation: Number of episodes per complete evaluation (default: 50)
        min_episodes_threshold: Minimum episodes required to include an evaluation (default: 30)
    
    Returns:
        Dictionary with evaluation periods as keys and success rates as values
    """
    # Sort episodes by episode number
    sorted_episodes = sorted(episode_data, key=lambda x: x[0])
    
    evaluations = {}
    current_evaluation = 1
    current_episodes = []
    episode_start_idx = 0
    
    for i, (episode_num, success) in enumerate(sorted_episodes):
        current_episodes.append(success)
        
        # Check if we've completed an evaluation period
        if len(current_episodes) == episodes_per_evaluation:
            success_rate = np.mean(current_episodes)
            start_episode = sorted_episodes[episode_start_idx][0]
            end_episode = sorted_episodes[i][0]
            evaluations[f"Eval_{current_evaluation}"] = {
                'success_rate': success_rate,
                'episode_count': len(current_episodes),
                'total_successes': sum(current_episodes),
                'episode_range': f"{start_episode}-{end_episode}",
                'is_complete': True
            }
            current_evaluation += 1
            current_episodes = []
            episode_start_idx = i + 1
    
    # Handle remaining episodes if any
    if current_episodes:
        if len(current_episodes) >= min_episodes_threshold:
            success_rate = np.mean(current_episodes)
            start_episode = sorted_episodes[episode_start_idx][0]
            end_episode = sorted_episodes[-1][0]
            evaluations[f"Eval_{current_evaluation}"] = {
                'success_rate': success_rate,
                'episode_count': len(current_episodes),
                'total_successes': sum(current_episodes),
                'episode_range': f"{start_episode}-{end_episode}",
                'is_complete': False
            }
        else:
            # Too few episodes, exclude this evaluation
            print(f"Warning: Excluding evaluation {current_evaluation} with only {len(current_episodes)} episodes (< {min_episodes_threshold} threshold)")
    
    return evaluations

def analyze_rollout_folder(folder_path, ood_indicator=False):
    """
    Analyze all rollouts in a folder structure, grouping episodes into evaluations.
    
    Args:
        folder_path: Path to the rollouts folder
        ood_indicator: Whether this is OOD data
    
    Returns:
        Dictionary with task evaluation data
    """
    results = defaultdict(lambda: defaultdict(dict))
    
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
                
                # Collect episode data for this task
                episode_data = []
                for video_file in files:
                    if not video_file.endswith('.mp4'):
                        continue
                    success = extract_success_from_filename(video_file)
                    episode_num = extract_episode_number(video_file)
                    if success is not None and episode_num is not None:
                        episode_data.append((episode_num, success))
                
                # Group episodes into evaluations
                if episode_data:
                    evaluations = group_episodes_by_evaluation(episode_data)
                    results[task_name][seed] = evaluations
    
    # If no seeds found, fall back to a flat recursive scan and use a pseudo-seed 0
    if not seeds_found:
        for root, dirs, files in os.walk(folder_path):
            # Collect episode data for each task
            task_episodes = defaultdict(list)
            for video_file in files:
                if not video_file.endswith('.mp4'):
                    continue
                success = extract_success_from_filename(video_file)
                episode_num = extract_episode_number(video_file)
                if success is None or episode_num is None:
                    continue
                # Try to extract task name from filename first
                task_match = re.search(r'--task=([^\.]+)', video_file)
                if task_match:
                    task_name = task_match.group(1)
                else:
                    # Fallback: use the immediate directory name as task
                    task_name = os.path.basename(root)
                task_episodes[task_name].append((episode_num, success))
            
            # Group episodes into evaluations for each task
            for task_name, episode_data in task_episodes.items():
                if episode_data:
                    evaluations = group_episodes_by_evaluation(episode_data)
                    results[task_name][0] = evaluations
    
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
    """Calculate success rates and statistics for each task across evaluations."""
    stats = {}
    
    for task_name, seed_data in results.items():
        task_stats = {
            'seeds': [],
            'evaluations': {},
            'overall_mean': 0.0,
            'overall_std': 0.0,
            'total_evaluations': 0,
            'total_episodes': 0,
            'total_successes': 0
        }
        
        all_evaluation_rates = []
        
        for seed in sorted(seed_data.keys()):
            evaluations = seed_data[seed]
            task_stats['seeds'].append(seed)
            
            for eval_name, eval_data in evaluations.items():
                if eval_name not in task_stats['evaluations']:
                    task_stats['evaluations'][eval_name] = []
                
                task_stats['evaluations'][eval_name].append(eval_data['success_rate'])
                all_evaluation_rates.append(eval_data['success_rate'])
                task_stats['total_episodes'] += eval_data['episode_count']
                task_stats['total_successes'] += eval_data['total_successes']
        
        # Calculate overall statistics
        if all_evaluation_rates:
            task_stats['overall_mean'] = np.mean(all_evaluation_rates)
            task_stats['overall_std'] = np.std(all_evaluation_rates)
            task_stats['total_evaluations'] = len(all_evaluation_rates)
        
        # Calculate mean and std for each evaluation period across seeds
        for eval_name, rates in task_stats['evaluations'].items():
            if rates:
                task_stats['evaluations'][eval_name] = {
                    'mean': np.mean(rates),
                    'std': np.std(rates),
                    'count': len(rates),
                    'rates': rates
                }
        
        stats[task_name] = task_stats
    
    return stats

def calculate_rephrase_statistics(rephrase_results):
    """Calculate success rates and statistics for rephrase experiments with evaluations."""
    stats = {}
    
    for experiment_name, task_data in rephrase_results.items():
        experiment_stats = {}
        
        for task_name, task_results in task_data.items():
            if isinstance(task_results, dict) and any(isinstance(v, dict) and 'success_rate' in v for v in task_results.values()):
                # This is already in evaluation format (from analyze_rollout_folder_recursive)
                experiment_stats[task_name] = task_results
            else:
                # This is old format - convert to evaluation format
                if task_results:  # Only process if there are results
                    # Convert old format to evaluation format
                    episode_data = []
                    for i, success in enumerate(task_results):
                        episode_data.append((i + 1, success))
                    
                    evaluations = group_episodes_by_evaluation(episode_data)
                    experiment_stats[task_name] = evaluations
        
        stats[experiment_name] = experiment_stats
    
    return stats

def create_evaluation_plots(stats_in_dist, stats_ood, rephrase_stats, robomonkey_stats, stats_in_dist_test=None, stats_ood_test=None, output_dir='./analysis_plots', filter_insufficient=True):
    """Create ONE plot showing mean and std across 4 evaluation periods for each task."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all stats for comprehensive analysis
    all_stats = {}
    all_stats.update(stats_in_dist)
    all_stats.update(stats_ood)
    all_stats.update(rephrase_stats)
    all_stats.update(robomonkey_stats)
    if stats_in_dist_test:
        all_stats.update(stats_in_dist_test)
    if stats_ood_test:
        all_stats.update(stats_ood_test)
    
    if not all_stats:
        print("No data found for evaluation plotting")
        return
    
    # Collect data for plotting: each task gets mean and std across its 4 evaluation periods
    plot_data = []
    
    for experiment_name, task_stats in all_stats.items():
        for task_name, task_data in task_stats.items():
            if isinstance(task_data, dict) and any(isinstance(v, dict) and 'success_rate' in v for v in task_data.values()):
                # Get the 4 main evaluation periods (Eval_1, Eval_2, Eval_3, Eval_4)
                eval_rates = []
                for eval_name in ['Eval_1', 'Eval_2', 'Eval_3', 'Eval_4']:
                    if eval_name in task_data and isinstance(task_data[eval_name], dict) and 'success_rate' in task_data[eval_name]:
                        eval_rates.append(task_data[eval_name]['success_rate'])
                
                if len(eval_rates) >= 2:  # Need at least 2 evaluations to calculate std
                    plot_data.append({
                        'Experiment': experiment_name,
                        'Task': task_name.replace('_', ' ').title(),
                        'Mean': np.mean(eval_rates),
                        'Std': np.std(eval_rates),
                        'Evaluations': len(eval_rates)
                    })
    
    if not plot_data:
        print("No evaluation data found")
        return
    
    # Create the single plot
    plt.figure(figsize=(16, 10))
    
    df = pd.DataFrame(plot_data)
    
    # Filter tasks based on the filter_insufficient parameter
    if filter_insufficient:
        # Only include tasks with multiple experiment types
        task_experiment_counts = df.groupby('Task')['Experiment'].nunique()
        tasks_with_multiple_experiments = task_experiment_counts[task_experiment_counts > 1].index.tolist()
        
        print(f"Filtering tasks: {len(tasks_with_multiple_experiments)} tasks have multiple experiment types")
        print(f"Tasks with multiple experiments: {tasks_with_multiple_experiments}")
        
        # Filter dataframe to only include tasks with multiple experiments
        df = df[df['Task'].isin(tasks_with_multiple_experiments)]
        
        if df.empty:
            print("No tasks with multiple experiments found after filtering")
            return
    
    # Get unique experiments and tasks
    experiments = sorted(df['Experiment'].unique())
    tasks = sorted(df['Task'].unique())
    
    # Create grouped bar plot
    x_pos = np.arange(len(tasks))
    width = 0.8 / len(experiments)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))
    
    for i, experiment in enumerate(experiments):
        exp_data = df[df['Experiment'] == experiment]
        
        task_means = []
        task_stds = []
        
        for task in tasks:
            task_data = exp_data[exp_data['Task'] == task]
            if not task_data.empty:
                task_means.append(task_data['Mean'].iloc[0])
                task_stds.append(task_data['Std'].iloc[0])
            else:
                task_means.append(0)
                task_stds.append(0)
        
        bars = plt.bar(x_pos + i * width, task_means, width, 
                      yerr=task_stds, capsize=3, alpha=0.8,
                      color=colors[i], edgecolor='black', linewidth=0.5,
                      label=experiment)
        
        # Add value labels
        for j, (bar, mean_val, std_val) in enumerate(zip(bars, task_means, task_stds)):
            if mean_val > 0:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                        f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Tasks', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate (Mean ± Std across 4 evaluations)', fontsize=14, fontweight='bold')
    plt.title('Mean and Standard Deviation Across 4 Evaluation Periods (50 episodes each)', fontsize=16, fontweight='bold')
    plt.xticks(x_pos + width * (len(experiments) - 1) / 2, 
              tasks, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_mean_std_across_4_periods.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create separate plots for specific experiment types
    create_separate_experiment_plots(all_stats, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("MEAN AND STD ACROSS EVALUATION PERIODS")
    print("="*80)
    for _, row in df.iterrows():
        print(f"{row['Experiment']} - {row['Task']}: {row['Mean']:.3f} ± {row['Std']:.3f} (n={row['Evaluations']} evals)")
    
    # Print detailed evaluation info for incomplete evaluations
    print("\n" + "="*80)
    print("DETAILED EVALUATION INFORMATION")
    print("="*80)
    for experiment_name, task_stats in all_stats.items():
        for task_name, task_data in task_stats.items():
            if isinstance(task_data, dict) and any(isinstance(v, dict) and 'success_rate' in v for v in task_data.values()):
                incomplete_evals = []
                complete_evals = []
                for eval_name, eval_data in task_data.items():
                    if isinstance(eval_data, dict) and 'success_rate' in eval_data:
                        if eval_data.get('is_complete', True):
                            complete_evals.append(eval_name)
                        else:
                            incomplete_evals.append(f"{eval_name} ({eval_data['episode_count']} episodes)")
                
                if incomplete_evals:
                    print(f"{experiment_name} - {task_name.replace('_', ' ').title()}:")
                    print(f"  Complete: {', '.join(complete_evals)}")
                    print(f"  Incomplete: {', '.join(incomplete_evals)}")
    
    return df

def create_separate_experiment_plots(all_stats, output_dir):
    """Create separate plots for specific experiment types."""
    
    # Define experiment groups
    experiment_groups = {
        'openpi_original_rephrase': [],
        'openpi_original_no_transform': []
    }
    
    # Group experiments by type
    for experiment_name, task_stats in all_stats.items():
        if 'rephrase' in experiment_name and 'openpi_original' in experiment_name:
            experiment_groups['openpi_original_rephrase'].append(experiment_name)
        elif 'no_transform' in experiment_name and 'openpi_original' in experiment_name:
            experiment_groups['openpi_original_no_transform'].append(experiment_name)
    
    # Create plots for each group
    for group_name, experiment_list in experiment_groups.items():
        if not experiment_list:
            continue
            
        # Collect data for this group
        group_data = []
        for experiment_name in experiment_list:
            task_stats = all_stats[experiment_name]
            for task_name, task_data in task_stats.items():
                if isinstance(task_data, dict) and any(isinstance(v, dict) and 'success_rate' in v for v in task_data.values()):
                    # Get the 4 main evaluation periods (Eval_1, Eval_2, Eval_3, Eval_4)
                    eval_rates = []
                    for eval_name in ['Eval_1', 'Eval_2', 'Eval_3', 'Eval_4']:
                        if eval_name in task_data and isinstance(task_data[eval_name], dict) and 'success_rate' in task_data[eval_name]:
                            eval_rates.append(task_data[eval_name]['success_rate'])
                    
                    if len(eval_rates) >= 2:  # Need at least 2 evaluations to calculate std
                        group_data.append({
                            'Experiment': experiment_name,
                            'Task': task_name.replace('_', ' ').title(),
                            'Mean': np.mean(eval_rates),
                            'Std': np.std(eval_rates),
                            'Evaluations': len(eval_rates)
                        })
        
        if not group_data:
            continue
            
        # Create the plot for this group
        plt.figure(figsize=(14, 8))
        
        df_group = pd.DataFrame(group_data)
        
        # Get unique experiments and tasks
        experiments = sorted(df_group['Experiment'].unique())
        tasks = sorted(df_group['Task'].unique())
        
        # Create grouped bar plot
        x_pos = np.arange(len(tasks))
        width = 0.8 / len(experiments)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))
        
        for i, experiment in enumerate(experiments):
            exp_data = df_group[df_group['Experiment'] == experiment]
            
            task_means = []
            task_stds = []
            
            for task in tasks:
                task_data = exp_data[exp_data['Task'] == task]
                if not task_data.empty:
                    task_means.append(task_data['Mean'].iloc[0])
                    task_stds.append(task_data['Std'].iloc[0])
                else:
                    task_means.append(0)
                    task_stds.append(0)
            
            bars = plt.bar(x_pos + i * width, task_means, width, 
                          yerr=task_stds, capsize=3, alpha=0.8,
                          color=colors[i], edgecolor='black', linewidth=0.5,
                          label=experiment)
            
            # Add value labels
            for j, (bar, mean_val, std_val) in enumerate(zip(bars, task_means, task_stds)):
                if mean_val > 0:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                            f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Tasks', fontsize=14, fontweight='bold')
        plt.ylabel('Success Rate (Mean ± Std across evaluations)', fontsize=14, fontweight='bold')
        plt.title(f'{group_name.replace("_", " ").title()} - Mean and Std Across Evaluation Periods', fontsize=16, fontweight='bold')
        plt.xticks(x_pos + width * (len(experiments) - 1) / 2, 
                  tasks, rotation=45, ha='right', fontsize=10)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        safe_group_name = group_name.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f'evaluation_mean_std_{safe_group_name}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary for this group
        print(f"\n{group_name.replace('_', ' ').title()} Summary:")
        print("-" * 50)
        for _, row in df_group.iterrows():
            print(f"  {row['Experiment']} - {row['Task']}: {row['Mean']:.3f} ± {row['Std']:.3f} (n={row['Evaluations']} evals)")

def create_bar_plots(stats_in_dist, stats_ood, rephrase_stats, robomonkey_stats, stats_in_dist_test=None, stats_ood_test=None, output_dir='./analysis_plots', filter_insufficient=True):
    """Create bar plots for success rates including rephrase experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame for plotting
    plot_data = []
    
    # Add OpenPI original/no_transform data
    for experiment_name, task_data in stats_in_dist.items():
        for task_name, task_stats in task_data.items():
            plot_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'Experiment': experiment_name,
                'Success Rate': task_stats['success_rate'],
                'Episode Count': task_stats['episode_count'],
                'Data Type': 'Original'
            })
    
    # Add original/rephrase data (if exists)
    for experiment_name, task_data in stats_ood.items():
        for task_name, task_stats in task_data.items():
            plot_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'Experiment': experiment_name,
                'Success Rate': task_stats['success_rate'],
                'Episode Count': task_stats['episode_count'],
                'Data Type': 'Original'
            })
    
    # Add OpenPI original/no_transform TEST data
    stats_in_dist_test = stats_in_dist_test or {}
    for experiment_name, task_data in stats_in_dist_test.items():
        for task_name, task_stats in task_data.items():
            plot_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'Experiment': experiment_name,
                'Success Rate': task_stats['success_rate'],
                'Episode Count': task_stats['episode_count'],
                'Data Type': 'Test'
            })
    
    # Add original/rephrase TEST data (if exists)
    stats_ood_test = stats_ood_test or {}
    for experiment_name, task_data in stats_ood_test.items():
        for task_name, task_stats in task_data.items():
            plot_data.append({
                'Task': task_name.replace('_', ' ').title(),
                'Experiment': experiment_name,
                'Success Rate': task_stats['success_rate'],
                'Episode Count': task_stats['episode_count'],
                'Data Type': 'Test'
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

    
    df = pd.DataFrame(plot_data)
    
    # Create color palette for different experiments
    experiment_colors = {
        'openpi_original_no_transform': '#2E86AB',   # Blue
        'openpi_original_rephrase': '#A23B72',       # Pink
        'openpi_rephrase_no_transform': '#FF6B35',   # Orange
        'openpi_rephrase_rephrase': '#8B5CF6',       # Purple
        'openpi_instruct_aug_Original': '#8B5CF6',   # Legacy support
        'RoboMonkey': '#E74C3C',                     # Red
    }
    # Colors for new TEST experiments
    experiment_colors['openpi_original_no_transform_test'] = '#7FB3D5'  # Light Blue
    experiment_colors['openpi_original_rephrase_test'] = '#D988BC'      # Light Pink
    
    # Add colors for all other experiments (rephrase, etc.)
    other_experiments = [exp for exp in df['Experiment'].unique() 
                        if exp not in experiment_colors.keys()]
    if other_experiments:
        other_colors = plt.cm.Set3(np.linspace(0, 1, len(other_experiments)))
        for i, exp in enumerate(other_experiments):
            experiment_colors[exp] = other_colors[i]
    
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
    
    # Create separate subplots for each experiment
    experiments = sorted(df_filtered['Experiment'].unique())
    n_experiments = len(experiments)
    
    # Calculate subplot layout
    n_cols = min(3, n_experiments)  # Max 3 columns
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    if n_experiments == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_experiments > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Create individual bar plots for each experiment
    for i, experiment in enumerate(experiments):
        ax = axes[i] if n_experiments > 1 else axes[0]
        
        exp_data = df_filtered[df_filtered['Experiment'] == experiment]
        
        # Calculate mean success rate for each task in this experiment
        task_means = []
        task_stds = []
        task_names = []
        
        for task in all_tasks:
            task_exp_data = exp_data[exp_data['Task'] == task]
            if not task_exp_data.empty:
                task_means.append(task_exp_data['Success Rate'].mean())
                task_stds.append(task_exp_data['Success Rate'].std() if len(task_exp_data) > 1 else 0)
                task_names.append(task)
            else:
                task_means.append(0)
                task_stds.append(0)
                task_names.append(task)
        
        # Create bar plot
        x_pos = np.arange(len(task_names))
        bars = ax.bar(x_pos, task_means, yerr=task_stds, capsize=3, alpha=0.8,
                     color=experiment_colors[experiment], edgecolor='black', linewidth=0.8)
        
        # Add value labels on bars
        for j, (bar, mean_val, std_val) in enumerate(zip(bars, task_means, task_stds)):
            if mean_val > 0:  # Only label non-zero bars
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.02,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # If this is RoboMonkey, also annotate episode IDs used per task
        if experiment == 'RoboMonkey':
            for j, task in enumerate(task_names):
                ids = ROBOMONKEY_IDS.get(task.replace(' ', '_').lower(), [])
                if ids:
                    ids_text = ','.join(str(x) for x in ids[:3])
                    ax.text(j, max(task_means[j], 0.02) + 0.05,
                           f'IDs: {ids_text}'+('…' if len(ids) > 3 else ''),
                             ha='center', va='bottom', fontsize=8, rotation=90, color='#333333')
    
        # Customize subplot
        ax.set_xlabel('Tasks', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'{experiment.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for i in range(n_experiments, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rates_all_experiments.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create separate plots for original vs rephrase vs test data
    original_data = df_filtered[df_filtered['Data Type'] == 'Original']
    rephrase_data = df_filtered[df_filtered['Data Type'] == 'Rephrase']
    test_data = df_filtered[df_filtered['Data Type'] == 'Test']
    robomonkey_data = df_filtered[df_filtered['Data Type'] == 'RoboMonkey']
    
    if not original_data.empty:
        # Original data plot - grouped bars like rephrase plot
        plt.figure(figsize=(16, 8))
        
        # Get all original experiments
        original_experiments = sorted(original_data['Experiment'].unique())
        
        # Create grouped bar plot for original experiments
        all_tasks_original = sorted(original_data['Task'].unique())
        x_pos = np.arange(len(all_tasks_original))
        width = 0.8 / len(original_experiments)
        
        for i, experiment in enumerate(original_experiments):
            exp_data = original_data[original_data['Experiment'] == experiment]
            
            # Calculate means for each task
            task_means = []
            task_stds = []
            for task in all_tasks_original:
                task_data = exp_data[exp_data['Task'] == task]
                if not task_data.empty:
                    task_means.append(task_data['Success Rate'].mean())
                    task_stds.append(task_data['Success Rate'].std() if len(task_data) > 1 else 0)
                else:
                    task_means.append(0)
                    task_stds.append(0)
            
            bars = plt.bar(x_pos + i * width, task_means, width, 
                          yerr=task_stds, capsize=3, alpha=0.8, 
                              color=experiment_colors[experiment],
                          edgecolor='black', linewidth=0.8,
                              label=experiment)
                
                # Add value labels
            for j, (bar, mean_val, std_val) in enumerate(zip(bars, task_means, task_stds)):
                if mean_val > 0:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                            f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.xlabel('Tasks', fontsize=12, fontweight='bold')
        plt.ylabel('Success Rate', fontsize=12, fontweight='bold')
        plt.title('Original OpenPI Results', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width * (len(original_experiments) - 1) / 2, 
                  all_tasks_original, rotation=45, ha='right', fontsize=10)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
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
    
    if not test_data.empty:
        # Test data plot - include both original and test experiments
        plt.figure(figsize=(16, 10))
        
        # Combine original and test data for comparison
        original_and_test_data = df_filtered[df_filtered['Data Type'].isin(['Original', 'Test'])]
        
        # Get all experiments (original + test)
        all_experiments = sorted(original_and_test_data['Experiment'].unique())
        
        # Create grouped bar plot for all experiments
        all_tasks_combined = sorted(original_and_test_data['Task'].unique())
        x_pos = np.arange(len(all_tasks_combined))
        width = 0.8 / len(all_experiments)
        
        for i, experiment in enumerate(all_experiments):
            exp_data = original_and_test_data[original_and_test_data['Experiment'] == experiment]
            
            # Calculate means for each task
            task_means = []
            for task in all_tasks_combined:
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
        plt.title('Original vs Test Experiments - Success Rates Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width * (len(all_experiments) - 1) / 2, 
                  all_tasks_combined, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'success_rates_test.png'), dpi=300, bbox_inches='tight')
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

def print_summary_statistics(stats_in_dist, stats_ood, rephrase_stats, robomonkey_stats, stats_in_dist_test=None, stats_ood_test=None):
    """Print detailed summary statistics with evaluation periods."""
    print("="*80)
    print("OPENPI - SUCCESS RATE ANALYSIS (EVALUATION PERIODS)")
    print("="*80)
    
    def print_experiment_stats(stats_dict, title):
        if not stats_dict:
            return
        print(f"\n{title}:")
        print("-" * 50)
        for experiment_name, task_data in stats_dict.items():
            print(f"\nExperiment: {experiment_name}")
            for task_name, task_stats in task_data.items():
                print(f"  Task: {task_name.replace('_', ' ').title()}")
                if isinstance(task_stats, dict) and any(isinstance(v, dict) and 'success_rate' in v for v in task_stats.values()):
                    # New evaluation format
                    all_rates = [eval_data['success_rate'] for eval_data in task_stats.values() if isinstance(eval_data, dict) and 'success_rate' in eval_data]
                    total_episodes = sum(eval_data['episode_count'] for eval_data in task_stats.values() if isinstance(eval_data, dict) and 'episode_count' in eval_data)
                    total_successes = sum(eval_data['total_successes'] for eval_data in task_stats.values() if isinstance(eval_data, dict) and 'total_successes' in eval_data)
                    if all_rates:
                        print(f"    Overall Success Rate: {np.mean(all_rates):.3f} ± {np.std(all_rates):.3f}")
                        print(f"    Evaluation Periods: {len(all_rates)}")
                        print(f"    Episodes: {total_episodes}")
                        print(f"    Successes: {total_successes}")
                        for eval_name, eval_data in task_stats.items():
                            if isinstance(eval_data, dict) and 'success_rate' in eval_data:
                                print(f"      {eval_name}: {eval_data['success_rate']:.3f} ({eval_data['episode_count']} episodes)")
                else:
                    # Old format fallback
                    print(f"    Success Rate: {task_stats.get('success_rate', 0):.3f}")
                    print(f"    Episodes: {task_stats.get('episode_count', 0)}")
                    print(f"    Successes: {task_stats.get('total_successes', 0)}")
    
    print_experiment_stats(stats_in_dist, "OPENPI INSTRUCTION AUGMENTED TASKS")
    print_experiment_stats(stats_ood, "OPENPI OUT-OF-DISTRIBUTION TASKS")
    print_experiment_stats(stats_in_dist_test, "OPENPI INSTRUCTION AUGMENTED TEST TASKS")
    print_experiment_stats(stats_ood_test, "OPENPI OUT-OF-DISTRIBUTION TEST TASKS")
    print_experiment_stats(rephrase_stats, "REPHRASE EXPERIMENTS")
    print_experiment_stats(robomonkey_stats, "ROBOMONKEY EXPERIMENTS")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY:")
    print("="*80)
    
    all_rates_by_type = {}
    
    def collect_rates(stats_dict, type_name):
        rates = []
        for experiment_data in stats_dict.values():
            for task_stats in experiment_data.values():
                if isinstance(task_stats, dict) and any(isinstance(v, dict) and 'success_rate' in v for v in task_stats.values()):
                    task_rates = [eval_data['success_rate'] for eval_data in task_stats.values() if isinstance(eval_data, dict) and 'success_rate' in eval_data]
                    rates.extend(task_rates)
                else:
                    rates.append(task_stats.get('success_rate', 0))
        if rates:
            all_rates_by_type[type_name] = rates
    
    collect_rates(stats_in_dist, "OpenPI Instruction Augmented")
    collect_rates(stats_ood, "OpenPI OOD")
    collect_rates(stats_in_dist_test, "OpenPI Instruction Augmented TEST")
    collect_rates(stats_ood_test, "OpenPI OOD TEST")
    collect_rates(rephrase_stats, "Rephrase Experiments")
    collect_rates(robomonkey_stats, "RoboMonkey Experiments")
    
    for type_name, rates in all_rates_by_type.items():
        if rates:
            print(f"{type_name} Average Success Rate: {np.mean(rates):.3f} ± {np.std(rates):.3f}")
            print(f"{type_name} Total Evaluations: {len(rates)}")

def analyze_verifier_scores_from_pickles(base_path="./"):
    """Analyze verifier scores from pickle files in rollout directories."""
    verifier_data = defaultdict(lambda: defaultdict(lambda: {'success': [], 'failure': []}))
    
    # Find all pickle files in rollout directories
    pickle_files = []
    for item in os.listdir(base_path):
        if item.startswith('rollouts_'):
            rollouts_dir = os.path.join(base_path, item)
            if os.path.isdir(rollouts_dir):
                # Find all pickle files recursively
                pattern = os.path.join(rollouts_dir, "**", "*.pkl")
                pickle_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(pickle_files)} pickle files to analyze")
    
    for pickle_file in pickle_files:
        try:
            with open(pickle_file, 'rb') as f:
                episode_data = pickle.load(f)
            
            # Extract task name from filename
            filename = os.path.basename(pickle_file)
            task_match = re.search(r'--task=([^\.]+)', filename)
            if not task_match:
                continue
            task_name = task_match.group(1)
            
            # Extract success/failure from filename
            success_match = re.search(r'--success=([^\-]+)', filename)
            is_success = success_match.group(1) == 'True' if success_match else False
            
            # Extract experiment info from directory path
            relative_path = os.path.relpath(os.path.dirname(pickle_file), base_path)
            experiment_label = create_folder_label_from_path(relative_path)
            
            # Extract verifier scores and timesteps
            verifier_scores = episode_data.get('verifier_scores', [])
            step_timestamps = episode_data.get('step_timestamps', [])
            
            # Filter out None scores and create valid data pairs
            valid_data = []
            for score, timestamp in zip(verifier_scores, step_timestamps):
                if score is not None:
                    valid_data.append((timestamp, score))
            
            if valid_data:
                # Store data based on success/failure
                if is_success:
                    verifier_data[experiment_label][task_name]['success'].extend(valid_data)
                else:
                    verifier_data[experiment_label][task_name]['failure'].extend(valid_data)
                print(f"  Loaded {len(valid_data)} verifier scores from {filename} (success={is_success})")
                
        except Exception as e:
            print(f"Error loading {pickle_file}: {e}")
            continue
    
    return verifier_data

def create_folder_label_from_path(relative_path):
    """Create experiment label from pickle file path."""
    path_parts = relative_path.split(os.sep)
    
    if len(path_parts) >= 2:
        base_name = path_parts[0]  # e.g., 'rollouts_openpi_original'
        
        # Parse the base name
        if 'robomonkey' in base_name.lower():
            prefix = 'robomonkey'
        elif 'openpi_original' in base_name:
            prefix = 'openpi_original'
        elif 'openpi_rephrase' in base_name:
            prefix = 'openpi_rephrase'
        else:
            prefix = base_name.replace('rollouts_', '')
        
        # Parse transform type and subdirectories
        if len(path_parts) >= 3:
            transform_type = path_parts[1]  # e.g., 'transform_no_transform'
            if transform_type.startswith('transform_'):
                transform_clean = transform_type.replace('transform_', '')
            else:
                transform_clean = transform_type
            
            # Include subdirectory information
            if len(path_parts) > 2:
                subdir_parts = path_parts[2:]
                label = f"{prefix}_{transform_clean}_{'_'.join(subdir_parts)}"
            else:
                label = f"{prefix}_{transform_clean}"
        else:
            label = prefix
    else:
        label = relative_path
    
    return label

def plot_verifier_scores_over_time(verifier_data, output_dir='./analysis_plots'):
    """Create plots showing verifier scores over timesteps for each task and experiment."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a comprehensive plot with all experiments
    plt.figure(figsize=(20, 12))
    
    # Color palette for different experiments
    experiment_colors = {}
    all_experiments = list(verifier_data.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_experiments)))
    for i, exp in enumerate(all_experiments):
        experiment_colors[exp] = colors[i]
    
    # Get all unique tasks across all experiments
    all_tasks = set()
    for exp_data in verifier_data.values():
        all_tasks.update(exp_data.keys())
    all_tasks = sorted(list(all_tasks))
    
    # Create subplots for each task
    n_tasks = len(all_tasks)
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    for task_idx, task_name in enumerate(all_tasks):
        plt.subplot(n_rows, n_cols, task_idx + 1)
        
        task_display_name = task_name.replace('_', ' ').title()
        
        for exp_name, exp_data in verifier_data.items():
            if task_name in exp_data:
                task_data = exp_data[task_name]
                
                # Plot success and failure data separately
                for outcome, data_list in [('success', task_data['success']), ('failure', task_data['failure'])]:
                    if not data_list:
                        continue
                    
                    # Group data by timestep and calculate statistics
                    timestep_scores = defaultdict(list)
                    for timestamp, score in data_list:
                        timestep_scores[timestamp].append(score)
                    
                    # Calculate mean and std for each timestep
                    timesteps = sorted(timestep_scores.keys())
                    means = []
                    stds = []
                    valid_timesteps = []
                    
                    for ts in timesteps:
                        scores = timestep_scores[ts]
                        if len(scores) > 0:
                            means.append(np.mean(scores))
                            stds.append(np.std(scores))
                            valid_timesteps.append(ts)
                    
                    if valid_timesteps:
                        # Choose color based on outcome
                        color = 'green' if outcome == 'success' else 'red'
                        label_suffix = 'Success' if outcome == 'success' else 'Failure'
                        
                        # Plot with error bars (lighter variance)
                        plt.errorbar(valid_timesteps, means, yerr=stds, 
                                   label=f"{exp_name} {label_suffix} (n={len(data_list)})",
                                   color=color,
                                   alpha=0.6, capsize=3, capthick=1,  # Lighter error bars
                                   elinewidth=1,  # Thinner error bar lines
                                   markeredgewidth=0.5)  # Thinner cap lines
                        
                        # Add trend line (darker mean)
                        if len(valid_timesteps) > 1:
                            z = np.polyfit(valid_timesteps, means, 1)
                            p = np.poly1d(z)
                            plt.plot(valid_timesteps, p(valid_timesteps), 
                                   color=color, linestyle='-', alpha=1.0, linewidth=2.5)  # Darker trend line
        
        plt.xlabel('Timestep', fontsize=10)
        plt.ylabel('Verifier Score', fontsize=10)
        plt.title(f'{task_display_name}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.ylim(-0.1, 0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'verifier_scores_over_time.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual plots for each experiment
    for exp_name, exp_data in verifier_data.items():
        if not exp_data:
            continue
            
        plt.figure(figsize=(16, 10))
        
        exp_tasks = list(exp_data.keys())
        n_exp_tasks = len(exp_tasks)
        n_cols_exp = min(3, n_exp_tasks)
        n_rows_exp = (n_exp_tasks + n_cols_exp - 1) // n_cols_exp
        
        for task_idx, task_name in enumerate(exp_tasks):
            plt.subplot(n_rows_exp, n_cols_exp, task_idx + 1)
            
            task_data = exp_data[task_name]
            task_display_name = task_name.replace('_', ' ').title()
            
            # Plot success and failure data separately
            for outcome, data_list in [('success', task_data['success']), ('failure', task_data['failure'])]:
                if not data_list:
                    continue
                
                # Group data by timestep
                timestep_scores = defaultdict(list)
                for timestamp, score in data_list:
                    timestep_scores[timestamp].append(score)
                
                # Calculate statistics for each timestep
                timesteps = sorted(timestep_scores.keys())
                means = []
                stds = []
                counts = []
                valid_timesteps = []
                
                for ts in timesteps:
                    scores = timestep_scores[ts]
                    if len(scores) > 0:
                        means.append(np.mean(scores))
                        stds.append(np.std(scores))
                        counts.append(len(scores))
                        valid_timesteps.append(ts)
                
                if valid_timesteps:
                    # Choose color based on outcome
                    color = 'green' if outcome == 'success' else 'red'
                    label_suffix = 'Success' if outcome == 'success' else 'Failure'
                    
                    # Plot with error bars (lighter variance)
                    plt.errorbar(valid_timesteps, means, yerr=stds, 
                               label=f"{label_suffix} (n={len(data_list)})",
                               color=color,
                               alpha=0.6, capsize=3, capthick=1,  # Lighter error bars
                               elinewidth=1,  # Thinner error bar lines
                               markeredgewidth=0.5)  # Thinner cap lines
                    
                    # Add trend line (darker mean)
                    if len(valid_timesteps) > 1:
                        z = np.polyfit(valid_timesteps, means, 1)
                        p = np.poly1d(z)
                        plt.plot(valid_timesteps, p(valid_timesteps), 
                               color=color, linestyle='-', alpha=1.0, linewidth=2.5)  # Darker trend line
                
                    # Add sample count annotations
                    for i, (ts, count) in enumerate(zip(valid_timesteps, counts)):
                        if i % max(1, len(valid_timesteps) // 5) == 0:  # Show every 5th annotation
                            plt.annotate(f'n={count}', (ts, means[i]), 
                                       xytext=(0, 10), textcoords='offset points',
                                       fontsize=8, ha='center')
            
            plt.xlabel('Timestep', fontsize=10)
            plt.ylabel('Verifier Score', fontsize=10)
            plt.title(f'{task_display_name}', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.1, 0.4)
        
        plt.suptitle(f'Verifier Scores Over Time - {exp_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'verifier_scores_{exp_name}.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    return verifier_data

def plot_verifier_score_distributions(verifier_data, output_dir='./analysis_plots'):
    """Create histogram plots showing the distribution of verifier scores for success vs failure."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots for each experiment
    for exp_name, exp_data in verifier_data.items():
        if not exp_data:
            continue
            
        plt.figure(figsize=(20, 12))
        
        exp_tasks = list(exp_data.keys())
        n_exp_tasks = len(exp_tasks)
        n_cols_exp = min(3, n_exp_tasks)
        n_rows_exp = (n_exp_tasks + n_cols_exp - 1) // n_cols_exp
        
        for task_idx, task_name in enumerate(exp_tasks):
            plt.subplot(n_rows_exp, n_cols_exp, task_idx + 1)
            
            task_data = exp_data[task_name]
            task_display_name = task_name.replace('_', ' ').title()
            
            # Extract all verifier scores for success and failure
            success_scores = [score for timestamp, score in task_data['success']]
            failure_scores = [score for timestamp, score in task_data['failure']]
            
            if success_scores or failure_scores:
                # Create histogram bins
                all_scores = success_scores + failure_scores
                min_score = min(all_scores) if all_scores else 0
                max_score = max(all_scores) if all_scores else 1
                
                # Create bins from -0.1 to 0.4 with 0.01 intervals
                bins = np.arange(-0.1, 0.41, 0.01)
                
                # Plot histograms
                if success_scores:
                    plt.hist(success_scores, bins=bins, alpha=0.7, color='green', 
                           label=f'Success (n={len(success_scores)})', density=False)
                
                if failure_scores:
                    plt.hist(failure_scores, bins=bins, alpha=0.7, color='red', 
                           label=f'Failure (n={len(failure_scores)})', density=False)
                
                plt.xlabel('Verifier Score', fontsize=10)
                plt.ylabel('Count', fontsize=10)
                plt.title(f'{task_display_name}', fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=8)
                plt.xlim(-0.1, 0.4)
                
                # Add statistics text
                if success_scores:
                    success_mean = np.mean(success_scores)
                    success_std = np.std(success_scores)
                    plt.text(0.02, 0.95, f'Success: μ={success_mean:.3f}, σ={success_std:.3f}', 
                           transform=plt.gca().transAxes, fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
                
                if failure_scores:
                    failure_mean = np.mean(failure_scores)
                    failure_std = np.std(failure_scores)
                    plt.text(0.02, 0.85, f'Failure: μ={failure_mean:.3f}, σ={failure_std:.3f}', 
                           transform=plt.gca().transAxes, fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.suptitle(f'Verifier Score Distributions - {exp_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'verifier_distributions_{exp_name}.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create a combined plot with all experiments
    plt.figure(figsize=(20, 12))
    
    # Get all unique tasks across all experiments
    all_tasks = set()
    for exp_data in verifier_data.values():
        all_tasks.update(exp_data.keys())
    all_tasks = sorted(list(all_tasks))
    
    # Create subplots for each task
    n_tasks = len(all_tasks)
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    for task_idx, task_name in enumerate(all_tasks):
        plt.subplot(n_rows, n_cols, task_idx + 1)
        
        task_display_name = task_name.replace('_', ' ').title()
        
        # Collect all scores for this task across all experiments
        all_success_scores = []
        all_failure_scores = []
        
        for exp_name, exp_data in verifier_data.items():
            if task_name in exp_data:
                task_data = exp_data[task_name]
                all_success_scores.extend([score for timestamp, score in task_data['success']])
                all_failure_scores.extend([score for timestamp, score in task_data['failure']])
        
        if all_success_scores or all_failure_scores:
            # Create bins from -0.1 to 0.4 with 0.01 intervals
            bins = np.arange(-0.1, 0.41, 0.01)
            
            # Plot histograms
            if all_success_scores:
                plt.hist(all_success_scores, bins=bins, alpha=0.7, color='green', 
                       label=f'Success (n={len(all_success_scores)})', density=False)
            
            if all_failure_scores:
                plt.hist(all_failure_scores, bins=bins, alpha=0.7, color='red', 
                       label=f'Failure (n={len(all_failure_scores)})', density=False)
            
            plt.xlabel('Verifier Score', fontsize=10)
            plt.ylabel('Count', fontsize=10)
            plt.title(f'{task_display_name}', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.xlim(-0.1, 0.4)
            
            # Add statistics text
            if all_success_scores:
                success_mean = np.mean(all_success_scores)
                success_std = np.std(all_success_scores)
                plt.text(0.02, 0.95, f'Success: μ={success_mean:.3f}, σ={success_std:.3f}', 
                       transform=plt.gca().transAxes, fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            
            if all_failure_scores:
                failure_mean = np.mean(all_failure_scores)
                failure_std = np.std(all_failure_scores)
                plt.text(0.02, 0.85, f'Failure: μ={failure_mean:.3f}, σ={failure_std:.3f}', 
                       transform=plt.gca().transAxes, fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.suptitle('Verifier Score Distributions - All Experiments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'verifier_distributions_all_experiments.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return verifier_data

def discover_rollout_folders(base_path="./"):
    """Discover all folders that start with 'rollouts' and return their analysis results."""
    all_results = {}
    
    # Find all directories that start with 'rollouts'
    for item in os.listdir(base_path):
        if item.startswith('rollouts_'):
            rollouts_dir = os.path.join(base_path, item)
            if os.path.isdir(rollouts_dir):
                print(f"Discovering rollouts in: {rollouts_dir}")
                
                # Analyze this rollouts directory recursively
                results = analyze_rollout_folder_recursive(rollouts_dir, item)
                all_results.update(results)
    
    return all_results

def analyze_rollout_folder_recursive(base_path, base_name):
    """Recursively analyze a rollouts folder and create meaningful labels with evaluation grouping."""
    results = {}
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        # Check if this directory has video files
        video_files = [f for f in files if f.endswith('.mp4')]
        if not video_files:
            continue
            
        # Create a label based on the path structure
        relative_path = os.path.relpath(root, base_path)
        label = create_folder_label(base_name, relative_path)
        
        # Analyze video files in this directory with evaluation grouping
        experiment_results = defaultdict(dict)
        
        # Group videos by task first
        task_episodes = defaultdict(list)
        for video_file in video_files:
            success = extract_success_from_filename(video_file)
            episode_num = extract_episode_number(video_file)
            if success is not None and episode_num is not None:
                # Extract task name from filename
                task_match = re.search(r'--task=([^\.]+)', video_file)
                if task_match:
                    task_name = task_match.group(1)
                    task_episodes[task_name].append((episode_num, success))
        
        # Group episodes into evaluations for each task
        for task_name, episode_data in task_episodes.items():
            if episode_data:
                evaluations = group_episodes_by_evaluation(episode_data)
                experiment_results[task_name] = evaluations
        
        # Only add if we have results
        if experiment_results:
            results[label] = experiment_results
            print(f"  Found {len(video_files)} videos in {relative_path} -> labeled as '{label}'")
    
    return results

def create_folder_label(base_name, relative_path):
    """Create a meaningful label from folder structure, preserving all subdirectory names."""
    # Parse the base name (e.g., 'rollouts_openpi_original')
    if 'robomonkey' in base_name.lower():
        prefix = 'robomonkey'
    elif 'openpi_original' in base_name:
        prefix = 'openpi_original'
    elif 'openpi_rephrase' in base_name:
        prefix = 'openpi_rephrase'
    else:
        prefix = base_name.replace('rollouts_', '')
    
    # Parse the relative path
    path_parts = relative_path.split(os.sep)
    
    # Handle different path structures
    if len(path_parts) >= 1:
        # Clean up transform type if it starts with 'transform_'
        if path_parts[0].startswith('transform_'):
            transform_clean = path_parts[0].replace('transform_', '')
        else:
            transform_clean = path_parts[0]
            
        # Include all remaining path parts as separate components
        if len(path_parts) > 1:
            # Keep all subdirectory names as-is
            subdir_parts = path_parts[1:]
            label = f"{prefix}_{transform_clean}_{'_'.join(subdir_parts)}"
        else:
            label = f"{prefix}_{transform_clean}"
    else:
        label = prefix
    
    return label

def main():
    """Main analysis function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze OpenPI success rates with optional filtering')
    parser.add_argument('--include-insufficient', action='store_true', 
                       help='Include tasks with only one experiment type (default: filter them out)')
    parser.add_argument('--output-dir', default='./analysis_plots',
                       help='Output directory for plots (default: ./analysis_plots)')
    args = parser.parse_args()
    
    print("Analyzing All Rollout Folders Automatically...")
    print(f"Filter insufficient results: {not args.include_insufficient}")
    
    # Automatically discover and analyze all rollout folders
    print("\nDiscovering all rollout folders...")
    all_rollout_results = discover_rollout_folders()
    
    # Calculate statistics for all discovered experiments with evaluation grouping
    print("\nCalculating statistics for all experiments with evaluation grouping...")
    all_stats = calculate_rephrase_statistics(all_rollout_results)
    
    # Separate experiments by type for plotting
    stats_in_dist = {}
    stats_ood = {}
    stats_in_dist_test = {}
    stats_ood_test = {}
    stats_rephrase = {}
    stats_robomonkey = {}
    
    for experiment_name, task_data in all_stats.items():
        if 'robomonkey' in experiment_name.lower():
            stats_robomonkey[experiment_name] = task_data
        elif 'test' in experiment_name:
            if 'no_transform' in experiment_name:
                stats_in_dist_test[experiment_name] = task_data
            else:
                stats_ood_test[experiment_name] = task_data
        elif 'rephrase' in experiment_name and 'openpi_rephrase' in experiment_name:
            stats_rephrase[experiment_name] = task_data
        elif 'openpi_original' in experiment_name:
            if 'no_transform' in experiment_name:
                stats_in_dist[experiment_name] = task_data
            else:
                stats_ood[experiment_name] = task_data
        else:
            # Default to rephrase category for other experiments
            stats_rephrase[experiment_name] = task_data
    
    # Create plots
    print("\nCreating plots...")
    
    # Create evaluation-based plots
    print("\nCreating evaluation period plots...")
    evaluation_summary = create_evaluation_plots(stats_in_dist, stats_ood, stats_rephrase, stats_robomonkey,
                                  stats_in_dist_test=stats_in_dist_test,
                                  stats_ood_test=stats_ood_test,
                                              output_dir=args.output_dir,
                                              filter_insufficient=not args.include_insufficient)
    
    # Analyze verifier scores from pickle files
    print("\nAnalyzing verifier scores from pickle files...")
    verifier_data = analyze_verifier_scores_from_pickles()
    
    if verifier_data:
        print("\nCreating verifier score plots...")
        plot_verifier_scores_over_time(verifier_data, output_dir=args.output_dir)
        
        print("\nCreating verifier score distribution plots...")
        plot_verifier_score_distributions(verifier_data, output_dir=args.output_dir)
    
    # Print summary
    print_summary_statistics(stats_in_dist, stats_ood, stats_rephrase, stats_robomonkey, stats_in_dist_test, stats_ood_test)
    
    print(f"\nAnalysis complete! Plots saved to '{args.output_dir}/'")
    print("Files generated:")
    print("  - evaluation_mean_std_across_4_periods.png (Combined plot: Shows mean and std across evaluation periods for all experiments)")
    print("  - evaluation_mean_std_openpi_original_rephrase.png (NEW: Separate plot for rephrase experiments)")
    print("  - evaluation_mean_std_openpi_original_no_transform.png (NEW: Separate plot for no_transform experiments)")
    print("  - success_rates_all_experiments.png")
    print("  - success_rates_original.png")
    print("  - success_rates_rephrase.png")
    print("  - success_rates_test.png")
    print("  - success_rates_robomonkey.png")
    if verifier_data:
        print("  - verifier_scores_over_time.png (Verifier scores over timesteps for all experiments)")
        for exp_name in verifier_data.keys():
            print(f"  - verifier_scores_{exp_name}.png (Verifier scores for {exp_name})")

if __name__ == "__main__":
    main()
