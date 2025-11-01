#!/usr/bin/env python3
"""
Script to generate evaluation plots comparing lang_1_sample_1 and robomonkey folders.
Creates a grouped bar plot showing mean and std success rates across evaluation periods.
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

def analyze_folder(folder_path):
    """
    Analyze rollouts in a folder, grouping episodes into evaluations.
    
    Args:
        folder_path: Path to the rollouts folder
    
    Returns:
        Dictionary with task evaluation data
    """
    results = defaultdict(dict)
    
    # Collect episode data for each task
    task_episodes = defaultdict(list)
    
    for root, dirs, files in os.walk(folder_path):
        for video_file in files:
            if not video_file.endswith('.mp4'):
                continue
            
            success = extract_success_from_filename(video_file)
            episode_num = extract_episode_number(video_file)
            
            if success is None or episode_num is None:
                continue
            
            # Extract task name from filename
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
            results[task_name] = evaluations
    
    return results

def create_comparison_plot(lang1_data, robomonkey_data, output_path='./analysis_plots/evaluation_mean_std_action_chunks_comparison.png'):
    """
    Create a grouped bar plot comparing lang_1_sample_1 and robomonkey.
    
    Args:
        lang1_data: Dictionary with task evaluation data for lang_1_sample_1
        robomonkey_data: Dictionary with task evaluation data for robomonkey
        output_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Collect data for plotting
    plot_data = []
    
    # Get all unique tasks
    all_tasks = set(lang1_data.keys()) | set(robomonkey_data.keys())
    all_tasks = sorted(all_tasks)
    
    # Process lang_1_sample_1 data
    for task_name in all_tasks:
        if task_name in lang1_data:
            task_data = lang1_data[task_name]
            eval_rates = []
            for eval_name in ['Eval_1', 'Eval_2', 'Eval_3', 'Eval_4']:
                if eval_name in task_data and isinstance(task_data[eval_name], dict) and 'success_rate' in task_data[eval_name]:
                    eval_rates.append(task_data[eval_name]['success_rate'])
            
            if len(eval_rates) >= 2:  # Need at least 2 evaluations to calculate std
                plot_data.append({
                    'Condition': 'action chunks',
                    'Task': task_name.replace('_', ' ').title(),
                    'Mean': np.mean(eval_rates),
                    'Std': np.std(eval_rates),
                    'N': len(eval_rates)
                })
    
    # Process robomonkey data
    for task_name in all_tasks:
        if task_name in robomonkey_data:
            task_data = robomonkey_data[task_name]
            eval_rates = []
            for eval_name in ['Eval_1', 'Eval_2', 'Eval_3', 'Eval_4']:
                if eval_name in task_data and isinstance(task_data[eval_name], dict) and 'success_rate' in task_data[eval_name]:
                    eval_rates.append(task_data[eval_name]['success_rate'])
            
            if len(eval_rates) >= 2:  # Need at least 2 evaluations to calculate std
                plot_data.append({
                    'Condition': 'action_chunk_1',
                    'Task': task_name.replace('_', ' ').title(),
                    'Mean': np.mean(eval_rates),
                    'Std': np.std(eval_rates),
                    'N': len(eval_rates)
                })
    
    if not plot_data:
        print("No data to plot")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    df = pd.DataFrame(plot_data)
    
    # Grouped bar plot: tasks on X-axis, multiple bars per task
    tasks = sorted(df['Task'].unique())
    conditions = ['action chunks', 'action_chunk_1']
    
    x_pos = np.arange(len(tasks))
    width = 0.8 / len(conditions)
    
    colors = ['#2E86AB', '#A23B72']  # Different colors for the two conditions
    
    for i, condition in enumerate(conditions):
        condition_data = df[df['Condition'] == condition]
        
        task_means = []
        task_stds = []
        
        for task in tasks:
            task_data = condition_data[condition_data['Task'] == task]
            if not task_data.empty:
                task_means.append(task_data['Mean'].iloc[0])
                task_stds.append(task_data['Std'].iloc[0])
            else:
                task_means.append(0)
                task_stds.append(0)
        
        bars = plt.bar(x_pos + i * width, task_means, width,
                      yerr=task_stds, capsize=3, alpha=0.8,
                      color=colors[i], edgecolor='black', linewidth=0.5,
                      label=condition)
        
        # Add value labels
        for bar, mean_val, std_val in zip(bars, task_means, task_stds):
            if mean_val > 0:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                        f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontsize=6)
    
    plt.xlabel('Tasks', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate (Mean ± Std across evaluations)', fontsize=14, fontweight='bold')
    plt.title('Action Chunks Comparison - Mean and Std Across Evaluation Periods', 
             fontsize=16, fontweight='bold')
    plt.xticks(x_pos + width * (len(conditions) - 1) / 2, 
              tasks, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plot for lang_1_sample_1 and robomonkey')
    parser.add_argument('--base_dir', type=str, 
                       default='./rollouts_openpi_original/transform_rephrase',
                       help='Base directory containing the folders')
    parser.add_argument('--output_dir', type=str, 
                       default='./analysis_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Paths to the two folders
    lang1_path = os.path.join(args.base_dir, 'lang_1_sample_1')
    robomonkey_path = os.path.join(args.base_dir, 'robomonkey')
    
    # Check if folders exist
    if not os.path.exists(lang1_path):
        print(f"Error: Folder not found: {lang1_path}")
        return
    
    if not os.path.exists(robomonkey_path):
        print(f"Error: Folder not found: {robomonkey_path}")
        return
    
    print(f"Analyzing {lang1_path}...")
    lang1_data = analyze_folder(lang1_path)
    print(f"Found {len(lang1_data)} tasks")
    
    print(f"\nAnalyzing {robomonkey_path}...")
    robomonkey_data = analyze_folder(robomonkey_path)
    print(f"Found {len(robomonkey_data)} tasks")
    
    # Create comparison plot
    output_path = os.path.join(args.output_dir, 'evaluation_mean_std_action_chunks_comparison.png')
    create_comparison_plot(lang1_data, robomonkey_data, output_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()

