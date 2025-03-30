import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from datetime import datetime

def analyze_rollouts(rollout_dir="./rollouts"):
    """
    Analyze CLIP scores from rollout data in the specified directory.
    
    Args:
        rollout_dir: Path to the directory containing rollout data
        
    Returns:
        tuple: (results dictionary, time_series_data dictionary)
    """
    if not os.path.exists(rollout_dir):
        print(f"Directory {rollout_dir} does not exist.")
        return None, None
    
    # Get clip filter folders (True/False)
    clip_filter_folders = [d for d in os.listdir(rollout_dir) 
                         if os.path.isdir(os.path.join(rollout_dir, d)) and d.startswith("clip_filter_")]
    
    if not clip_filter_folders:
        print(f"No clip_filter folders found in {rollout_dir}")
        return None, None
    
    results = {}
    time_series_data = {}
    
    # Process each clip filter folder
    for clip_filter_folder in clip_filter_folders:
        clip_filter_path = os.path.join(rollout_dir, clip_filter_folder)
        
        # Get language transformation folders (original, synonym, etc.)
        lang_transform_folders = [d for d in os.listdir(clip_filter_path) 
                                if os.path.isdir(os.path.join(clip_filter_path, d))]
        
        # Process each language transformation folder
        for lang_folder in lang_transform_folders:
            folder_path = os.path.join(clip_filter_path, lang_folder)
            folder_name = f"{clip_filter_folder}/{lang_folder}"
            
            folder_results, folder_time_series = process_folder(folder_path, folder_name)
            if folder_results:
                results[folder_name] = folder_results
                time_series_data[folder_name] = folder_time_series
    
    # Print and visualize results
    if results:
        print_results_summary(results)
        create_visualization(results)
        create_time_series_plots(time_series_data, rollout_dir)
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(rollout_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create score-length correlation plots
        create_score_length_plots(results, plots_dir)
    
    return results, time_series_data

def process_folder(folder_path, folder_name):
    """Process a single folder of rollout data."""
    print(f"Analyzing folder: {folder_path}")
    
    # Find all pickle files
    pkl_files = glob(os.path.join(folder_path, "*.pkl"))
    if not pkl_files:
        print(f"No pickle files found in {folder_path}")
        return None, None
    
    success_scores = []
    failure_scores = []
    success_actions = []
    failure_actions = []
    success_lengths = []  # Add episode lengths
    failure_lengths = []  # Add episode lengths
    
    # For time series analysis
    success_time_series = []
    failure_time_series = []
    success_action_series = []
    failure_action_series = []
    
    for pkl_file in pkl_files:
        # Extract success info from filename
        filename = os.path.basename(pkl_file)
        is_success = "success=True" in filename
        
        # Load the pickle file
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            score_list = data["score_list"]
            action_list = data["action_list"]
        
        # Calculate average score for this trajectory (using non-zero scores only)
        if score_list:
            non_zero_scores = [score[0] for score in score_list if score[0] != 0]
            avg_score = np.mean(non_zero_scores) if non_zero_scores else 0
            episode_length = len(score_list)  # Get episode length
            
            if is_success:
                success_scores.append(avg_score)
                success_lengths.append(episode_length)
                success_time_series.append(score_list)
                success_action_series.append(action_list)
            else:
                failure_scores.append(avg_score)
                failure_lengths.append(episode_length)
                failure_time_series.append(score_list)
                failure_action_series.append(action_list)
    
    # Calculate statistics
    folder_results = {
        "success_count": len(success_scores),
        "failure_count": len(failure_scores),
        "success_rate": len(success_scores) / (len(success_scores) + len(failure_scores)) if (len(success_scores) + len(failure_scores)) > 0 else 0,
        "avg_success_score": np.mean(success_scores) if success_scores else 0,
        "avg_failure_score": np.mean(failure_scores) if failure_scores else 0,
        "std_success_score": np.std(success_scores) if success_scores else 0,
        "std_failure_score": np.std(failure_scores) if failure_scores else 0,
        "success_scores": success_scores,  # Add individual scores
        "failure_scores": failure_scores,
        "success_lengths": success_lengths,  # Add episode lengths
        "failure_lengths": failure_lengths,
    }
    
    # Store time series data
    folder_time_series = {
        "success_time_series": success_time_series,
        "failure_time_series": failure_time_series,
        "success_action_series": success_action_series,
        "failure_action_series": failure_action_series
    }
    
    return folder_results, folder_time_series

def print_results_summary(results):
    """Print a summary of the results."""
    print("\nResults Summary:")
    for folder, stats in results.items():
        print(f"\n{folder}:")
        print(f"  Success rate: {stats['success_rate']:.2%} ({stats['success_count']}/{stats['success_count'] + stats['failure_count']})")
        print(f"  Avg success score: {stats['avg_success_score']:.4f} ± {stats['std_success_score']:.4f}")
        print(f"  Avg failure score: {stats['avg_failure_score']:.4f} ± {stats['std_failure_score']:.4f}")

def create_visualization(results):
    """Create bar chart visualizations for the results."""
    # Bar chart for success vs failure scores
    folders = list(results.keys())
    success_scores = [results[f]["avg_success_score"] for f in folders]
    failure_scores = [results[f]["avg_failure_score"] for f in folders]
    success_std = [results[f]["std_success_score"] for f in folders]
    failure_std = [results[f]["std_failure_score"] for f in folders]
    
    x = np.arange(len(folders))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, success_scores, width, label='Success', yerr=success_std, capsize=5)
    rects2 = ax.bar(x + width/2, failure_scores, width, label='Failure', yerr=failure_std, capsize=5)
    
    ax.set_ylabel('Average CLIP Score')
    ax.set_title('CLIP Scores by Outcome and Condition')
    ax.set_xticks(x)
    # Modify x-axis labels to be more readable
    folder_labels = [f.replace('clip_filter_', 'Filter ').replace('/', '\n') for f in folders]
    ax.set_xticklabels(folder_labels)
    ax.legend()
    
    # Add success rate as text on top of bars
    for i, folder in enumerate(folders):
        ax.text(i, max(success_scores[i], failure_scores[i]) + 0.02, 
                f"{results[folder]['success_rate']:.1%}", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join("./rollouts", "plots", "clip_score_analysis.png"))
    plt.close()

def create_time_series_plots(time_series_data, rollout_dir):
    """
    Create time series plots showing average CLIP score at each timestep
    for successful and failed trajectories, including variance.
    """
    # Create a directory for the plots
    plots_dir = os.path.join(rollout_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create combined plot with subplots
    create_combined_time_series_plot(time_series_data, plots_dir)
    
    # Create individual plots for each transformation type
    create_individual_time_series_plots(time_series_data, plots_dir)

def create_combined_time_series_plot(time_series_data, plots_dir):
    """Create a combined plot with subplots for each transformation type."""
    n_folders = len(time_series_data)
    fig, axes = plt.subplots(n_folders, 1, figsize=(12, 5 * n_folders), sharex=True)
    
    # If there's only one folder, axes won't be an array
    if n_folders == 1:
        axes = [axes]
    
    for i, (folder, data) in enumerate(time_series_data.items()):
        plot_time_series(axes[i], folder, data, is_subplot=True)
    
    # Set common x-axis label
    plt.xlabel('Timestep')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(plots_dir, "time_series_analysis.png"))
    plt.close()

def create_individual_time_series_plots(time_series_data, plots_dir):
    """Create individual plots for each transformation type."""
    for folder, data in time_series_data.items():
        # Skip if no data
        if not data["success_time_series"] and not data["failure_time_series"]:
            continue
        
        # Replace slashes with underscores for filename
        safe_folder_name = folder.replace('/', '_')
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.gca()  # Get current axis
        plot_time_series(ax, folder, data, is_subplot=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{safe_folder_name}_time_series.png"))
        plt.close()

def plot_time_series(ax, folder, data, is_subplot=True):
    """Plot time series data for a single folder."""
    success_series = data["success_time_series"]
    failure_series = data["failure_time_series"]
    success_action_series = data["success_action_series"]
    failure_action_series = data["failure_action_series"]
    
    # Find the maximum length across all trajectories
    max_len_success = max([len(series) for series in success_series]) if success_series else 0
    max_len_failure = max([len(series) for series in failure_series]) if failure_series else 0
    max_len = max(max_len_success, max_len_failure)
    
    # For each timestep, collect all non-zero scores and actions across trajectories
    success_scores_by_timestep = [[] for _ in range(max_len)]
    failure_scores_by_timestep = [[] for _ in range(max_len)]
    success_actions_by_timestep = [[] for _ in range(max_len)]
    failure_actions_by_timestep = [[] for _ in range(max_len)]
    
    # Accumulate non-zero scores and actions for each timestep
    for series, action_series in zip(success_series, success_action_series):
        for t, (score, action) in enumerate(zip(series, action_series)):
            if score[0] != 0:  # Only include non-zero scores
                success_scores_by_timestep[t].append(score)
                success_actions_by_timestep[t].append(action)
    
    for series, action_series in zip(failure_series, failure_action_series):
        for t, (score, action) in enumerate(zip(series, action_series)):
            if score[0] != 0:  # Only include non-zero scores
                failure_scores_by_timestep[t].append(score)
                failure_actions_by_timestep[t].append(action)
    
    # Calculate averages and standard deviations from non-zero scores and actions
    avg_success_score = np.zeros(max_len)
    avg_failure_score = np.zeros(max_len)
    std_success_score = np.zeros(max_len)
    std_failure_score = np.zeros(max_len)
    
    # For actions, we'll store the norm of the action vector
    avg_success_action_norm = np.zeros(max_len)
    avg_failure_action_norm = np.zeros(max_len)
    std_success_action_norm = np.zeros(max_len)
    std_failure_action_norm = np.zeros(max_len)
    
    for t in range(max_len):
        if success_scores_by_timestep[t]:
            avg_success_score[t] = np.mean(success_scores_by_timestep[t])
            if len(success_scores_by_timestep[t]) > 1:
                std_success_score[t] = np.std(success_scores_by_timestep[t])
            
            actions = np.array(success_actions_by_timestep[t])
            if len(actions) > 0:
                # Calculate norm of each action vector
                action_norms = np.linalg.norm(actions, axis=1)
                avg_success_action_norm[t] = np.mean(action_norms)
                if len(actions) > 1:
                    std_success_action_norm[t] = np.std(action_norms)
        
        if failure_scores_by_timestep[t]:
            avg_failure_score[t] = np.mean(failure_scores_by_timestep[t])
            if len(failure_scores_by_timestep[t]) > 1:
                std_failure_score[t] = np.std(failure_scores_by_timestep[t])
            
            actions = np.array(failure_actions_by_timestep[t])
            if len(actions) > 0:
                # Calculate norm of each action vector
                action_norms = np.linalg.norm(actions, axis=1)
                avg_failure_action_norm[t] = np.mean(action_norms)
                if len(actions) > 1:
                    std_failure_action_norm[t] = np.std(action_norms)
    
    # Create figure with two subplots
    if not is_subplot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    else:
        ax1 = ax
        ax2 = ax.twinx()  # Create a twin axis for actions
    
    timesteps = np.arange(1, max_len + 1)
    
    # Plot scores
    if np.any(avg_success_score > 0):
        ax1.plot(timesteps, avg_success_score, 'g-', label=f'Success Score (n={len(success_series)})', linewidth=2)
        ax1.fill_between(
            timesteps, 
            avg_success_score - std_success_score, 
            avg_success_score + std_success_score, 
            color='g', 
            alpha=0.2
        )
    
    if np.any(avg_failure_score > 0):
        ax1.plot(timesteps, avg_failure_score, 'r-', label=f'Failure Score (n={len(failure_series)})', linewidth=2)
        ax1.fill_between(
            timesteps, 
            avg_failure_score - std_failure_score, 
            avg_failure_score + std_failure_score, 
            color='r', 
            alpha=0.2
        )
    
    # Plot action norms with shaded variance
    if np.any(avg_success_action_norm != 0):
        ax2.plot(timesteps, avg_success_action_norm, 'b--', label=f'Success Action Norm', linewidth=1)
        ax2.fill_between(
            timesteps,
            avg_success_action_norm - std_success_action_norm,
            avg_success_action_norm + std_success_action_norm,
            color='b',
            alpha=0.1
        )
    if np.any(avg_failure_action_norm != 0):
        ax2.plot(timesteps, avg_failure_action_norm, 'm--', label=f'Failure Action Norm', linewidth=1)
        ax2.fill_between(
            timesteps,
            avg_failure_action_norm - std_failure_action_norm,
            avg_failure_action_norm + std_failure_action_norm,
            color='m',
            alpha=0.1
        )
    
    # Add count of non-zero scores to the legend
    non_zero_success_counts = [len(scores) for scores in success_scores_by_timestep]
    non_zero_failure_counts = [len(scores) for scores in failure_scores_by_timestep]
    
    ax1.set_ylabel('Average CLIP Score (non-zero only)')
    ax2.set_ylabel('Action Norm')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              title=f'Non-zero scores: Success max={max(non_zero_success_counts) if non_zero_success_counts else 0}, Failure max={max(non_zero_failure_counts) if non_zero_failure_counts else 0}')
    
    # Add success rate to the title
    success_rate = len(success_series) / (len(success_series) + len(failure_series)) if (len(success_series) + len(failure_series)) > 0 else 0
    ax1.set_title(f'{folder} - CLIP Score and Action Norm Over Time (Success Rate: {success_rate:.1%})')
    
    if not is_subplot:
        ax1.set_xlabel('Timestep')
        plt.tight_layout()

def create_score_length_plots(results, plots_dir):
    """Create scatter plots showing the relationship between average CLIP scores and episode lengths."""
    # Create a figure with subplots for each condition
    n_folders = len(results)
    fig, axes = plt.subplots(n_folders, 1, figsize=(12, 6 * n_folders))
    
    # If there's only one folder, make axes iterable
    if n_folders == 1:
        axes = [axes]
    
    # Plot each condition
    for i, (folder, stats) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot success and failure points
        if stats["success_scores"]:
            ax.scatter(stats["success_scores"], stats["success_lengths"], 
                      c='g', label='Success', alpha=0.6)
            # Add trend line for success
            z = np.polyfit(stats["success_scores"], stats["success_lengths"], 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(stats["success_scores"]), max(stats["success_scores"]), 100)
            ax.plot(x_range, p(x_range), "g--", alpha=0.8)
        
        if stats["failure_scores"]:
            ax.scatter(stats["failure_scores"], stats["failure_lengths"], 
                      c='r', label='Failure', alpha=0.6)
            # Add trend line for failure
            z = np.polyfit(stats["failure_scores"], stats["failure_lengths"], 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(stats["failure_scores"]), max(stats["failure_scores"]), 100)
            ax.plot(x_range, p(x_range), "r--", alpha=0.8)
        
        # Calculate correlations
        if stats["success_scores"]:
            success_corr = np.corrcoef(stats["success_scores"], stats["success_lengths"])[0,1]
        else:
            success_corr = 0
        if stats["failure_scores"]:
            failure_corr = np.corrcoef(stats["failure_scores"], stats["failure_lengths"])[0,1]
        else:
            failure_corr = 0
        
        # Add labels and title
        ax.set_xlabel('Average CLIP Score')
        ax.set_ylabel('Episode Length')
        folder_label = folder.replace('clip_filter_', 'Filter ').replace('/', ' - ')
        ax.set_title(f'{folder_label}\nCorrelations - Success: {success_corr:.2f}, Failure: {failure_corr:.2f}')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "score_length_correlation.png"))
    plt.close()

if __name__ == "__main__":
    results, time_series_data = analyze_rollouts()