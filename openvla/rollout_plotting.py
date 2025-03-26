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
    # Get the most recent date folder
    date_path, latest_date = get_latest_date_folder(rollout_dir)
    if date_path is None:
        return None, None
    
    # Get analysis folders (original and transformed language folders)
    analysis_folders = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
    
    results = {}
    time_series_data = {}
    
    # Process each folder
    for folder in analysis_folders:
        folder_results, folder_time_series = process_folder(os.path.join(date_path, folder), folder)
        if folder_results:
            results[folder] = folder_results
            time_series_data[folder] = folder_time_series
    
    # Print and visualize results
    print_results_summary(results)
    
    if results:
        create_visualization(results)
        create_time_series_plots(time_series_data, rollout_dir, latest_date)
    
    return results, time_series_data

def get_latest_date_folder(rollout_dir):
    """Get the most recent date folder from the rollouts directory."""
    if not os.path.exists(rollout_dir):
        print(f"Directory {rollout_dir} does not exist.")
        return None, None
    
    # Find all date folders
    date_folders = [d for d in os.listdir(rollout_dir) if os.path.isdir(os.path.join(rollout_dir, d))]
    if not date_folders:
        print(f"No date folders found in {rollout_dir}")
        return None, None
    
    # Sort by date (assuming YYYY_MM_DD format)
    date_folders.sort(key=lambda x: datetime.strptime(x, "%Y_%m_%d") if len(x.split("_")) == 3 else datetime.min)
    latest_date = date_folders[-1]
    
    return os.path.join(rollout_dir, latest_date), latest_date

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
    
    # For time series analysis
    success_time_series = []
    failure_time_series = []
    
    for pkl_file in pkl_files:
        # Extract success info from filename
        filename = os.path.basename(pkl_file)
        is_success = "success=True" in filename
        
        # Load the pickle file
        with open(pkl_file, "rb") as f:
            score_list = pickle.load(f)
        
        # Calculate average score for this trajectory (using non-zero scores only)
        if score_list:
            non_zero_scores = [score for score in score_list if score != 0]
            avg_score = np.mean(non_zero_scores) if non_zero_scores else 0
            
            if is_success:
                success_scores.append(avg_score)
                success_time_series.append(score_list)
            else:
                failure_scores.append(avg_score)
                failure_time_series.append(score_list)
    
    # Calculate statistics
    folder_results = {
        "success_count": len(success_scores),
        "failure_count": len(failure_scores),
        "success_rate": len(success_scores) / (len(success_scores) + len(failure_scores)) if (len(success_scores) + len(failure_scores)) > 0 else 0,
        "avg_success_score": np.mean(success_scores) if success_scores else 0,
        "avg_failure_score": np.mean(failure_scores) if failure_scores else 0,
        "std_success_score": np.std(success_scores) if success_scores else 0,
        "std_failure_score": np.std(failure_scores) if failure_scores else 0,
    }
    
    # Store time series data
    folder_time_series = {
        "success_time_series": success_time_series,
        "failure_time_series": failure_time_series
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
    ax.set_xticklabels(folders)
    ax.legend()
    
    # Add success rate as text on top of bars
    for i, folder in enumerate(folders):
        ax.text(i, max(success_scores[i], failure_scores[i]) + 0.02, 
                f"{results[folder]['success_rate']:.1%}", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join("./rollouts", "clip_score_analysis.png"))
    plt.close()

def create_time_series_plots(time_series_data, rollout_dir, date):
    """
    Create time series plots showing average CLIP score at each timestep
    for successful and failed trajectories, including variance.
    """
    # Create a directory for the plots
    plots_dir = os.path.join(rollout_dir, date, "plots")
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
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.gca()  # Get current axis
        plot_time_series(ax, folder, data, is_subplot=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{folder}_time_series.png"))
        plt.close()

def plot_time_series(ax, folder, data, is_subplot=True):
    """Plot time series data for a single folder."""
    success_series = data["success_time_series"]
    failure_series = data["failure_time_series"]
    
    # Find the maximum length across all trajectories
    max_len_success = max([len(series) for series in success_series]) if success_series else 0
    max_len_failure = max([len(series) for series in failure_series]) if failure_series else 0
    max_len = max(max_len_success, max_len_failure)
    
    # For each timestep, collect all non-zero scores across trajectories
    success_scores_by_timestep = [[] for _ in range(max_len)]
    failure_scores_by_timestep = [[] for _ in range(max_len)]
    
    # Accumulate non-zero scores for each timestep
    for series in success_series:
        for t, score in enumerate(series):
            if score != 0:  # Only include non-zero scores
                success_scores_by_timestep[t].append(score)
    
    for series in failure_series:
        for t, score in enumerate(series):
            if score != 0:  # Only include non-zero scores
                failure_scores_by_timestep[t].append(score)
    
    # Calculate averages and standard deviations from non-zero scores
    avg_success = np.zeros(max_len)
    avg_failure = np.zeros(max_len)
    std_success = np.zeros(max_len)
    std_failure = np.zeros(max_len)
    
    for t in range(max_len):
        if success_scores_by_timestep[t]:
            avg_success[t] = np.mean(success_scores_by_timestep[t])
            if len(success_scores_by_timestep[t]) > 1:
                std_success[t] = np.std(success_scores_by_timestep[t])
        
        if failure_scores_by_timestep[t]:
            avg_failure[t] = np.mean(failure_scores_by_timestep[t])
            if len(failure_scores_by_timestep[t]) > 1:
                std_failure[t] = np.std(failure_scores_by_timestep[t])
    
    # Plot the time series with shaded variance
    timesteps = np.arange(1, max_len + 1)
    
    if np.any(avg_success > 0):
        ax.plot(timesteps, avg_success, 'g-', label=f'Success (n={len(success_series)})', linewidth=2)
        ax.fill_between(
            timesteps, 
            avg_success - std_success, 
            avg_success + std_success, 
            color='g', 
            alpha=0.2
        )
    
    if np.any(avg_failure > 0):
        ax.plot(timesteps, avg_failure, 'r-', label=f'Failure (n={len(failure_series)})', linewidth=2)
        ax.fill_between(
            timesteps, 
            avg_failure - std_failure, 
            avg_failure + std_failure, 
            color='r', 
            alpha=0.2
        )
    
    # Add count of non-zero scores to the legend
    non_zero_success_counts = [len(scores) for scores in success_scores_by_timestep]
    non_zero_failure_counts = [len(scores) for scores in failure_scores_by_timestep]
    
    ax.set_ylabel('Average CLIP Score (non-zero only)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title=f'Non-zero scores: Success max={max(non_zero_success_counts) if non_zero_success_counts else 0}, Failure max={max(non_zero_failure_counts) if non_zero_failure_counts else 0}')
    
    # Add success rate to the title
    success_rate = len(success_series) / (len(success_series) + len(failure_series)) if (len(success_series) + len(failure_series)) > 0 else 0
    ax.set_title(f'{folder} - CLIP Score Over Time (Success Rate: {success_rate:.1%})')
    
    if not is_subplot:
        ax.set_xlabel('Timestep')

if __name__ == "__main__":
    results, time_series_data = analyze_rollouts()