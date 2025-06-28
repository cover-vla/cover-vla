import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from datetime import datetime
import re # Import re for parsing filenames
from collections import defaultdict

def analyze_rollouts(rollout_dir="./rollouts_oracle"):
    """
    Analyze CLIP scores from rollout data in the specified directory.
    Assumes the structure: ./rollouts/{condition_folder}/...pkl
    where {condition_folder} might be 'original', 'synonym', 'clip_filtered_original', etc.

    Args:
        rollout_dir: Path to the directory containing rollout data

    Returns:
        tuple: (results dictionary, time_series_data dictionary)
    """
    if not os.path.exists(rollout_dir):
        print(f"Directory {rollout_dir} does not exist.")
        return None, None

    # Get all subdirectories directly under rollout_dir, these are the conditions
    # Exclude the 'plots' directory if it exists
    condition_folders = [d for d in os.listdir(rollout_dir)
                         if os.path.isdir(os.path.join(rollout_dir, d)) and d != "plots"]

    if not condition_folders:
        print(f"No condition folders found in {rollout_dir}")
        return None, None

    results = {}
    time_series_data = {}

    # Process each condition folder
    for folder_name in condition_folders:
        folder_path = os.path.join(rollout_dir, folder_name)

        print(f"\n--- Analyzing Condition: {folder_name} ---")
        folder_results, folder_time_series = process_folder(folder_path, folder_name)
        if folder_results:
            results[folder_name] = folder_results
            time_series_data[folder_name] = folder_time_series

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(rollout_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Print and visualize results
    if results:
        print_results_summary(results)
        create_visualization(results, plots_dir)
        create_time_series_plots(time_series_data, plots_dir)
        # create_score_length_plots(results, plots_dir)
        # create_global_score_length_plot(results, plots_dir)
        # create_rate_vs_score_plot(results, plots_dir)
        create_task_success_rate_plots(rollout_dir)

    return results, time_series_data

def process_folder(folder_path, folder_name):
    """Process a single folder of rollout data."""
    print(f"Processing folder: {folder_path}")

    # Find all pickle files
    pkl_files = glob(os.path.join(folder_path, "*.pkl"))
    if not pkl_files:
        print(f"No pickle files found in {folder_path}")
        return None, None

    success_scores = []
    failure_scores = []
    success_actions = [] # Keep track of actions for time series
    failure_actions = [] # Keep track of actions for time series
    success_lengths = []
    failure_lengths = []

    # For time series analysis
    success_time_series = []
    failure_time_series = []
    success_action_series = [] # Keep separate for time series plotting
    failure_action_series = [] # Keep separate for time series plotting

    for pkl_file in pkl_files:
        filename = os.path.basename(pkl_file)

        # --- Filename Parsing ---
        success_match = re.search(r'success=(True|False)', filename)
        if not success_match:
            print(f"Warning: Could not parse success status from filename: {filename}. Skipping.")
            continue
        is_success = success_match.group(1) == 'True'

        # --- Load Data ---
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                if "score_list" not in data or "action_list" not in data:
                     print(f"Warning: Missing 'score_list' or 'action_list' in {filename}. Skipping.")
                     continue
                score_list = data["score_list"]
                action_list = data["action_list"]
        except Exception as e:
             print(f"Error loading or reading {pkl_file}: {e}. Skipping.")
             continue

        # --- Basic Validation ---
        if not isinstance(score_list, list) or not isinstance(action_list, list):
            print(f"Warning: 'score_list' or 'action_list' is not a list in {filename}. Skipping.")
            continue
        if len(score_list) != len(action_list):
             print(f"Warning: Mismatch between score_list ({len(score_list)}) and action_list ({len(action_list)}) lengths in {filename}. Skipping.")
             continue
        # We handle empty score_list below

        # --- Calculate Trajectory Avg Score ---
        # Convert to numpy array to easily handle potential None/NaNs and zeros
        scores_np = np.array(score_list, dtype=float) # Use float for NaN compatibility
        # Replace 0 with NaN, then calculate nanmean
        scores_np[scores_np == 0] = np.nan
        # np.nanmean ignores NaNs. Returns NaN if scores_np contains only NaNs after replacement.
        avg_score = np.nanmean(scores_np)

        episode_length = len(score_list) # Get episode length

        # --- Store Results ---
        if is_success:
            success_scores.append(avg_score) # avg_score might be NaN here
            success_lengths.append(episode_length)
            success_time_series.append(score_list) # Store original list for time series plot
            success_action_series.append(action_list)
        else:
            failure_scores.append(avg_score) # avg_score might be NaN here
            failure_lengths.append(episode_length)
            failure_time_series.append(score_list) # Store original list for time series plot
            failure_action_series.append(action_list)


    # Calculate overall statistics using nanmean/nanstd
    # These functions return NaN if the input list is empty or contains only NaNs.
    avg_success_overall = np.nanmean(success_scores) # Automatically handles NaNs in the list
    avg_failure_overall = np.nanmean(failure_scores) # Automatically handles NaNs in the list
    # Ensure at least 2 valid (non-NaN) points for std dev calculation
    std_success_overall = np.nanstd([s for s in success_scores if not np.isnan(s)]) if len([s for s in success_scores if not np.isnan(s)]) > 1 else np.nan
    std_failure_overall = np.nanstd([s for s in failure_scores if not np.isnan(s)]) if len([s for s in failure_scores if not np.isnan(s)]) > 1 else np.nan

    folder_results = {
        "success_count": len(success_scores), # Total count including potential NaNs
        "failure_count": len(failure_scores),
        "success_rate": len(success_scores) / (len(success_scores) + len(failure_scores)) if (len(success_scores) + len(failure_scores)) > 0 else 0,
        # Store the overall calculated nanmean/nanstd (might be NaN)
        "avg_success_score": avg_success_overall,
        "avg_failure_score": avg_failure_overall,
        "std_success_score": std_success_overall,
        "std_failure_score": std_failure_overall,
        # Keep the raw lists containing potential NaNs
        "success_scores": success_scores,
        "failure_scores": failure_scores,
        "success_lengths": success_lengths,
        "failure_lengths": failure_lengths,
    }

    # Store time series data
    folder_time_series = {
        "success_time_series": success_time_series,
        "failure_time_series": failure_time_series,
        "success_action_series": success_action_series, # Add actions here
        "failure_action_series": failure_action_series  # Add actions here
    }

    return folder_results, folder_time_series

def print_results_summary(results):
    """Print a summary of the results."""
    print("\n--- Results Summary ---")
    for folder, stats in results.items():
        print(f"\nCondition: {folder}")
        print(f"  Success rate: {stats['success_rate']:.2%} ({stats['success_count']}/{stats['success_count'] + stats['failure_count']})")
        # Directly use the pre-calculated overall averages/stds (which might be NaN)
        print(f"  Avg success score: {stats['avg_success_score']:.4f} ± {stats['std_success_score']:.4f}")
        print(f"  Avg failure score: {stats['avg_failure_score']:.4f} ± {stats['std_failure_score']:.4f}")
        if stats['success_lengths']:
             # Filter NaNs before calculating length stats if necessary, though lengths shouldn't be NaN
             valid_success_lengths = [l for i, l in enumerate(stats['success_lengths']) if not np.isnan(stats['success_scores'][i])] if stats['success_scores'] else stats['success_lengths']
             if valid_success_lengths:
                  print(f"  Avg success length: {np.mean(valid_success_lengths):.1f} ± {np.std(valid_success_lengths):.1f}")
             else:
                  print(f"  Avg success length: N/A (no valid scores)")
        if stats['failure_lengths']:
             valid_failure_lengths = [l for i, l in enumerate(stats['failure_lengths']) if not np.isnan(stats['failure_scores'][i])] if stats['failure_scores'] else stats['failure_lengths']
             if valid_failure_lengths:
                  print(f"  Avg failure length: {np.mean(valid_failure_lengths):.1f} ± {np.std(valid_failure_lengths):.1f}")
             else:
                  print(f"  Avg failure length: N/A (no valid scores)")

def create_visualization(results, plots_dir): # Added plots_dir parameter
    """Create bar chart visualizations for the results."""
    folders = list(results.keys())
    if not folders: return # Skip if no results

    success_scores = [results[f]["avg_success_score"] for f in folders]
    failure_scores = [results[f]["avg_failure_score"] for f in folders]
    success_std = [results[f]["std_success_score"] for f in folders]
    failure_std = [results[f]["std_failure_score"] for f in folders]

    x = np.arange(len(folders))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(folders) * 1.5), 7)) # Adjust width based on number of folders
    rects1 = []
    rects2 = []
    for i, folder in enumerate(folders):
        r1 = ax.bar(x[i] - width/2, success_scores[i], width, label=f'{folder} Success', yerr=success_std[i], capsize=5, color='mediumseagreen')
        r2 = ax.bar(x[i] + width/2, failure_scores[i], width, label=f'{folder} Failure', yerr=failure_std[i], capsize=5, color='lightcoral')
        rects1.append(r1)
        rects2.append(r2)

    if "oracle" in plots_dir:
        ax.set_ylabel('Average Oracle Score')
        ax.set_title('Average Oracle Scores by Outcome and Condition')
    else:
        ax.set_ylabel('Average CLIP Score (non-zero only)')
        ax.set_title('Average CLIP Scores by Outcome and Condition')
    ax.set_xticks(x)
    
    # Adjust x-tick label properties based on the number of folders
    num_folders = len(folders)
    if num_folders > 10:  # Many folders
        rotation_angle = 60
        label_ha = "right"
        label_fontsize = 8
        rotation_mode = "anchor"
    elif num_folders > 5:  # Medium number of folders
        rotation_angle = 45
        label_ha = "right"
        label_fontsize = 9
        rotation_mode = "anchor"
    else:  # Few folders
        rotation_angle = 0
        label_ha = "center" # Center alignment for no rotation
        label_fontsize = 10
        rotation_mode = None

    ax.set_xticklabels(
        folders,
        rotation=rotation_angle,
        ha=label_ha,
        fontsize=label_fontsize,
        rotation_mode=rotation_mode
    )
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Add success rate as text on top of bars
    for i, folder in enumerate(folders):
        # Position text above the higher bar + error bar
        height1 = success_scores[i] + success_std[i] if success_std[i] else success_scores[i]
        height2 = failure_scores[i] + failure_std[i] if failure_std[i] else failure_scores[i]
        max_height = max(height1, height2)
        ax.text(i, max_height + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), # Adjust offset relative to y-axis range
                f"{results[folder]['success_rate']:.1%}",
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "clip_score_analysis.png")) # Use plots_dir
    print(f"Saved score bar chart to {os.path.join(plots_dir, 'clip_score_analysis.png')}")
    plt.close()

def create_time_series_plots(time_series_data, plots_dir): # Added plots_dir parameter
    """
    Create time series plots showing average CLIP score at each timestep
    for successful and failed trajectories, including variance.
    """
    if not time_series_data: return # Skip if no data

    # Create combined plot with subplots
    create_combined_time_series_plot(time_series_data, plots_dir)

    # Create individual plots for each condition type
    create_individual_time_series_plots(time_series_data, plots_dir)

def create_combined_time_series_plot(time_series_data, plots_dir):
    """Create a combined plot with subplots for each condition type."""
    n_folders = len(time_series_data)
    if n_folders == 0: return

    # Adjust layout: 2 columns if more than 1 folder
    ncols = 2 if n_folders > 1 else 1
    nrows = (n_folders + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows), sharex=True, squeeze=False) # Ensure axes is always 2D

    axes_flat = axes.flatten() # Flatten for easy iteration

    for i, (folder, data) in enumerate(time_series_data.items()):
        plot_time_series(axes_flat[i], folder, data, is_subplot=True)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])


    # Set common x-axis label (centered at the bottom)
    fig.supxlabel('Timestep', y=0.02) # Adjust y position if needed
    if "oracle" in plots_dir:
        fig.supylabel('Average Oracle Score / Action Norm', x=0.01) # Add shared Y label
    else:
        fig.supylabel('Average CLIP Score / Action Norm', x=0.01) # Add shared Y label
    fig.suptitle("Time Series Analysis per Condition", fontsize=16, y=0.99) # Add overall title

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97]) # Adjust layout to prevent overlap

    # Save the figure
    save_path = os.path.join(plots_dir, "time_series_analysis_combined.png")
    plt.savefig(save_path)
    print(f"Saved combined time series plot to {save_path}")
    plt.close()

def create_individual_time_series_plots(time_series_data, plots_dir):
    """Create individual plots for each condition type."""
    for folder, data in time_series_data.items():
        # Skip if no data
        if not data["success_time_series"] and not data["failure_time_series"]:
            continue

        # Replace problematic characters for filename
        safe_folder_name = re.sub(r'[\\/*?:"<>|]', '_', folder) # Replace invalid chars

        fig = plt.figure(figsize=(10, 6))
        ax = fig.gca() # Get current axis
        plot_time_series(ax, folder, data, is_subplot=False) # Use the single plotting function

        plt.tight_layout()
        save_path = os.path.join(plots_dir, f"{safe_folder_name}_time_series.png")
        plt.savefig(save_path)
        print(f"Saved individual time series plot for {folder} to {save_path}")
        plt.close()

def plot_time_series(ax, folder, data, is_subplot=True):
    """Plot time series data (scores and action norms) for a single folder on the given axes."""
    success_series = data["success_time_series"] # Original lists, may contain None/NaN/0
    failure_series = data["failure_time_series"]
    success_action_series = data["success_action_series"]
    failure_action_series = data["failure_action_series"]

    # Find the maximum length across all trajectories for this condition
    max_len_success = max([len(series) for series in success_series]) if success_series else 0
    max_len_failure = max([len(series) for series in failure_series]) if failure_series else 0
    max_len = max(max_len_success, max_len_failure)

    if max_len == 0: # No data to plot for this folder
         ax.set_title(f'{folder} (No data)')
         ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
         return

    # --- Accumulate data per timestep, replacing 0 scores with NaN ---
    success_scores_by_timestep = [[] for _ in range(max_len)]
    failure_scores_by_timestep = [[] for _ in range(max_len)]
    success_actions_by_timestep = [[] for _ in range(max_len)] # Actions corresponding to NON-NAN scores
    failure_actions_by_timestep = [[] for _ in range(max_len)] # Actions corresponding to NON-NAN scores

    for series, action_series in zip(success_series, success_action_series):
        for t, (score, action) in enumerate(zip(series, action_series)):
             score_float = float(score) if score is not None else np.nan # Convert to float, handle None
             if score_float == 0: score_float = np.nan # Replace 0 with NaN
             if not np.isnan(score_float): # Only store non-NaN scores and corresponding actions
                success_scores_by_timestep[t].append(score_float)
                success_actions_by_timestep[t].append(action) # Store corresponding action

    for series, action_series in zip(failure_series, failure_action_series):
        for t, (score, action) in enumerate(zip(series, action_series)):
             score_float = float(score) if score is not None else np.nan # Convert to float, handle None
             if score_float == 0: score_float = np.nan # Replace 0 with NaN
             if not np.isnan(score_float): # Only store non-NaN scores and corresponding actions
                 failure_scores_by_timestep[t].append(score_float)
                 failure_actions_by_timestep[t].append(action) # Store corresponding action


    # --- Calculate averages and standard deviations using nanmean/nanstd ---
    avg_success_score = np.full(max_len, np.nan)
    avg_failure_score = np.full(max_len, np.nan)
    std_success_score = np.full(max_len, np.nan)
    std_failure_score = np.full(max_len, np.nan)

    avg_success_action_norm = np.full(max_len, np.nan)
    avg_failure_action_norm = np.full(max_len, np.nan)
    std_success_action_norm = np.full(max_len, np.nan)
    std_failure_action_norm = np.full(max_len, np.nan)

    valid_success_counts = np.zeros(max_len, dtype=int) # Count of non-NaN scores
    valid_failure_counts = np.zeros(max_len, dtype=int)

    for t in range(max_len):
        # Success stats
        scores_t_success = success_scores_by_timestep[t]
        actions_t_success = success_actions_by_timestep[t]
        if scores_t_success: # If list is not empty (already filtered for NaNs)
            valid_success_counts[t] = len(scores_t_success)
            avg_success_score[t] = np.nanmean(scores_t_success) # nanmean handles potential remaining NaNs robustly
            if len(scores_t_success) > 1:
                 std_success_score[t] = np.nanstd(scores_t_success)

            if actions_t_success: # If corresponding actions exist
                 # Calculate norms only for valid actions corresponding to non-NaN scores
                 action_norms_success = np.linalg.norm(np.array(actions_t_success), axis=1)
                 avg_success_action_norm[t] = np.nanmean(action_norms_success)
                 if len(actions_t_success) > 1:
                     std_success_action_norm[t] = np.nanstd(action_norms_success)

        # Failure stats
        scores_t_failure = failure_scores_by_timestep[t]
        actions_t_failure = failure_actions_by_timestep[t]
        if scores_t_failure: # If list is not empty (already filtered for NaNs)
            valid_failure_counts[t] = len(scores_t_failure)
            avg_failure_score[t] = np.nanmean(scores_t_failure)
            if len(scores_t_failure) > 1:
                std_failure_score[t] = np.nanstd(scores_t_failure)

            if actions_t_failure: # If corresponding actions exist
                 action_norms_failure = np.linalg.norm(np.array(actions_t_failure), axis=1)
                 avg_failure_action_norm[t] = np.nanmean(action_norms_failure)
                 if len(actions_t_failure) > 1:
                     std_failure_action_norm[t] = np.nanstd(action_norms_failure)

    # --- Plotting on the provided axes (remains the same) ---
    ax1 = ax # Use the primary axis for scores
    ax2 = ax1.twinx() # Create twin axis for action norms

    timesteps = np.arange(1, max_len + 1)

    if np.any(~np.isnan(avg_success_score)):
        ax1.plot(timesteps, avg_success_score, 'g-', label=f'Success (n={len(success_series)})', linewidth=2)
        ax1.fill_between(timesteps, avg_success_score - std_success_score, avg_success_score + std_success_score,
                            color='g', alpha=0.2, where=~np.isnan(avg_success_score)) # Only shade where data exists

    if np.any(~np.isnan(avg_failure_score)):
        ax1.plot(timesteps, avg_failure_score, 'r-', label=f'Failure (n={len(failure_series)})', linewidth=2)
        ax1.fill_between(timesteps, avg_failure_score - std_failure_score, avg_failure_score + std_failure_score,
                            color='r', alpha=0.2, where=~np.isnan(avg_failure_score))

    # Plot Action Norms
    if np.any(~np.isnan(avg_success_action_norm)):
        ax2.plot(timesteps, avg_success_action_norm, 'b--', label=f'Success Action Norm', linewidth=1.5)
        ax2.fill_between(timesteps, avg_success_action_norm - std_success_action_norm, avg_success_action_norm + std_success_action_norm,
                            color='b', alpha=0.1, where=~np.isnan(avg_success_action_norm))

    if np.any(~np.isnan(avg_failure_action_norm)):
        ax2.plot(timesteps, avg_failure_action_norm, 'm--', label=f'Failure Action Norm', linewidth=1.5)
        ax2.fill_between(timesteps, avg_failure_action_norm - std_failure_action_norm, avg_failure_action_norm + std_failure_action_norm,
                            color='m', alpha=0.1, where=~np.isnan(avg_failure_action_norm))

    # --- Formatting (remains the same) ---

    ax1.set_ylabel('Avg Verifier Score')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')

    ax2.set_ylabel('Avg Action Norm')
    ax2.tick_params(axis='y', labelcolor='black')

    if not is_subplot:
        ax1.yaxis.set_label_coords(-0.08, 0.5)
        ax2.yaxis.set_label_coords(1.08, 0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend_loc = 'upper left' if not is_subplot else 'best'
    bbox = (1.05, 1) if not is_subplot else None
    # Use valid counts (non-NaN scores) for the legend title
    ax1.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc, bbox_to_anchor=bbox, fontsize='small',
               title=f'Max Valid Samples/Step:\nSuccess: {np.max(valid_success_counts) if valid_success_counts.size > 0 else 0}, Failure: {np.max(valid_failure_counts) if valid_failure_counts.size > 0 else 0}')


    success_rate = len(success_series) / (len(success_series) + len(failure_series)) if (len(success_series) + len(failure_series)) > 0 else 0
    ax1.set_title(f'{folder} (Success Rate: {success_rate:.1%})', fontsize=10)

    if not is_subplot:
        ax1.set_xlabel('Timestep')

def create_score_length_plots(results, plots_dir): # Added plots_dir parameter
    """Create scatter plots showing the relationship between average CLIP scores and episode lengths."""
    n_folders = len(results)
    if n_folders == 0: return

    # Adjust layout: 2 columns if more than 1 folder
    ncols = 2 if n_folders > 1 else 1
    nrows = (n_folders + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False) # Ensure axes is always 2D
    axes_flat = axes.flatten() # Flatten for easy iteration

    for i, (folder, stats) in enumerate(results.items()):
        ax = axes_flat[i]

        # Plot success and failure points
        success_has_data = stats.get("success_scores") and stats.get("success_lengths")
        failure_has_data = stats.get("failure_scores") and stats.get("failure_lengths")

        if success_has_data:
            ax.scatter(stats["success_scores"], stats["success_lengths"],
                        c='g', label='Success', alpha=0.6, s=30) # Adjust size 's'
            # Add trend line only if more than 1 point
            if len(stats["success_scores"]) > 1:
                z = np.polyfit(stats["success_scores"], stats["success_lengths"], 1)
                p = np.poly1d(z)
                x_min, x_max = min(stats["success_scores"]), max(stats["success_scores"])
                x_range = np.linspace(x_min, x_max, 2) # Just 2 points needed for line
                ax.plot(x_range, p(x_range), "g--", alpha=0.8)

        if failure_has_data:
            ax.scatter(stats["failure_scores"], stats["failure_lengths"],
                        c='r', label='Failure', alpha=0.6, s=30)
            # Add trend line only if more than 1 point
            if len(stats["failure_scores"]) > 1:
                z = np.polyfit(stats["failure_scores"], stats["failure_lengths"], 1)
                p = np.poly1d(z)
                x_min, x_max = min(stats["failure_scores"]), max(stats["failure_scores"])
                x_range = np.linspace(x_min, x_max, 2)
                ax.plot(x_range, p(x_range), "r--", alpha=0.8)

        # Calculate correlations if data exists and has variance
        success_corr = np.nan
        if success_has_data and len(stats["success_scores"]) > 1 and np.std(stats["success_scores"]) > 1e-6 and np.std(stats["success_lengths"]) > 1e-6:
             success_corr = np.corrcoef(stats["success_scores"], stats["success_lengths"])[0,1]

        failure_corr = np.nan
        if failure_has_data and len(stats["failure_scores"]) > 1 and np.std(stats["failure_scores"]) > 1e-6 and np.std(stats["failure_lengths"]) > 1e-6:
             failure_corr = np.corrcoef(stats["failure_scores"], stats["failure_lengths"])[0,1]

        # Add labels and title
        if "oracle" in plots_dir:
            ax.set_xlabel('Average Oracle Score')
        else:
            ax.set_xlabel('Average CLIP Score (non-zero only)')
        ax.set_ylabel('Episode Length')
        folder_label = folder # Use folder name directly
        ax.set_title(f'{folder_label}\nCorrelations - Success: {success_corr:.2f}, Failure: {failure_corr:.2f}')
        ax.grid(True, linestyle='--', alpha=0.4)
        if success_has_data or failure_has_data: # Only show legend if there's data
             ax.legend(loc='best')

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.suptitle("Score vs Episode Length Correlation", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout
    save_path = os.path.join(plots_dir, "score_length_correlation.png")
    plt.savefig(save_path)
    print(f"Saved score-length correlation plot to {save_path}")
    plt.close()

def create_global_score_length_plot(results, plots_dir):
    """
    Creates a single scatter plot showing score vs episode length
    across all conditions.
    """
    if not results:
        print("No results available to create global score-length plot.")
        return

    all_success_scores_raw = []
    all_success_lengths_raw = []
    all_failure_scores_raw = []
    all_failure_lengths_raw = []

    # Aggregate data from all conditions (may include NaNs)
    for folder, stats in results.items():
        all_success_scores_raw.extend(stats.get("success_scores", []))
        all_success_lengths_raw.extend(stats.get("success_lengths", []))
        all_failure_scores_raw.extend(stats.get("failure_scores", []))
        all_failure_lengths_raw.extend(stats.get("failure_lengths", []))

    # Convert to numpy arrays and filter NaNs for plotting and analysis
    all_success_scores_np = np.array(all_success_scores_raw, dtype=float)
    all_success_lengths_np = np.array(all_success_lengths_raw, dtype=float)
    valid_success_mask = ~np.isnan(all_success_scores_np)
    all_success_scores_valid = all_success_scores_np[valid_success_mask]
    all_success_lengths_valid = all_success_lengths_np[valid_success_mask] # Lengths corresponding to valid scores

    all_failure_scores_np = np.array(all_failure_scores_raw, dtype=float)
    all_failure_lengths_np = np.array(all_failure_lengths_raw, dtype=float)
    valid_failure_mask = ~np.isnan(all_failure_scores_np)
    all_failure_scores_valid = all_failure_scores_np[valid_failure_mask]
    all_failure_lengths_valid = all_failure_lengths_np[valid_failure_mask] # Lengths corresponding to valid scores


    if len(all_success_scores_valid) == 0 and len(all_failure_scores_valid) == 0:
        print("No valid score/length data found across all conditions.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot success points (use valid data)
    if len(all_success_scores_valid) > 0:
        ax.scatter(all_success_scores_valid, all_success_lengths_valid,
                   c='g', label='Success (All Conditions)', alpha=0.5, s=25)
        # Add trend line if enough points and scores vary
        if len(all_success_scores_valid) > 1 and np.std(all_success_scores_valid) > 1e-9:
             try:
                 z = np.polyfit(all_success_scores_valid, all_success_lengths_valid, 1)
                 p = np.poly1d(z)
                 x_min, x_max = min(all_success_scores_valid), max(all_success_scores_valid)
                 x_range = np.linspace(x_min, x_max, 2)
                 ax.plot(x_range, p(x_range), "g--", alpha=0.9, linewidth=2)
             except np.linalg.LinAlgError:
                 print("Warning: polyfit failed for global success data. Skipping trendline.")
        elif len(all_success_scores_valid) > 1:
            print("Warning: Skipping global success trendline due to constant score values.")

    # Plot failure points (use valid data)
    if len(all_failure_scores_valid) > 0:
        ax.scatter(all_failure_scores_valid, all_failure_lengths_valid,
                   c='r', label='Failure (All Conditions)', alpha=0.5, s=25)
        # Add trend line if enough points and scores vary
        if len(all_failure_scores_valid) > 1 and np.std(all_failure_scores_valid) > 1e-9:
            try:
                z = np.polyfit(all_failure_scores_valid, all_failure_lengths_valid, 1)
                p = np.poly1d(z)
                x_min, x_max = min(all_failure_scores_valid), max(all_failure_scores_valid)
                x_range = np.linspace(x_min, x_max, 2)
                ax.plot(x_range, p(x_range), "r--", alpha=0.9, linewidth=2)
            except np.linalg.LinAlgError:
                 print("Warning: polyfit failed for global failure data. Skipping trendline.")
        elif len(all_failure_scores_valid) > 1:
            print("Warning: Skipping global failure trendline due to constant score values.")

    # Calculate overall correlations using valid data
    success_corr_global = np.nan
    if len(all_success_scores_valid) > 1 and np.std(all_success_scores_valid) > 1e-6 and np.std(all_success_lengths_valid) > 1e-6:
         try:
             success_corr_global = np.corrcoef(all_success_scores_valid, all_success_lengths_valid)[0,1]
         except Exception as e:
             print(f"Warning: Global success correlation calculation failed: {e}")

    failure_corr_global = np.nan
    if len(all_failure_scores_valid) > 1 and np.std(all_failure_scores_valid) > 1e-6 and np.std(all_failure_lengths_valid) > 1e-6:
         try:
             failure_corr_global = np.corrcoef(all_failure_scores_valid, all_failure_lengths_valid)[0,1]
         except Exception as e:
             print(f"Warning: Global failure correlation calculation failed: {e}")


    # Add labels, title, legend, grid
    if "oracle" in plots_dir:
        ax.set_xlabel('Average Oracle Score')
    else:
        ax.set_xlabel('Average CLIP Score (non-zero only)')
    ax.set_ylabel('Episode Length')
    ax.set_title(f'Global Score vs Episode Length (All Conditions)\nOverall Correlations - Success: {success_corr_global:.2f}, Failure: {failure_corr_global:.2f}')
    ax.grid(True, linestyle='--', alpha=0.5)
    if len(all_success_scores_valid) > 0 or len(all_failure_scores_valid) > 0:
        ax.legend(loc='best')

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "global_score_length_scatter.png")
    plt.savefig(save_path)
    print(f"Saved global score-length scatter plot to {save_path}")
    plt.close()

def create_rate_vs_score_plot(results, plots_dir):
    """
    Creates a scatter plot showing Success Rate vs Overall Average Score
    for each condition.
    """
    if not results:
        print("No results available to create rate vs score plot.")
        return

    conditions = []
    success_rates = []
    overall_avg_scores = []

    # Calculate overall average score and collect success rates for each condition
    for folder, stats in results.items():
        success_count = stats['success_count']
        failure_count = stats['failure_count']
        total_count = success_count + failure_count

        if total_count == 0:
            continue # Skip conditions with no data

        # Use the pre-calculated overall averages (which might be NaN)
        avg_success = stats['avg_success_score']
        avg_failure = stats['avg_failure_score']

        # Calculate weighted average, treating NaN components as having 0 contribution if the count is > 0
        # If both success_count and failure_count are 0, total_count is 0, skipped above.
        # If success_count > 0 but avg_success is NaN, treat its contribution as 0.
        # If failure_count > 0 but avg_failure is NaN, treat its contribution as 0.
        weighted_sum = 0
        if success_count > 0 and not np.isnan(avg_success):
            weighted_sum += avg_success * success_count
        if failure_count > 0 and not np.isnan(avg_failure):
            weighted_sum += avg_failure * failure_count

        if total_count > 0 : # Should always be true here
             overall_avg = weighted_sum / total_count
        else: # Should not happen, but as fallback
             overall_avg = np.nan # Or 0? Let's use NaN to indicate no valid average

        # Only plot points where we could calculate a valid overall average
        if not np.isnan(overall_avg):
            conditions.append(folder)
            success_rates.append(stats['success_rate'])
            overall_avg_scores.append(overall_avg)
        else:
            print(f"Warning: Skipping condition '{folder}' in rate_vs_score plot due to NaN overall average score.")


    if not success_rates:
        print("No data points to plot for rate vs score after filtering.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Create the scatter plot
    scatter = ax.scatter(overall_avg_scores, success_rates, c=np.arange(len(conditions)), cmap='tab10', s=60, alpha=0.8)

    # Add labels and title
    if "oracle" in plots_dir:
        ax.set_xlabel('Overall Average Oracle Score (per condition, NaN avg excluded)')
    else:
        ax.set_xlabel('Overall Average CLIP Score (per condition, NaN avg excluded)')
    ax.set_ylabel('Success Rate (per condition)')
    ax.set_title('Success Rate vs. Overall Average Score by Condition')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Improve y-axis formatting (percentage)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_ylim(bottom=0, top=max(1.05, max(success_rates)*1.1) if success_rates else 1.05) # Ensure y-axis starts at 0 and goes slightly above max rate

    # Add a legend
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.8)
    ax.legend(handles, conditions, title="Conditions", loc='best', fontsize='small')


    plt.tight_layout()
    save_path = os.path.join(plots_dir, "rate_vs_score_scatter.png")
    plt.savefig(save_path)
    print(f"Saved rate vs score scatter plot to {save_path}")
    plt.close()

def create_task_success_rate_plots(rollout_dir):
    """
    For each unique task (language instruction), plot the success rate across all rephrase types (folders).
    X-axis: rephrase type (sorted numerically)
    Y-axis: success rate for that task
    Title and filename: use the task description
    Save plots in rollout_dir/plots/task_success_rate/
    """
    import matplotlib.pyplot as plt
    import os
    import re
    from glob import glob
    from collections import defaultdict

    # Find all transformation folders (exclude plots)
    transformation_folders = [d for d in os.listdir(rollout_dir)
                             if os.path.isdir(os.path.join(rollout_dir, d)) and d != "plots"]
    if not transformation_folders:
        print(f"No transformation folders found in {rollout_dir}")
        return

    # Aggregate: task -> rephrase -> [success/failure]
    task_rephrase_results = defaultdict(lambda: defaultdict(list))

    for trans in transformation_folders:
        if trans == "no_transform_1":
            rephrase_num = 0
        else:
            rephrase_match = re.search(r'rephrase_(\d+)', trans)
            if not rephrase_match:
                continue
            rephrase_num = int(rephrase_match.group(1))
        folder_path = os.path.join(rollout_dir, trans)
        pkl_files = glob(os.path.join(folder_path, "*.pkl"))
        for pkl_file in pkl_files:
            filename = os.path.basename(pkl_file)
            # Extract task
            task_match = re.search(r'task=([^.]*)', filename)
            if not task_match:
                continue
            task = task_match.group(1)
            # Extract success
            success_match = re.search(r'success=(True|False)', filename)
            if not success_match:
                continue
            is_success = success_match.group(1) == 'True'
            task_rephrase_results[task][rephrase_num].append(is_success)

    # Create output dir
    task_plot_dir = os.path.join(rollout_dir, "plots", "task_success_rate")
    os.makedirs(task_plot_dir, exist_ok=True)

    # For each task, plot success rate vs rephrase type
    for task, rephrase_dict in task_rephrase_results.items():
        rephrase_nums = sorted(rephrase_dict.keys())
        success_rates = []
        counts = []
        for r in rephrase_nums:
            results = rephrase_dict[r]
            if results:
                rate = sum(results) / len(results)
                success_rates.append(rate)
                counts.append(len(results))
            else:
                success_rates.append(0)
                counts.append(0)
        plt.figure(figsize=(max(8, len(rephrase_nums) * 1.2), 6))
        # Plot all as green line first (excluding no_transform_1 if present)
        plot_rephrase_nums = [r for r in rephrase_nums if r != 0]
        plot_success_rates = [rate for r, rate in zip(rephrase_nums, success_rates) if r != 0]
        plt.plot(plot_rephrase_nums, plot_success_rates, marker='o', color='mediumseagreen', label='Rephrases')

        # Plot no_transform_1 (oracle) in red if present
        if 0 in rephrase_nums:
            idx = rephrase_nums.index(0)
            # plt.plot([0], [success_rates[idx]], marker='o', color='red', markersize=10, label='oracle (no_transform)')
            # Draw a horizontal dashed line at the oracle accuracy
            plt.axhline(y=success_rates[idx], color='red', linestyle='--', alpha=0.7, label='oracle (no_transform)')

        plt.xticks(sorted(rephrase_nums))
        plt.xlabel('Number of Samples (Rephrases)')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1.05)
        plt.title(task.replace('_', ' '))
        # Annotate with counts
        for x, y, n in zip(rephrase_nums, success_rates, counts):
            plt.text(x, y + 0.03, f'n={n}', ha='center', fontsize=8)
        plt.legend()
        plt.tight_layout()
        # Sanitize filename
        safe_task = re.sub(r'[^a-zA-Z0-9_\-]', '_', task)[:80]
        out_path = os.path.join(task_plot_dir, f'{safe_task}_success_rate.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved success rate plot for task '{task}' to {out_path}")

def create_task_success_rate_plots_combined(rollouts_oracle_dir, rollouts_dir):
    """
    For each unique task, plot the success rate across all rephrase types (folders) for both oracle and designed verifiers.
    - Red horizontal line: oracle policy (no_transform_1, from rollouts_oracle)
    - Green line: oracle verifier (rephrases, from rollouts_oracle)
    - Blue line: designed verifier (rephrases, from rollouts)
    Save plots in ./plots/task_success_rate_combined/
    """
    import matplotlib.pyplot as plt
    import os
    import re
    from glob import glob
    from collections import defaultdict

    def aggregate_task_rephrase_results(rollout_dir):
        # Returns: task -> rephrase_num -> [success/failure]
        transformation_folders = [d for d in os.listdir(rollout_dir)
                                 if os.path.isdir(os.path.join(rollout_dir, d)) and d != "plots"]
        task_rephrase_results = defaultdict(lambda: defaultdict(list))
        for trans in transformation_folders:
            if trans == "no_transform_1":
                rephrase_num = 0
            else:
                rephrase_match = re.search(r'rephrase_(\d+)', trans)
                if not rephrase_match:
                    continue
                rephrase_num = int(rephrase_match.group(1))
            folder_path = os.path.join(rollout_dir, trans)
            pkl_files = glob(os.path.join(folder_path, "*.pkl"))
            for pkl_file in pkl_files:
                filename = os.path.basename(pkl_file)
                task_match = re.search(r'task=([^.]*)', filename)
                if not task_match:
                    continue
                task = task_match.group(1)
                success_match = re.search(r'success=(True|False)', filename)
                if not success_match:
                    continue
                is_success = success_match.group(1) == 'True'
                task_rephrase_results[task][rephrase_num].append(is_success)
        return task_rephrase_results

    # Aggregate results
    oracle_results = aggregate_task_rephrase_results(rollouts_oracle_dir)
    designed_results = aggregate_task_rephrase_results(rollouts_dir)

    # Output dir
    combined_plot_dir = os.path.join("./plots", "task_success_rate_combined")
    os.makedirs(combined_plot_dir, exist_ok=True)

    # Union of all tasks
    all_tasks = set(oracle_results.keys()) | set(designed_results.keys())

    for task in all_tasks:
        # Oracle policy (no_transform_1, rephrase_num=0)
        oracle_policy_rate = None
        if 0 in oracle_results.get(task, {}):
            results = oracle_results[task][0]
            if results:
                oracle_policy_rate = sum(results) / len(results)
        # Oracle verifier (rephrases, from rollouts_oracle)
        oracle_rephrase_nums = sorted([k for k in oracle_results.get(task, {}).keys() if k != 0])
        oracle_rephrase_rates = [sum(oracle_results[task][k])/len(oracle_results[task][k]) if oracle_results[task][k] else 0 for k in oracle_rephrase_nums]
        oracle_counts = [len(oracle_results[task][k]) for k in oracle_rephrase_nums]
        # Designed verifier (rephrases, from rollouts)
        designed_rephrase_nums = sorted(designed_results.get(task, {}).keys())
        designed_rephrase_rates = [sum(designed_results[task][k])/len(designed_results[task][k]) if designed_results[task][k] else 0 for k in designed_rephrase_nums]
        designed_counts = [len(designed_results[task][k]) for k in designed_rephrase_nums]

        plt.figure(figsize=(max(8, max(len(oracle_rephrase_nums), len(designed_rephrase_nums)) * 1.2), 6))
        # Oracle verifier (green)
        if oracle_rephrase_nums:
            plt.plot(oracle_rephrase_nums, oracle_rephrase_rates, marker='o', color='mediumseagreen', label='Oracle Verifier (rephrases)')
        # Designed verifier (blue)
        if designed_rephrase_nums:
            plt.plot(designed_rephrase_nums, designed_rephrase_rates, marker='o', color='royalblue', label='Designed Verifier (rephrases)')
        # Oracle policy (red horizontal line)
        if oracle_policy_rate is not None:
            plt.axhline(y=oracle_policy_rate, color='red', linestyle='--', alpha=0.7, label='Oracle Policy (no_transform)')
        # X ticks
        all_rephrase_nums = sorted(set(oracle_rephrase_nums) | set(designed_rephrase_nums))
        plt.xticks(all_rephrase_nums)
        plt.xlabel('Number of Samples (Rephrases)')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1.05)
        plt.title(task.replace('_', ' '))
        # Annotate with counts (oracle counts above, designed below)
        for x, y, n in zip(oracle_rephrase_nums, oracle_rephrase_rates, oracle_counts):
            plt.text(x, y + 0.03, f'n={n}', ha='center', fontsize=8, color='mediumseagreen')
        for x, y, n in zip(designed_rephrase_nums, designed_rephrase_rates, designed_counts):
            plt.text(x, y - 0.06, f'n={n}', ha='center', fontsize=8, color='royalblue')
        plt.legend()
        plt.tight_layout()
        # Sanitize filename
        safe_task = re.sub(r'[^a-zA-Z0-9_\-]', '_', task)[:80]
        out_path = os.path.join(combined_plot_dir, f'{safe_task}_success_rate_combined.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved combined success rate plot for task '{task}' to {out_path}")

if __name__ == "__main__":

    path_to_rollouts_oracle = "./rollouts_oracle"
    path_to_rollouts_clip = "./rollouts_hack"
    
    # results_oracle, time_series_data_oracle = analyze_rollouts(path_to_rollouts_oracle)
    results_clip, time_series_data_clip = analyze_rollouts(path_to_rollouts_clip)

    # Call the new combined plot function
    # create_task_success_rate_plots_combined(path_to_rollouts_oracle, path_to_rollouts_clip)
