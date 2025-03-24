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
    """
    # Get the most recent date folder if not specified
    if not os.path.exists(rollout_dir):
        print(f"Directory {rollout_dir} does not exist.")
        return
    
    # Find all date folders
    date_folders = [d for d in os.listdir(rollout_dir) if os.path.isdir(os.path.join(rollout_dir, d))]
    if not date_folders:
        print(f"No date folders found in {rollout_dir}")
        return
    
    # Sort by date (assuming YYYY_MM_DD format)
    date_folders.sort(key=lambda x: datetime.strptime(x, "%Y_%m_%d") if len(x.split("_")) == 3 else datetime.min)
    latest_date = date_folders[-1]
    
    # Check for original and transformed language folders
    date_path = os.path.join(rollout_dir, latest_date)
    analysis_folders = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
    
    results = {}
    
    for folder in analysis_folders:
        folder_path = os.path.join(date_path, folder)
        print(f"Analyzing folder: {folder_path}")
        
        # Find all pickle files
        pkl_files = glob(os.path.join(folder_path, "*.pkl"))
        if not pkl_files:
            print(f"No pickle files found in {folder_path}")
            continue
        
        success_scores = []
        failure_scores = []
        
        for pkl_file in pkl_files:
            # Extract success info from filename
            filename = os.path.basename(pkl_file)
            is_success = "success=True" in filename
            
            # Load the pickle file
            with open(pkl_file, "rb") as f:
                score_list = pickle.load(f)
            
            # Calculate average score for this trajectory
            if score_list:
                avg_score = np.mean(score_list)
                if is_success:
                    success_scores.append(avg_score)
                else:
                    failure_scores.append(avg_score)
        
        # Calculate statistics
        results[folder] = {
            "success_count": len(success_scores),
            "failure_count": len(failure_scores),
            "success_rate": len(success_scores) / (len(success_scores) + len(failure_scores)) if (len(success_scores) + len(failure_scores)) > 0 else 0,
            "avg_success_score": np.mean(success_scores) if success_scores else 0,
            "avg_failure_score": np.mean(failure_scores) if failure_scores else 0,
            "std_success_score": np.std(success_scores) if success_scores else 0,
            "std_failure_score": np.std(failure_scores) if failure_scores else 0,
        }
    
    # Print results
    print("\nResults Summary:")
    for folder, stats in results.items():
        print(f"\n{folder}:")
        print(f"  Success rate: {stats['success_rate']:.2%} ({stats['success_count']}/{stats['success_count'] + stats['failure_count']})")
        print(f"  Avg success score: {stats['avg_success_score']:.4f} ± {stats['std_success_score']:.4f}")
        print(f"  Avg failure score: {stats['avg_failure_score']:.4f} ± {stats['std_failure_score']:.4f}")
    
    # Create visualization
    if results:
        create_visualization(results)
    
    return results

def create_visualization(results):
    """Create visualizations for the results"""
    plt.figure(figsize=(12, 6))
    
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

if __name__ == "__main__":
    analyze_rollouts()