#!/usr/bin/env python3
"""
Script to analyze success rates grouped by language number.
Takes a folder as input and creates plots for each language number (lang_1, lang_2, etc.).
For lang_1: creates one plot with all lang_1_sample_* folders
For other language numbers: combines them into one plot
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
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

def extract_language_number(folder_name):
    """Extract language number from folder name (e.g., 'lang_1_sample_2' -> 1)."""
    match = re.search(r'lang[_\s]*(\d+)', folder_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def analyze_folder_by_language(folder_path):
    """
    Analyze a folder and group data by language number.
    
    Args:
        folder_path: Path to the folder containing lang_X_sample_Y subfolders
    
    Returns:
        Dictionary: {lang_number: {sample_folder: evaluations}}
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return {}
    
    lang_data = defaultdict(dict)
    
    # Walk through subdirectories
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if not os.path.isdir(item_path):
            continue
        
        # Extract language number
        lang_num = extract_language_number(item)
        if lang_num is None:
            # Skip folders that don't match lang_X pattern (e.g., 'robomonkey')
            continue
        
        # Collect episode data from this sample folder
        episode_data = []
        for root, dirs, files in os.walk(item_path):
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
            lang_data[lang_num][item] = evaluations
            print(f"  Found {len(episode_data)} episodes in {item} (lang_{lang_num})")
    
    return dict(lang_data)

def create_plot_for_language(lang_data, lang_number, output_dir, folder_name):
    """
    Create a plot for a specific language number showing all samples.
    
    Args:
        lang_data: Dictionary {sample_folder: evaluations}
        lang_number: Language number (e.g., 1, 2, etc.)
        output_dir: Output directory for plots
        folder_name: Name of the parent folder (for plot title)
    """
    if not lang_data:
        print(f"  No data for lang_{lang_number}, skipping plot")
        return
    
    # Collect data for plotting
    plot_data = []
    
    for sample_folder, evaluations in lang_data.items():
        # Get the 4 main evaluation periods (Eval_1, Eval_2, Eval_3, Eval_4)
        eval_rates = []
        for eval_name in ['Eval_1', 'Eval_2', 'Eval_3', 'Eval_4']:
            if eval_name in evaluations and isinstance(evaluations[eval_name], dict) and 'success_rate' in evaluations[eval_name]:
                eval_rates.append(evaluations[eval_name]['success_rate'])
        
        if len(eval_rates) >= 1:  # Need at least 1 evaluation
            plot_data.append({
                'Sample': sample_folder.replace('_', ' ').title(),
                'Mean': np.mean(eval_rates),
                'Std': np.std(eval_rates) if len(eval_rates) > 1 else 0.0,
                'Evaluations': len(eval_rates)
            })
    
    if not plot_data:
        print(f"  No evaluation data for lang_{lang_number}, skipping plot")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Sort samples by name for consistent ordering
    plot_data_sorted = sorted(plot_data, key=lambda x: x['Sample'])
    
    samples = [item['Sample'] for item in plot_data_sorted]
    means = [item['Mean'] for item in plot_data_sorted]
    stds = [item['Std'] for item in plot_data_sorted]
    
    x_pos = np.arange(len(samples))
    width = 0.7
    
    bars = plt.bar(x_pos, means, width, yerr=stds, capsize=5, alpha=0.8,
                   color='steelblue', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, means, stds):
        height = bar.get_height()
        label_text = f'{mean_val:.3f}'
        if std_val > 0:
            label_text += f'±{std_val:.3f}'
        plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                label_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize plot
    plt.xlabel('Sample Folder', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate (Mean ± Std across evaluations)', fontsize=14, fontweight='bold')
    plt.title(f'Language {lang_number} - Success Rates Across All Samples\n({folder_name})', 
             fontsize=16, fontweight='bold')
    plt.xticks(x_pos, samples, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    safe_folder_name = folder_name.replace('/', '_').replace(' ', '_')
    safe_lang_num = f"lang_{lang_number}"
    out_path = os.path.join(output_dir, f'success_rates_{safe_folder_name}_{safe_lang_num}.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {out_path}")

def create_combined_plot_for_other_languages(all_other_lang_data, output_dir, folder_name):
    """
    Create a combined plot for all language numbers except lang_1.
    
    Args:
        all_other_lang_data: Dictionary {lang_number: {sample_folder: evaluations}}
        output_dir: Output directory for plots
        folder_name: Name of the parent folder (for plot title)
    """
    if not all_other_lang_data:
        print("  No data for other languages, skipping combined plot")
        return
    
    # Collect data for plotting
    plot_data = []
    
    for lang_number, lang_data in sorted(all_other_lang_data.items()):
        for sample_folder, evaluations in lang_data.items():
            # Get the 4 main evaluation periods
            eval_rates = []
            for eval_name in ['Eval_1', 'Eval_2', 'Eval_3', 'Eval_4']:
                if eval_name in evaluations and isinstance(evaluations[eval_name], dict) and 'success_rate' in evaluations[eval_name]:
                    eval_rates.append(evaluations[eval_name]['success_rate'])
            
            if len(eval_rates) >= 1:
                plot_data.append({
                    'Label': f'Lang_{lang_number}_{sample_folder.replace("lang_", "").replace("_", " ").title()}',
                    'Mean': np.mean(eval_rates),
                    'Std': np.std(eval_rates) if len(eval_rates) > 1 else 0.0,
                    'Evaluations': len(eval_rates)
                })
    
    if not plot_data:
        print("  No evaluation data for other languages, skipping combined plot")
        return
    
    # Create the plot
    plt.figure(figsize=(16, 8))
    
    # Sort by label for consistent ordering
    plot_data_sorted = sorted(plot_data, key=lambda x: x['Label'])
    
    labels = [item['Label'] for item in plot_data_sorted]
    means = [item['Mean'] for item in plot_data_sorted]
    stds = [item['Std'] for item in plot_data_sorted]
    
    x_pos = np.arange(len(labels))
    width = 0.7
    
    bars = plt.bar(x_pos, means, width, yerr=stds, capsize=5, alpha=0.8,
                   color='coral', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, means, stds):
        height = bar.get_height()
        label_text = f'{mean_val:.3f}'
        if std_val > 0:
            label_text += f'±{std_val:.3f}'
        plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                label_text, ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Customize plot
    plt.xlabel('Language Sample', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate (Mean ± Std across evaluations)', fontsize=14, fontweight='bold')
    plt.title(f'Other Languages (Lang 2+) - Success Rates\n({folder_name})', 
             fontsize=16, fontweight='bold')
    plt.xticks(x_pos, labels, rotation=45, ha='right', fontsize=9)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    safe_folder_name = folder_name.replace('/', '_').replace(' ', '_')
    out_path = os.path.join(output_dir, f'success_rates_{safe_folder_name}_other_languages.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved combined plot: {out_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze success rates grouped by language number')
    parser.add_argument('folder_path', type=str, 
                       help='Path to folder containing lang_X_sample_Y subfolders')
    parser.add_argument('--output-dir', type=str, default='./analysis_plots',
                       help='Output directory for plots (default: ./analysis_plots)')
    args = parser.parse_args()
    
    folder_path = args.folder_path
    output_dir = args.output_dir
    
    print("="*80)
    print(f"Analyzing folder: {folder_path}")
    print("="*80)
    
    # Analyze folder and group by language number
    lang_data_all = analyze_folder_by_language(folder_path)
    
    if not lang_data_all:
        print("No language data found!")
        return
    
    print(f"\nFound data for {len(lang_data_all)} language number(s): {sorted(lang_data_all.keys())}")
    
    # Separate lang_1 from others
    lang_1_data = lang_data_all.get(1, {})
    other_lang_data = {k: v for k, v in lang_data_all.items() if k != 1}
    
    folder_name = os.path.basename(folder_path.rstrip('/'))
    
    # Create plot for lang_1
    if lang_1_data:
        print(f"\nCreating plot for lang_1 ({len(lang_1_data)} samples)...")
        create_plot_for_language(lang_1_data, 1, output_dir, folder_name)
    
    # Create combined plot for other languages
    if other_lang_data:
        print(f"\nCreating combined plot for other languages ({len(other_lang_data)} language numbers)...")
        create_combined_plot_for_other_languages(other_lang_data, output_dir, folder_name)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for lang_number in sorted(lang_data_all.keys()):
        lang_data = lang_data_all[lang_number]
        print(f"\nLang_{lang_number}:")
        for sample_folder, evaluations in sorted(lang_data.items()):
            eval_rates = []
            for eval_name in ['Eval_1', 'Eval_2', 'Eval_3', 'Eval_4']:
                if eval_name in evaluations and isinstance(evaluations[eval_name], dict) and 'success_rate' in evaluations[eval_name]:
                    eval_rates.append(evaluations[eval_name]['success_rate'])
            
            if eval_rates:
                mean_rate = np.mean(eval_rates)
                std_rate = np.std(eval_rates) if len(eval_rates) > 1 else 0.0
                print(f"  {sample_folder}: {mean_rate:.3f} ± {std_rate:.3f} (n={len(eval_rates)} evals)")
    
    print(f"\nAnalysis complete! Plots saved to '{output_dir}/'")

if __name__ == "__main__":
    main()

