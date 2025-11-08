#!/usr/bin/env python3
"""
Script to analyze success rates by task description (rephrase performance).
For each task (based on original_task_description), shows which task descriptions perform best.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import glob

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_task_description_performance(base_path):
    """
    Analyze success rates by task description from pickle files.
    
    Args:
        base_path: Path to the rollouts directory (e.g., test_rollouts_openpi_rephrase_selection)
    
    Returns:
        Dictionary: task_name -> {task_description -> {'success': count, 'total': count, 'rate': float}}
    """
    # Find all pickle files
    pattern = os.path.join(base_path, "**", "*.pkl")
    pickle_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Structure: task_name (original) -> task_description (used) -> {'success': [], 'total': []}
    task_data = defaultdict(lambda: defaultdict(lambda: {'success': 0, 'total': 0}))
    
    for pickle_file in pickle_files:
        try:
            with open(pickle_file, 'rb') as f:
                episode_data = pickle.load(f)
            
            # Extract original task description
            original_task = episode_data.get('original_task_description')
            if not original_task:
                continue
            
            # Extract used task description (the rephrase that was actually used)
            used_task_desc = episode_data.get('used_task_description')
            if not used_task_desc:
                # Fallback to selected_instructions if available
                selected_instructions = episode_data.get('selected_instructions', [])
                if selected_instructions:
                    used_task_desc = selected_instructions[-1]  # Use last selected instruction
                else:
                    continue
            
            # Extract success status
            success = episode_data.get('success', False)
            
            # Count this episode
            task_data[original_task][used_task_desc]['total'] += 1
            if success:
                task_data[original_task][used_task_desc]['success'] += 1
                
        except Exception as e:
            print(f"Error loading {pickle_file}: {e}")
            continue
    
    # Calculate success rates
    results = defaultdict(dict)
    for task_name, descriptions in task_data.items():
        for desc, counts in descriptions.items():
            if counts['total'] > 0:
                rate = counts['success'] / counts['total']
                results[task_name][desc] = {
                    'success_rate': rate,
                    'success_count': counts['success'],
                    'total_count': counts['total']
                }
    
    return results

def plot_task_description_performance(results, output_dir='./analysis_plots/task_description_plots'):
    """
    Create plots showing success rates for different task descriptions per task.
    
    Args:
        results: Dictionary from analyze_task_description_performance
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    for task_name, descriptions in results.items():
        if not descriptions:
            continue
        
        # Prepare data for plotting
        descs = list(descriptions.keys())
        rates = [descriptions[d]['success_rate'] for d in descs]
        success_counts = [descriptions[d]['success_count'] for d in descs]
        total_counts = [descriptions[d]['total_count'] for d in descs]
        
        # Sort by success rate (descending)
        sorted_data = sorted(zip(descs, rates, success_counts, total_counts), 
                           key=lambda x: x[1], reverse=True)
        descs_sorted, rates_sorted, success_counts_sorted, total_counts_sorted = zip(*sorted_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create bar plot
        bars = ax.bar(range(len(descs_sorted)), rates_sorted, alpha=0.7)
        
        # Color bars based on success rate (green for high, red for low)
        colors = ['green' if r >= 0.7 else 'orange' if r >= 0.5 else 'red' for r in rates_sorted]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for i, (rate, success, total) in enumerate(zip(rates_sorted, success_counts_sorted, total_counts_sorted)):
            ax.text(i, rate + 0.01, f'{rate:.2f}\n({success}/{total})', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Task Description (Rephrases)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        
        # Format task name for title
        task_title = task_name.replace('_', ' ').title()
        ax.set_title(f'Task Description Performance: {task_title}', fontsize=14, fontweight='bold')
        
        # Set x-axis labels (truncate long descriptions)
        x_labels = []
        for desc in descs_sorted:
            if len(desc) > 50:
                x_labels.append(desc[:47] + '...')
            else:
                x_labels.append(desc)
        
        ax.set_xticks(range(len(descs_sorted)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        
        # Set y-axis limits
        ax.set_ylim([0, max(rates_sorted) * 1.2 if rates_sorted else 1.0])
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot
        safe_task_name = task_name.lower().replace(' ', '_').replace('/', '_')
        out_path = os.path.join(output_dir, f'{safe_task_name}_task_description_performance.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(out_path)
        
        print(f"  Saved: {out_path} (showing {len(descs_sorted)} different descriptions)")
    
    print(f"\nTask description performance plots saved to: {output_dir}")
    return saved_paths

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze task description performance from pickle files')
    parser.add_argument('--base_path', type=str, 
                       default='./test_rollouts_openpi_rephrase_selection',
                       help='Path to rollouts directory containing pickle files')
    parser.add_argument('--output_dir', type=str,
                       default='./analysis_plots/task_description_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print(f"Analyzing task description performance from: {args.base_path}")
    results = analyze_task_description_performance(args.base_path)
    
    print(f"\nFound {len(results)} tasks")
    for task_name, descriptions in results.items():
        print(f"  {task_name}: {len(descriptions)} different descriptions")
    
    print(f"\nCreating plots...")
    plot_task_description_performance(results, output_dir=args.output_dir)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()


