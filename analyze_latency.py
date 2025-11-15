#!/usr/bin/env python3
"""
Script to analyze latency and throughput data from rollout pickle files.
"""

import pickle
import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path

def load_latency_data(base_path):
    """Load all latency data from pickle files."""
    pkl_files = glob.glob(f'{base_path}/*latency*/*.pkl')
    
    results = []
    
    for pkl_file in sorted(pkl_files):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract directory name and episode info
            dir_name = os.path.basename(os.path.dirname(pkl_file))
            episode_name = os.path.basename(pkl_file).replace('.pkl', '')
            
            # Extract latency stats
            latency_stats = data.get('latency_stats', {})
            throughput = data.get('throughput', {})
            
            # Get VLA latency and throughput
            vla_latency = latency_stats.get('vla_total', {})
            vla_throughput = throughput.get('vla_total_inferences_per_second', None)
            
            # Get image-text encoder latency and throughput
            image_text_latency = latency_stats.get('verifier_image_text_encoder', {})
            image_text_throughput = throughput.get('verifier_image_text_encoder_inferences_per_second', None)
            
            # Get action encoder latency and throughput
            action_encoder_latency = latency_stats.get('verifier_action_encoder', {})
            action_encoder_throughput = throughput.get('verifier_action_encoder_inferences_per_second', None)
            
            # Extract mean and std if available (convert from seconds to milliseconds)
            # Latency values are in seconds, convert to milliseconds (* 1000)
            vla_mean_ms = vla_latency.get('mean', None) * 1000 if isinstance(vla_latency, dict) and vla_latency.get('mean') is not None else None
            vla_std_ms = vla_latency.get('std', None) * 1000 if isinstance(vla_latency, dict) and vla_latency.get('std') is not None else None
            
            image_text_mean_ms = image_text_latency.get('mean', None) * 1000 if isinstance(image_text_latency, dict) and image_text_latency.get('mean') is not None else None
            image_text_std_ms = image_text_latency.get('std', None) * 1000 if isinstance(image_text_latency, dict) and image_text_latency.get('std') is not None else None
            
            action_encoder_mean_ms = action_encoder_latency.get('mean', None) * 1000 if isinstance(action_encoder_latency, dict) and action_encoder_latency.get('mean') is not None else None
            action_encoder_std_ms = action_encoder_latency.get('std', None) * 1000 if isinstance(action_encoder_latency, dict) and action_encoder_latency.get('std') is not None else None
            
            result = {
                'directory': dir_name,
                'episode': episode_name,
                'vla_latency_mean': vla_mean_ms,
                'vla_latency_std': vla_std_ms,
                'vla_throughput': vla_throughput,
                'image_text_latency_mean': image_text_mean_ms,
                'image_text_latency_std': image_text_std_ms,
                'image_text_throughput': image_text_throughput,
                'action_encoder_latency_mean': action_encoder_mean_ms,
                'action_encoder_latency_std': action_encoder_std_ms,
                'action_encoder_throughput': action_encoder_throughput,
                'success': data.get('success', None),
                'episode_length': data.get('episode_length', None),
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    return results

def create_summary_table(results):
    """Create a summary table from the results."""
    df = pd.DataFrame(results)
    
    # Group by directory and calculate averages
    summary_data = []
    
    for dir_name in df['directory'].unique():
        dir_df = df[df['directory'] == dir_name]
        
        # Calculate mean and std across all episodes in this directory
        summary = {
            'Directory': dir_name,
            'Num Episodes': len(dir_df),
            'VLA Latency (ms)': f"{dir_df['vla_latency_mean'].mean():.1f} ± {dir_df['vla_latency_std'].mean():.1f}" if dir_df['vla_latency_mean'].notna().any() else 'N/A',
            'VLA Throughput (inf/s)': f"{dir_df['vla_throughput'].mean():.2f}" if dir_df['vla_throughput'].notna().any() else 'N/A',
            'Image-Text Encoder Latency (ms)': f"{dir_df['image_text_latency_mean'].mean():.2f} ± {dir_df['image_text_latency_std'].mean():.2f}" if dir_df['image_text_latency_mean'].notna().any() else 'N/A',
            'Image-Text Encoder Throughput (inf/s)': f"{dir_df['image_text_throughput'].mean():.2f}" if dir_df['image_text_throughput'].notna().any() else 'N/A',
            'Action Encoder Latency (ms)': f"{dir_df['action_encoder_latency_mean'].mean():.2f} ± {dir_df['action_encoder_latency_std'].mean():.2f}" if dir_df['action_encoder_latency_mean'].notna().any() else 'N/A',
            'Action Encoder Throughput (inf/s)': f"{dir_df['action_encoder_throughput'].mean():.2f}" if dir_df['action_encoder_throughput'].notna().any() else 'N/A',
        }
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Also create per-episode detailed table
    detailed_df = df[[
        'directory', 'episode', 
        'vla_latency_mean', 'vla_throughput',
        'image_text_latency_mean', 'image_text_throughput',
        'action_encoder_latency_mean', 'action_encoder_throughput',
        'success', 'episode_length'
    ]].copy()
    
    # Format the latency and throughput columns for better readability
    detailed_df['vla_latency_mean'] = detailed_df['vla_latency_mean'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    detailed_df['image_text_latency_mean'] = detailed_df['image_text_latency_mean'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    detailed_df['action_encoder_latency_mean'] = detailed_df['action_encoder_latency_mean'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    detailed_df['vla_throughput'] = detailed_df['vla_throughput'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    detailed_df['image_text_throughput'] = detailed_df['image_text_throughput'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    detailed_df['action_encoder_throughput'] = detailed_df['action_encoder_throughput'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    detailed_df.columns = [
        'Directory', 'Episode',
        'VLA Latency (ms)', 'VLA Throughput (inf/s)',
        'Image-Text Encoder Latency (ms)', 'Image-Text Encoder Throughput (inf/s)',
        'Action Encoder Latency (ms)', 'Action Encoder Throughput (inf/s)',
        'Success', 'Episode Length'
    ]
    
    return summary_df, detailed_df

def print_table(df, title):
    """Print a nicely formatted table."""
    print(f"\n{'='*120}")
    print(f"{title:^120}")
    print(f"{'='*120}")
    print(df.to_string(index=False))
    print(f"{'='*120}\n")

if __name__ == '__main__':
    base_path = 'RoboMonkey/openvla-mini/experiments/robot/simpler/bashes/rollouts_openpi_original/transform_rephrase'
    
    print(f"Loading latency data from: {base_path}")
    results = load_latency_data(base_path)
    
    if not results:
        print("No data found!")
        exit(1)
    
    print(f"Loaded {len(results)} episodes")
    
    # Create tables
    summary_df, detailed_df = create_summary_table(results)
    
    # Print summary table
    print_table(summary_df, "SUMMARY: Average Latency and Throughput by Directory")
    
    # Print detailed table
    print_table(detailed_df, "DETAILED: Latency and Throughput per Episode")
    
    # Calculate overall statistics
    print(f"\n{'='*120}")
    print(f"{'OVERALL STATISTICS':^120}")
    print(f"{'='*120}")
    
    df = pd.DataFrame(results)
    
    components = [
        ('VLA', 'vla_latency_mean', 'vla_throughput'),
        ('Image-Text Encoder', 'image_text_latency_mean', 'image_text_throughput'),
        ('Action Encoder', 'action_encoder_latency_mean', 'action_encoder_throughput'),
    ]
    
    for comp_name, lat_col, thr_col in components:
        if df[lat_col].notna().any():
            mean_lat = df[lat_col].mean()
            std_lat = df[lat_col].std()
            mean_thr = df[thr_col].mean() if df[thr_col].notna().any() else None
            # Format based on component (VLA needs 1 decimal, others 2 decimals)
            if comp_name == 'VLA':
                lat_format = f"{mean_lat:.1f} ± {std_lat:.1f}"
            else:
                lat_format = f"{mean_lat:.2f} ± {std_lat:.2f}"
            print(f"{comp_name:30s} | Latency: {lat_format} ms | Throughput: {mean_thr:.2f} inf/s" if mean_thr else f"{comp_name:30s} | Latency: {lat_format} ms | Throughput: N/A")
    
    print(f"{'='*120}\n")
    
    # Save to CSV files
    summary_csv_path = 'latency_summary_table.csv'
    detailed_csv_path = 'latency_detailed_table.csv'
    
    summary_df.to_csv(summary_csv_path, index=False)
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    print(f"Tables saved to:")
    print(f"  - Summary: {summary_csv_path}")
    print(f"  - Detailed: {detailed_csv_path}\n")

