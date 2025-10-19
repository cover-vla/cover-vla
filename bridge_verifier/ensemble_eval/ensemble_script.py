#!/usr/bin/env python3
"""
Script to evaluate how ensemble size affects accuracy.
Tests with 1-5 models (top 5 performing: epochs 7, 9, 11, 6, 12).
"""

import subprocess
import os
import sys
import re

# Model configurations (top 5 performing models based on accuracy)
# Accuracies: epoch_7=0.740, epoch_9=0.700, epoch_11=0.680, epoch_6=0.660, epoch_12=0.640
MODELS = [
    ("downloads/bridge_4096_6e5_64_epoch_7_trainloss_2.8373_valloss_1.7805.pt", "hf-hub:timm/ViT-L-16-SigLIP2-384", True),  # 0.740
    ("downloads/bridge_4096_6e5_64_epoch_8_trainloss_2.3440_valloss_1.4271.pt", "hf-hub:timm/ViT-L-16-SigLIP2-384", True),  # 0.700
    ("downloads/bridge_4096_6e5_64_epoch_9_trainloss_1.9012_valloss_1.0189.pt", "hf-hub:timm/ViT-L-16-SigLIP2-384", True),  # 0.700
]

# Common parameters
DATASET = "bridge_dataset_with_rephrases.json"
IMAGES_FOLDER = "bridge_dataset_with_rephrases_images"
HISTORY_LENGTH = 10
NUM_SAMPLES = 50
ACTION_POOL_SIZE = 20

def run_ensemble_evaluation(num_models, models_to_use):
    """Run ensemble evaluation with specified number of models"""
    
    print(f"\n{'='*60}")
    print(f"Testing with {num_models} model(s)")
    print(f"{'='*60}")
    
    # Build command with new flexible argument format
    cmd = [
        "python3", "ensemble_eval/ensemble_inference.py",
        "--bridge_dataset", DATASET,
        "--images_folder", IMAGES_FOLDER,
        "--history_length", str(HISTORY_LENGTH),
        "--num_samples", str(NUM_SAMPLES),
        "--action_pool_size", str(ACTION_POOL_SIZE),
        "--use_transformer",  # All models use transformer
    ]
    
    # Add model paths
    cmd.append("--model_paths")
    for model_path, _, _ in models_to_use:
        cmd.append(model_path)
    
    # Add backbones
    cmd.append("--backbones")
    for _, backbone, _ in models_to_use:
        cmd.append(backbone)
    
    print(f"Models:")
    for i, (model_path, _, _) in enumerate(models_to_use, 1):
        print(f"  {i}. {os.path.basename(model_path)}")
    print()
    
    # Run command and capture output
    log_file = f"logs/ensemble_size_{num_models}_models.log"
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/root/vla-clip/bridge_verifier"
            )
        
        if result.returncode == 0:
            print(f"✅ Results saved to {log_file}")
            return log_file
        else:
            print(f"❌ Error running evaluation (exit code: {result.returncode})")
            return None
            
    except Exception as e:
        print(f"❌ Exception during evaluation: {e}")
        return None

def extract_metrics(log_file):
    """Extract accuracy and mean rank from log file"""
    if not os.path.exists(log_file):
        return None, None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract accuracy
        accuracy_match = re.search(r'Overall accuracy: ([\d.]+)', content)
        accuracy = float(accuracy_match.group(1)) if accuracy_match else None
        
        # Extract mean rank
        rank_match = re.search(r'Mean rank of ground truth action history: ([\d.]+)', content)
        mean_rank = float(rank_match.group(1)) if rank_match else None
        
        return accuracy, mean_rank
    except Exception as e:
        print(f"Error extracting metrics from {log_file}: {e}")
        return None, None

def main():
    os.chdir("/root/vla-clip/bridge_verifier")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    print("="*60)
    print("Ensemble Size Analysis")
    print("="*60)
    print(f"Dataset: {DATASET}")
    print(f"Images: {IMAGES_FOLDER}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Action pool size: {ACTION_POOL_SIZE}")
    print(f"Total models available: {len(MODELS)}")
    print()
    
    results = []
    
    # Test with different ensemble sizes from 1 to 5 models
    ensemble_sizes = [1, 2, 3, 4, 5]
    
    for num_models in ensemble_sizes:
        if num_models <= len(MODELS):
            log_file = run_ensemble_evaluation(num_models, MODELS[:num_models])
            if log_file:
                accuracy, mean_rank = extract_metrics(log_file)
                results.append((num_models, accuracy, mean_rank))
        else:
            print(f"⚠️  Skipping {num_models} models (only {len(MODELS)} available)")
            break
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY: Ensemble Size vs Accuracy")
    print("="*60)
    print(f"{'Ensemble Size':<15} {'Accuracy':<12} {'Mean Rank':<12}")
    print("-"*60)
    
    for num_models, accuracy, mean_rank in results:
        acc_str = f"{accuracy:.3f}" if accuracy is not None else "N/A"
        rank_str = f"{mean_rank:.3f}" if mean_rank is not None else "N/A"
        print(f"{num_models} model(s){' '*(7)}{acc_str:<12} {rank_str:<12}")
    
    print("="*60)
    
    # Calculate improvement
    if len(results) >= 2 and results[0][1] is not None and results[-1][1] is not None:
        baseline_acc = results[0][1]
        ensemble_acc = results[-1][1]
        improvement = ensemble_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        num_baseline = results[0][0]
        num_ensemble = results[-1][0]
        
        print(f"\nImprovement from {num_baseline} to {num_ensemble} models:")
        print(f"  Absolute: {improvement:+.3f}")
        print(f"  Relative: {improvement_pct:+.1f}%")
        print("="*60)
    
    # Save summary to file
    summary_file = "logs/ensemble_size_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Ensemble Size Analysis Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Ensemble Size':<15} {'Accuracy':<12} {'Mean Rank':<12}\n")
        f.write("-"*60 + "\n")
        for num_models, accuracy, mean_rank in results:
            acc_str = f"{accuracy:.3f}" if accuracy is not None else "N/A"
            rank_str = f"{mean_rank:.3f}" if mean_rank is not None else "N/A"
            f.write(f"{num_models} model(s){' '*(7)}{acc_str:<12} {rank_str:<12}\n")
        f.write("="*60 + "\n")
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()
