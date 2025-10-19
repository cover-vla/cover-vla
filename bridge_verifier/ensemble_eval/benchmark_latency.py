#!/usr/bin/env python3
"""
Benchmark latency for different components of the ensemble inference.
Measures visual encoder, text encoder, and trainable components separately.
"""

import torch
import time
import numpy as np
from PIL import Image
import os
import sys
import argparse
import json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from efficient_ensemble_merged import EfficientEnsembleMerged
from open_clip import create_model_from_pretrained, get_tokenizer


class LatencyBenchmark:
    def __init__(self, merged_checkpoint_path, device='cuda', warmup_runs=5, benchmark_runs=20):
        """
        Initialize latency benchmark
        
        Args:
            merged_checkpoint_path: Path to merged checkpoint
            device: Device to run on
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
        """
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        
        print("="*80)
        print("Ensemble Inference Latency Benchmark")
        print("="*80)
        print(f"Device: {device}")
        print(f"Warmup runs: {warmup_runs}")
        print(f"Benchmark runs: {benchmark_runs}")
        
        # Load ensemble
        print(f"\nLoading ensemble from: {os.path.basename(merged_checkpoint_path)}")
        self.ensemble = EfficientEnsembleMerged(merged_checkpoint_path, device=device)
        
        # Create dummy data for benchmarking
        print("\nPreparing dummy data...")
        self.dummy_image = Image.new('RGB', (384, 384), color='red')
        self.dummy_instruction = "pick up the red block and place it in the box"
        self.dummy_action = np.random.randn(10, 7).astype(np.float32)
        
    def time_function(self, func, *args, **kwargs):
        """Time a function execution"""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        elapsed = (end - start) * 1000  # Convert to ms
        
        return elapsed, result
    
    def benchmark_visual_encoder(self):
        """Benchmark visual encoder latency"""
        print("\n" + "="*80)
        print("1. Visual Encoder Latency")
        print("="*80)
        
        # Preprocess image and convert to bf16 to match frozen encoder
        img_tensor = self.ensemble.preprocess(self.dummy_image).unsqueeze(0).to(self.device)
        img_tensor = img_tensor.to(torch.bfloat16)
        
        # Warmup
        print(f"Warmup ({self.warmup_runs} runs)...")
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = self.ensemble.siglip_model.encode_image(img_tensor)
        
        # Benchmark
        print(f"Benchmarking ({self.benchmark_runs} runs)...")
        latencies = []
        for _ in range(self.benchmark_runs):
            elapsed, _ = self.time_function(
                lambda: self.ensemble.siglip_model.encode_image(img_tensor)
            )
            latencies.append(elapsed)
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        print(f"\nResults:")
        print(f"  Mean: {mean_latency:.2f} ms")
        print(f"  Std:  {std_latency:.2f} ms")
        print(f"  Min:  {min_latency:.2f} ms")
        print(f"  Max:  {max_latency:.2f} ms")
        
        return {
            'mean': mean_latency,
            'std': std_latency,
            'min': min_latency,
            'max': max_latency,
            'all_runs': latencies
        }
    
    def benchmark_text_encoder(self):
        """Benchmark text encoder latency"""
        print("\n" + "="*80)
        print("2. Text Encoder Latency")
        print("="*80)
        
        # Tokenize text
        text_tokens = self.ensemble.tokenizer(
            [self.dummy_instruction],
            context_length=self.ensemble.siglip_model.context_length
        ).to(self.device)
        
        # Warmup
        print(f"Warmup ({self.warmup_runs} runs)...")
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = self.ensemble.siglip_model.encode_text(text_tokens)
        
        # Benchmark
        print(f"Benchmarking ({self.benchmark_runs} runs)...")
        latencies = []
        for _ in range(self.benchmark_runs):
            elapsed, _ = self.time_function(
                lambda: self.ensemble.siglip_model.encode_text(text_tokens)
            )
            latencies.append(elapsed)
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        print(f"\nResults:")
        print(f"  Mean: {mean_latency:.2f} ms")
        print(f"  Std:  {std_latency:.2f} ms")
        print(f"  Min:  {min_latency:.2f} ms")
        print(f"  Max:  {max_latency:.2f} ms")
        
        return {
            'mean': mean_latency,
            'std': std_latency,
            'min': min_latency,
            'max': max_latency,
            'all_runs': latencies
        }
    
    def benchmark_feature_extraction(self):
        """Benchmark combined feature extraction (visual + text)"""
        print("\n" + "="*80)
        print("3. Feature Extraction Latency (Visual + Text)")
        print("="*80)
        
        img_tensor = self.ensemble.preprocess(self.dummy_image).unsqueeze(0).to(self.device)
        text_tokens = self.ensemble.tokenizer(
            [self.dummy_instruction],
            context_length=self.ensemble.siglip_model.context_length
        ).to(self.device)
        
        # Warmup
        print(f"Warmup ({self.warmup_runs} runs)...")
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = self.ensemble.extract_shared_features(img_tensor, text_tokens)
        
        # Benchmark
        print(f"Benchmarking ({self.benchmark_runs} runs)...")
        latencies = []
        for _ in range(self.benchmark_runs):
            elapsed, _ = self.time_function(
                self.ensemble.extract_shared_features,
                img_tensor,
                text_tokens
            )
            latencies.append(elapsed)
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        print(f"\nResults:")
        print(f"  Mean: {mean_latency:.2f} ms")
        print(f"  Std:  {std_latency:.2f} ms")
        print(f"  Min:  {min_latency:.2f} ms")
        print(f"  Max:  {max_latency:.2f} ms")
        
        return {
            'mean': mean_latency,
            'std': std_latency,
            'min': min_latency,
            'max': max_latency,
            'all_runs': latencies
        }
    
    def benchmark_trainable_component_breakdown(self):
        """Benchmark individual trainable components"""
        print("\n" + "="*80)
        print("4. Trainable Component Breakdown (per model, single sample)")
        print("="*80)
        
        # Extract features once
        img_tensor = self.ensemble.preprocess(self.dummy_image).unsqueeze(0).to(self.device)
        text_tokens = self.ensemble.tokenizer(
            [self.dummy_instruction],
            context_length=self.ensemble.siglip_model.context_length
        ).to(self.device)
        
        with torch.no_grad():
            patch_features, text_features = self.ensemble.extract_shared_features(img_tensor, text_tokens)
        
        history_tensor = torch.tensor(self.dummy_action, dtype=torch.float32).unsqueeze(0).to(self.device)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)
        
        # Test with first model
        model_idx = 0
        components = self.ensemble.trainable_models[model_idx]
        
        results = {}
        
        # 1. Text-aware visual extraction
        print("\n  a) Text-aware visual extraction...")
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = components['text_aware_visual_extraction'](patch_features, text_features)
        
        latencies = []
        for _ in range(self.benchmark_runs):
            elapsed, _ = self.time_function(
                components['text_aware_visual_extraction'],
                patch_features,
                text_features
            )
            latencies.append(elapsed)
        
        results['text_aware_visual_extraction'] = {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'all_runs': latencies
        }
        print(f"     Mean: {results['text_aware_visual_extraction']['mean']:.2f} ms")
        
        # Get intermediate features for next steps
        with torch.no_grad():
            text_aware_features = components['text_aware_visual_extraction'](patch_features, text_features)
        
        # 2. Vision pooling
        print("\n  b) Vision pooling...")
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = components['vision_poolings'](text_aware_features)
        
        latencies = []
        for _ in range(self.benchmark_runs):
            elapsed, _ = self.time_function(
                components['vision_poolings'],
                text_aware_features
            )
            latencies.append(elapsed)
        
        results['vision_poolings'] = {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'all_runs': latencies
        }
        print(f"     Mean: {results['vision_poolings']['mean']:.2f} ms")
        
        # 3. Text pooling
        print("\n  c) Text pooling...")
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = components['text_pooling'](text_features)
        
        latencies = []
        for _ in range(self.benchmark_runs):
            elapsed, _ = self.time_function(
                components['text_pooling'],
                text_features
            )
            latencies.append(elapsed)
        
        results['text_pooling'] = {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'all_runs': latencies
        }
        print(f"     Mean: {results['text_pooling']['mean']:.2f} ms")
        
        # Get tokens for projection
        with torch.no_grad():
            vision_token = components['vision_poolings'](text_aware_features)
            text_token = components['text_pooling'](text_features)
            combined_features = torch.cat([text_token, vision_token], dim=-1)
        
        # 4. Input projection
        print("\n  d) Input projection...")
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = components['input_projection'](combined_features)
        
        latencies = []
        for _ in range(self.benchmark_runs):
            elapsed, _ = self.time_function(
                components['input_projection'],
                combined_features
            )
            latencies.append(elapsed)
        
        results['input_projection'] = {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'all_runs': latencies
        }
        print(f"     Mean: {results['input_projection']['mean']:.2f} ms")
        
        # 5. Action encoder
        action_histories = history_tensor.float()
        
        if self.ensemble.use_transformer:
            print("\n  e) Action encoder (Transformer):")
            
            # Single step encoder
            print("     - Single step encoder...")
            for _ in range(self.warmup_runs):
                with torch.no_grad():
                    _ = components['single_step_action_encoder'](action_histories)
            
            latencies = []
            for _ in range(self.benchmark_runs):
                elapsed, _ = self.time_function(
                    components['single_step_action_encoder'],
                    action_histories
                )
                latencies.append(elapsed)
            
            results['single_step_action_encoder'] = {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'all_runs': latencies
            }
            print(f"       Mean: {results['single_step_action_encoder']['mean']:.2f} ms")
            
            # Trajectory encoder
            print("     - Trajectory encoder...")
            with torch.no_grad():
                encoded_steps = components['single_step_action_encoder'](action_histories)
                encoded_steps_permuted = encoded_steps.permute(1, 0, 2)
                padding_mask = (action_histories[:, :, 0] == components['action_padding_value'])
            
            for _ in range(self.warmup_runs):
                with torch.no_grad():
                    _ = components['trajectory_encoder'](encoded_steps_permuted, src_key_padding_mask=padding_mask)
            
            latencies = []
            for _ in range(self.benchmark_runs):
                elapsed, _ = self.time_function(
                    components['trajectory_encoder'],
                    encoded_steps_permuted,
                    src_key_padding_mask=padding_mask
                )
                latencies.append(elapsed)
            
            results['trajectory_encoder'] = {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'all_runs': latencies
            }
            print(f"       Mean: {results['trajectory_encoder']['mean']:.2f} ms")
            
            # Total action encoder time
            results['action_encoder_total'] = {
                'mean': results['single_step_action_encoder']['mean'] + results['trajectory_encoder']['mean']
            }
            
        else:
            print("\n  e) Action encoder (MLP)...")
            batch_size = action_histories.shape[0]
            flat_actions = action_histories.reshape(batch_size, -1)
            
            for _ in range(self.warmup_runs):
                with torch.no_grad():
                    _ = components['complex_action_encoder'](flat_actions)
            
            latencies = []
            for _ in range(self.benchmark_runs):
                elapsed, _ = self.time_function(
                    components['complex_action_encoder'],
                    flat_actions
                )
                latencies.append(elapsed)
            
            results['complex_action_encoder'] = {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'all_runs': latencies
            }
            results['action_encoder_total'] = {
                'mean': results['complex_action_encoder']['mean']
            }
            print(f"     Mean: {results['complex_action_encoder']['mean']:.2f} ms")
        
        # Calculate total
        total_mean = (
            results['text_aware_visual_extraction']['mean'] +
            results['vision_poolings']['mean'] +
            results['text_pooling']['mean'] +
            results['input_projection']['mean'] +
            results['action_encoder_total']['mean']
        )
        results['total'] = {'mean': total_mean}
        
        print(f"\n  Total (single model, single sample): {total_mean:.2f} ms")
        
        return results
    
    def benchmark_trainable_components(self, num_histories=[1, 5, 10, 20]):
        """Benchmark trainable components with different batch sizes"""
        print("\n" + "="*80)
        print("5. Trainable Components Latency (batch processing)")
        print("="*80)
        
        # Extract features once
        img_tensor = self.ensemble.preprocess(self.dummy_image).unsqueeze(0).to(self.device)
        text_tokens = self.ensemble.tokenizer(
            [self.dummy_instruction],
            context_length=self.ensemble.siglip_model.context_length
        ).to(self.device)
        
        with torch.no_grad():
            patch_features, text_features = self.ensemble.extract_shared_features(img_tensor, text_tokens)
        
        results = {}
        
        for batch_size in num_histories:
            print(f"\n  Testing with {batch_size} action histories (BATCHED)...")
            
            # Create action histories batch
            action_histories = [self.dummy_action for _ in range(batch_size)]
            action_histories_array = np.array(action_histories)
            action_histories_batch = torch.tensor(action_histories_array, dtype=torch.float32).to(self.device)
            
            # Warmup
            for _ in range(self.warmup_runs):
                with torch.no_grad():
                    for model_idx in range(self.ensemble.num_models):
                        _ = self.ensemble.get_embeddings_from_model_batch(
                            model_idx, patch_features, text_features, action_histories_batch
                        )
            
            # Benchmark
            latencies = []
            for _ in range(self.benchmark_runs):
                start_time = time.perf_counter()
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                with torch.no_grad():
                    for model_idx in range(self.ensemble.num_models):
                        _ = self.ensemble.get_embeddings_from_model_batch(
                            model_idx, patch_features, text_features, action_histories_batch
                        )
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                elapsed = (end_time - start_time) * 1000
                latencies.append(elapsed)
            
            mean_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            per_sample = mean_latency / (batch_size * self.ensemble.num_models)
            
            print(f"    Total: {mean_latency:.2f} ± {std_latency:.2f} ms")
            print(f"    Per sample (1 model): {per_sample:.2f} ms")
            
            results[batch_size] = {
                'total_mean': mean_latency,
                'total_std': std_latency,
                'per_sample': per_sample,
                'all_runs': latencies
            }
        
        return results
    
    def benchmark_end_to_end(self, num_histories=[1, 5, 10, 20]):
        """Benchmark complete end-to-end inference"""
        print("\n" + "="*80)
        print("6. End-to-End Inference Latency")
        print("="*80)
        
        results = {}
        
        for batch_size in num_histories:
            print(f"\n  Testing with {batch_size} action histories...")
            
            # Create action pool
            action_pool = [self.dummy_action for _ in range(batch_size)]
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = self.ensemble.predict(self.dummy_image, self.dummy_instruction, action_pool)
            
            # Benchmark
            latencies = []
            for _ in range(self.benchmark_runs):
                elapsed, _ = self.time_function(
                    self.ensemble.predict,
                    self.dummy_image,
                    self.dummy_instruction,
                    action_pool
                )
                latencies.append(elapsed)
            
            mean_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            print(f"    Mean: {mean_latency:.2f} ± {std_latency:.2f} ms")
            
            results[batch_size] = {
                'mean': mean_latency,
                'std': std_latency,
                'all_runs': latencies
            }
        
        return results
    
    def run_all_benchmarks(self, batch_sizes=[1, 5, 10, 20]):
        """Run all benchmarks and compile results"""
        results = {
            'device': self.device,
            'num_models': self.ensemble.num_models,
            'warmup_runs': self.warmup_runs,
            'benchmark_runs': self.benchmark_runs,
            'batch_sizes': batch_sizes,
        }
        
        # Run benchmarks
        results['visual_encoder'] = self.benchmark_visual_encoder()
        results['text_encoder'] = self.benchmark_text_encoder()
        results['feature_extraction'] = self.benchmark_feature_extraction()
        results['trainable_component_breakdown'] = self.benchmark_trainable_component_breakdown()
        results['trainable_components'] = self.benchmark_trainable_components(batch_sizes)
        results['end_to_end'] = self.benchmark_end_to_end(batch_sizes)
        
        return results
    
    def print_summary(self, results):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        print(f"\nConfiguration:")
        print(f"  Device: {results['device']}")
        print(f"  Number of models in ensemble: {results['num_models']}")
        print(f"  Benchmark runs: {results['benchmark_runs']}")
        
        print(f"\n1. Encoder Latencies (single image/text):")
        print(f"  Visual Encoder:  {results['visual_encoder']['mean']:.2f} ms")
        print(f"  Text Encoder:    {results['text_encoder']['mean']:.2f} ms")
        print(f"  Combined (extract_features): {results['feature_extraction']['mean']:.2f} ms")
        
        print(f"\n2. Trainable Component Breakdown (per model, single sample):")
        breakdown = results['trainable_component_breakdown']
        print(f"  Text-aware visual extraction:  {breakdown['text_aware_visual_extraction']['mean']:.2f} ms")
        print(f"  Vision pooling:                {breakdown['vision_poolings']['mean']:.2f} ms")
        print(f"  Text pooling:                  {breakdown['text_pooling']['mean']:.2f} ms")
        print(f"  Input projection:              {breakdown['input_projection']['mean']:.2f} ms")
        if 'single_step_action_encoder' in breakdown:
            print(f"  Action encoder (Transformer):")
            print(f"    - Single step encoder:       {breakdown['single_step_action_encoder']['mean']:.2f} ms")
            print(f"    - Trajectory encoder:        {breakdown['trajectory_encoder']['mean']:.2f} ms")
            print(f"    - Total:                     {breakdown['action_encoder_total']['mean']:.2f} ms")
        else:
            print(f"  Action encoder (MLP):          {breakdown['action_encoder_total']['mean']:.2f} ms")
        print(f"  Total (1 model, 1 sample):     {breakdown['total']['mean']:.2f} ms")
        
        print(f"\n3. Trainable Components (batch processing):")
        print(f"  {'Batch Size':<12} {'Total Time':<15} {'Per Sample':<15}")
        print(f"  {'-'*12} {'-'*15} {'-'*15}")
        for batch_size in results['batch_sizes']:
            data = results['trainable_components'][batch_size]
            print(f"  {batch_size:<12} {data['total_mean']:>10.2f} ms   {data['per_sample']:>10.2f} ms")
        
        print(f"\n4. End-to-End Inference:")
        print(f"  {'Batch Size':<12} {'Total Time':<15} {'Throughput':<20}")
        print(f"  {'-'*12} {'-'*15} {'-'*20}")
        for batch_size in results['batch_sizes']:
            data = results['end_to_end'][batch_size]
            throughput = 1000.0 / data['mean']  # samples per second
            print(f"  {batch_size:<12} {data['mean']:>10.2f} ms   {throughput:>10.2f} samples/s")
        
        # Calculate breakdown for a typical case (batch_size=20)
        if 20 in results['batch_sizes']:
            print(f"\n5. Latency Breakdown (batch_size=20):")
            feature_time = results['feature_extraction']['mean']
            trainable_time = results['trainable_components'][20]['total_mean']
            fusion_time = results['end_to_end'][20]['mean'] - feature_time - trainable_time
            total_time = results['end_to_end'][20]['mean']
            
            print(f"  Feature Extraction:     {feature_time:>8.2f} ms ({feature_time/total_time*100:>5.1f}%)")
            print(f"  Trainable Components:   {trainable_time:>8.2f} ms ({trainable_time/total_time*100:>5.1f}%)")
            
            # Add detailed breakdown of trainable components
            breakdown = results['trainable_component_breakdown']
            per_model_time = breakdown['total']['mean']
            expected_trainable = per_model_time * results['num_models'] * 20  # 3 models × 20 samples
            
            print(f"    └─ Per-sample breakdown:")
            print(f"       Text-aware visual:   {breakdown['text_aware_visual_extraction']['mean']:>6.2f} ms ({breakdown['text_aware_visual_extraction']['mean']/per_model_time*100:>5.1f}%)")
            print(f"       Vision pooling:      {breakdown['vision_poolings']['mean']:>6.2f} ms ({breakdown['vision_poolings']['mean']/per_model_time*100:>5.1f}%)")
            print(f"       Text pooling:        {breakdown['text_pooling']['mean']:>6.2f} ms ({breakdown['text_pooling']['mean']/per_model_time*100:>5.1f}%)")
            print(f"       Input projection:    {breakdown['input_projection']['mean']:>6.2f} ms ({breakdown['input_projection']['mean']/per_model_time*100:>5.1f}%)")
            print(f"       Action encoder:      {breakdown['action_encoder_total']['mean']:>6.2f} ms ({breakdown['action_encoder_total']['mean']/per_model_time*100:>5.1f}%)")
            print(f"       Total (1 model):     {per_model_time:>6.2f} ms (100.0%)")
            
            print(f"  Fusion & Scoring:       {fusion_time:>8.2f} ms ({fusion_time/total_time*100:>5.1f}%)")
            print(f"  {'Total:':<24} {total_time:>8.2f} ms (100.0%)")
        
        print("\n" + "="*80)


def save_results(results, output_file):
    """Save results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark ensemble inference latency')
    
    parser.add_argument('--merged_checkpoint', type=str, required=True,
                       help='Path to merged checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda or cpu)')
    parser.add_argument('--warmup_runs', type=int, default=5,
                       help='Number of warmup iterations')
    parser.add_argument('--benchmark_runs', type=int, default=20,
                       help='Number of benchmark iterations')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 5, 10, 20],
                       help='Batch sizes to test')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.merged_checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.merged_checkpoint}")
        exit(1)
    
    # Run benchmark
    benchmark = LatencyBenchmark(
        args.merged_checkpoint,
        device=args.device,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs
    )
    
    results = benchmark.run_all_benchmarks(batch_sizes=args.batch_sizes)
    benchmark.print_summary(results)
    
    # Save results
    save_results(results, args.output)

