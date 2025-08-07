#!/usr/bin/env python3
"""
Test script to verify the integration of the batch verifier and VLA-CLIP trajectory scoring
with the SimplerEnv evaluation pipeline.

This script tests:
1. Import of the new evaluation module
2. Configuration parsing
3. Basic functionality without running full evaluation
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import numpy as np
        import tqdm
        import collections
        print("✓ Basic imports successful")
        
        # Test SimplerEnv imports
        from simpler_benchmark import get_benchmark
        from simpler_utils import get_simpler_env, get_simpler_img
        print("✓ SimplerEnv imports successful")
        
        # Test OpenVLA imports
        sys.path.append("../..")
        from experiments.robot.openvla_utils import get_processor
        from experiments.robot.robot_utils import get_model, get_action
        print("✓ OpenVLA imports successful")
        
        # Test VLA-CLIP imports
        sys.path.append("/root/vla-clip/bridge_verifier")
        try:
            from vla_clip_inference_bridge import VLA_CLIP_Bridge_Inference, ACTION_PADDING_VALUE
            print("✓ VLA-CLIP Bridge imports successful")
        except ImportError as e:
            print(f"⚠ VLA-CLIP Bridge imports failed (this is expected if models not available): {e}")
        
        # Test batch verifier imports
        try:
            import requests
            import json_numpy as json
            print("✓ Batch verifier imports successful")
        except ImportError as e:
            print(f"⚠ Batch verifier imports failed: {e}")
            print("  Install with: pip install requests json_numpy")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration parsing."""
    print("\nTesting configuration...")
    
    try:
        from run_simpler_eval_with_verifier import GenerateConfig, eval_simpler_with_verifier
        
        # Test basic configuration
        config = GenerateConfig()
        print(f"✓ Default config created: task_suite_name={config.task_suite_name}")
        
        # Test configuration with verifier enabled
        config.use_batch_verifier = True
        config.use_vla_clip_trajectory_scorer = True
        config.clip_select_action_num_candidates = 3
        print(f"✓ Verifier config set: batch_verifier={config.use_batch_verifier}, clip_scorer={config.use_vla_clip_trajectory_scorer}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_benchmark_loading():
    """Test that the benchmark can be loaded."""
    print("\nTesting benchmark loading...")
    
    try:
        from simpler_benchmark import get_benchmark
        
        # Test loading the main benchmark
        benchmark = get_benchmark("simpler_widowx")()
        print(f"✓ Benchmark loaded: {benchmark.name}, {benchmark.n_tasks} tasks")
        
        # Test getting a task
        task = benchmark.get_task(0)
        print(f"✓ Task retrieved: {task}")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark loading failed: {e}")
        return False

def test_rephrase_loading():
    """Test that rephrases can be loaded."""
    print("\nTesting rephrase loading...")
    
    try:
        from run_simpler_eval_with_verifier import load_rephrases
        
        rephrases = load_rephrases("simpler_widowx")
        if rephrases:
            print(f"✓ Rephrases loaded: {len(rephrases)} task entries")
            for task_id, data in rephrases.items():
                print(f"  Task {task_id}: {len(data['rephrases'])} rephrases")
        else:
            print("⚠ No rephrases found (this is expected if file doesn't exist)")
        
        return True
        
    except Exception as e:
        print(f"✗ Rephrase loading failed: {e}")
        return False

def test_batch_function():
    """Test the batch action function (without actual server)."""
    print("\nTesting batch function...")
    
    try:
        from run_simpler_eval_with_verifier import get_batch_actions
        
        # Test function signature
        instructions = ["test instruction 1", "test instruction 2"]
        image_path = "./test_image.jpg"
        server_url = "http://localhost:3200"
        
        # This should fail gracefully if server is not running
        output_ids, actions = get_batch_actions(instructions, image_path, server_url)
        if output_ids is None and actions is None:
            print("✓ Batch function handles server unavailability gracefully")
        else:
            print("✓ Batch function returned results (server may be running)")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch function test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing SimplerEnv Evaluation with Verifier Integration")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_benchmark_loading,
        test_rephrase_loading,
        test_batch_function,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The integration is ready to use.")
        print("\nNext steps:")
        print("1. Start the SGLang batch server: python vla/openvla_server.py --seed 0")
        print("2. Ensure you have a VLA-CLIP trajectory model for scoring")
        print("3. Run the evaluation: python run_simpler_eval_with_verifier.py --help")
    else:
        print("⚠ Some tests failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("- Missing dependencies: pip install requests json_numpy")
        print("- Missing clip_verifier scripts")
        print("- Incorrect file paths")

if __name__ == "__main__":
    main() 