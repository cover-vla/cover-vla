#!/usr/bin/env python3
"""
Test script for the Pi0Policy batch API server.

Usage:
    1. Start the server: python pi0_batch_server.py
    2. Run this test: python test_pi0_batch_api.py
"""

import requests
import json
import numpy as np
import os

def test_batch_api():
    """Test the batch API with multiple instructions."""
    
    # Configuration
    api_url = "http://localhost:3200/batch"
    image_path = "/root/pi/vla-clip/RoboMonkey/openvla-mini/experiments/robot/simpler/test_bridge_image.jpg"
    
    # Test instructions
    instructions = [
        "open the drawer",
        "close the drawer",
        "pick up the cup"
    ]
    
    print("="*80)
    print("Testing Pi0Policy Batch API")
    print("="*80)
    print(f"API URL: {api_url}")
    print(f"Image: {image_path}")
    print(f"Number of instructions: {len(instructions)}")
    print("\nInstructions:")
    for i, instr in enumerate(instructions):
        print(f"  {i+1}. {instr}")
    print("="*80)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return
    
    # Prepare request payload
    payload = {
        "instructions": instructions,
        "image_path": image_path,
        "temperature": 0.0
    }
    
    print("\nSending request to API...")
    try:
        response = requests.post(
            api_url,
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        response.raise_for_status()
        
        print("✓ Request successful!")
        
        # Parse response
        result = response.json()
        output_ids = np.array(result["output_ids"])
        actions = np.array(result["actions"])
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Output IDs shape: {output_ids.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Expected shape: [{len(instructions)}, 4, 7]")
        
        # Verify shape
        if actions.shape == (len(instructions), 4, 7):
            print("✓ Action shape is correct!")
        else:
            print(f"✗ Action shape mismatch! Expected [{len(instructions)}, 4, 7], got {actions.shape}")
        
        # Display actions for each instruction
        print("\n" + "="*80)
        print("RAW ACTIONS (normalized)")
        print("="*80)
        
        for i, instruction in enumerate(instructions):
            print(f"\nInstruction {i+1}: \"{instruction}\"")
            print("-" * 80)
            
            for step in range(4):
                action = actions[i, step]
                print(f"  Step {step+1}/4: {action}")
                print(f"    Position delta (x, y, z): [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
                print(f"    Rotation (roll, pitch, yaw): [{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}]")
                print(f"    Gripper (0=close, 1=open): {action[6]:.4f}")
        
        print("\n" + "="*80)
        print("✓ Test completed successfully!")
        print("="*80)
        
        return actions
        
    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Could not connect to API server.")
        print("  Make sure the server is running: python pi0_batch_server.py")
    except requests.exceptions.Timeout:
        print("✗ ERROR: Request timed out.")
    except requests.exceptions.HTTPError as e:
        print(f"✗ ERROR: HTTP error: {e}")
        print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_batch_api()

