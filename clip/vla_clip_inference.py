import torch
import clip
from PIL import Image
import h5py
import os
import numpy as np
import random
from tqdm import tqdm
from model import TextAwareVisualExtraction, ModelConfig
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt
import cv2
import argparse


class VLA_CLIP_Inference:
    def __init__(self, model_path, trajectory_mode=True, trajectory_length=20, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Load the base CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        
        # Set mode (trajectory or single action)
        self.trajectory_mode = trajectory_mode
        self.trajectory_length = trajectory_length
        
        # Initialize the VLA_CLIP model
        self.model = self._init_model(model_path, device)
        self.device = device
    
    def _init_model(self, model_path, device):
        # Initialize the appropriate model based on mode
        if self.trajectory_mode:
            from finetune_trajectory import VLA_CLIP
        else:
            from finetune import VLA_CLIP
        
        # Create model configuration
        from model import ModelConfig
        model_config = ModelConfig(clip_model=self.clip_model)
        
        # Set trajectory parameters if in trajectory mode
        if self.trajectory_mode:
            model_config.trajectory_length = self.trajectory_length
        
        # Initialize model
        model = VLA_CLIP(model_config).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()  # Set to evaluation mode
        model.float()  # Ensure model is in float32 precision
        return model
    
    def predict(self, image, instruction, possible_actions, action_history=None):
        """
        Predict the most likely action given an image and instruction
        
        Args:
            image: PIL Image or numpy array
            instruction: String instruction
            possible_actions: List of numerical action arrays
            action_history: Optional list of past actions (for trajectory mode)
            
        Returns:
            predicted_action: The most likely action
            action_scores: Dictionary mapping actions to scores
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize instruction
        text_tokens = clip.tokenize([instruction]).to(self.device)
        
        # Ensure all actions are numpy arrays
        processed_actions = [np.array(action) if not isinstance(action, np.ndarray) else action 
                            for action in possible_actions]
        
        # Handle trajectory mode
        if self.trajectory_mode:
            # Create action trajectories for each possible action
            action_trajectories = []
            
            # If no action history is provided, create a default one with padding
            if action_history is None:
                # Create a padding trajectory with the last element being the action
                for action in processed_actions:
                    # Create a trajectory of padding values
                    padding = np.ones((self.trajectory_length-1, action.shape[0])) * -5.0
                    # Add the current action as the last element
                    trajectory = np.vstack([padding, action.reshape(1, -1)])
                    action_trajectories.append(trajectory)
            else:
                # Use the provided action history
                history = action_history[-self.trajectory_length+1:] if len(action_history) > self.trajectory_length-1 else action_history
                
                # Pad history if needed
                if len(history) < self.trajectory_length-1:
                    padding_length = self.trajectory_length-1 - len(history)
                    # Use first action for padding if available, otherwise use -5.0
                    if len(history) > 0:
                        padding = np.array([history[0]] * padding_length)
                    else:
                        padding = np.ones((padding_length, processed_actions[0].shape[0])) * -5.0
                    history = np.vstack([padding, history]) if len(history) > 0 else padding
                
                # Create trajectories for each possible action
                for action in processed_actions:
                    trajectory = np.vstack([history, action.reshape(1, -1)])
                    action_trajectories.append(trajectory)
            
            # Forward pass through the model
            with torch.no_grad():
                # Run inference with trajectories
                image_logits, _ = self.model(image_tensor, text_tokens, action_trajectories)
                
                # Get scores from image_logits
                scores = image_logits.cpu().numpy()[0]
                
                # Get predicted action
                predicted_idx = scores.argmax()
                predicted_action = possible_actions[predicted_idx]
        else:
            # Single action mode
            with torch.no_grad():
                # Run inference with single actions
                image_logits, _ = self.model(image_tensor, text_tokens, processed_actions)
                
                # Get scores from image_logits
                scores = image_logits.cpu().numpy()[0]
                
                # Get predicted action
                predicted_idx = scores.argmax()
                predicted_action = possible_actions[predicted_idx]
        
        # Create a dictionary of action scores
        action_score_dict = {str(i): float(score) for i, score in enumerate(scores)}
        
        return predicted_action, action_score_dict

def load_libero_dataset(libero_base_path):
    """Load the LIBERO dataset from all folders in the datasets directory"""
    dataset_dict = {}
    
    # Get all subdirectories in the datasets folder
    dataset_folders = [f for f in os.listdir(libero_base_path) 
                      if os.path.isdir(os.path.join(libero_base_path, f))]
    
    # Process each dataset folder
    for folder in dataset_folders:
        folder_path = os.path.join(libero_base_path, folder)
        print(f"\nLoading dataset folder: {folder}")
        
        # Process each task file in the folder
        for task in os.listdir(folder_path):
            if not task.endswith('.hdf5'):
                continue
                
            # Extract task name without .hdf5 extension as the language instruction
            language_instruction = task.replace('.hdf5', '')
            dataset_dict[language_instruction] = {
                'actions': [],
                'images': []
            }
            
            task_path = os.path.join(folder_path, task)
            print(f"Loading task: {task}")
            
            with h5py.File(task_path, 'r') as f:
                for demo_key in f['data'].keys():
                    demo_data = f['data'][demo_key]
                    
                    # Get actions data
                    actions = demo_data['actions'][()]
                    dataset_dict[language_instruction]['actions'].append(actions)
                    
                    # Get observation data
                    obs_group = demo_data['obs']
                    obs_data = obs_group['agentview_rgb'][()]
                    dataset_dict[language_instruction]['images'].append(obs_data)
                    
            print(f"Processed {task}: {len(dataset_dict[language_instruction]['actions'])} demonstrations")
    
    return dataset_dict

def sample_and_test(dataset_dict, model_path, trajectory_mode=True, num_samples=10, action_pool_size=20):
    """
    Sample random data points and test the model with a limited action pool
    
    Args:
        dataset_dict: Dictionary containing the dataset
        model_path: Path to the trained model
        trajectory_mode: Whether to use trajectory mode or single action mode
        num_samples: Number of samples to test
        action_pool_size: Size of the action pool for each test sample
    """
    # Initialize the inference model
    inference_model = VLA_CLIP_Inference(model_path, trajectory_mode=trajectory_mode)
    
    # Flatten the dataset into (image, instruction, action) tuples
    samples = []
    all_actions = []
    action_histories = {}  # Store action histories for each (instruction, demo_idx)
    
    for instruction, data in dataset_dict.items():
        images = data['images']
        actions = data['actions']
        for i, img_seq in enumerate(images):
            # Initialize history for this demonstration
            history_key = (instruction, i)
            action_histories[history_key] = []
            
            for j, frame in enumerate(img_seq):
                # Add current action to history
                current_action = actions[i][j]
                
                # Store the sample with its history key
                samples.append((frame, instruction, current_action, history_key, j))
                all_actions.append(current_action)
                
                # Update history after using it
                action_histories[history_key].append(current_action)
    
    # Randomly sample data points
    random.seed(42)  # For reproducibility
    test_samples = random.sample(samples, min(num_samples, len(samples)))
    
    # Test each sample with a limited action pool
    results = []
    for image, instruction, ground_truth, history_key, frame_idx in tqdm(test_samples, desc="Testing samples"):
        # Create a random pool of actions
        # 1. Start with the ground truth
        action_pool = [ground_truth]
        
        # 2. Randomly sample from all actions
        other_actions = [a for a in all_actions if not np.array_equal(a, ground_truth)]
        if other_actions and action_pool_size > 1:
            # Sample random actions to fill the pool
            sampled_actions = random.sample(other_actions, 
                                          min(action_pool_size - 1, len(other_actions)))
            action_pool.extend(sampled_actions)
        
        # Shuffle the action pool
        random.shuffle(action_pool)
        
        # Find index of ground truth in the action pool
        ground_truth_idx = None
        for i, action in enumerate(action_pool):
            if np.array_equal(action, ground_truth):
                ground_truth_idx = i
                break
        
        # Get action history (excluding current frame)
        if trajectory_mode and frame_idx > 0:
            action_history = action_histories[history_key][:frame_idx]
        else:
            action_history = None
        
        # Run prediction
        predicted_action, scores = inference_model.predict(
            image, instruction, action_pool, action_history=action_history
        )
        
        # Check if prediction is correct
        is_correct = np.array_equal(predicted_action, ground_truth)
        
        results.append({
            'image': image,
            'instruction': instruction,
            'ground_truth': ground_truth,
            'ground_truth_idx': ground_truth_idx,
            'prediction': predicted_action,
            'action_pool_size': len(action_pool),
            'scores': scores,
            'correct': is_correct
        })
    
    return results

def display_results(results):
    """Display the test results with enhanced metrics"""
    correct = 0
    rank_sum = 0
    
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"Instruction: {result['instruction']}")
        print(f"Action pool size: {result['action_pool_size']}")
        
        # Print ground truth index and action
        print(f"Ground truth index: {result['ground_truth_idx']}")
        print(f"Ground truth action: {result['ground_truth']}")
        print(f"Predicted action: {result['prediction']}")
        
        # Check if prediction matches ground truth
        is_correct = result['correct']
        if is_correct:
            correct += 1
            print(f"Correct: True")
        else:
            print(f"Correct: False")
            
            # Calculate L2 distance for continuous actions
            l2_dist = np.linalg.norm(result['ground_truth'] - result['prediction'])
            print(f"L2 distance: {l2_dist:.4f}")
        
        # Find rank of ground truth in scores
        if 'ground_truth_idx' in result and result['ground_truth_idx'] is not None:
            gt_idx = result['ground_truth_idx']
            action_scores = [(i, float(score)) for i, (action, score) in enumerate(result['scores'].items())]
            action_scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (idx, _) in enumerate(action_scores):
                if str(gt_idx) == idx:
                    rank_sum += rank + 1
                    print(f"Ground truth rank: {rank + 1}/{result['action_pool_size']}")
                    break
        
        # Print top 3 scores with indices
        sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
        print("Top predictions (index: score):")
        for j, (action_idx, score) in enumerate(sorted_scores[:3]):
            if j < 3:  # Only show top 3
                print(f"  {action_idx}: {score:.4f}")
    
    accuracy = correct / len(results)
    mean_rank = rank_sum / len(results) if results else 0
    
    print(f"\nOverall accuracy: {accuracy:.2f} ({correct}/{len(results)})")
    print(f"Mean rank of ground truth: {mean_rank:.2f}")
    
    # Optional: Display some images with predictions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(results):
            ax.imshow(results[i]['image'])
            ax.set_title(f"Instruction: {results[i]['instruction'][:20]}...\n" +
                         f"GT idx: {results[i]['ground_truth_idx']}, Correct: {results[i]['correct']}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    print("Sample images saved to 'prediction_samples.png'")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='VLA-CLIP Inference')
    parser.add_argument('--model_path', type=str, default="finetuned_clip_trajectory.pt",
                        help='Path to the trained model file')
    parser.add_argument('--trajectory_mode', type=bool, default=None,
                        help='Whether to use trajectory mode (if None, will be determined from model name)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to test')
    parser.add_argument('--action_pool_size', type=int, default=20,
                        help='Size of the action pool for each test sample')
    parser.add_argument('--libero_path', type=str, default="/home/xilun/LIBERO/libero/datasets",
                        help='Path to LIBERO dataset')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Path to LIBERO dataset
    libero_path = args.libero_path
    
    # Path to trained model
    model_path = args.model_path
    
    # Set trajectory mode based on model type if not explicitly specified
    trajectory_mode = args.trajectory_mode
    if trajectory_mode is None:
        trajectory_mode = "trajectory" in model_path
    
    # Load dataset
    print("Loading dataset...")
    dataset_dict = load_libero_dataset(libero_path)
    
    # Sample and test with a limited action pool
    print(f"\nTesting model on random samples with limited action pool (trajectory mode: {trajectory_mode})...")
    results = sample_and_test(
        dataset_dict, 
        model_path, 
        trajectory_mode=trajectory_mode, 
        num_samples=args.num_samples, 
        action_pool_size=args.action_pool_size
    )
    
    # Display results
    display_results(results)