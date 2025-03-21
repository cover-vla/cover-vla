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
    def __init__(self, model_path, trajectory_mode=True, trajectory_length=20, device="cuda" if torch.cuda.is_available() else "cpu", use_transformer=False):
        # Load the base CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        
        # Set mode (trajectory or single action)
        self.trajectory_mode = trajectory_mode
        self.trajectory_length = trajectory_length
        
        # Initialize the VLA_CLIP model
        self.model = self._init_model(model_path, device, use_transformer)
        self.device = device
        
                # Get CLIP's image preprocessing pipeline
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def _init_model(self, model_path, device, use_transformer):
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
        model = VLA_CLIP(model_config, use_transformer=use_transformer).to(device)
        
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
            possible_actions: List of numerical action arrays or action trajectories
            action_history: Optional action history (for trajectory mode)
            
        Returns:
            predicted_action: The most likely action
            action_scores: Dictionary mapping actions to scores
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize instruction
        if isinstance(instruction, str):
            text_tokens = clip.tokenize(instruction).to(self.device)
        else:
            # Assume it's already tokenized
            text_tokens = instruction.to(self.device)
        
        # Handle trajectory mode
        if self.trajectory_mode:
            # Forward pass through the model
            with torch.no_grad():
                # Convert action trajectories to tensors
                action_tensors = []
                for traj in possible_actions:
                    # Convert to tensor
                    tensor = torch.tensor(traj, dtype=torch.float32).to(self.device)
                    action_tensors.append(tensor)
                
                # Stack tensors into a batch
                action_batch = torch.stack(action_tensors)
                # Run inference with trajectories
                image_logits, _ = self.model(image_tensor, text_tokens, action_batch)
                
                # Get scores from image_logits
                scores = image_logits.cpu().numpy()[0]
                
                # Get predicted action
                predicted_idx = scores.argmax()
                predicted_action = possible_actions[predicted_idx]
        else:
            # Single action mode
            with torch.no_grad():
                # Convert actions to tensors
                action_tensors = []
                for action in possible_actions:
                    # Convert to tensor
                    tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
                    action_tensors.append(tensor)
                
                # Stack tensors into a batch
                action_batch = torch.stack(action_tensors)
                
                # Run inference with single actions
                image_logits, _ = self.model(image_tensor, text_tokens, action_batch)
                
                # Get scores from image_logits
                scores = image_logits.cpu().numpy()[0]
                
                # Get predicted action
                predicted_idx = scores.argmax()
                predicted_action = possible_actions[predicted_idx]
        
        # Create a dictionary of action scores for all actions in the pool
        action_scores = {str(i): float(scores[i]) for i in range(len(scores))}
        
        return predicted_action, action_scores

def load_libero_dataset(libero_base_path):
    """Load the LIBERO dataset from all folders in the datasets directory"""
    dataset_dict = {}
    
    # Get all subdirectories in the datasets folder
    dataset_folders = [f for f in os.listdir(libero_base_path) 
                      if os.path.isdir(os.path.join(libero_base_path, f))]
    dataset_folders = ["libero_spatial"]
    # Process each dataset folder
    for folder in dataset_folders:
        folder_path = os.path.join(libero_base_path, folder)
        print(f"\nLoading dataset folder: {folder}")
        
        # Process each task file in the folder
        for task in os.listdir(folder_path):
            if not task.endswith('.hdf5'):
                continue
                
            # Extract task name without .hdf5 extension as the language instruction
            language_instruction = task.replace('.hdf5', '').replace('_', ' ')
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

def sample_and_test(dataset_dict, model_path, trajectory_mode=True, num_samples=10, action_pool_size=20, trajectory_length=20, use_transformer=False):
    """
    Sample random data points and test the model with a limited action pool
    
    Args:
        dataset_dict: Dictionary containing the dataset
        model_path: Path to the trained model
        trajectory_mode: Whether to use trajectory mode or single action mode
        num_samples: Number of samples to test
        action_pool_size: Size of the action pool for each test sample
        trajectory_length: Length of action trajectories (for trajectory mode)
    """
    # Initialize the inference model
    inference_model = VLA_CLIP_Inference(model_path, trajectory_mode=trajectory_mode, trajectory_length=trajectory_length, use_transformer=use_transformer)
    
    # Create a dataset similar to how it's done in finetune_trajectory.py
    samples = []
    action_trajectories = []
    
    # Flatten the dataset into (image, text, action_trajectory) triplets
    for instruction, data in dataset_dict.items():
        images = data['images']
        actions = data['actions']

        for i, img_seq in enumerate(images):
            for j, frame in enumerate(img_seq):
                # Create action trajectory (past actions)
                action_trajectory = []   
                # Add past actions, padding with zeros if not enough history
                for k in range(j - trajectory_length + 1, j):
                    if k < 0:
                        if j > 0:  # If we have at least one valid action
                            action_trajectory.append(actions[i][0])  # Use the first action
                        else:
                            if k == j - 1:  # Last position before current frame
                                # Use zeros (neutral action) instead of padding token
                                action_trajectory.append(np.zeros(actions[i][0].shape))
                            else:
                                action_trajectory.append(np.ones(actions[i][0].shape) * -5.0)  # Padding for earlier positions
                    else:
                        action_trajectory.append(actions[i][k])
                
                # Add current action to complete the trajectory
                action_trajectory.append(actions[i][j])
                
                # Convert to numpy array for consistency
                action_trajectory = np.array(action_trajectory)
                
                # Verify that not all values are padding
                if np.all(action_trajectory[:, 0] == -5.0):
                    # Force at least one non-padding value
                    action_trajectory[-1] = np.zeros(action_trajectory[-1].shape)
                
                # Add to samples
                samples.append((frame, instruction, action_trajectory))
                action_trajectories.append(action_trajectory)
    
    # Randomly sample test points
    random.seed(42)  # For reproducibility
    sampled_indices = random.sample(range(len(samples)), min(num_samples, len(samples)))
    
    # Test each sampled point
    results = []
    for idx in tqdm(sampled_indices, desc="Testing samples"):
        ground_truth_frame, ground_truth_instruction, ground_truth_action_trajectory = samples[idx]
        
        # Create action pool with ground truth action
        action_trajectory_pool = [ground_truth_action_trajectory]
        
        # Sample additional action trajectories for the pool
        other_trajectories = [a for i, a in enumerate(action_trajectories) 
                             if i != idx and not np.array_equal(a, ground_truth_action_trajectory)]
        
        if other_trajectories and action_pool_size > 1:
            sampled_trajectories = random.sample(other_trajectories, min(action_pool_size - 1, len(other_trajectories)))
            action_trajectory_pool.extend(sampled_trajectories)
        
        # Shuffle the action pool
        random.shuffle(action_trajectory_pool)
        
        # Find index of ground truth in the shuffled action pool
        ground_truth_idx = None
        for i, traj in enumerate(action_trajectory_pool):
            if np.array_equal(traj, ground_truth_action_trajectory):
                ground_truth_idx = i
                break
        
        # Run prediction
        predicted_action, scores = inference_model.predict(
            ground_truth_frame, ground_truth_instruction, action_trajectory_pool, action_history=ground_truth_action_trajectory
        )
        
        # Check if prediction is correct
        is_correct = np.array_equal(predicted_action, ground_truth_action_trajectory)
        
        results.append({
            'image': ground_truth_frame,
            'instruction': ground_truth_instruction,
            'ground_truth': ground_truth_action_trajectory,
            'ground_truth_idx': ground_truth_idx,  # This is now the index in the shuffled pool
            'prediction': predicted_action,
            'action_pool_size': len(action_trajectory_pool),
            'scores': scores,
            'correct': is_correct,
        })
    
    return results

def display_results(results):
    """Display the results of the sample_and_test function"""
    correct = 0
    ranks = []
    
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"Instruction: {result['instruction']}")
        print(f"Action pool size: {result['action_pool_size']}")
        print(f"Ground truth index: {result['ground_truth_idx']}")
        print(f"Ground truth action: {result['ground_truth'][-1]}")
        print(f"Predicted action: {result['prediction'][-1]}")
        print(f"Correct: {result['correct']}")
        
        # Calculate L2 distance between prediction and ground truth
        if isinstance(result['prediction'], np.ndarray) and isinstance(result['ground_truth'], np.ndarray):
            # Flatten arrays for distance calculation
            pred_flat = result['prediction'].flatten()
            gt_flat = result['ground_truth'].flatten()
            
            # Ensure same length
            min_len = min(len(pred_flat), len(gt_flat))
            l2_dist = np.linalg.norm(pred_flat[:min_len] - gt_flat[:min_len])
            print(f"L2 distance: {l2_dist:.4f}")
        
        # Display top predictions
        scores = result['scores']
        sorted_scores = sorted([(int(k), v) for k, v in scores.items()], key=lambda x: x[1], reverse=True)
        print("Top predictions (index: score):")
        for idx, score in sorted_scores[:5]:  # Show top 5 predictions
            print(f"  {idx}: {score:.4f}")
        
        # Calculate rank of ground truth
        if result['ground_truth_idx'] is not None:
            gt_idx_str = str(result['ground_truth_idx'])
            if gt_idx_str in scores:
                gt_score = scores[gt_idx_str]
                rank = sum(1 for _, score in scores.items() if float(score) > gt_score) + 1
                ranks.append(rank)
        
        if result['correct']:
            correct += 1
    
    # Calculate overall accuracy and mean rank
    accuracy = correct / len(results) if results else 0
    mean_rank = sum(ranks) / len(ranks) if ranks else 0
    
    print(f"\nOverall accuracy: {accuracy:.2f} ({correct}/{len(results)})")
    print(f"Mean rank of ground truth: {mean_rank:.2f}")
    
    # # Optional: Display some images with predictions
    # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # axes = axes.flatten()
    
    # for i, ax in enumerate(axes):
    #     if i < len(results):
    #         ax.imshow(results[i]['image'])
    #         ax.set_title(f"Instruction: {results[i]['instruction'][:20]}...\n" +
    #                      f"GT idx: {results[i]['ground_truth_idx']}, Correct: {results[i]['correct']}")
    #         ax.axis('off')
    
    # plt.tight_layout()
    # plt.savefig('prediction_samples.png')
    # print("Sample images saved to 'prediction_samples.png'")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='VLA-CLIP Inference')
    parser.add_argument('--model_path', type=str, default="finetuned_clip_trajectory.pt",
                        help='Path to the trained model file')
    parser.add_argument('--trajectory_mode', action='store_true',
                        help='Whether to use trajectory mode')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of samples to test')
    parser.add_argument('--action_pool_size', type=int, default=30,
                        help='Size of the action pool for each test sample')
    parser.add_argument('--libero_path', type=str, default="/home/xilun/LIBERO/libero/datasets",
                        help='Path to LIBERO dataset')
    parser.add_argument('--augmented_path', type=str, default=None,
                        help='Path to the augmented dataset pickle file')
    parser.add_argument('--trajectory_length', type=int, default=20,
                        help='Length of action trajectories (for trajectory mode)')
    parser.add_argument('--use_transformer', action='store_true',
                        help='Whether to use the transformer model')
    
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
    if args.augmented_path:
        # Load augmented dataset from pickle file
        import pickle
        with open(args.augmented_path, 'rb') as f:
            dataset_dict = pickle.load(f)
        print(f"Loaded augmented dataset from {args.augmented_path}")

    else:
        # Load original LIBERO dataset
        dataset_dict = load_libero_dataset(libero_path)
    
    # Sample and test with a limited action pool
    print(f"\nTesting model on random samples with limited action pool (trajectory mode: {trajectory_mode})...")
    results = sample_and_test(
        dataset_dict, 
        model_path, 
        trajectory_mode=trajectory_mode, 
        num_samples=args.num_samples, 
        action_pool_size=args.action_pool_size,
        trajectory_length=args.trajectory_length,
        use_transformer=args.use_transformer
    )
    
    # Display results
    display_results(results)