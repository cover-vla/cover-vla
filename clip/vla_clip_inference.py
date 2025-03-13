import torch
import clip
from PIL import Image
import h5py
import os
import numpy as np
import random
from tqdm import tqdm
from model import TextAwareVisualExtraction
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt
import cv2
class VLA_CLIP_Inference:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Load the base CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        
        # Initialize the VLA_CLIP model
        self.model = self._init_model(model_path, device)
        self.device = device
    
    def _init_model(self, model_path, device):
        # Initialize the full VLA_CLIP model
        from finetune import VLA_CLIP
        model = VLA_CLIP(self.clip_model).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()  # Set to evaluation mode
        model.float()  # Ensure model is in float32 precision
        return model
    
    def predict(self, image, instruction, possible_actions):
        """
        Predict the most likely action given an image and instruction
        
        Args:
            image: PIL Image or numpy array
            instruction: String instruction
            possible_actions: List of possible action strings or arrays
            
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
        
        # Convert image tensor to float32 to match expected type
        image_tensor = image_tensor.float()
        
        # Forward pass through the model
        with torch.no_grad():
            # Convert model to float32 to ensure consistent precision
            self.model.float()
            
            # Process actions to match the format expected by the model
            # Make sure all actions are in a consistent format (list or numpy array)
            processed_actions = []
            for action in possible_actions:
                if isinstance(action, np.ndarray):
                    processed_actions.append(action.tolist())
                elif isinstance(action, str):
                    # Try to convert string representation of list back to list
                    try:
                        # Handle string representations of lists
                        if action.startswith('[') and action.endswith(']'):
                            processed_actions.append(eval(action))
                        else:
                            # If it's just a text action, encode it as a simple vector
                            # This is a fallback and might not work well with the model
                            processed_actions.append([float(ord(c))/255 for c in action[:10]])
                    except:
                        # Fallback for pure text actions
                        processed_actions.append([0.0] * 7)  # Default size matching other actions
                else:
                    processed_actions.append(action)
            
            # Make sure all actions have the same length
            max_len = max(len(a) for a in processed_actions)
            for i in range(len(processed_actions)):
                if len(processed_actions[i]) < max_len:
                    processed_actions[i] = processed_actions[i] + [0.0] * (max_len - len(processed_actions[i]))
            
            # Pass the processed actions directly to the model
            image_logits, action_logits = self.model(image_tensor, text_tokens, processed_actions)
            
            # Get scores from image_logits (image to action matching)
            scores = image_logits.cpu().numpy()[0]  # First batch item
            
            # Get predicted action
            predicted_idx = scores.argmax()
            predicted_action = possible_actions[predicted_idx]
        
        # Create a dictionary of action scores
        action_score_dict = {str(action): float(score) for action, score in zip(possible_actions, scores)}
        
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

def sample_and_test(dataset_dict, model_path, num_samples=10):
    """Sample random data points and test the model"""
    # Initialize the inference model
    inference_model = VLA_CLIP_Inference(model_path)
    
    # Flatten the dataset into (image, instruction, action) tuples
    samples = []
    for instruction, data in dataset_dict.items():
        images = data['images']
        actions = data['actions']
        for i, img_seq in enumerate(images):
            for j, frame in enumerate(img_seq):
                samples.append((frame, instruction, actions[i][j]))
    
    # Randomly sample data points
    random.seed(42)  # For reproducibility
    test_samples = random.sample(samples, min(num_samples, len(samples)))
    
    # Get actions from the dataset
    all_actions = []
    for _, _, action in samples:
        all_actions.append(action)
    
    # Test each sample
    results = []
    for image, instruction, ground_truth in tqdm(test_samples, desc="Testing samples"):
        predicted_action, scores = inference_model.predict(image, instruction, all_actions)
        results.append({
            'image': image,
            'instruction': instruction,
            'ground_truth': ground_truth,
            'prediction': predicted_action,
            'scores': scores
        })
    
    return results

def display_results(results):
    """Display the test results"""
    correct = 0
    
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"Instruction: {result['instruction']}")
        print(f"Ground truth: {result['ground_truth']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Correct: {result['ground_truth'] == result['prediction']}")
        
        # Print top 3 scores
        sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
        print("Top predictions:")
        for action, score in sorted_scores[:3]:
            print(f"  {action}: {score:.4f}")
        
        if np.linalg.norm(result['ground_truth'] - result['prediction']) < 0.1:
            correct += 1
    
    accuracy = correct / len(results)
    print(f"\nOverall accuracy: {accuracy:.2f} ({correct}/{len(results)})")
    
    # Optional: Display some images with predictions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(results):
            ax.imshow(cv2.rotate(results[i]['image'], cv2.ROTATE_180))
            ax.set_title(f"GT: {results[i]['ground_truth']}\nPred: {results[i]['prediction']}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    print("Sample images saved to 'prediction_samples.png'")

if __name__ == "__main__":
    # Path to LIBERO dataset
    libero_path = "/home/xilun/LIBERO/libero/datasets"
    
    # Path to trained model
    model_path = "finetuned_clip.pt"
    
    # Load dataset
    print("Loading dataset...")
    dataset_dict = load_libero_dataset(libero_path)
    
    # Sample and test
    print("\nTesting model on random samples...")
    results = sample_and_test(dataset_dict, model_path, num_samples=2)
    
    # Display results
    display_results(results)