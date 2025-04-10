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
import argparse
import torch.nn.functional as F
from clip.simple_tokenizer import SimpleTokenizer



class VLA_CLIP_Inference:
    def __init__(self, model_path, trajectory_mode=True, trajectory_length=20, device="cuda" if torch.cuda.is_available() else "cpu", use_transformer=False):
        # Load the base CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        
        # Set mode (trajectory or single action)
        self.trajectory_mode = trajectory_mode
        self.trajectory_length = trajectory_length
        
        self.tokenizer = SimpleTokenizer()
        
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
    
    def decode_text_embed(self, text_embed, topk=1):
        """
        Decodes optimized text embeddings to the nearest token IDs using top-k sampling.
        Cleans special tokens and stops decoding at <|endoftext|>.
        """
        with torch.no_grad():
            token_embeddings = self.clip_model.token_embedding.weight  # [vocab_size, dim]
            token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)

            text_embed = text_embed.squeeze(0)  # [seq_len, dim]
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

            decoded_token_ids = []

            for token_vec in text_embed:
                similarities = token_vec @ token_embeddings.T  # [vocab_size]
                if topk == 1:
                    token_id = similarities.argmax().item()
                else:
                    topk_vals, topk_idxs = similarities.topk(topk)
                    token_id = topk_idxs[torch.randint(0, topk, (1,)).item()].item()
                decoded_token_ids.append(token_id)

                # Stop if <|endoftext|>
                if token_id == self.tokenizer.encoder.get("<|endoftext|>"):
                    break

            # Remove <|startoftext|> if present at the beginning
            if decoded_token_ids and decoded_token_ids[0] == self.tokenizer.encoder.get("<|startoftext|>"):
                decoded_token_ids = decoded_token_ids[1:]
                
            if decoded_token_ids and decoded_token_ids[-1] == self.tokenizer.encoder.get("<|endoftext|>"):
                decoded_token_ids = decoded_token_ids[:-1]

            # Decode to string
            return self.tokenizer.decode(decoded_token_ids)

    
    def optimize_action_and_instruction(self, 
                                        image, 
                                        instruction, 
                                        num_iterations=500, 
                                        lr=1e-3, 
                                        temperature=0.07, 
                                        reg_weight=0.1,
                                        topk=5):
        """
        Jointly optimize action and instruction to maximize CLIP similarity with a fixed image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                # Tokenize text and get learnable embeddings
        
        tokenized = clip.tokenize(instruction).to(self.device)
        original_text_embed = self.clip_model.token_embedding(tokenized).detach().clone()
        text_embed = original_text_embed + 0.1 * torch.randn_like(original_text_embed)
        text_embed.requires_grad_()


        # Learnable raw action vector (before tanh)
        action_dim = self.model.action_dim
        raw_action = torch.randn(action_dim, device=self.device, requires_grad=True)
        original_raw_action = raw_action.clone().detach()

        # Optimizer over both
        optimizer = torch.optim.Adam([raw_action, text_embed], lr=lr)

        best_similarity = float('-inf')
        best_action = None
        best_text_embed = None

        # Encoder for custom text embedding
        def encode_text_from_embedding(embedding):
            x = embedding + self.clip_model.positional_embedding
            x = x.permute(1, 0, 2)
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.clip_model.ln_final(x)
            x = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)]
            return x @ self.clip_model.text_projection

        for step in range(num_iterations):
            optimizer.zero_grad()

            # Text feature
            text_features = encode_text_from_embedding(text_embed)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Image feature (fixed)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Combined CLIP feature
            patch_features, _ = self.model.extract_clip_features(image_tensor, tokenized)
            text_aware_features = self.model.text_aware_visual_extraction(patch_features, text_features.unsqueeze(1))
            vision_token = self.model.vision_poolings(text_aware_features)
            text_token = self.model.text_pooling(text_features.unsqueeze(1))
            combined_features = torch.cat([text_token, vision_token], dim=-1)
            combined_features = self.model.input_projection(combined_features)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

            # Project action
            action = torch.tanh(raw_action)
            projected_action = self.model.action_projection(action)
            projected_action = projected_action / projected_action.norm(dim=-1, keepdim=True)

            # CLIP similarity score
            similarity = (combined_features @ projected_action.T).squeeze() / temperature

            # Regularization to keep close to original instruction & action
            text_reg = F.mse_loss(text_embed, original_text_embed)
            action_reg = F.mse_loss(raw_action, original_raw_action)
            loss = -similarity + reg_weight * (text_reg + action_reg)

            loss.backward()
            optimizer.step()

            if similarity.item() > best_similarity:
                best_similarity = similarity.item()
                best_action = action.detach().clone().cpu().numpy()
                best_text_embed = text_embed.detach().clone()

            # if step % 100 == 0:
            #     print(f"Step {step:03d} | sim: {similarity.item():.4f}")
                
        # Decode optimized text embedding
        decoded_text = self.decode_text_embed(best_text_embed, topk=topk)

        return best_action, decoded_text, best_similarity


    
    def optimize_action_gradient(self, image, instruction, vla_action, num_iterations=500, lr=1e-3, temperature=0.07, reg_weight=0.1):
        """
        Optimize action vector using gradient-based optimization of CLIP similarity.
        Fixes image and instruction; optimizes action (bounded via tanh) to align with image-text embedding.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        tokenized = clip.tokenize(instruction).to(self.device)

        with torch.no_grad():
            patch_features, text_features = self.model.extract_clip_features(image_tensor, tokenized)
            text_aware_features = self.model.text_aware_visual_extraction(patch_features, text_features)
            vision_token = self.model.vision_poolings(text_aware_features)
            text_token = self.model.text_pooling(text_features)
            combined_features = torch.cat([text_token, vision_token], dim=-1)
            combined_features = self.model.input_projection(combined_features)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

        # Initialize raw action tensor (before tanh)
        raw_action = torch.tensor(vla_action, dtype=torch.float32, device=self.device, requires_grad=True)
        original_raw_action = raw_action.clone().detach()

        optimizer = torch.optim.Adam([raw_action], lr=lr)

        best_similarity = float('-inf')
        best_action = None

        for step in range(num_iterations):
            optimizer.zero_grad()

            # Apply tanh to keep action in [-1, 1]
            action = torch.tanh(raw_action)

            projected_action = self.model.action_projection(action)
            projected_action = projected_action / projected_action.norm(dim=-1, keepdim=True)

            similarity = (combined_features @ projected_action.T).squeeze() / temperature
            reg_loss = F.mse_loss(raw_action, original_raw_action)  # regularize unbounded latent
            loss = -similarity + reg_weight * reg_loss

            loss.backward()
            optimizer.step()
            
            # if step % 10 == 0:
            #     print(f"Step {step:03d} | sim: {similarity.item():.4f}")
                # print(f"Step {step:03d} | action: {action.detach().clone()}")

            if similarity.item() > best_similarity:
                best_similarity = similarity.item()
                best_action = action.detach().cpu().numpy()  # save the *bounded* action

        return best_action, best_similarity


    def online_predict(self, image, instruction, actions, task_description_list=[], softmax=False, beta=0.5):
        """
        Output the image logits for the given image, instruction, and action.
        """

        text_input = clip.tokenize(instruction).to(self.device) if isinstance(instruction, str) else instruction.to(self.device)
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_tensors = []
            for action in actions:
                tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
                action_tensors.append(tensor)
            
            action_batch = torch.stack(action_tensors)
            image_logits, _ = self.model(image_tensor, text_input, action_batch)
            
            scores = image_logits.cpu().numpy()[0]
            if softmax:
                weights = np.exp(scores / beta)
                weights = weights / np.sum(weights)
                sampled_idx = np.random.choice(len(weights), p=weights)
            else:
                sampled_idx = scores.argmax()
            
            predicted_action = actions[sampled_idx]
            predicted_task_description = task_description_list[sampled_idx] if task_description_list else None

        return scores, predicted_action, predicted_task_description
    
    def predict(self, image, instruction, possible_actions, action_history=None):
        """
        Predict the most likely action given an image and instruction
        
        Args:
            image: PIL Image or numpy array
            instruction: String instruction
            possible_actions: List of numerical action arrays or action trajectories
            
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
            text_tokens = self.model.clip.tokenize(instruction).to(self.device)
        else:
            # Assume it's already tokenized
            text_tokens = instruction.to(self.device)
            
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

def sample_and_test(dataset_dict, model_path, trajectory_mode=False, num_samples=10, action_pool_size=20, trajectory_length=20, use_transformer=False):
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
    
    if trajectory_mode:
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
    else:
        samples = []
        action_trajectories = []
        for instruction, data in dataset_dict.items():
            images = data['images']
            actions = data['actions']
            for i, img_seq in enumerate(images):
                for j, frame in enumerate(img_seq):
                    samples.append((frame, instruction, actions[i][j]))
                    action_trajectories.append(actions[i][j])
    
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
            ground_truth_frame, ground_truth_instruction, action_trajectory_pool,
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
        # print(f"Ground truth action: {result['ground_truth'][0]} to {result['ground_truth'][-1]}")
        # print(f"Predicted action: {result['prediction'][0]} to {result['prediction'][-1]}")
        print(f"Ground truth action: {result['ground_truth']}")
        print(f"Predicted action: {result['prediction']}")
        print (f"Score: {result['scores']}")
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
    parser.add_argument('--action_pool_size', type=int, default=2,
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