import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm
from model import TextAwareVisualExtraction, ModelConfig
from finetune_trajectory_bridge_ddp import VLA_SigLIP2_Bridge
import argparse
import json

ACTION_PADDING_VALUE = -5.0  # Define padding value globally


class VLA_SigLIP_Bridge_Inference:
    def __init__(self, model_path, history_length, backbone='hf-hub:timm/ViT-L-16-SigLIP2-384', 
                 use_transformer=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize VLA-SigLIP inference model for Bridge dataset
        
        Args:
            model_path: Path to trained model weights
            history_length: Action history length used during training
            backbone: SigLIP2 model backbone (must match training, default: ViT-L-16-SigLIP2-384 ~1B params)
            use_transformer: Whether model uses transformer encoder
            device: Device to run inference on
        """
        self.device = device
        self.history_length = history_length
        self.use_transformer = use_transformer
        self.backbone = backbone
        
        # Load the base SigLIP2 model and preprocessing
        print(f"Loading SigLIP2 model: {backbone}")
        self.siglip_model, self.preprocess = create_model_from_pretrained(backbone)
        self.siglip_model = self.siglip_model.to(device)
        # Note: VLA_SigLIP2_Bridge constructor will handle freezing and bf16 conversion
        
        # Load the tokenizer
        self.tokenizer = get_tokenizer(backbone)
        
        # Initialize the VLA_SigLIP2_Bridge model
        self.model = self._init_model(model_path, device)
        
        # Bridge dataset uses single agent view image
        self.image_key = 'agent_view_image_file'
    
    def _init_model(self, model_path, device):
        """Initialize the trajectory model for Bridge dataset"""
        # ModelConfig needs history_length and action_dim (7 for Bridge)
        model_config = ModelConfig(
            clip_model=self.siglip_model, 
            history_length=self.history_length, 
            action_dim=7
        )
        
        # Initialize model using VLA_SigLIP2_Bridge from training script
        # Note: VLA_SigLIP2_Bridge constructor already sets frozen encoder to bf16
        model = VLA_SigLIP2_Bridge(model_config, use_transformer=self.use_transformer).to(device)
        
        # Load trained weights (precision is preserved from checkpoint)
        print(f"Loading model weights from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Handle both full checkpoint and state_dict only formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"Loaded checkpoint from epoch {epoch}")
                
                # Display additional checkpoint information if available
                if 'train_loss' in checkpoint and 'val_loss' in checkpoint:
                    print(f"  Training loss: {checkpoint['train_loss']:.4f}")
                    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
                
                if 'train_img2act_top1_acc' in checkpoint:
                    print(f"  Training img2act accuracy: {checkpoint['train_img2act_top1_acc']:.3f}")
                if 'val_img2act_top1_acc' in checkpoint:
                    print(f"  Validation img2act accuracy: {checkpoint['val_img2act_top1_acc']:.3f}")
                
                if 'best_val_loss' in checkpoint:
                    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
                
                if 'global_step' in checkpoint:
                    print(f"  Global step: {checkpoint['global_step']}")
            else:
                # Assume it's just a state_dict (legacy format)
                print("Loading legacy checkpoint format (model weights only)")
                model.load_state_dict(checkpoint)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        model.eval()  # Set to evaluation mode
        # Precision: Frozen encoder in bf16, trainable parts in fp32 (as preserved from checkpoint)
        return model
    
    def predict(self, image, instruction, possible_action_histories, batch_size=None):
        """
        Predict the most likely action history given an agent view image and instruction.
        Supports both single input and batch processing for efficiency.
        
        Args:
            image: PIL Image or numpy array (single agent view image)
            instruction: String instruction or list of instructions
            possible_action_histories: List of numpy action history arrays (Shape [H, D])
            batch_size: Optional batch size for processing histories. If None, processes all at once.
            
        Returns:
            predicted_history: The most likely action history (numpy array)
            history_scores: Dictionary mapping history index (str) to score (float)
        """
        # Preprocess the image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize instruction(s)
        if isinstance(instruction, list):
            assert len(instruction) == len(possible_action_histories), \
                "Number of instructions must match number of action histories."
            # Tokenize each instruction
            text_tokens = self.tokenizer(instruction, context_length=self.siglip_model.context_length).to(self.device)
        else:
            # Single instruction - tokenize once
            text_tokens = self.tokenizer([instruction], context_length=self.siglip_model.context_length).to(self.device)
        
        with torch.no_grad():
            num_histories = len(possible_action_histories)
            
            # Process in batches if batch_size is specified
            if batch_size is not None and num_histories > batch_size:
                all_scores = []
                for i in range(0, num_histories, batch_size):
                    end_idx = min(i + batch_size, num_histories)
                    batch_histories = possible_action_histories[i:end_idx]
                    
                    history_batch = torch.tensor(np.array(batch_histories), dtype=torch.float32).to(self.device)
                    batch_len = history_batch.shape[0]
                    img_batch = img_tensor.repeat(batch_len, 1, 1, 1)
                    
                    # Handle text tokens for this batch
                    if text_tokens.shape[0] == 1:
                        text_batch = text_tokens.repeat(batch_len, 1)
                    else:
                        text_batch = text_tokens[i:end_idx]
                    
                    # Forward pass
                    image_logits, _ = self.model(img_batch, text_batch, history_batch)
                    batch_scores = torch.diag(image_logits).cpu().numpy()
                    all_scores.append(batch_scores)
                
                scores = np.concatenate(all_scores)
            else:
                # Process all at once
                history_batch = torch.tensor(np.array(possible_action_histories), dtype=torch.float32).to(self.device)
                img_batch = img_tensor.repeat(num_histories, 1, 1, 1)
                
                # Repeat text tokens if needed
                if text_tokens.shape[0] != num_histories:
                    text_tokens = text_tokens.repeat(num_histories, 1)
                
                # Use the bridge model's forward method
                image_logits, action_logits = self.model(img_batch, text_tokens, history_batch)
                
                # Get similarity scores from the model output
                scores = torch.diag(image_logits).cpu().numpy()
            
            predicted_idx = scores.argmax()
            predicted_history = possible_action_histories[predicted_idx]
        
        history_scores = {str(i): float(scores[i]) for i in range(len(scores))}
        return predicted_history, history_scores
    
    def get_history_score(self, image, instruction, action_history):
        """
        Calculates the VLA-SigLIP cosine similarity score between an agent view image/instruction
        and a given action history by leveraging the model's forward method.
        
        Args:
            image: PIL Image or numpy array (single agent view image)
            instruction: String instruction
            action_history: Numpy array action history (H, D), potentially padded
            
        Returns:
            score: Float similarity score tensor (on the model's device)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        if isinstance(instruction, str):
            text_tokens = self.tokenizer([instruction], context_length=self.siglip_model.context_length).to(self.device)
        else:
            text_tokens = instruction.to(self.device)
            if text_tokens.ndim == 1:
                text_tokens = text_tokens.unsqueeze(0)
        
        history_tensor = torch.tensor(action_history, dtype=torch.float32).unsqueeze(0).to(self.device)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)
        
        with torch.no_grad():
            image_logits, action_logits = self.model(img_tensor, text_tokens, history_tensor)
            # Return the similarity score (diagonal element)
            score = image_logits[0, 0]
        
        return score


def sample_and_test_bridge(bridge_dataset_dict, model_path, history_length, backbone='hf-hub:timm/ViT-L-16-SigLIP2-384',
                           use_transformer=False, num_samples=10, action_pool_size=20, images_folder=None, inference_batch_size=None):
    """
    Sample and test on Bridge dataset with SigLIP model
    
    Args:
        bridge_dataset_dict: Dictionary loaded from bridge dataset JSON
        model_path: Path to trained model
        history_length: Action history length
        backbone: SigLIP2 model backbone
        use_transformer: Whether model uses transformer encoder
        num_samples: Number of samples to test
        action_pool_size: Size of action pool for each test
        images_folder: Path to images folder
        inference_batch_size: Batch size for processing action histories (None = process all at once)
    """
    inference_model = VLA_SigLIP_Bridge_Inference(
        model_path,
        history_length=history_length,
        backbone=backbone,
        use_transformer=use_transformer
    )
    
    # Extract samples from bridge dataset format
    action_histories = bridge_dataset_dict['action_histories']
    instructions = bridge_dataset_dict['instructions']
    samples = bridge_dataset_dict['samples']
    
    print("Processing bridge dataset samples...")
    all_samples = []
    all_histories = []
    
    for sample in tqdm(samples, desc="Processing samples"):
        action_history_id = sample.get('action_history_id')
        instruction_id = sample.get('instruction_id')
        agent_view_image_file = sample.get('agent_view_image_file')
        
        if not all([action_history_id, instruction_id, agent_view_image_file]):
            continue
            
        # Get action history and instruction
        action_hist = np.array(action_histories[action_history_id])
        instruction = instructions[instruction_id]
        
        # Load image
        if images_folder:
            image_path = os.path.join(images_folder, agent_view_image_file)
        else:
            # Default to 10episodes_imgs folder
            image_path = os.path.join(os.path.dirname(model_path), '../../10episodes_imgs', agent_view_image_file)
        
        if not os.path.exists(image_path):
            continue
            
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        all_samples.append((image, instruction, action_hist))
        all_histories.append(action_hist)
    
    if not all_samples:
        print("Error: No valid samples found in the dataset.")
        return []
    
    if not all_histories:
        print("Error: No valid action histories found for pooling.")
        return []
    
    print(f"Total samples for testing: {len(all_samples)}")
    print(f"Total histories for pooling: {len(all_histories)}")
    
    random.seed(42)
    sampled_indices = random.sample(range(len(all_samples)), min(num_samples, len(all_samples)))
    results = []
    
    for idx in tqdm(sampled_indices, desc="Testing samples"):
        image, gt_instruction, gt_action_hist = all_samples[idx]
        
        # Create action history pool
        action_history_pool = [gt_action_hist]
        gt_pos_hist_added = True
        num_needed = action_pool_size - len(action_history_pool)
        
        if num_needed > 0:
            candidate_pool = [h for h in all_histories if not np.array_equal(h, gt_action_hist)]
            if len(candidate_pool) > 0:
                num_to_sample = min(num_needed, len(candidate_pool))
                sampled_histories = random.sample(candidate_pool, num_to_sample)
                action_history_pool.extend(sampled_histories)
            else:
                print(f"Warning: Not enough unique histories in dataset to fill pool for sample {idx}. Pool size: {len(action_history_pool)}")
        
        random.shuffle(action_history_pool)
        
        # Find ground truth index in pool
        ground_truth_idx_in_pool = None
        for i, hist in enumerate(action_history_pool):
            if np.array_equal(hist, gt_action_hist):
                ground_truth_idx_in_pool = i
                break
        
        if ground_truth_idx_in_pool is None and gt_pos_hist_added:
            print(f"Warning: Ground truth action history lost during pooling/shuffling for sample {idx}? This shouldn't happen.")
        
        # Make prediction
        predicted_history, scores = inference_model.predict(
            image, gt_instruction, action_history_pool, batch_size=inference_batch_size
        )
        
        is_correct = np.array_equal(predicted_history, gt_action_hist)
        
        results.append({
            'image': image,
            'instruction': gt_instruction,
            'ground_truth_action': gt_action_hist,
            'ground_truth_idx': ground_truth_idx_in_pool,
            'prediction': predicted_history,
            'action_pool_size': len(action_history_pool),
            'scores': scores,
            'correct': is_correct,
        })
    
    return results


def display_results(results):
    """Display the results of the sample_and_test function"""
    correct = 0
    ranks = []
    l2_distances = []

    print("\n--- Evaluation Results ---")
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Instruction: {result['instruction']}")
        print(f"  Action pool size: {result['action_pool_size']}")
        print(f"  Ground truth action index in pool: {result['ground_truth_idx']}")

        print(f"  Correct (Predicted == GT Action): {result['correct']}")

        # Calculate L2 distance between prediction and ground truth action history
        if isinstance(result['prediction'], np.ndarray) and isinstance(result['ground_truth_action'], np.ndarray):
            pred_flat = result['prediction'].flatten()
            gt_flat = result['ground_truth_action'].flatten()
            l2_dist = np.linalg.norm(pred_flat - gt_flat)
            print(f"  L2 distance (Pred vs GT Action): {l2_dist:.4f}")
            l2_distances.append(l2_dist)

        # Display top predictions from the pool
        scores = result['scores']
        try:
            # Convert string keys to int for sorting if necessary
            scores_int_keys = {int(k): v for k, v in scores.items()}
            sorted_scores = sorted(scores_int_keys.items(), key=lambda item: item[1], reverse=True)
            print("  Top predictions (pool_index: score):")
            for pool_idx, score in sorted_scores[:5]:
                print(f"    {pool_idx}: {score:.4f}")

            # Calculate rank of ground truth action history
            if result['ground_truth_idx'] is not None:
                gt_score = scores_int_keys.get(result['ground_truth_idx'], -float('inf'))
                rank = sum(1 for score in scores_int_keys.values() if score > gt_score) + 1
                ranks.append(rank)
                print(f"  Rank of Ground Truth Action: {rank}")

        except ValueError:
            print("  Error processing scores for ranking (invalid format?).")

        if result['correct']:
            correct += 1

    # Calculate overall accuracy and mean rank
    accuracy = correct / len(results) if results else 0
    mean_rank = np.mean(ranks) if ranks else float('nan')
    mean_l2 = np.mean(l2_distances) if l2_distances else float('nan')

    print("-" * 25)
    print(f"Overall accuracy: {accuracy:.3f} ({correct}/{len(results)})")
    print(f"Mean rank of ground truth action history: {mean_rank:.3f}")
    print(f"Mean L2 distance (Prediction vs GT Action): {mean_l2:.4f}")
    print("-" * 25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VLA-SigLIP Bridge Dataset Inference')
    
    # Model and Training Params
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained bridge model file (.pt)')
    parser.add_argument('--backbone', type=str, default='hf-hub:timm/ViT-L-16-SigLIP2-384',
                       help='SigLIP2 model backbone (must match training, default: ViT-L-16-SigLIP2-384 ~1B params)')
    parser.add_argument('--history_length', type=int, default=10,
                       help='Action history length used during training (must match dataset)')
    parser.add_argument('--use_transformer', action='store_true', 
                       help='Specify if the loaded model uses a Transformer action encoder')

    # Dataset
    parser.add_argument('--bridge_dataset', type=str, required=True,
                       help='Path to the bridge dataset .json file')
    parser.add_argument('--images_folder', type=str, required=True,
                       help='Path to folder containing agent view images')

    # Evaluation Params
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to test for evaluation')
    parser.add_argument('--action_pool_size', type=int, default=20,
                       help='Size of the action history pool (including GT) for each test sample')
    parser.add_argument('--inference_batch_size', type=int, default=None,
                       help='Batch size for processing action histories during inference (default: None, process all at once)')

    args = parser.parse_args()

    # Load bridge dataset
    print(f"Loading bridge dataset: {args.bridge_dataset}")
    if not os.path.exists(args.bridge_dataset):
        print(f"Error: Dataset file not found at {args.bridge_dataset}")
        exit(1)
    
    try:
        with open(args.bridge_dataset, 'r') as f:
            dataset_dict = json.load(f)
        print(f"Loaded bridge dataset with {len(dataset_dict.get('samples', []))} samples.")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit(1)

    # Verify images folder
    if not os.path.exists(args.images_folder):
        print(f"Error: Images folder not found at {args.images_folder}")
        exit(1)

    # Verify model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)

    # Run evaluation
    print(f"\nStarting evaluation on Bridge dataset with SigLIP...")
    print(f"Backbone: {args.backbone}")
    print(f"History length: {args.history_length}")
    print(f"Use transformer: {args.use_transformer}")
    if args.inference_batch_size:
        print(f"Inference batch size: {args.inference_batch_size}")
    else:
        print(f"Inference batch size: Process all at once")
    
    results = sample_and_test_bridge(
        bridge_dataset_dict=dataset_dict,
        model_path=args.model_path,
        history_length=args.history_length,
        backbone=args.backbone,
        use_transformer=args.use_transformer,
        num_samples=args.num_samples,
        action_pool_size=args.action_pool_size,
        images_folder=args.images_folder,
        inference_batch_size=args.inference_batch_size
    )

    # Display results
    display_results(results)

