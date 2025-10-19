import torch
import clip
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm
from model import TextAwareVisualExtraction, ModelConfig
from clip_class import VLA_CLIP_Bridge
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import argparse
import torch.nn.functional as F
from clip.simple_tokenizer import SimpleTokenizer
import json

ACTION_PADDING_VALUE = -5.0 # Define padding value globally

class VLA_CLIP_Bridge_Inference:
    def __init__(self, model_path, history_length, use_transformer=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Load the base CLIP model
        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        
        # --- Configuration specific to trajectory model ---
        self.trajectory_mode = True # Hardcoded for this script version
        self.history_length = history_length
        self.use_transformer = use_transformer
        # --------------------------------------------------
        
        self.tokenizer = SimpleTokenizer()
        
        # Initialize the VLA_CLIP_Bridge model for trajectories
        self.model = self._init_model(model_path, device)
        self.device = device
        
        # Get CLIP's image preprocessing pipeline
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])
        
        # Bridge dataset uses single agent view image
        self.image_key = 'agent_view_image_file'
    
    def _init_model(self, model_path, device):
        # Initialize the trajectory model for Bridge dataset
        # ModelConfig needs history_length and action_dim (7 for Bridge)
        model_config = ModelConfig(clip_model=self.clip_model, history_length=self.history_length, action_dim=7)
        
        # Initialize model using the imported VLA_CLIP_Bridge from finetune_trajectory_bridge_ddp
        model = VLA_CLIP_Bridge(model_config, use_transformer=self.use_transformer).to(device)
        
        # Load trained weights
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()  # Set to evaluation mode
        model.float()  # Ensure model is in float32 precision
        print("Model loaded successfully.")
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

    def predict(self, image, instruction, possible_action_histories):
        """
        Predict the most likely action history given an agent view image and instruction.
        Args:
            image: PIL Image or numpy array (single agent view image)
            instruction: String instruction
            possible_action_histories: List of numpy action history arrays (Shape [H, D])
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
            assert len(instruction) == len(possible_action_histories), "Number of instructions must match number of action histories."
            # Tokenize each instruction and stack into a batch
            text_tokens = clip.tokenize(instruction).to(self.device)
        else:
            text_tokens = clip.tokenize([instruction]).to(self.device)  # Make it batch size 1

        with torch.no_grad():
            history_batch = torch.tensor(np.array(possible_action_histories), dtype=torch.float32).to(self.device)
            num_histories = history_batch.shape[0]
            img_batch = img_tensor.repeat(num_histories, 1, 1, 1)
            # text_tokens should be (num_histories, seq_len)
            if text_tokens.shape[0] != num_histories:
                text_tokens = text_tokens.repeat(num_histories, 1)
            
            # Use the bridge model's forward method
            image_logits, action_logits = self.model(img_batch, text_tokens, history_batch)
            
            # Get similarity scores from the model output
            # For bridge model, we use the image_logits diagonal (similarity between image and its corresponding action)
            scores = torch.diag(image_logits).cpu().numpy()
            
            predicted_idx = scores.argmax()
            predicted_history = possible_action_histories[predicted_idx]
        
        history_scores = {str(i): float(scores[i]) for i in range(len(scores))}
        return predicted_history, history_scores

    def get_history_score(self, image, instruction, action_history):
        """
        Calculates the VLA-CLIP cosine similarity score between an agent view image/instruction
        and a given action history by leveraging the model's forward method.
        Args:
            image: PIL Image or numpy array (single agent view image)
            instruction: String instruction.
            action_history: Numpy array action history (H, D), potentially padded.
        Returns:
            score: Float similarity score tensor (on the model's device).
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        if isinstance(instruction, str):
            text_tokens = clip.tokenize([instruction]).to(self.device)
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

def sample_and_test_bridge(bridge_dataset_dict, model_path, history_length, use_transformer=False, num_samples=10, action_pool_size=20, images_folder=None):
    """
    Sample and test on Bridge dataset
    Args:
        bridge_dataset_dict: Dictionary loaded from bridge dataset JSON
        model_path: Path to trained model
        history_length: Action history length
        use_transformer: Whether model uses transformer encoder
        num_samples: Number of samples to test
        action_pool_size: Size of action pool for each test
        images_folder: Path to images folder
    """
    inference_model = VLA_CLIP_Bridge_Inference(model_path,
                                               history_length=history_length,
                                               use_transformer=use_transformer)
    
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
            image, gt_instruction, action_history_pool
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

        # Optional: Print start/end of histories if too long
        gt_str = f"Start: {result['ground_truth_action'][0]}, End: {result['ground_truth_action'][-1]}"
        pred_str = f"Start: {result['prediction'][0]}, End: {result['prediction'][-1]}"
        # print(f"  Ground truth action hist: {gt_str}")
        # print(f"  Predicted action hist:  {pred_str}")

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
    # This block is for running vla_clip_inference_bridge.py directly for evaluation.
    
    parser = argparse.ArgumentParser(description='VLA-CLIP Bridge Dataset Inference')
    # Model and Training Params
    parser.add_argument('--model_path', type=str, 
                       default='/root/vla-clip-bridge/clip_verifier/scripts/bridge_rephrases_epoch_20.pt',
                       help='Path to the trained bridge model file (.pt)')
    parser.add_argument('--history_length', type=int, default=10,
                       help='Action history length used during training (must match dataset)')
    parser.add_argument('--use_transformer', action='store_true', 
                       help='Specify if the loaded model uses a Transformer action encoder')

    # Dataset
    parser.add_argument('--bridge_dataset', type=str, 
                       default='/root/vla-clip-bridge/10episodes.json',
                       help='Path to the bridge dataset .json file')
    parser.add_argument('--images_folder', type=str,
                       default='/root/vla-clip-bridge/10episodes_imgs',
                       help='Path to folder containing agent view images')

    # Evaluation Params
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to test for evaluation')
    parser.add_argument('--action_pool_size', type=int, default=20,
                       help='Size of the action history pool (including GT) for each test sample')

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
    print(f"\nStarting evaluation on Bridge dataset...")
    results = sample_and_test_bridge(
        bridge_dataset_dict=dataset_dict,
        model_path=args.model_path,
        history_length=args.history_length,
        use_transformer=args.use_transformer,
        num_samples=args.num_samples,
        action_pool_size=args.action_pool_size,
        images_folder=args.images_folder
    )

    # Display results
    display_results(results)