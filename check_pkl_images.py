import pickle
import numpy as np
from PIL import Image
import os

# Path to the pickle file - you'll need to specify the correct path
pkl_path = "clip_verifier/augmented_datasets/libero_spatial_all_diverse.pkl"  # Update this path

# Check if file exists
if os.path.exists(pkl_path):
    print(f"Loading pickle file: {pkl_path}")
    
    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded data type: {type(data)}")
    print(f"Number of instructions: {len(data)}")
    
    # Get the first instruction and its samples
    first_instruction = list(data.keys())[0]
    print(f"First instruction: {first_instruction}")
    
    first_data = data[first_instruction]
    print(f"First data keys: {first_data.keys()}")
    
    if 'samples' in first_data:
        samples = first_data['samples']
        print(f"Number of samples: {len(samples)}")
        
        if len(samples) > 0:
            first_sample = samples[0]
            print(f"First sample keys: {first_sample.keys()}")
            
            if 'images' in first_sample:
                images = first_sample['images']
                print(f"Image keys: {list(images.keys())}")
                
                # Save agentview image as example
                if 'agentview_rgb' in images:
                    agentview_img = images['agentview_rgb']
                    print(f"Agentview image shape: {agentview_img.shape}")
                    print(f"Agentview image dtype: {agentview_img.dtype}")
                    print(f"Agentview image value range: {agentview_img.min()} to {agentview_img.max()}")
                    
                    # Convert to PIL Image and save
                    if agentview_img.dtype != np.uint8:
                        # Normalize to 0-255 if needed
                        if agentview_img.max() <= 1.0:
                            agentview_img = (agentview_img * 255).astype(np.uint8)
                        else:
                            agentview_img = agentview_img.astype(np.uint8)
                    
                    pil_img = Image.fromarray(agentview_img)
                    output_path = "example_agentview_image.png"
                    pil_img.save(output_path)
                    print(f"Saved example agentview image to: {output_path}")
                    print(f"Saved image dimensions: {pil_img.size}")
                
                # Save handview image as example
                if 'eye_in_hand_rgb' in images:
                    handview_img = images['eye_in_hand_rgb']
                    print(f"Handview image shape: {handview_img.shape}")
                    print(f"Handview image dtype: {handview_img.dtype}")
                    print(f"Handview image value range: {handview_img.min()} to {handview_img.max()}")
                    
                    # Convert to PIL Image and save
                    if handview_img.dtype != np.uint8:
                        # Normalize to 0-255 if needed
                        if handview_img.max() <= 1.0:
                            handview_img = (handview_img * 255).astype(np.uint8)
                        else:
                            handview_img = handview_img.astype(np.uint8)
                    
                    pil_img = Image.fromarray(handview_img)
                    output_path = "example_handview_image.png"
                    pil_img.save(output_path)
                    print(f"Saved example handview image to: {output_path}")
                    print(f"Saved image dimensions: {pil_img.size}")
                    
else:
    print(f"Pickle file not found: {pkl_path}")
    print("Available pickle files in clip_verifier/augmented_datasets/:")
    if os.path.exists("clip_verifier/augmented_datasets/"):
        for file in os.listdir("clip_verifier/augmented_datasets/"):
            if file.endswith('.pkl'):
                print(f"  - {file}") 