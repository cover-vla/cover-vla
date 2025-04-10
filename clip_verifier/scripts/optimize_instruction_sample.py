import torch
import clip
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# 2. Load and preprocess image
image = Image.open("/home/xilun/vla-clip/openvla/original_image_online_predict.png")
image_tensor = preprocess(image).unsqueeze(0).to(device)

# 3. Tokenize and extract token embeddings (with gradient)
original_text = "move the red block"
tokenized = clip.tokenize([original_text]).to(device)
original_embedding = clip_model.token_embedding(tokenized)
embedding = original_embedding.detach().clone().requires_grad_(True)

# Constants for optimization
TEMPERATURE = 0.07  # CLIP's default temperature
REGULARIZATION_WEIGHT = 0.1

# 4. Custom text encoder that accepts embeddings
def encode_text_from_embedding(clip_model, embedding):
    x = embedding + clip_model.positional_embedding
    x = x.permute(1, 0, 2)  # (seq_len, batch, dim)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # (batch, seq_len, dim)
    x = clip_model.ln_final(x)
    x = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)]
    return x @ clip_model.text_projection

# 5. Encode image once
with torch.no_grad():
    image_features = clip_model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# 6. Function to get closest tokens
def get_nearest_tokens(embedding):
    # Compute similarity between our embedding and all token embeddings
    token_embeds = clip_model.token_embedding.weight
    similarity = F.cosine_similarity(
        embedding.unsqueeze(2), 
        token_embeds.t().unsqueeze(0).unsqueeze(0), 
        dim=-1
    )
    top_tokens = similarity.argmax(dim=-1)
    return top_tokens

# 7. Optimize instruction embedding
optimizer = torch.optim.Adam([embedding], lr=1e-2)
best_similarity = float('-inf')
best_embedding = None

print(f"Initial text: {original_text}")
for step in range(200):
    optimizer.zero_grad()
    
    # Get text features
    text_features = encode_text_from_embedding(clip_model, embedding)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute CLIP score with temperature
    similarity = (image_features @ text_features.T).squeeze() / TEMPERATURE
    
    # Add regularization to stay close to original embedding
    reg_loss = F.mse_loss(embedding, original_embedding)
    loss = -similarity + REGULARIZATION_WEIGHT * reg_loss
    
    loss.backward()
    optimizer.step()
    
    if similarity.item() > best_similarity:
        best_similarity = similarity.item()
        best_embedding = embedding.detach().clone()
    
    if step % 20 == 0:
        print(f"Step {step:03d} | CLIP Score: {similarity.item():.4f} | Reg Loss: {reg_loss.item():.4f}")

# Print final results
print("\nOptimization complete!")
print(f"Best CLIP score: {best_similarity:.4f}")

# Try to decode the optimized embedding
with torch.no_grad():
    nearest_tokens = get_nearest_tokens(best_embedding)
    # You would need CLIP's vocabulary to convert these indices to actual tokens
    print("Token indices of optimized text:", nearest_tokens.cpu().numpy().tolist())

