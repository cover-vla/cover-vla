import torch
import clip
from PIL import Image

# Load the model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load an image and a set of text prompts
image = preprocess(Image.open("OpenVLA_img.png")).unsqueeze(0).to(device)
texts = clip.tokenize(["lift carrot", "lift vegetable"]).to(device)
print (image.shape)
# Compute image and text features
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)

# Compute similarity
image_similarity = (image_features @ text_features.T).softmax(dim=-1)
text_similarity = (text_features @ image_features.T).softmax(dim=-1)
print("Image Similarity Scores:", image_similarity.cpu().numpy())
print("Text Similarity Scores:", text_similarity.cpu().numpy())
