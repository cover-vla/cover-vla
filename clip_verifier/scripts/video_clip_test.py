# pip install transformers==4.52 av decord torch --upgrade
import av, torch, numpy as np, os
from transformers import AutoProcessor, AutoModel, CLIPTokenizer, CLIPTextModel
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1) TEXT EMBEDDING (from filenames) -----------------------
def extract_text_from_filename(filename):
    match = re.search(r'task=([^.]*)\\.mp4$', filename)
    if not match:
        match = re.search(r'task=([^.]*)\.mp4$', filename)
    if match:
        return match.group(1).replace('_', ' ')
    else:
        return None

mp4_dir = "/home/xilun/vla-clip/openvla/rollouts/no_transform_1"
mp4_files = [f for f in os.listdir(mp4_dir) if f.endswith('.mp4')]
text_prompts_set = set()
for fname in mp4_files:
    text = extract_text_from_filename(fname)
    if text:
        text_prompts_set.add(text)
text_prompts = sorted(list(text_prompts_set))

print("Text prompts:")
for t in text_prompts:
    print(t)

tok  = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
txtm = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
text_inputs = tok(text_prompts, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_embs = txtm(**text_inputs).pooler_output

# --- 2) VIDEO FRAMES ---------------------------------------------------------
def sample_8_frames(path):
    container = av.open(path); num = container.streams.video[0].frames
    idx = np.linspace(0, num-1, 8).astype(int)
    frames = [f.to_ndarray(format="rgb24") for i,f in enumerate(container.decode(video=0)) if i in idx]
    return frames

# Use the first video file for the video embedding
def get_first_video_path():
    for i, f in enumerate(sorted(os.listdir(mp4_dir))):
        if f.endswith('.mp4') and i > 5:
            print (f"Using video: {f}")
            return os.path.join(mp4_dir, f)
    raise RuntimeError("No mp4 files found")

video_path = get_first_video_path()
frames = sample_8_frames(video_path)

# --- 3) VIDEO EMBEDDING with X-CLIP -----------------------------------------
proc  = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
xclip = AutoModel.from_pretrained("microsoft/xclip-base-patch32").to(device).eval()

inputs = proc(videos=list(frames), return_tensors="pt").to(device)
with torch.no_grad():
    vid_emb = xclip.get_video_features(**inputs)

# --- 4) SIMILARITY -----------------------------------------------------------
sims = torch.nn.functional.cosine_similarity(vid_emb, text_embs)
for i, (sim, text) in enumerate(zip(sims, text_prompts)):
    print(f"{i+1:2d}. cos-sim = {sim.item():.3f} | {text}")
