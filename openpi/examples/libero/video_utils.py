import time
import os
import imageio
from PIL import Image, ImageDraw
import numpy as np
import json
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
    
    
def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, 
                       captions=None, transform = None, task_suite_name = None):
    
    rollout_dir = f"./examples/libero/rollouts/pi0/{task_suite_name}/{transform}/{idx}/"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )
    base_filename = f"{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}"
    mp4_path = f"{rollout_dir}/{base_filename}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=15)
    
    margin = 20
    line_height = 15 
    top_margin = 10  
    gap = 5          

    save_dic = {
        "transform": transform,
        "task_description": task_description,
        "success": success,
        "mp4_path": mp4_path,
        "captions": captions,
    }
    
    def wrap_text(text, width):

        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            extra = 1 if current_line else 0
            if current_length + extra + len(word) <= width:
                current_line.append(word)
                current_length += extra + len(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    first_img = rollout_images[0]
    max_text_width = first_img.shape[1] - (2 * margin)
    chars_per_line = max_text_width // 6

    banner_text_static = f"Orig. Text - {task_description}"
    static_lines = wrap_text(banner_text_static, chars_per_line)
    
    max_caption_lines = 0
    for frame_idx, caption in enumerate(captions):
        caption_text = f"Frame {frame_idx} - Trans. Text - {caption}"
        caption_lines = wrap_text(caption_text, chars_per_line)
        max_caption_lines = max(max_caption_lines, len(caption_lines))
        
    fixed_banner_height = top_margin + len(static_lines) * line_height + gap + max_caption_lines * line_height + top_margin

    for frame_idx, img in enumerate(rollout_images):
        extended_img = np.full((img.shape[0] + fixed_banner_height, img.shape[1], 3), 255, dtype=np.uint8)
        extended_img[int(fixed_banner_height):, :] = img
        
        pil_img = Image.fromarray(extended_img)
        draw = ImageDraw.Draw(pil_img)
        
        y = top_margin
        for line in static_lines:
            draw.text((margin, y), line, fill=(0, 0, 0))
            y += line_height
        
        y += gap
        caption_text = f"Frame {frame_idx} - Trans. Text - {captions[frame_idx]}"
        caption_lines = wrap_text(caption_text, chars_per_line)
        for line in caption_lines:
            draw.text((margin, y), line, fill=(0, 0, 0))
            y += line_height
        
        video_writer.append_data(np.array(pil_img))
    
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
        
    with open(f"{rollout_dir}/{base_filename}.json", "w") as f:
        json.dump(save_dic, f)
    
    return mp4_path