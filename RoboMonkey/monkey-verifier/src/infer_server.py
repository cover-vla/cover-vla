import os
import time
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from transformers import AutoTokenizer, set_seed
import uvicorn
import json_numpy as json
import torch
import einops

from reward_model_utils import (
    pad_sequence_from_left,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    DisableLogger,
    logger,
)
from lora_utils import print_trainable_parameters, DEFAULT_PAD_TOKEN
from models.reward_model import RewardConfig, RewardModel
from action_processing import ActionTokenizer
from llava import conversation as conversation_lib
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.train.train import smart_tokenizer_and_embedding_resize

class RobotRewardModel:
    def __init__(self):
        # Parse arguments from environment variables and defaults based on the shell script
        model_args = ModelArguments(
            model_name_or_path=os.path.join(os.environ.get("MODEL_DIR", "./model_dir"), 
                                          "llava-v1.5-7b/sft_model/"),
            vision_tower="openai/clip-vit-large-patch14-336",
            mm_vision_select_layer=-2,
            mm_use_im_start_end=False,
            mm_use_im_patch_token=False,
            version="v1"
        )
        
        data_args = DataArguments(
            image_aspect_ratio='pad',
            is_multimodal=True,
            reward_prompt_file="./prompts/robot_reward_prompt.txt"
        )
        
        training_args = TrainingArguments(
            model_max_length=2048,
            query_len=1280,
            response_len=768,
            bits=16,
            lora_r=64,
            lora_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            output_dir=os.path.join(os.environ.get("MODEL_DIR", "./model_dir"), 
                                   "llava-v1.5-7b"),
            freeze_mm_mlp_adapter=True,
            group_by_length=False,
            bf16=True,
            seed=42
        )

        # Set seed for deterministic behavior
        set_seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            truncation_side="right",
            use_fast=False,
        )

        # Handle tokenizer configuration
        if model_args.version == "v0":
            if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="[PAD]"),
                    tokenizer=tokenizer,
                    model=None,
                )
        elif model_args.version == "v0.5":
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
            if model_args.version in conversation_lib.conv_templates:
                conversation_lib.default_conversation = conversation_lib.conv_templates[
                    model_args.version
                ]
            else:
                conversation_lib.default_conversation = conversation_lib.conv_templates[
                    "vicuna_v1"
                ]

        # Initialize model
        if model_args.vision_tower is not None:
            config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)

            with DisableLogger():
                args = type('Args', (), {})()
                for key, value in vars(model_args).items():
                    setattr(args, key, value)
                for key, value in vars(data_args).items():
                    setattr(args, key, value)
                for key, value in vars(training_args).items():
                    setattr(args, key, value)
                
                model = RewardModel(
                    args=args,
                    config=config,
                    qlora=True,
                    checkpoint_dir=os.path.join(os.environ.get("MODEL_DIR", "./model_dir"), "lora_adapter"),
                    tokenizer=tokenizer,
                ).to(torch.bfloat16)

            model.backbone_model.config.use_cache = False  # Disable for deterministic behavior
            print_trainable_parameters(args, model)
            print("Loaded model")

            with DisableLogger():
                model_temp = model.backbone_model

            vision_tower = model_temp.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()

            data_args.image_processor = vision_tower.image_processor
            model_temp.config.mm_use_im_start_end = model_args.mm_use_im_start_end

            self.tokenizer = tokenizer
            self.model = model
            self.model.eval()
            self.data_args = data_args

    def _left_pad_helper(self, ex_input_ids, batch_size):
        input_ids = [seq for seq in ex_input_ids]
        input_ids = pad_sequence_from_left(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = einops.rearrange(
            input_ids,
            "(bsz num_candidates) max_seq_len -> bsz num_candidates max_seq_len",
            num_candidates=batch_size,
        )
        return input_ids

    def get_rewards(self, instruction, image_path, actions):
        batch_size = len(actions)
        action_tokenizer = ActionTokenizer(self.tokenizer)
        conv_mode = "vicuna_v1"
        conv_template = conv_templates[conv_mode].copy()

        action_in_ids = []
        instruction = instruction.lower().rstrip('.')

        for action in actions:
            action_id = np.array(action)
            if type(action_id[0]) == float:
                action_id = action_tokenizer(action)
            action_holder = ' '.join(['placeholder'] * 7)  # seven identical tokens
            
            inp = (f"shows the current observation from the robot's wrist-mounted camera. "
                   f"The robot manipulation arm is attempting to {instruction}. "
                   f"What action should the robot take to effectively accomplish the task? "
                   f"ASSISTANT: The robot should take the action: {action_holder} </s> "
                   f"USER: Please evaluate the quality of the robot action. "
                   f"A good robot action should consider different factors, "
                   f"especially interactions with surrounding objects and human preferences.\n"
                   f"ASSISTANT: Based on how humans would control the robot arm and the "
                   f"awareness of the situation, the quality score of the robot action is")
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv = conv_template.copy()
            conv.append_message(conv.roles[0], inp)
            prompt = conv.get_prompt()
            prompt = prompt.replace("<image>", " placeholder ")
            
            in_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length + 2,
                truncation=True,
            ).input_ids

            first_image_idx = (in_ids == 12983).nonzero()  # Token ID of "placeholder" is 12983
            start_idx = first_image_idx[0][1].item()
            in_ids[0, start_idx: start_idx + 1] = -200

            action_indices = (in_ids == 12983).nonzero()  # Token ID of "placeholder" is 12983
            start_idx = action_indices[0][1].item()
            in_ids[0, start_idx:start_idx + 7] = torch.tensor(action_id - 1000)

            in_ids = in_ids[:, :-1]
            in_ids = torch.tensor(in_ids, dtype=torch.long).squeeze(0)
            action_in_ids.append(in_ids)

        input_ids = self._left_pad_helper(action_in_ids, batch_size).squeeze(0)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        # Process image
        processor = self.data_args.image_processor
        image = Image.open(image_path).convert("RGB")

        if self.data_args.image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        images = image.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        model_inputs = {
            "input_ids": input_ids.cuda(0).to(torch.int64),
            "attention_mask": attention_mask.cuda(0).to(torch.int64),
            "images": images.cuda(0).to(torch.bfloat16)
        }
        
        with torch.no_grad():
            scores = self.model.forward(**model_inputs)
        
        return scores.rewards.detach().cpu().tolist()


# FastAPI application
app = FastAPI()
reward_model = None


@app.on_event("startup")
async def startup_event():
    global reward_model
    reward_model = RobotRewardModel()


@app.get("/")
async def read_root():
    return {"message": "RM server up"}


@app.post("/process")
async def process_data(request: Request):
    body = await request.body()
    data = json.loads(body)

    instruction = data.get("instruction")
    image_path = data.get("image_path")
    action = data.get("action")

    if not isinstance(instruction, str):
        raise HTTPException(status_code=400, detail="Instruction must be a string")
    if not isinstance(image_path, str):
        raise HTTPException(status_code=400, detail="Image path must be a string")

    action_array = np.array(action)

    if action_array.ndim != 2:
        raise HTTPException(status_code=400, detail="Action must be a 2D array")

    start_time = time.time()

    rewards = reward_model.get_rewards(instruction, image_path, action_array)
    
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    return {"rewards": rewards}


if __name__ == "__main__":
    # Set environment variables from shell script defaults
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "model_dir"))
    os.environ.setdefault("GPUS_PER_NODE", "1")
    
    uvicorn.run(app, host="0.0.0.0", port=3100)