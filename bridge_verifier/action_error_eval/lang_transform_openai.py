from openai import OpenAI
import random
import string
import json
import re
import os
import numpy as np
import cv2
import base64
from PIL import Image
import io

class LangTransform:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key = self.api_key)
        self.gpt_transforms = ['synonym', 'antonym', 'negation', 
                               'verb_noun_shuffle', 'in_set', 'out_set', 'rephrase']

    ### MAIN FUNCTIONS
    
    def extract_reworded_instructions(self, text):
        lines = text.strip().splitlines()
        instructions = []
        recording = False

        for line in lines:
            line = line.strip()

            if line == "Reworded Instructions:":
                recording = True
                continue

            if recording:
                if line and line.split()[0][:-1].isdigit() and line.split()[0].endswith('.'):
                    # Remove the leading number and dot
                    instruction_text = ' '.join(line.split()[1:])
                    instructions.append(instruction_text)
                elif line == "":
                    continue
                else:
                    break

        return instructions

    def get_rephrase_batch(self, instruction, batch_number=1):
        instruction = f"""
            Given the original instruction: "{instruction}", generate {batch_number} reworded instructions that:
            - Convey the same objective as the original
            - Try to generate a mixture of easy and hard rephrases. You can try some simple ones with only 1 or 2 words different, and also include some rephrases that are more challenging. 
            - Try to paraphrase both the noun and the verb.
            
            Format your response as:

            Original Instruction:
            <The user-provided instruction>

            Reworded Instructions:
            1. <Alternative phrasing 1>
            2. <Alternative phrasing 2>
            ...
            {batch_number}. <Alternative phrasing {batch_number}>
            """
        return instruction

    def transform(self, curr_instruction, transform_type, batch_number=1, image=None):
        
        if transform_type in self.gpt_transforms:
            response = self.gpt_transform(curr_instruction, transform_type, batch_number=1, image=image)
        elif transform_type == 'random_shuffle':
            response = self.rand_shuffle_transform(curr_instruction)
        elif transform_type == 'no_transform':
            response = curr_instruction
        
        if batch_number > 1:
            batch_responses = self.gpt_transform(curr_instruction, transform_type=None, batch_number=batch_number, image=image)
            # print (batch_responses)
            return self.extract_reworded_instructions(batch_responses)
        else:
            return response
    
    ### TRANSFORM FUNCTIONS
    def sample_transform(self, instruction, transform_type):
        json_transforms = self.get_json_transforms()
        sampled_transforms = json_transforms[instruction][transform_type]
        return random.choice(sampled_transforms)
    
    def get_json_transforms(self):
        json_transforms = self.open_json('./experiments/robot/libero/negative_set.json')
        return json_transforms
    
    def encode_image_for_openai(self, image: np.ndarray) -> str:
        # Convert to PIL
        pil_image = Image.fromarray(image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')

        # Get raw bytes and encode to base64
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"

    
    def rand_shuffle_transform(self, instruction):
        while True:
            words = instruction.split()
            end_punctuation = ''
            
            if words[-1][-1] in string.punctuation:
                end_punctuation = words[-1][-1]
                words[-1] = words[-1][:-1]

            random.shuffle(words)

            shuffled_instruction = ' '.join(words) + end_punctuation
            if shuffled_instruction != instruction:
                return shuffled_instruction

    def gpt_transform(self, instruction, transform_type, batch_number, image=None):
        if transform_type is not None:
            system_prompt = self.get_system_prompt(transform_type)
            if transform_type == 'out_set':
                t = 0.8
                in_set_words = self.get_set_of_words()
                instruction = f"Given this instruction: {instruction}, please reword it without using any of the following words {in_set_words}"
            elif transform_type == 'in_set':
                t = 0
                unique_words = self.get_set_of_words()
                instruction = f"Given this instruction: {instruction}, please reword it using only the following words {unique_words}"
            else:
                t = 0.5
        else: 
            system_prompt = self.get_system_prompt('rephrase_batch')
            
        if batch_number > 1:
            t = 0.5
            instruction = self.get_rephrase_batch(instruction, batch_number=batch_number)

        # Create the messages list with content array
        message_content = [{'type': 'text', 'text': instruction}]
            
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": message_content
            }
        ]
        # print ("API key: ",self.client.api_key)
        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model = 'gpt-4o',
                    messages = messages,
                    temperature = t,
                    max_tokens = 5000
                )
                return response.choices[0].message.content  # <-- returns here if successful, loop ends
            except Exception as e:
                print(f"Error: {e}")
                # loop continues to next attempt


    ### HELPER FUNCTIONS
    # def get_api_key(self):
    #     with open('./api_keys.txt', 'r') as file:
    #         file_contents = file.readlines()
    #         return file_contents[0].strip()
        
    def get_system_prompt(self, transform_type):
        with open(f'/root/vla-clip/clip_verifier/system_prompts/{transform_type}.txt', 'r', encoding='utf-8') as file:
            return file.read()
        
    def get_set_of_words(self, path_to_ep_stat = '/root/vla-clip/clip_verifier/scripts/unique_words.json'):
        set_of_words = self.open_json(path_to_ep_stat)
        return set_of_words
    
    def open_json(self, path_to_json):
        with open(path_to_json, 'r') as f:
            return json.load(f)
    
