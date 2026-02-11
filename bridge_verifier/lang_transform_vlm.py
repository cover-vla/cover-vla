from openai import OpenAI
import random
import string
import json
import re
import os
from pathlib import Path
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
            Given the original instruction: "{instruction}", and the appeneded image, generate {batch_number} reworded instructions that convey the same objective.

            Guidelines for rephrasing:
            1. Use simple, clear words and actions (focus on verbs and nouns)
            2. Remove adverbs whenever possible
            3. Keep descriptions concise but complete
            4. Infer and include object colors when they can be reasonably deduced (e.g., apples are typically red, strawberries are red)
            5. Use diverse vocabulary across rephrases (vary nouns, verbs, and adjectives)
            6. Ensure each rephrase maintains the same core meaning and task objective
            7. Try to generate as diverse as possible rephrases.
            8. Consider the image when generating the rephrased instructions.
            
            Examples:
            Original: "put apple on the desk"
            Reworded: "pick up the red apple and place it on the desk", "take the apple and put it on the desk", "place the red fruit on the desk"
            
            Original: "put cooking pot in the green basket"
            Reworded: "move the silver cooking pot to the green basket", "take the cooking pot and put it in the green basket", "put the utensil into the green basket"
            
            Original: "put strawberry on top of the fridge"
            Reworded: "put the red fruit on the fridge", "place the red berry on the top of the fridge", "set the red berry on the top of the refrigerator"
            
            Original: "lift the water bottle and place it on the desk"
            Reworded: "pick up the transparent bottle and place it on the wooden desk", "take the hydration bottle and put it on the desk", "place the water on the desk"
            
            
            Guidelines for generation: 
            1. You need to consider both image and instruction when generating the rephrased instructions.
            2. You need to first generate a description of the image in your own words, and then think about what does the language instruction mean in the context of the image.
            
            
            Format your response as:
            <Description of the image>
            <Meaning of the instruction in the context of the image>
            Original: <Nouns> as many as possible potential replacements: <Nouns>
            Original: <Verbs> as many as possible potential replacements: <Verbs>
            Original: <Adjectives> as many as possible potential replacements: <Adjectives>
            Original: <Adverbs>

            Original Instruction:
            {instruction}

            Reworded Instructions:
            1. <Alternative phrasing 1>
            2. <Alternative phrasing 2>
            ...
            {batch_number}. <Alternative phrasing {batch_number}>
            
            Important: Ensure all rephrased instructions avoid adverbs, use diverse vocabulary, and maintain the same objective as the original.
            """
        return instruction

    def transform(self, curr_instruction, batch_number=1, image=None):
        
        if batch_number > 1:
            batch_responses = self.gpt_transform(curr_instruction, batch_number=batch_number, image=image)
            print (batch_responses)
            print ("--------------------------------")
            return self.extract_reworded_instructions(batch_responses)


    def gpt_transform(self, instruction, batch_number, image=None):

        system_prompt = self.get_system_prompt()
            
        if batch_number > 1:
            t = 0.8
            instruction = self.get_rephrase_batch(instruction, batch_number=batch_number)
        else:
            t = 0.8  # Set default temperature even when batch_number is 1

        # Create the messages list with content array
        message_content = [{'type': 'text', 'text': instruction}]
        
        # Add image to message content if provided
        if image is not None:
            image_base64 = self._encode_image(image)
            if image_base64:
                message_content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{image_base64}'
                    }
                })
            
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
    
    def _encode_image(self, image):
        """
        Encode image to base64 string.
        Supports multiple input formats:
        - File path (str): path to image file
        - numpy array: image array (BGR or RGB)
        - PIL Image: PIL Image object
        - base64 string: already encoded image
        
        Returns base64 encoded string or None if encoding fails.
        """
        try:
            # If already a base64 string (data URL format)
            if isinstance(image, str) and image.startswith('data:image'):
                # Extract base64 part if it's a data URL
                return image.split(',')[1] if ',' in image else image
            
            # If it's a file path
            if isinstance(image, str) and os.path.isfile(image):
                with open(image, 'rb') as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            
            # If it's a numpy array
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Assume BGR, convert to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                
                # Convert to PIL Image
                pil_image = Image.fromarray(image_rgb)
                
                # Convert to bytes
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
                
                return base64.b64encode(image_bytes).decode('utf-8')
            
            # If it's a PIL Image
            if isinstance(image, Image.Image):
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
                return base64.b64encode(image_bytes).decode('utf-8')
            
            print(f"Warning: Unsupported image type: {type(image)}")
            return None
            
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
        
    def get_system_prompt(self):
        prompt_path = Path(__file__).resolve().parent / "system_prompts" / "rephrase_batch.txt"
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read()
    
