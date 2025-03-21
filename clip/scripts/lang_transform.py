from openai import OpenAI
import random
import string
import json
import re
import os
import numpy as np

class LangTransform:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key = self.api_key)
        self.gpt_transforms = ['synonym', 'antonym', 'negation', 
                               'verb_noun_shuffle', 'in_set', 'out_set']

    ### MAIN FUNCTIONS

    def transform(self, curr_instruction, transform_type):
        
        if transform_type in self.gpt_transforms:
            return self.gpt_transform(curr_instruction, transform_type)
        elif transform_type == 'random_shuffle':
            return self.rand_shuffle_transform(curr_instruction)
        elif transform_type == 'no_transform':
            return curr_instruction
    
    ### TRANSFORM FUNCTIONS
    def sample_transform(self, instruction, transform_type):
        json_transforms = self.get_json_transforms()
        sampled_transforms = json_transforms[instruction][transform_type]
        return random.choice(sampled_transforms)
    
    def get_json_transforms(self):
        json_transforms = self.open_json('./experiments/robot/libero/negative_set.json')
        return json_transforms
    
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

    def gpt_transform(self, instruction, transform_type):
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
            t = 0.1
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [{'type' : 'text', 'text' : instruction}]
            }
        ]
        # print ("API key: ",self.client.api_key)
        response = self.client.chat.completions.create(
            model = 'gpt-4o',
            messages = messages,
            temperature = t,
            max_tokens = 250
        )
        return response.choices[0].message.content


    ### HELPER FUNCTIONS
    # def get_api_key(self):
    #     with open('./api_keys.txt', 'r') as file:
    #         file_contents = file.readlines()
    #         return file_contents[0].strip()
        
    def get_system_prompt(self, transform_type):
        with open(f'../system_prompts/{transform_type}.txt', 'r', encoding='utf-8') as file:
            return file.read()
        
    def get_set_of_words(self, path_to_ep_stat = './experiments/robot/libero/unique_words.json'):
        set_of_words = self.open_json(path_to_ep_stat)
        return set_of_words
    
    def open_json(self, path_to_json):
        with open(path_to_json, 'r') as f:
            return json.load(f)
    
