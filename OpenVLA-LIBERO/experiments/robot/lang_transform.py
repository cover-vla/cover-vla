from openai import OpenAI
import random
import string
import json
import re
import os
import numpy as np

class LangTransform:
    def __init__(self):
        self.api_key = self.get_api_key()
        self.client = OpenAI(api_key = self.api_key)
        self.gpt_transforms = ['synonym', 'antonym', 'negation', 
                               'verb_noun_shuffle', 'in_set', 'out_set']

    ### MAIN FUNCTIONS

    def transform(self, curr_instruction, transform_type, untransformed_eval_path = None, original_instruction = None, prev_action = np.zeros((7)), curr_action = np.zeros((7))):
        # instruction = self.get_curr_instruction(curr_instruction)
        
        if transform_type in self.gpt_transforms:
            return self.gpt_transform(curr_instruction, transform_type, untransformed_eval_path)
        elif transform_type == 'random_shuffle':
            return self.rand_shuffle_transform(curr_instruction)
        elif transform_type == 'interpolation':
            return self.interpolation_transform(original_instruction=original_instruction, curr_instruction=curr_instruction, curr_action=curr_action, prev_action=prev_action)
        elif transform_type == 'no_transform':
            return curr_instruction

    def interpolation_transform(self, original_instruction, curr_instruction, curr_action, prev_action):
        # check if action are close to each other
        if np.linalg.norm(curr_action[:-1] - prev_action[:-1]) > 0.01 or curr_action[-1] != prev_action[-1]:
            return curr_instruction
        else:
            if original_instruction is not None:
                # parse original instruction as strings 
                parsed_instruction = original_instruction.split()
                # parse curr_instruction as strings
                parsed_curr_instruction = curr_instruction.split()
                current_instruction_length = len(parsed_curr_instruction)
                if current_instruction_length != len(parsed_instruction):
                    new_instruction_string = parsed_curr_instruction + [parsed_instruction[current_instruction_length]]
                    return ' '.join(new_instruction_string)
                else:
                    return original_instruction
            else:
                return curr_instruction.split()[0]
    
    ### TRANSFORM FUNCTIONS
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

    def gpt_transform(self, instruction, transform_type, untransformed_eval_path):
        system_prompt = self.get_system_prompt(transform_type)
        if transform_type == 'out_set':
            t = 0.8
            in_set_words = self.get_set_of_words(untransformed_eval_path)
            instruction = f"Given this instruction: {instruction}, please reword it without using any of the following words {in_set_words}"
        elif transform_type == 'in_set':
            t = 0
            unique_words = self.get_set_of_words(untransformed_eval_path)
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
    def get_api_key(self):
        key = os.getenv('OPENAI_API_KEY')
        print ("using API key: ", key)
        return key
        
    def get_system_prompt(self, transform_type):
        with open(f'./experiments/robot/libero/system_prompts/{transform_type}.txt', 'r', encoding='utf-8') as file:
            return file.read()
        
    def get_set_of_words(self, path_to_ep_stat):
        set_of_words = self.open_json(path_to_ep_stat)
        unique_words = set(set_of_words)
        return unique_words
    
    def open_json(self, path_to_json):
        with open(path_to_json, 'r') as f:
            return json.load(f)
    
    def get_curr_instruction(self, instruction):
        match = re.search(r"Current instruction: (.*)", instruction)
        if match:
            return match.group(1)            
        else:
            print("No 'Current instruction' part found.")
    
    def format_transformation(self, curr_instruction, transformed_instruction):
        return re.sub(r"(Current instruction: ).*", rf"\1{transformed_instruction}", curr_instruction)


## DEMO

# lang_transform = LangTransform()
# instruction = """Task goal: close the red jar.
# Current instruction: approach the lid by moving towards the back right corner of the table and then lower the gripper to just above the lid."""
# print('Original', instruction)
# print()
# print('synonym', lang_transform.transform(instruction, 'synonym'))
# print()
# print('antonym', lang_transform.transform(instruction, 'antonym'))
# print()
# print('negation', lang_transform.transform(instruction, 'negation'))
# print()
# print('verb noun shuffle', lang_transform.transform(instruction, 'verb_noun_shuffle'))
# print()
# print('random shuffle', lang_transform.transform(instruction, 'random_shuffle'))
# # print()
# # print('in set', lang_transform.transform(instruction, 'in_set'))
# print()
# print('out set', lang_transform.transform(instruction, 'out_set'))

