import requests
import json_numpy as json
from PIL import Image
import numpy as np
import os

def get_batch_actions(instruction, image_path, batch_size=3, temperature=1.0):
    image_path = os.path.abspath(image_path)
    payload = {
        "instruction": instruction,
        "image_path": image_path,
        "batch_size": batch_size,
        "temperature": temperature
    }

    res = requests.post(
        "http://localhost:3200/batch",
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'}
    )
    res.raise_for_status()
    return np.array(json.loads(res.text)["output_ids"]), np.array(json.loads(res.text)["actions"])

instruction = "close the drawer"
image_path = "vla/example.jpg"

actions = get_batch_actions(
    instruction=instruction,
    image_path=image_path,
    batch_size=3,
    temperature=1.0
)

print("Discrete Action Tokens: \n", actions[0])
print("Continuous actions: \n", actions[1])

