from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import uvicorn
import numpy as np
import sglang as sgl
from token2action import TokenActionConverter, image_qa
import json_numpy as json
from typing import List, Optional
import argparse

app = FastAPI()
converter = TokenActionConverter()

class BatchRequest(BaseModel):
    instruction: str
    image_path: str
    batch_size: Optional[int] = 3
    temperature: Optional[float] = 1.0

def process_batch(instruction: str, image_path: str, batch_size: int, temperature: float):
    """Run a batch of inference and return output token IDs and actions."""
    prompts = [{
        "image_path": image_path,
        "question": f"In: What action should the robot take to {instruction}?\nOut:"
    }] * batch_size

    states = image_qa.run_batch(prompts, max_new_tokens=7, temperature=temperature)
    
    output_ids = [np.array(s.get_meta_info("action")["output_ids"]) for s in states]
    actions = [np.array(converter.token_to_action(ids)) for ids in output_ids]
    return output_ids, actions

@app.get("/")
async def root():
    return {"message": "Batch processing server is running"}

@app.post("/batch")
async def handle_batch(request: Request):
    try:
        data = json.loads(await request.body())

        instruction = data.get("instruction")
        image_path = data.get("image_path")
        if not isinstance(instruction, str) or not isinstance(image_path, str):
            raise HTTPException(status_code=400, detail="Both 'instruction' and 'image_path' must be strings")

        batch_size = int(data.get("batch_size", 3))
        temperature = float(data.get("temperature", 1.0))

        output_ids, actions = process_batch(instruction, image_path, batch_size, temperature)
        return Response(content=json.dumps({"output_ids": output_ids, "actions": actions}),
                        media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    runtime = sgl.Runtime(
        model_path="openvla/openvla-7b",
        tokenizer_path="openvla/openvla-7b",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        random_seed=args.seed,
    )
    sgl.set_default_backend(runtime)
    uvicorn.run(app, host="0.0.0.0", port=3200)
