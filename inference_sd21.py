# %%
import torch
from diffusers import DiffusionPipeline
import os
import json

# only needed if using GPT-4 for LLM
# import openai 
# import re

# %%
cfg = {}
cfg['manual_steps'] = True
cfg['gpu_number'] = '0'
cfg['device'] = f'cuda:{cfg["gpu_number"]}'
cfg['save_dir'] = './output/sample20/sd21'# './output/sample_base'#'./output'
cfg['ip_dir'] = './dataset'
cfg['json_file'] ='dataset_25.json'#'sample_data.json'# 
# Ensure the save directory exists
os.makedirs(cfg['save_dir'], exist_ok=True)

# Check for CUDA availability
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. This script requires a GPU.")

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", 
    torch_dtype=torch.float16
)
generator = torch.Generator(device=cfg['device']).manual_seed(0)

# %%
pipeline = pipeline.to(cfg['device'])

# %%
with open(os.path.join(cfg['ip_dir'],cfg['json_file']), 'r') as f:
    tasks_data = json.load(f)

# copy dataset_processed.json to the output directory

with open(os.path.join(cfg['save_dir'] , cfg['json_file'] ), 'w') as f:
    json.dump(tasks_data, f, indent=4)


for task_id, task_details in tasks_data.items():
    goal = task_details['goal']
    step_texts = task_details['steps']
    
    task_dir = os.path.join(cfg['save_dir'], str(task_id))
    os.makedirs(task_dir, exist_ok=True)

    MAX_STEPS = 6
    if len(step_texts) > MAX_STEPS:
        print(f"Warning: Task {task_id} has more than {MAX_STEPS} steps. Truncating.")
        step_texts = step_texts[:MAX_STEPS]
    elif len(step_texts) < MAX_STEPS:
        padding = [""] * (MAX_STEPS - len(step_texts))
        step_texts.extend(padding)

    print(f"Starting image generation for task: {task_id} - {goal}")

    for i, step in enumerate(step_texts):
        if not step: # Skip empty steps
            continue

        prompt = step
        
        with torch.autocast(device_type="cuda", 
                            dtype=torch.float16,
                            enabled=True):
            image = pipeline(
                prompt,
                num_inference_steps=50,
                guidance_scale=7,
                generator=generator,
            ).images[0]
        
        output_image_fname = f"step_{i}.png"
        save_path = os.path.join(task_dir, output_image_fname)
        image.save(save_path)
        print(f"Saved: {save_path}")

print("Script finished.")
