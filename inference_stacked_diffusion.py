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
cfg['save_dir'] = './output/sample20/StackedDiff'#'./output/sample'#'./output'
cfg['ip_dir'] = './dataset'
cfg['json_file'] =  'dataset_25.json'#'sample_data.json'# 'dataset_processed.json'
# Ensure the save directory exists
os.makedirs(cfg['save_dir'], exist_ok=True)

# Check for CUDA availability
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. This script requires a GPU.")

pipeline = DiffusionPipeline.from_pretrained(
    "sachit-menon/illustrated_instructions", 
    custom_pipeline="snt_pipeline", 
    trust_remote_code=True
)
tokenizer = pipeline.model.tokenizer
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

    prompt = f"Goal: {goal}."

    MAX_STEPS = 6
    if len(step_texts) > MAX_STEPS:
        print(f"Warning: Task {task_id} has more than {MAX_STEPS} steps. Truncating.")
        step_texts = step_texts[:MAX_STEPS]
    elif len(step_texts) < MAX_STEPS:
        padding = [""] * (MAX_STEPS - len(step_texts))
        step_texts.extend(padding)

    for i, step in enumerate(step_texts):
        step_texts[i] = f'{i+1}. {step}'

    input_texts = [prompt]
    input_texts.extend(step_texts)

    prompts = tokenizer(
        input_texts,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    prompts = prompts.unsqueeze(0).to(cfg['device'])

    print(f"Starting image generation for task: {task_id} - {goal}")

    with torch.autocast(device_type="cuda", 
                        dtype=torch.float16,
                        enabled=True):
        output_images = pipeline(
            prompts,
            image=torch.zeros((1,3,256,256)).to(cfg['device']),
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7,
            generator=generator,
        ).images
    
    print(f"Generation complete for task {task_id}. Saving {len(output_images)} images.")

    for i in range(len(step_texts)):
        if step_texts[i] == f'{i+1}. ': # Skip empty steps
            continue
        
        output_image_fname = f"step_{i}.png"
        save_path = os.path.join(task_dir, output_image_fname)
        
        if output_images and i < len(output_images):
            output_images[i].save(save_path)
            print(f"Saved: {save_path}")
        else:
            print(f"Warning: No image generated for step {i+1} in task {task_id}")

print("Script finished.")
