import torch
from PIL import Image
import os
import json
import numpy as np
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_dataset(dataset_path: Path) -> dict:
    """Loads the dataset from a JSON file."""
    if not dataset_path.exists():
        logging.error(f"Missing combined task data JSON file at '{dataset_path}'.")
        return {}
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in '{dataset_path}'.")
        return {}
    except Exception as e:
        logging.error(f"Error loading dataset from '{dataset_path}': {e}")
        return {}

def _get_image_files(task_dir: Path) -> list[Path]:
    """Gets sorted image files from a task directory."""
    if not task_dir.is_dir():
        logging.warning(f"Task directory '{task_dir}' not found. Skipping.")
        return []
    
    image_files = sorted(
        [f for f in task_dir.iterdir() if f.name.lower().startswith('step_') and f.suffix.lower() in ('.png', '.jpg', '.jpeg')],
        key=lambda x: int(x.name.split('_')[1].split('.')[0])
    )
    return image_files

def calculate_clip_score(
    dataset_path: Path, 
    image_base_dir: Path, 
    model: CLIPModel, 
    processor: CLIPProcessor, 
    device: str
) -> float:
    """
    Calculates the average CLIP score for text-image alignment across all tasks.
    The scores are scaled to be between 0 and 1.

    Args:
        dataset_path (Path): The path to the main JSON file (e.g., 'dataset_processed.json').
        image_base_dir (Path): The base directory containing the task subdirectories
        model (CLIPModel): The pre-trained CLIP model.
        processor (CLIPProcessor): The CLIP processor for text and images.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        float: The aggregated average CLIP score (0-1 range).
    """
    all_clip_scores = []
    
    dataset_processed = _load_dataset(dataset_path)
    if not dataset_processed:
        return 0.0

    for task_id_str, task_data in tqdm(dataset_processed.items(), desc="Processing tasks for CLIP score"):
        task_dir = image_base_dir / task_id_str 

        image_files = _get_image_files(task_dir)
        if not image_files:
            logging.warning(f"No images named 'step_X.png/jpg' found in task directory '{task_dir}'. Skipping task '{task_id_str}'.")
            continue

        if "steps" not in task_data or not isinstance(task_data["steps"], list):
            logging.warning(f"Task '{task_id_str}' is missing 'steps' or it's not a list. Skipping.")
            continue
        
        captions = task_data.get("steps", [])

        # The number of images is the ground truth.
        # Only consider captions for which an image was actually generated.
        num_images = len(image_files)
        captions = captions[:num_images]

        # This check is now more for logging/debugging
        if len(captions) != num_images:
            logging.warning(f"After aligning to {num_images} images, caption count is {len(captions)} for task '{task_id_str}'. There might be an issue with the input data.")
            # We proceed with the minimum length to be safe
            min_len = min(len(captions), num_images)
            image_files = image_files[:min_len]
            captions = captions[:min_len]

        for i, img_file_path in enumerate(image_files):
            try:
                step_text = captions[i]
                image = Image.open(img_file_path).convert("RGB")
                
                inputs = processor(text=[step_text], images=image, return_tensors="pt", padding=True, truncation=True).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image # This is typically scaled by 100
                    score = logits_per_image.item()
                    
                    # Normalize to 0-1 range.
                    # Since logits_per_image are typically 100 * cosine_similarity (and clipped at 0),
                    # dividing by 100 gives a score between 0 and 1.
                    normalized_score = max(0.0, score / 100.0)
                    all_clip_scores.append(normalized_score)

            except IndexError:
                logging.error(f"Index out of bounds for captions/images in task '{task_id_str}'. "
                                 "This might indicate an issue with caption-image correspondence.")
                continue
            except FileNotFoundError:
                logging.error(f"Image file not found: '{img_file_path}' in task '{task_id_str}'. Skipping.")
                continue
            except Exception as e:
                logging.error(f"Error processing '{img_file_path}' in task '{task_id_str}': {e}")
                continue

    if not all_clip_scores:
        logging.info("No CLIP scores were calculated. Check warnings/errors above.")
        return 0.0

    aggregated_average_score = np.mean(all_clip_scores)
    return aggregated_average_score

# --- Main Execution ---
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load CLIP model and processor
    model_name = "openai/clip-vit-base-patch32"
    
    # Use torch.float32 for CPU or if fp16 causes issues on some GPUs, otherwise stick to fp16
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    logging.info(f"Loading CLIP model '{model_name}' on device '{device}' with dtype '{dtype}'.")
    try:
        clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True, torch_dtype=dtype).to(device)
        clip_processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        logging.critical(f"Failed to load CLIP model or processor: {e}")
        exit()
        
    clip_model.eval()

    output_directory = Path('./output/sample20/StackedDiff')#Path('./output/sample20/StackedDiff')#Path('./output/sd21')#Path('./output/StackedDiff') #Path('./output/sd21')  #Path('./output/flux_schnell')#Path('./output/sample') 
    dataset_json_path = output_directory / 'dataset_25.json'#'sample_data.json' # 
    output_directory.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting CLIP score calculation for output directory: {output_directory}")
    logging.info(f"Using dataset JSON file: {dataset_json_path}")

    # Calculate and print the CLIP score
    clip_score = calculate_clip_score(dataset_json_path, output_directory, clip_model, clip_processor, device)
    
    result_text = f"Aggregated Average CLIP Score (Text-Image Alignment): {clip_score:.4f}\n"
    print(f"\n{result_text}")

    # Write the result to a file
    results_file_path = output_directory / 'results.txt'
    try:
        with open(results_file_path, 'a') as f:
            f.write(result_text)
        logging.info(f"Results written to: {results_file_path}")
    except Exception as e:
        logging.error(f"Failed to write results to file '{results_file_path}': {e}")