import torch
from PIL import Image
import json
import numpy as np
from tqdm.auto import tqdm
from transformers import ViltProcessor, ViltForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_dataset(dataset_path: Path) -> dict:
    if not dataset_path.exists():
        logging.error(f"Missing dataset JSON at {dataset_path}")
        return {}
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return {}

def _get_image_files(task_dir: Path) -> list[Path]:
    if not task_dir.exists():
        logging.warning(f"Task dir '{task_dir}' not found.")
        return []
    
    return sorted(
        [f for f in task_dir.iterdir() if f.name.startswith('step_') and f.suffix.lower() in ['.png', '.jpg', '.jpeg']],
        key=lambda x: int(x.stem.split('_')[1])
    )

def calculate_vqa_alignment_score(
    dataset_path: Path,
    image_base_dir: Path,
    model: ViltForQuestionAnswering,
    processor: ViltProcessor,
    embedding_model: SentenceTransformer,
    device: str
) -> float:
    dataset = _load_dataset(dataset_path)
    if not dataset:
        return 0.0

    similarities = []

    for task_id, task_data in tqdm(dataset.items(), desc="VQA Alignment Evaluation"):
        task_dir = image_base_dir / task_id
        image_files = _get_image_files(task_dir)
        steps = task_data.get("steps", [])

        if not isinstance(steps, list) or not image_files:
            continue

        steps = steps[:len(image_files)]
        image_files = image_files[:len(steps)]

        for idx, (step_text, img_path) in enumerate(zip(steps, image_files)):
            question = "What is happening in this image?"

            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(image, question, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    pred_idx = logits.argmax(-1).item()
                    answer = model.config.id2label[pred_idx]

                # Semantic similarity between generated answer and step caption
                emb_answer = embedding_model.encode(answer, convert_to_tensor=True)
                emb_step = embedding_model.encode(step_text, convert_to_tensor=True)

                similarity = util.pytorch_cos_sim(emb_answer, emb_step).item()
                similarities.append(similarity)

            except Exception as e:
                logging.error(f"Error in task {task_id}, step {idx}: {e}")
                continue

    if not similarities:
        logging.warning("No similarity scores computed.")
        return 0.0

    return float(np.mean(similarities))


# --- Main ---
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Loading models...")
    try:
        vqa_model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa",
    use_safetensors=True
).to(device).eval()
        vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Light & fast
    except Exception as e:
        logging.critical(f"Model load failure: {e}")
        exit()

    output_directory = Path('./output/sample20/StackedDiff')#Path('./output/sd21')#Path('./output/StackedDiff') #Path('./output/sd21')  #Path('./output/flux_schnell')#Path('./output/sample') 
    dataset_json_path = output_directory / 'dataset_25.json'#'sample_data.json' # 
    output_directory.mkdir(parents=True, exist_ok=True)

    score = calculate_vqa_alignment_score(dataset_json_path, output_directory, vqa_model, vqa_processor, embedding_model, device)
    result_text = f"Average VQA-Step Semantic Alignment Score: {score:.4f}\n"
    print(result_text)

    try:
        with open(output_directory / 'vqa_semantic_results.txt', 'a') as f:
            f.write(result_text)
    except Exception as e:
        logging.error(f"Failed to write results: {e}")
