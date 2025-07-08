import torch
from PIL import Image
import os
from tqdm.auto import tqdm
import numpy as np
import dreamsim 

# Suppress warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_perceptual_distance(output_dir, dreamsim_model, preprocess, device):
    """
    Calculates the aggregated average DreamSim distance for all task sequences
    in the output directory.

    Args:
        output_dir (str): The directory containing the generated images for each task.
        dreamsim_model: The loaded DreamSim model object (the feature extractor).
        preprocess: The preprocessing function for DreamSim.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        float: The aggregated average DreamSim distance.
    """
    task_distance_score = []
    task_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    for task_dir_name in tqdm(task_dirs, desc="Processing tasks for DreamSim coherence"):
        task_path = os.path.join(output_dir, task_dir_name)
        
        image_files = sorted(
            [f for f in os.listdir(task_path) if f.endswith(('.png', '.jpg', '.jpeg')) and f.startswith('step_')],
            key=lambda x: int(x.split('_')[1].split('.')[0]) # Ensure numerical sorting
        )
        
        if len(image_files) < 2:
            print(f"Skipping task {task_dir_name} as it has fewer than 2 images.")
            continue

        # Load images as PIL objects and preprocess them
        preprocessed_images = []
        for p in image_files: # Iterate through filenames
            img_path = os.path.join(task_path, p)
            img = Image.open(img_path).convert("RGB")
            # Preprocess and move to device
            preprocessed_img_tensor = preprocess(img).to(device)
            preprocessed_images.append(preprocessed_img_tensor)
        
        pairwise_distance_score = []
        # Calculate pairwise DreamSim distances between consecutive images
        for i in range(len(preprocessed_images) - 1):
            img1_tensor = preprocessed_images[i]
            img2_tensor = preprocessed_images[i+1]
            
            # Use the dreamsim_model directly for similarity calculation
            # It expects batched tensors
            perceptual_distance = model(img1_tensor, img2_tensor).item()
            pairwise_distance_score.append(perceptual_distance)
            
        if pairwise_distance_score: 
            average_task_distance = np.mean(pairwise_distance_score)
            task_distance_score.append(average_task_distance)

    if not task_distance_score:
        return 0.0

    aggregated_average_distance = np.mean(task_distance_score)
    return aggregated_average_distance

# --- Main Execution ---
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        print("Attempting to load DreamSim model...")
        # dreamsim is used to measure the perceptual distance between two images. A higher score means more different, lower means more similar
    
        model, preprocess = dreamsim.dreamsim(pretrained=True, dreamsim_type="dino_vitb16", device=device)
        print("DreamSim model loaded successfully.")
    except Exception as e:
        print("\nERROR: Failed to load DreamSim model!")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    output_directory =  './output/sd21' # './output/sample'#'./output/StackedDiff'#'./output/flux_schnell'

    # Pass the dreamsim model object and the preprocess function
    perceptual_distance = calculate_perceptual_distance(output_directory, model, preprocess, device)
    coherence_score = 1- perceptual_distance
    
    result_text = f"Aggregated Average DreamSim Similarity (Sequential Coherence Score): {coherence_score:.4f}\n"
    print(f"\n{result_text}")

    # Write the result to a file
    with open(os.path.join(output_directory, 'results.txt'), 'a') as f:
        f.write(result_text)
