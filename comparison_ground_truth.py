import torch
from PIL import Image
import os
from tqdm.auto import tqdm
import numpy as np
import dreamsim
import pandas as pd
import re # Import regex for more robust parsing

# Suppress warning for tokenizers (often seen with HuggingFace transformers, used by some models)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Suppress warning for Intel MKL (often seen with PyTorch on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_step_id_from_filename(filename):
    """
    Extracts the numerical step ID from a filename like 'step_0.png' or 'steps_12.jpg'.
    It's designed to be robust to 'step_' vs 'steps_' and various common image extensions.
    Returns the integer ID or None if the pattern is not found.
    """
    # Regex breakdown:
    # (steps?_|\d+)  -> Matches 'step_' or 'steps_' (case-insensitive due to .lower())
    #               -> OR matches one or more digits (\d+) potentially at the start if filename is like "0_img.png"
    # (?P<step_id>\d+) -> Named capture group 'step_id' to extract one or more digits for the step number
    # \.(png|jpg|jpeg|gif|bmp) -> Matches a dot followed by common image extensions
    match = re.match(r'(steps?_)(?P<step_id>\d+)\.(png|jpg|jpeg|gif|bmp)', filename.lower())
    if match:
        return int(match.group('step_id'))
    return None

def calculate_pairwise_similarity(ground_truth_root_dir, generated_root_dir, dreamsim_model, preprocess, device):
    """
    Calculates the pairwise DreamSim similarity between corresponding ground truth and
    generated images, and then aggregates task-wise and overall similarity.

    Args:
        ground_truth_root_dir (str): The root directory containing ground truth images.
        generated_root_dir (str): The root directory containing generated images.
        dreamsim_model: The loaded DreamSim model object (the feature extractor).
        preprocess: The preprocessing function for DreamSim.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary with task-wise average similarities.
            - float: The overall average similarity across all tasks.
    """
    task_similarities = {}
    all_pairwise_similarities = []

    print(f"GT Root Dir: {ground_truth_root_dir}")
    print(f"Gen Root Dir: {generated_root_dir}")

    # Get list of task subfolders (e.g., '0', '1', ...) from the ground truth directory.
    # Assumes task subfolders are numbered and present in both root directories.
    task_subfolders = sorted([d for d in os.listdir(ground_truth_root_dir) if os.path.isdir(os.path.join(ground_truth_root_dir, d))])
    print(f"Detected GT Task Subfolders: {task_subfolders}")

    for task_id in tqdm(task_subfolders, desc="Processing tasks for pairwise similarity"):
        gt_task_path = os.path.join(ground_truth_root_dir, task_id)
        gen_task_path = os.path.join(generated_root_dir, task_id)

        print(f"\n--- Processing Task: {task_id} ---")
        print(f"GT Task Path: {gt_task_path}")
        print(f"Gen Task Path: {gen_task_path}")

        # Check if task folders actually exist in both root directories
        if not os.path.isdir(gt_task_path):
            print(f"Skipping task {task_id}: Ground truth folder not found at {gt_task_path}.")
            continue
        if not os.path.isdir(gen_task_path):
            print(f"Skipping task {task_id}: Generated folder not found at {gen_task_path}.")
            continue

        # Dictionaries to store filenames mapped by their extracted step ID
        gt_images_by_step = {}
        for f in os.listdir(gt_task_path):
            step_id = get_step_id_from_filename(f)
            if step_id is not None:
                gt_images_by_step[step_id] = f # Store the original filename

        gen_images_by_step = {}
        for f in os.listdir(gen_task_path):
            step_id = get_step_id_from_filename(f)
            if step_id is not None:
                gen_images_by_step[step_id] = f # Store the original filename

        print(f"GT Image Files by Step ID for Task {task_id}: {gt_images_by_step}")
        print(f"Gen Image Files by Step ID for Task {task_id}: {gen_images_by_step}")

        # If no images were found in either directory for this task, skip
        if not gt_images_by_step and not gen_images_by_step:
            print(f"Skipping task {task_id}: No images found in both subfolders matching the step naming pattern.")
            continue

        pairwise_similarities_for_task = []
        
        # Find common step IDs present in both ground truth and generated images
        common_step_ids = sorted(list(set(gt_images_by_step.keys()).intersection(set(gen_images_by_step.keys()))))

        if not common_step_ids:
            print(f"Skipping task {task_id}: No common step IDs found between GT and generated images.")
            continue

        for step_id in common_step_ids:
            gt_img_name = gt_images_by_step[step_id]
            gen_img_name = gen_images_by_step[step_id]

            gt_img_path = os.path.join(gt_task_path, gt_img_name)
            gen_img_path = os.path.join(gen_task_path, gen_img_name)

            try:
                gt_img = Image.open(gt_img_path).convert("RGB")
                gen_img = Image.open(gen_img_path).convert("RGB")
            except FileNotFoundError:
                print(f"ERROR: Image file not found for step {step_id} in task {task_id}. GT: '{gt_img_path}', Gen: '{gen_img_path}'. Skipping this pair.")
                continue
            except Exception as e:
                print(f"ERROR: Could not open image for step {step_id} in task {task_id}. Details: {e}. Skipping this pair.")
                continue

            preprocessed_gt_img_tensor = preprocess(gt_img).to(device)
            preprocessed_gen_img_tensor = preprocess(gen_img).to(device)

            gt_tensor_for_model = preprocessed_gt_img_tensor
            gen_tensor_for_model = preprocessed_gen_img_tensor


            try:
                # Calculate perceptual distance using DreamSim model
                perceptual_distance = dreamsim_model(
                    gt_tensor_for_model,  # Pass the already-batched tensor
                    gen_tensor_for_model  # Pass the already-batched tensor
                ).item() # .item() extracts the scalar value from the 0-dim tensor
                
                # DreamSim returns distance, we want similarity (typically 1 - distance as distance is normalized 0-1)
                similarity = 1 - perceptual_distance 
                pairwise_similarities_for_task.append(similarity)
                all_pairwise_similarities.append(similarity)
            except Exception as e:
                print(f"ERROR: DreamSim model failed for step {step_id} in task {task_id}. Details: {e}. Skipping this pair.")
                continue # Continue to the next image pair if one fails

        if pairwise_similarities_for_task:
            task_avg_similarity = np.mean(pairwise_similarities_for_task)
            task_similarities[task_id] = task_avg_similarity
            print(f"Task {task_id} Average Similarity: {task_avg_similarity:.4f} (from {len(pairwise_similarities_for_task)} pairs)")
        else:
            task_similarities[task_id] = 0.0 # No common images processed for this task
            print(f"Task {task_id}: No pairwise similarities calculated.")

    # Calculate overall average similarity across all successful pairs
    overall_avg_similarity = np.mean(all_pairwise_similarities) if all_pairwise_similarities else 0.0
    return task_similarities, overall_avg_similarity

# --- Main Execution Block ---
if __name__ == '__main__':
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        print("Attempting to load DreamSim model...")
        # Load the DreamSim model and its associated preprocessing function
        # Using "dino_vitb16" as specified in your original code
        model, preprocess = dreamsim.dreamsim(pretrained=True, dreamsim_type="dino_vitb16", device=device)
        print("DreamSim model loaded successfully.")
    except Exception as e:
        print("\nERROR: Failed to load DreamSim model!")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()
        exit(1) # Exit if the model cannot be loaded, as it's a critical dependency

    # Define the root directories for your ground truth and generated images
    ground_truth_root_directory = r"D:\Uni\LangNVision\Project\Hallucination_in_VLMs\dataset\GroundTruth"
    generated_root_directory = r"D:\Uni\LangNVision\Project\Hallucination_in_VLMs\output\sample20\Stacked Diffusion" 
    
    # Extract the model name from the generated directory path for naming output files
    model_name = os.path.basename(os.path.normpath(generated_root_directory))

    # Define the base directory where results (e.g., Excel files) will be saved
    base_results_parent_dir = r"D:\Uni\LangNVision\Project\Hallucination_in_VLMs\results" 

    # Create a specific output directory for this model's results
    output_base_dir = os.path.join(base_results_parent_dir, model_name)
    os.makedirs(output_base_dir, exist_ok=True) # Create directory if it doesn't exist

    # Define the full path for the output Excel file
    excel_output_path = os.path.join(output_base_dir, f'{model_name}_ground_truth_comparison_results.xlsx')

    print(f"Ground Truth Directory: {ground_truth_root_directory}")
    print(f"Generated Directory: {generated_root_directory}")

    # Calculate the pairwise similarities
    task_similarities_dict, overall_similarity = calculate_pairwise_similarity(
        ground_truth_root_directory, generated_root_directory, model, preprocess, device
    )

    # Prepare data for the task-wise similarities DataFrame
    df_data = []
    for task_id, avg_sim in task_similarities_dict.items():
        df_data.append({'Task ID': task_id, 'Average Similarity': avg_sim})

    # Create DataFrame for task-wise similarities
    df_tasks = pd.DataFrame(df_data)

    # Create DataFrame for overall similarity
    df_overall = pd.DataFrame([{'Metric': 'Overall Average Similarity', 'Score': overall_similarity}])

    # Write the results to an Excel file with two sheets
    with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
        df_tasks.to_excel(writer, sheet_name='Task-wise Similarity', index=False)
        df_overall.to_excel(writer, sheet_name='Overall Similarity', index=False)

    print(f"\nResults saved to: {excel_output_path}")
    print(f"Overall Average DreamSim Similarity: {overall_similarity:.4f}")