import os
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
# Base directory where all your model output folders reside
BASE_OUTPUT_DIR = Path(r"D:\Uni\LangNVision\Project\Hallucination_in_VLMs\output\Sample")

# List of model output folder names relative to BASE_OUTPUT_DIR
# These names will also be used as column headers.
MODEL_FOLDERS = [
    "FLUX",  # Example: corresponds to 'Our Method' or similar
    "SD2.1",      # Replace with your actual model folder names
    "Stacked Diffusion"       # Replace with your actual model folder names
]


# Name of the JSON file containing step information
# Assumed to be located in the first MODEL_FOLDERS directory (e.g., BASE_OUTPUT_DIR/sample_flux/sample_data.json)
DATASET_JSON_NAME = "sample_data.json"

# Output directory for the generated comparison images
COMPARISON_OUTPUT_DIR = Path("./comparison_results")
COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Image dimensions for display
IMG_SIZE = 224  # Standard size like 224x224
PADDING = 20    # Padding around images and text
TEXT_COL_WIDTH = 350 # Width allocated for the 'Steps' text column

# Font sizes
GOAL_TITLE_FONT_SIZE = 30
HEADER_FONT_SIZE = 25
STEPS_FONT_SIZE = 20
FONT_PATH = "arial.ttf" # Or any font available on your system, e.g., "arial.ttf", "DejaVuSans.ttf"

# --- Utility Functions ---

def get_font(size, is_bold=False):
    """Loads a font. Tries common paths if not found directly."""
    try:
        # If you have a specific bold font file, specify it here
        # For simplicity, using the same font for bold/regular if no specific bold variant is provided
        return ImageFont.truetype(FONT_PATH, size)
    except IOError:
        print(f"Warning: Font '{FONT_PATH}' not found. Using default Pillow font.")
        return ImageFont.load_default()

def wrap_text(text, font, max_width):
    """Wraps text to fit within a maximum width."""
    # Dummy image for text size calculation (needed for ImageDraw.textbbox)
    draw = ImageDraw.Draw(Image.new("RGB", (1,1)))
    lines = []
    if not text.strip(): # Handle truly empty or whitespace-only strings
        return [""]

    words = text.split(' ')
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        # textbbox returns (left, top, right, bottom)
        bbox = draw.textbbox((0,0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))
    return lines

def load_full_dataset(json_path: Path):
    """Loads the entire dataset from the JSON file."""
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading JSON from {json_path}: {e}")
        return None


# --- Main Image Generation Function ---

def generate_task_comparison_image(task_id: str, all_tasks_data: dict):
    """
    Generates a single comparison image for a given task ID across all models.
    """
    task_data = all_tasks_data.get(task_id)

    if not task_data:
        print(f"Skipping task '{task_id}': Data not found in JSON.")
        return

    original_steps = task_data.get("steps", [])
    task_goal = task_data.get("goal", "Unknown Task").replace('"', '') # Clean up quotes if present

    # Filter out steps that are genuinely empty, or handle them as needed
    # If you want empty rows for empty steps, use: display_steps = original_steps
    # If you want to skip empty steps:
    display_steps = [step for step in original_steps if step.strip() != ""]
    
    # If after filtering, no steps remain, skip this task
    if not display_steps:
        print(f"Skipping task '{task_id}': No valid steps to display after filtering.")
        return

    num_rows = len(display_steps)
    num_cols = len(MODEL_FOLDERS) + 1 # +1 for the "Steps" column

    # Calculate font objects
    goal_font = get_font(GOAL_TITLE_FONT_SIZE, is_bold=True)
    header_font = get_font(HEADER_FONT_SIZE, is_bold=True)
    step_font = get_font(STEPS_FONT_SIZE)

    # Calculate heights
    draw_dummy = ImageDraw.Draw(Image.new("RGB", (1,1)))
    # For textbbox, it returns (left, top, right, bottom), so width = right-left, height = bottom-top
    goal_bbox = draw_dummy.textbbox((0,0), task_goal, font=goal_font)
    goal_title_display_height = (goal_bbox[3] - goal_bbox[1]) + PADDING # Height for the goal title plus padding below it
    
    header_row_height = HEADER_FONT_SIZE + PADDING # Height for column headers

    # Total image dimensions
    total_width = TEXT_COL_WIDTH + (len(MODEL_FOLDERS) * IMG_SIZE) + (num_cols + 1) * PADDING
    total_height = goal_title_display_height + header_row_height + (num_rows * IMG_SIZE) + (num_rows + 1) * PADDING

    composite_img = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(composite_img)

    # --- Draw Goal Title ---
    # Recalculate actual width to ensure perfect centering
    goal_title_actual_width = draw.textbbox((0,0), task_goal, font=goal_font)[2]
    goal_title_x = (total_width - goal_title_actual_width) / 2
    draw.text((goal_title_x, PADDING), task_goal, font=goal_font, fill='black')

    # --- Draw Column Headers ---
    current_x = PADDING
    header_y_pos = goal_title_display_height + PADDING # Y position for headers below goal title

    draw.text((current_x, header_y_pos), "Steps", font=header_font, fill='black')
    current_x += TEXT_COL_WIDTH + PADDING

    for model_name in MODEL_FOLDERS:
        model_name_width = draw.textbbox((0,0), model_name, font=header_font)[2]
        draw.text((current_x + (IMG_SIZE - model_name_width) / 2, header_y_pos), model_name, font=header_font, fill='black')
        current_x += IMG_SIZE + PADDING

    # --- Populate Rows with Step Text and Images ---
    for r_idx in tqdm(range(num_rows), desc=f"Populating images for task '{task_id}' - '{task_goal}'"):
        # Image row starts after headers
        y_pos = goal_title_display_height + header_row_height + PADDING + (r_idx * (IMG_SIZE + PADDING))

        # Draw Step Text
        step_text_content = display_steps[r_idx]
        wrapped_lines = wrap_text(step_text_content, step_font, TEXT_COL_WIDTH - 2 * PADDING)
        
        # Calculate vertical centering for text within its cell height (IMG_SIZE)
        # Account for line spacing in total text height
        total_text_height = len(wrapped_lines) * (STEPS_FONT_SIZE + 2) - 2 if wrapped_lines else 0
        text_y_offset = y_pos + (IMG_SIZE - total_text_height) / 2
        
        current_text_y = text_y_offset
        for line in wrapped_lines:
            draw.text((PADDING, current_text_y), line, font=step_font, fill='black')
            current_text_y += STEPS_FONT_SIZE + 2 # Small line spacing

        # Draw Images for each model
        current_x = TEXT_COL_WIDTH + PADDING

        # We assume step_X.png maps to original_steps[X] based on the original JSON structure.
        # So we need to find the original index of the step_text_content within original_steps.
        # This handles cases where empty steps might have been filtered out from display_steps.
        try:
            original_step_index = original_steps.index(step_text_content)
        except ValueError:
            # Fallback if the step text somehow isn't directly in original_steps (e.g., due to more complex filtering)
            original_step_index = r_idx 
            # print(f"Warning: Could not find '{step_text_content[:30]}...' in original_steps. Using display index {r_idx} for filename.")


        for model_folder in MODEL_FOLDERS:
            # Use original_step_index for image filename
            model_image_path = BASE_OUTPUT_DIR / model_folder / task_id / f"step_{original_step_index}.png"
            
            try:
                img = Image.open(model_image_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
                composite_img.paste(img, (int(current_x), int(y_pos)))
            except FileNotFoundError:
                print(f"Warning: Image not found for {model_folder} task {task_id} step {original_step_index}: {model_image_path}")
                draw.rectangle([(current_x, y_pos), (current_x + IMG_SIZE, y_pos + IMG_SIZE)], fill="lightgray", outline="black")
                draw.text((current_x + IMG_SIZE/2 - 30, y_pos + IMG_SIZE/2 - 10), "MISSING", font=get_font(12), fill="red")
            except Exception as e:
                print(f"Error loading or processing image {model_image_path}: {e}")
                draw.rectangle([(current_x, y_pos), (current_x + IMG_SIZE, y_pos + IMG_SIZE)], fill="lightgray", outline="black")
                draw.text((current_x + IMG_SIZE/2 - 30, y_pos + IMG_SIZE/2 - 10), "ERROR", font=get_font(12), fill="red")
            
            current_x += IMG_SIZE + PADDING

    # --- Save the composite image ---
    # Clean the task_goal string for use in filename, replacing problematic characters
    cleaned_task_goal = task_goal.replace(' ', '_')
    cleaned_task_goal = cleaned_task_goal.replace('/', '_')
    cleaned_task_goal = cleaned_task_goal.replace('\\', '_') # This is the line that caused the SyntaxError previously
    cleaned_task_goal = cleaned_task_goal.replace(':', '_') # Add colon replacement, common in titles
    cleaned_task_goal = cleaned_task_goal.replace('?', '_') # Add question mark replacement

    output_filename = f"comparison_task_{task_id}_{cleaned_task_goal}.png"
    output_path = COMPARISON_OUTPUT_DIR / output_filename
    composite_img.save(output_path)
    print(f"\nComparison image saved to: {output_path}")


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the BASE_OUTPUT_DIR is correctly set and exists
    if not BASE_OUTPUT_DIR.exists():
        print(f"Error: BASE_OUTPUT_DIR '{BASE_OUTPUT_DIR}' does not exist. Please check your path.")
        exit()

    # Load the full dataset (from the first model's directory for consistency)
    # This assumes all model directories contain the same DATASET_JSON_NAME file.
    json_path_for_all_tasks = BASE_OUTPUT_DIR / MODEL_FOLDERS[0] / DATASET_JSON_NAME
    all_tasks_data = load_full_dataset(json_path_for_all_tasks)

    if not all_tasks_data:
        print(f"Failed to load full dataset from {json_path_for_all_tasks}. Cannot proceed with all tasks.")
        exit()
    
    # Get all task IDs from the loaded data, sorted for consistent order
    all_task_ids = sorted(list(all_tasks_data.keys()))

    print(f"Found {len(all_task_ids)} tasks to process.")

    # Loop through each task and generate a comparison image
    for task_id in tqdm(all_task_ids, desc="Processing all tasks"):
        generate_task_comparison_image(task_id, all_tasks_data)

    print("\nAll task comparison images generated (or skipped if issues occurred).")