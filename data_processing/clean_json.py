import json

def clean_recipe_data(input_file_path, output_file_path):
    """
    Cleans a JSON file containing recipe data.
    Ensures 'steps' and 'steps_generated' arrays have at least 6 elements
    by appending empty strings if necessary.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_file_path (str): The path where the cleaned JSON data will be saved.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file_path}'. Please check the file format.")
        return

    for key, recipe in data.items():
        # Clean 'steps'
        if "steps" in recipe and isinstance(recipe["steps"], list):
            while len(recipe["steps"]) < 6:
                recipe["steps"].append("")

        # Clean 'steps_generated'
        if "steps_generated" in recipe and isinstance(recipe["steps_generated"], list):
            while len(recipe["steps_generated"]) < 6:
                recipe["steps_generated"].append("")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Cleaned data successfully saved to '{output_file_path}'")
    except IOError as e:
        print(f"Error: Could not write to file '{output_file_path}'. Reason: {e}")

# --- How to use the script ---
if __name__ == "__main__":
    # IMPORTANT: Replace "your_input_file.json" with the actual path to your JSON file
    input_json_file = r"D:\Uni\LangNVision\Project\hallucination_in_VLMs_for_manual_tasks\dataset\dataset_raw.json"
    output_json_file = r"D:\Uni\LangNVision\Project\hallucination_in_VLMs_for_manual_tasks\dataset\dataset_processed.json"

    clean_recipe_data(input_json_file, output_json_file)