import json
import os

def create_sample_json(output_dir=".", output_filename="all_tasks_data.json"):
    """
    Creates a sample JSON file with two tasks: Apple Crisp and Jamaican Callaloo With Shrimp.

    Args:
        output_dir (str): The directory where the JSON file should be created.
                          Defaults to the current directory (".").
        output_filename (str): The name of the JSON file to create.
                               Defaults to "all_tasks_data.json".
    """
    sample_data = {
        "0": {
            "steps": [
                "Preheat oven to 350°F (175°C).",
                "Peel, core, and slice apples; place in a baking dish.",
                "Mix sugar, cinnamon, and a pinch of salt; sprinkle over apples.",
                "Combine oats, flour, brown sugar, and butter; crumble over apples.",
                "Bake for 45 minutes or until golden brown.",
                "Serve warm with vanilla ice cream."
            ],
            "title": "Apple Crisp"
        },
        "1": { # Using "1" as the next task ID
            "steps": [
                "Heat the Coconut Oil in a wide pan over a medium flame, then add the Onion, Garlic, Scallion, and Ground Black Pepper. Reduce the heat to low for about 3-4 minutes.",
                "Add the Small Shrimp, stir well and cook for another 3 minutes.",
                "Turn the heat up to medium high and add the Jamaican Callaloo, Tomato, Scotch Bonnet Pepper, Fresh Thyme, and Sea Salt. After a couple minutes, add the Water and cook until tender.",
                "After about 10-12 minutes, taste for salt and adjust accordingly.",
                # Including the duplicated steps as provided in your request
                "Heat the Coconut Oil in a wide pan over a medium flame, then add the Onion, Garlic, Scallion, and Ground Black Pepper. Reduce the heat to low for about 3-4 minutes.",
                "Add the Small Shrimp, stir well and cook for another 3 minutes."
            ],
            "title": "Jamaican Callaloo With Shrimp"
        }
    }

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full path for the JSON file
    full_output_path = os.path.join(output_dir, output_filename)

    try:
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=4)
        print(f"Sample JSON '{output_filename}' created successfully in '{os.path.abspath(output_dir)}'.")
        print(f"It contains data for {len(sample_data)} tasks.")
    except IOError as e:
        print(f"Error writing JSON file to '{full_output_path}': {e}")

if __name__ == "__main__":
    # Example usage:
    # 1. Create in the current directory (default)
    print("Creating JSON in current directory:")
    create_sample_json()

    # 2. Create in a specific sub-directory named 'my_data_folder'
    path = r'D:\Uni\LangNVision\Project\generating-illustrated-instructions-reproduction\quick_inference\sample_data'
    print(f"\nCreating JSON in sub-directory '{path}':")
    create_sample_json(output_dir=path)
