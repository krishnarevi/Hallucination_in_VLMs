import os
from PIL import Image

def convert_webp_to_jpg(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.webp'):
                webp_path = os.path.join(root, file)
                jpg_path = os.path.splitext(webp_path)[0] + '.jpg'
                # print(f"{webp_path=}, {jpg_path=}")

                try:
                    with Image.open(webp_path) as img:
                        rgb_img = img.convert('RGB')  # Convert to RGB before saving as JPG
                        rgb_img.save(jpg_path, 'JPEG')
                        print(f"Converted: {webp_path} -> {jpg_path}")
                except Exception as e:
                    print(f"Failed to convert {webp_path}: {e}")

# Example usage
convert_webp_to_jpg(r'D:\Uni\LangNVision\Project\Hallucination_in_VLMs\dataset\GroundTruth')
