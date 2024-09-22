import os
import argparse
from PIL import Image
import json

def create_nb_info_json(folder_path, focal_length):
    # Define the images subfolder
    images_folder = os.path.join(folder_path, 'images')

    # Get the scene name (last entry in the folder path)
    scene_name = os.path.basename(folder_path)

    # Define the content for nb-info.json
    nb_info = {
        "id": "tanksandtemples_fastmap",
        "loader": "nerfbaselines.datasets.tanksandtemples_fastmap:load_tanksandtemples_fastmap_dataset",
        "scene": scene_name,
        "evaluation_protocol": "default",
        "type": "object-centric"
    }

    # Save the nb-info.json file
    nb_info_path = os.path.join(folder_path, 'nb-info.json')
    with open(nb_info_path, 'w') as f:
        json.dump(nb_info, f, indent=4)
    print(f"nb-info.json file created at: {nb_info_path}")

    # Load one of the JPG or PNG images to get its dimensions
    image_path = next(
        (os.path.join(images_folder, img) 
         for img in os.listdir(images_folder) 
         if img.lower().endswith(('.jpg', '.png'))),
        None
    )

    # Load one of the JPG images to get its dimensions
    # image_path = next((os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith('.JPG')) , None)
    if image_path is None:
        raise FileNotFoundError("No .JPG images found in the images folder.")

    with Image.open(image_path) as img:
        W, H = img.size  # Get image width and height

    # Save image dimensions and focal length to a text file
    info_txt_path = os.path.join(folder_path, 'image_info.txt')
    with open(info_txt_path, 'w') as f:
        f.write(f"H: {H}\nW: {W}\nFocal: {focal_length}\n")
    
    print(f"Image dimensions and focal length saved at: {info_txt_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate nb-info.json and save image dimensions.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images.")
    parser.add_argument("focal_length", type=float, help="Focal length of the camera.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create nb-info.json and save image dimensions
    create_nb_info_json(args.folder_path, args.focal_length)