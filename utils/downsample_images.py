import os
import argparse
from PIL import Image

# Function to downsample an image by a given factor
def downsample_image(img_path, save_path, factor):
    with Image.open(img_path) as img:
        # Get the original dimensions
        width, height = img.size
        # Downsample the image by the specified factor
        img_resized = img.resize((width // factor, height // factor), Image.ANTIALIAS)
        # Save the downsampled image
        img_resized.save(save_path)

def downsample_images(src_dir, factor):
    # Automatically create the destination directory `images_2` next to the source folder
    dst_dir = os.path.join(os.path.dirname(src_dir), 'images_2')

    # Create the destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Iterate over all .jpg files in the source directory
    for img_file in os.listdir(src_dir):
        if img_file.endswith(".jpg"):
            src_img_path = os.path.join(src_dir, img_file)
            dst_img_path = os.path.join(dst_dir, img_file)
            # Downsample and save the image
            downsample_image(src_img_path, dst_img_path, factor)

    print(f"Downsampling complete. Images saved in {dst_dir}")

if __name__ == "__main__":
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Downsample all .jpg images in a folder by a given factor.")
    parser.add_argument("src_dir", type=str, help="The source directory containing the images.")
    parser.add_argument("--factor", type=int, default=2, help="Downsampling factor (default is 2).")

    args = parser.parse_args()

    # Check if the provided source directory exists
    if not os.path.isdir(args.src_dir):
        print(f"Error: The directory {args.src_dir} does not exist.")
        exit(1)

    # Call the function to downsample images
    downsample_images(args.src_dir, args.factor)