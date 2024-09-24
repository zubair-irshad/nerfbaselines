import os
import json
import base64
import struct
import argparse
import matplotlib.pyplot as plt
import numpy as np

def decode_base64(encoded_str):
    """Decode a base64 string into a list of floats."""
    decoded_bytes = base64.b64decode(encoded_str)
    num_floats = len(decoded_bytes) // 4
    return struct.unpack(f"<{num_floats}f", decoded_bytes)

def load_metrics(subfolder):
    """Load and decode metrics from the results-200000.json file in the given subfolder."""
    result_file = os.path.join(subfolder, "results-200000.json")
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    metrics_raw = data["metrics_raw"]
    all_psnr = decode_base64(metrics_raw["psnr"])
    all_ssim = decode_base64(metrics_raw["ssim"])
    all_lpips = decode_base64(metrics_raw["lpips"])
    
    return all_psnr, all_ssim, all_lpips

def plot_metrics(psnr_colmap, ssim_colmap, lpips_colmap, psnr_fastmap, ssim_fastmap, lpips_fastmap, scene_name):
    """Plot PSNR, SSIM, and LPIPS for colmap and fastmap."""

    plt.figure(figsize=(15, 5))

    # Number of bars
    n = len(psnr_colmap)

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    index = np.arange(n)

    # Create the bar plot
    plt.bar(index, psnr_colmap, bar_width, label='Colmap', alpha=0.7, color='red')
    plt.bar(index + bar_width, psnr_fastmap, bar_width, label='Fastmap', alpha=0.7, color='blue')


    # Add labels and title
    plt.xlabel('Sample Index')
    plt.ylabel('PSNR Values')

    title = 'Per Image PSNR Comparison for Scene: ' + scene_name
    plt.title(title)
    plt.xticks(index + bar_width / 2, [f'{i}' for i in range(n)])  # Center the x labels
    plt.legend()

    # Show the plot
    plt.tight_layout()

    #only plot PSNR
    # plt.plot(psnr_colmap, color='red', label='zipnerf_colmap')
    # plt.plot(psnr_fastmap, color='blue', label='zipnerf_fastmap')
    #make a bar plot

    # plt.bar(range(len(psnr_colmap)), psnr_colmap, color='red', label='zipnerf_colmap')
    # plt.bar(range(len(psnr_fastmap)), psnr_fastmap, color='blue', label='zipnerf_fastmap')
    #per image there should be too bars side by side


    # title = 'Per Image PSNR for ' + scene_name
    # plt.title(title)
    # plt.xlabel('Evaluation Index')
    # plt.ylabel('PSNR')
    # plt.legend()

    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # # Plot PSNR
    # ax[0].plot(psnr_colmap, color='red', label='zipnerf_colmap')
    # ax[0].plot(psnr_fastmap, color='blue', label='zipnerf_fastmap')
    # ax[0].set_title('PSNR')
    # ax[0].set_xlabel('Evaluation Index')
    # ax[0].set_ylabel('PSNR')
    # ax[0].legend()

    # # Plot SSIM
    # ax[1].plot(ssim_colmap, color='red', label='zipnerf_colmap')
    # ax[1].plot(ssim_fastmap, color='blue', label='zipnerf_fastmap')
    # ax[1].set_title('SSIM')
    # ax[1].set_xlabel('Evaluation Index')
    # ax[1].set_ylabel('SSIM')
    # ax[1].legend()

    # # Plot LPIPS
    # ax[2].plot(lpips_colmap, color='red', label='zipnerf_colmap')
    # ax[2].plot(lpips_fastmap, color='blue', label='zipnerf_fastmap')
    # ax[2].set_title('LPIPS')
    # ax[2].set_xlabel('Evaluation Index')
    # ax[2].set_ylabel('LPIPS')
    # ax[2].legend()

    # plt.tight_layout()
    plt.savefig('results/'+'metrics'+scene_name+'.png')
    # plt.show()

def main(data_dir, scene):
    # subfolders = ["zipnerf_colmap_kitchen_by4", "zipnerf_fastmap_kitchen_by4"]

    subfolders = os.listdir(data_dir)

    psnr_colmap, ssim_colmap, lpips_colmap = [], [], []
    psnr_fastmap, ssim_fastmap, lpips_fastmap = [], [], []

    for subfolder in subfolders:
        folder_path = os.path.join(data_dir, subfolder)
        all_psnr, all_ssim, all_lpips = load_metrics(folder_path)

        print("subfolder:", subfolder)

        if 'colmap' in subfolder:
            print("in colmap")
        # if subfolder == "zipnerf_colmap_kitchen_by4":
            # psnr_colmap.append(psnr)
            # ssim_colmap.append(ssim)
            # lpips_colmap.append(lpips)
            psnr_colmap = all_psnr
            ssim_colmap = all_ssim
            lpips_colmap = all_lpips

        elif 'fastmap' in subfolder:
            print("in fastmap")
        # elif subfolder == "zipnerf_fastmap_kitchen_by4":
            # psnr_fastmap.append(psnr)
            # ssim_fastmap.append(ssim)
            # lpips_fastmap.append(lpips)
            psnr_fastmap = all_psnr
            ssim_fastmap = all_ssim
            lpips_fastmap = all_lpips

    plot_metrics(psnr_colmap, ssim_colmap, lpips_colmap, psnr_fastmap, ssim_fastmap, lpips_fastmap, scene)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot PSNR, SSIM, and LPIPS from result json files.')
    parser.add_argument('data_dir', type=str, help='Path to the directory containing result subfolders.')
    parser.add_argument('scene', type=str, help='Path to the directory containing result subfolders.')
    
    args = parser.parse_args()
    main(args.data_dir, args.scene)
