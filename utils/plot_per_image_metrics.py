import os
import json
import base64
import struct
import argparse
import matplotlib.pyplot as plt

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
    psnr = decode_base64(metrics_raw["psnr"])[0]
    ssim = decode_base64(metrics_raw["ssim"])[0]
    lpips = decode_base64(metrics_raw["lpips"])[0]
    
    return psnr, ssim, lpips

def plot_metrics(psnr_colmap, ssim_colmap, lpips_colmap, psnr_fastmap, ssim_fastmap, lpips_fastmap, scene_name):
    """Plot PSNR, SSIM, and LPIPS for colmap and fastmap."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot PSNR
    ax[0].plot(psnr_colmap, color='red', label='zipnerf_colmap')
    ax[0].plot(psnr_fastmap, color='blue', label='zipnerf_fastmap')
    ax[0].set_title('PSNR')
    ax[0].set_xlabel('Evaluation Index')
    ax[0].set_ylabel('PSNR')
    ax[0].legend()

    # Plot SSIM
    ax[1].plot(ssim_colmap, color='red', label='zipnerf_colmap')
    ax[1].plot(ssim_fastmap, color='blue', label='zipnerf_fastmap')
    ax[1].set_title('SSIM')
    ax[1].set_xlabel('Evaluation Index')
    ax[1].set_ylabel('SSIM')
    ax[1].legend()

    # Plot LPIPS
    ax[2].plot(lpips_colmap, color='red', label='zipnerf_colmap')
    ax[2].plot(lpips_fastmap, color='blue', label='zipnerf_fastmap')
    ax[2].set_title('LPIPS')
    ax[2].set_xlabel('Evaluation Index')
    ax[2].set_ylabel('LPIPS')
    ax[2].legend()

    plt.tight_layout()
    plt.savefig('results/'+'metrics'+scene_name+'.png')
    # plt.show()

def main(data_dir, scene):
    # subfolders = ["zipnerf_colmap_kitchen_by4", "zipnerf_fastmap_kitchen_by4"]

    subfolders = os.listdir(data_dir)

    psnr_colmap, ssim_colmap, lpips_colmap = [], [], []
    psnr_fastmap, ssim_fastmap, lpips_fastmap = [], [], []

    for subfolder in subfolders:
        folder_path = os.path.join(data_dir, subfolder)
        psnr, ssim, lpips = load_metrics(folder_path)

        if 'colmap' in subfolder:
        # if subfolder == "zipnerf_colmap_kitchen_by4":
            psnr_colmap.append(psnr)
            ssim_colmap.append(ssim)
            lpips_colmap.append(lpips)
        elif 'fastmap' in subfolder:
        # elif subfolder == "zipnerf_fastmap_kitchen_by4":
            psnr_fastmap.append(psnr)
            ssim_fastmap.append(ssim)
            lpips_fastmap.append(lpips)

    plot_metrics(psnr_colmap, ssim_colmap, lpips_colmap, psnr_fastmap, ssim_fastmap, lpips_fastmap, scene)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot PSNR, SSIM, and LPIPS from result json files.')
    parser.add_argument('data_dir', type=str, help='Path to the directory containing result subfolders.')
    parser.add_argument('scene', type=str, help='Path to the directory containing result subfolders.')
    
    args = parser.parse_args()
    main(args.data_dir, args.scene)
