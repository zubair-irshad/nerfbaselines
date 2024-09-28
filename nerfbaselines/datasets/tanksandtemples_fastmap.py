import json
import os
import logging
import shutil
import requests
from pathlib import Path
from typing import Union, Optional, TypeVar
import tarfile
from tqdm import tqdm
import tempfile
import numpy as np
from nerfbaselines import UnloadedDataset, DatasetNotFoundError
from nerfbaselines.datasets import dataset_index_select
from nerfbaselines.datasets.colmap import load_colmap_dataset
from nerfbaselines._constants import DATASETS_REPOSITORY
from nerfbaselines import new_cameras, new_dataset
import glob

T = TypeVar("T")
DATASET_NAME = "tanksandtemples_fastmap"
# BASE_URL = f"https://{DATASETS_REPOSITORY}/resolve/main/tanksandtemples"
# _URL = f"{BASE_URL}/{{scene}}.tar.gz"
# del _URL
# _URL2DOWN = f"{BASE_URL}/{{scene}}_2down.tar.gz"

SCENES = {
    # advanced
    "auditorium": True,
    "ballroom": True,
    "courtroom": True,
    "museum": True,
    "palace": True,
    "temple": True,

    # intermediate
    "family": True,
    "francis": True,
    "horse": True,
    "lighthouse": True,
    "m60": True,
    "panther": True,
    "playground": True,
    "train": True,

    # training
    "barn": True,
    "caterpillar": True,
    "church": True,
    "courthouse": True,
    "ignatius": True,
    "meetingroom": True,
    "truck": True,
}


def _assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value

def read_trajectory(filename):
    traj = []
    poses = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape = (4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype = float, sep=' \t')
            mat = mat[:3, :4]
            poses.append(mat)
            metastr = f.readline()
    poses = np.array(poses)
    return poses


def _load_cameras(path, mask=None):
    poses = []
    intrinsics = []
    image_sizes = []
    image_names = []

    # file_path = os.path.join(path, "correct_courthouse_c2w.log")


    file_pattern = os.path.join(path, "*fastmap*.log")
    files = glob.glob(file_pattern)
    file_path = files[0]

    poses = read_trajectory(file_path)

    info_file = os.path.join(path, 'image_info.txt')
    with open(info_file, 'r') as f:
        lines = f.readlines()
        h = int(lines[0].split()[1])
        w = int(lines[1].split()[1])
        focal = float(lines[2].split()[1])


    for i in range(poses.shape[0]):
        intrinsics.append(np.array([focal, focal, w / 2, h / 2], dtype=np.float32))
        image_sizes.append(np.array([w, h], dtype=np.int32))

    print("len(poses)", len(poses))
    print("poses", poses.shape)
    if mask is not None:
        # Filter poses, intrinsics, and image_sizes based on the mask
        poses = [poses[i] for i in range(len(poses)) if mask[i] == 1]
        intrinsics = [intrinsics[i] for i in range(len(intrinsics)) if mask[i] == 1]
        image_sizes = [image_sizes[i] for i in range(len(image_sizes)) if mask[i] == 1]
        poses = np.array(poses)
    print("poses after", poses.shape)

    return new_cameras(
        poses=poses,
        intrinsics=np.stack(intrinsics),
        image_sizes=np.stack(image_sizes),
        camera_models=np.zeros(len(poses), dtype=np.uint8))

def _select_indices_llff(image_names, llffhold=8):
    inds = np.argsort(image_names)
    # inds = inds[::-1]
    all_indices = np.arange(len(image_names))
    indices_train = inds[all_indices % llffhold != 0]
    indices_test = inds[all_indices % llffhold == 0]
    return indices_train, indices_test

def load_tanksandtemples_fastmap_dataset(path, downscale_factor: int = 2, split=None, mask_indices= True, **kwargs):
    # image_paths = [os.path.join(path, "images", name) for name in image_names]

    # downscale_loaded_factor = 1
    images_path = "images" if downscale_factor == 1 else f"images_{downscale_factor}"
    # Load all the .jpg files in ascending order
    #Check if .jpg or .JPG or .png
    
    image_paths = sorted(glob.glob(os.path.join(path, images_path, '*.jpg')))


    if len(image_paths) == 0:
        image_paths = sorted(glob.glob(os.path.join(path, images_path, '*.JPG')))

    if mask_indices:
        print("Load mask.txt... before filering number of iamges", len(image_paths))

        with open(os.path.join(path, 'courthouse_mask.txt'), 'r') as file:
            mask = list(map(int, file.read().split()))

        # Step 2: Filter image_paths based on the mask
        # Assuming image_paths is already defined
        filtered_image_paths = [image_paths[i] for i in range(len(image_paths)) if mask[i] == 1]

        image_paths = filtered_image_paths
        print("after filtering, number of images", len(image_paths))

    # print("image_paths", image_paths)

    if mask_indices:
        cameras = _load_cameras(path, mask)
    else:
        cameras = _load_cameras(path)


    dataset = new_dataset(
        image_paths=image_paths,
        cameras=cameras,
        metadata={
            "id": None,
            "color_space": "srgb",
            "evaluation_protocol": "default",
        })

    scene = next((x for x in SCENES if x in str(path)), None)

    dataset["metadata"]["id"] = DATASET_NAME
    dataset["metadata"]["scene"] = scene
    dataset["metadata"]["downscale_factor"] = downscale_factor
    # dataset["metadata"]["downscale_loaded_factor"] = downscale_loaded_factor
    dataset["metadata"]["type"] = "object-centric"
    dataset["metadata"]["evaluation_protocol"] = "default"

    indices_train, indices_test = _select_indices_llff(dataset["image_paths"])

    print("===========================================\n\n")
    print("train_indices", indices_train)
    print("test_indices_array", indices_test)
    print("===========================================\n\n")
    
    indices = indices_train if split == "train" else indices_test
    return dataset_index_select(dataset, indices)

# def load_tanksandtemples_fastmap_dataset(path: Union[Path, str], split: str, downscale_factor: int = 2, **kwargs) -> UnloadedDataset:

#     # Load TT dataset
#     images_path = "images" if downscale_factor == 1 else f"images_{downscale_factor}"
#     scene = next((x for x in SCENES if x in str(path)), None)
#     assert scene is not None, f"Scene not found in path {path}"

#     dataset = load_colmap_dataset(path, images_path=images_path, split=None, **kwargs)
#     dataset["metadata"]["id"] = DATASET_NAME
#     dataset["metadata"]["scene"] = scene
#     dataset["metadata"]["downscale_factor"] = downscale_factor
#     dataset["metadata"]["type"] = "object-centric"
#     dataset["metadata"]["evaluation_protocol"] = "default"
#     indices_train, indices_test = _select_indices_llff(dataset["image_paths"])

#     print("===========================================\n\n")
#     print("train_indices", indices_train)
#     print("test_indices_array", indices_test)
#     print("===========================================\n\n")
    
#     indices = indices_train if split == "train" else indices_test
#     return dataset_index_select(dataset, indices)



__all__ = ["load_tanksandtemples_fastmap_dataset"]
