import json
import os
import logging
from pathlib import Path
from typing import Union, TypeVar
import numpy as np
from nerfbaselines import DatasetNotFoundError
from nerfbaselines.datasets import dataset_index_select
from nerfbaselines.datasets.colmap import load_colmap_dataset
from nerfbaselines._constants import DATASETS_REPOSITORY
from ._common import download_dataset_wrapper, download_archive_dataset


T = TypeVar("T")
DATASET_NAME = "tanksandtemples"
BASE_URL = f"https://{DATASETS_REPOSITORY}/resolve/main/tanksandtemples"
_URL = f"{BASE_URL}/{{scene}}.tar.gz"
del _URL
_URL2DOWN = f"{BASE_URL}/{{scene}}_2down.tar.gz"
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


def _select_indices_llff(image_names, llffhold=8):
    inds = np.argsort(image_names)
    # inds = inds[::-1]
    all_indices = np.arange(len(image_names))
    indices_train = inds[all_indices % llffhold != 0]
    indices_test = inds[all_indices % llffhold == 0]
    return indices_train, indices_test


def load_tanksandtemples_dataset(path: Union[Path, str], split: str, downscale_factor: int = 2, **kwargs) -> UnloadedDataset:
    path = Path(path)
    if split:
        assert split in {"train", "test"}
    if DATASET_NAME not in str(path) or not any(s in str(path).lower() for s in SCENES):
        raise DatasetNotFoundError(f"{DATASET_NAME} and {set(SCENES.keys())} is missing from the dataset path: {path}")

    # Load TT dataset
    images_path = "images" if downscale_factor == 1 else f"images_{downscale_factor}"
    scene = next((x for x in SCENES if x in str(path)), None)
    assert scene is not None, f"Scene not found in path {path}"

    dataset = load_colmap_dataset(path, images_path=images_path, split=None, **kwargs)
    dataset["metadata"]["id"] = DATASET_NAME
    dataset["metadata"]["scene"] = scene
    dataset["metadata"]["downscale_factor"] = downscale_factor
    dataset["metadata"]["type"] = "object-centric"
    dataset["metadata"]["evaluation_protocol"] = "default"
    indices_train, indices_test = _select_indices_llff(dataset["image_paths"])

    print("===========================================\n\n")
    print("train_indices", indices_train)
    print("test_indices_array", indices_test)
    print("===========================================\n\n")
    
    indices = indices_train if split == "train" else indices_test
    return dataset_index_select(dataset, indices)


def download_tanksandtemples_dataset(path: str, output: Union[Path, str]) -> None:
    output = Path(output)
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError("Dataset path must be equal to 'tanksandtemples' or must start with 'tanksandtemples/'.")

    if path == DATASET_NAME:
        for scene in SCENES:
            download_tanksandtemples_dataset(f"{DATASET_NAME}/{scene}", output/scene)
        return

    scene = path.split("/")[-1]
    if SCENES.get(scene) is None:
        raise RuntimeError(f"Unknown scene {scene}")
    url = _URL2DOWN.format(scene=scene)
    downscale_factor = 2
    prefix = scene + "/"
    nb_info = {
        "id": dataset_name,
        "scene": scene,
        "loader": "colmap",
        "evaluation_protocol": "default",
        "type": "object-centric",
        "downscale_factor": downscale_factor,
        "loader_kwargs": {
            "images_path": f"images_{downscale_factor}",
        },
    }
    download_archive_dataset(url, output, 
                             archive_prefix=prefix, 
                             nb_info=nb_info,
                             callback=_write_splits,
                             file_type="tar.gz")
    logging.info(f"Downloaded {DATASET_NAME}/{scene} to {output}")


def load_tanksandtemples_dataset(path, *args, **kwargs):
    del args, kwargs
    raise RuntimeError(f"The dataset was likely downloaded with an older version of NerfBaselines. Please remove `{path}` and try again.")


__all__ = ["download_tanksandtemples_dataset"]
