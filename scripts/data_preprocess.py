import glob
import json
import os
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, List, Literal

import numpy as np
from PIL import Image
from tqdm import tqdm

from nvs.dataset import load_and_maybe_update_meta_info


def load_frames_from_meta_info(
    data_dir: str,
    meta_info: Dict[str, Any],
    frame_ids: List[int],
) -> Dict[str, Any]:
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    frames = meta_info["frames"]
    K = np.array(
        [
            [meta_info["fl_x"], 0, meta_info["cx"]],
            [0, meta_info["fl_y"], meta_info["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    c2ws = []
    for frame_id in frame_ids:
        frame = frames[frame_id]
        c2w = np.array(frame["transform_matrix"], dtype=np.float32) @ blender2opencv
        c2ws.append(c2w)

    return {
        "camtoworld": np.stack(c2ws),
        "K": K,
    }


def is_image_valid(filepath: str) -> bool:
    """Check if the image file is corrupted or not."""
    try:
        if filepath.endswith(".png") or filepath.endswith(".PNG"):
            # Quick Byte Check (Magic Numbers + EOF)
            with open(filepath, "rb") as f:
                start = f.read(8)
                f.seek(-12, os.SEEK_END)
                end = f.read(12)
            return start == b"\x89PNG\r\n\x1a\n" and end.endswith(b"IEND\xaeB\x60\x82")
        elif filepath.endswith(".jpg") or filepath.endswith(".JPG"):
            # Quick Byte Check (Magic Numbers + EOF)
            with open(filepath, "rb") as f:
                start = f.read(2)
                f.seek(-2, os.SEEK_END)
                end = f.read(2)
            return start == b"\xff\xd8" and end == b"\xff\xd9"
        else:
            # Slow Check by loading the pixels
            with Image.open(filepath) as img:
                img.load()
        return True
    except (IOError, ValueError):
        return False


def preprocess_and_cache_data(
    data_dir: str,
    cache_dir: str,
    verbose: bool = False,
    check_corrupt_images: bool = False,
    overwrite: bool = False,
):
    """Preprocess the data structured as follows:

    data_dir/
    |── transforms.json
    ├── images{_X}/
    |   ├── xxxx.{png,jpg}
    ...

    into

    cache_dir/
        data_dir/
        |── transforms.json (maybe updated)
        ├── images{_X}/ (soft linked to the original images)
        |   ├── xxxx.{png,jpg}
        ...

    We perform the following checks:
        - If "transforms.json" is missing, skip the scene.
        - If the image folder is missing, skip the scene.
        - When `check_corrupt_images=True`, try load image and remove corrupt images
          from `transforms.json`.
    """
    cache_data_dir = os.path.join(cache_dir, os.path.basename(data_dir.rstrip("/")))
    # dl3dv-140 store transforms.json in the nerfstudio or gaussian_splat directory
    if "dl3dv-140" in data_dir:
        data_dir = os.path.join(data_dir, "nerfstudio")

    if os.path.exists(os.path.join(cache_data_dir, "transforms.json")):
        if overwrite:
            os.rmdir(cache_data_dir)
        else:
            return

    json_path = os.path.join(data_dir, "transforms.json")

    # If "transforms.json" is missing, skip the scene.
    valid, meta_info = load_and_maybe_update_meta_info(json_path)
    if not valid:
        if verbose:
            print(f"[Skip] Invalid meta info: json_path = {json_path}")
        return
    image_subdir = meta_info["frames"][0]["file_path"].split("/")[0]

    # When `check_corrupt_images=True`, try load image and remove corrupt images
    # from `transforms.json`.
    if check_corrupt_images:
        frames = meta_info["frames"]
        valid_frames = []
        for frame in frames:
            img_path = os.path.join(data_dir, frame["file_path"])
            if not is_image_valid(img_path):
                print(f"[Corrupt] {img_path}")
            else:
                valid_frames.append(frame)
        meta_info["frames"] = valid_frames

    if len(meta_info["frames"]) == 0:
        if verbose:
            print("[Skip] Zero uncorrupted images.")
        return

    # Soft link the images
    os.makedirs(cache_data_dir, exist_ok=True)
    os.symlink(
        os.path.abspath(os.path.join(data_dir, image_subdir)),
        os.path.abspath(os.path.join(cache_data_dir, image_subdir)),
        target_is_directory=True,
    )

    # Save the updated meta info
    with open(os.path.join(cache_data_dir, "transforms.json"), "w") as f:
        json.dump(meta_info, f)


def preprocess_realestate10k(
    data_dir: str, cache_dir: str, split: Literal["train", "test"]
):
    data_dirs = sorted(glob.glob(f"{data_dir}/{split}/*"))
    with Pool(12) as p:
        _ = list(
            tqdm(
                p.imap_unordered(
                    partial(
                        preprocess_and_cache_data,
                        cache_dir=f"{cache_dir}/{split}/",
                        verbose=True,
                        # Do not remove corrupt images for the test set
                        # as we will use the indices to evaluate the performance.
                        # There shouldn't be any corrupt images anyway in this dataset.
                        check_corrupt_images=False,
                    ),
                    data_dirs,
                ),
                total=len(data_dirs),
            )
        )


if __name__ == "__main__":
    # Preprocess the realestate10k dataset
    data_dir = "/root/sfs_test/for-hzu/for-hzu/dataset/re10k"
    cache_dir = "/root/sfs_test/for-hzu/for-hzu/dataset/re10k/data_processed/realestate10k"
    preprocess_realestate10k(data_dir, cache_dir, split="train")
    preprocess_realestate10k(data_dir, cache_dir, split="test")
