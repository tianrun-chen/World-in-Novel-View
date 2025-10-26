import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob


def reorg_imgs(base_dir):
    """
    Reorganize all images in sequence folder into sequence/images folder.
    """
    for mode in ["train", "test"]:
        seqs = os.listdir(os.path.join(base_dir, mode))
        count = 0
        for seq in tqdm(seqs, desc=f"Reorganizing {mode} dataset"):
            seq_dir = os.path.join(base_dir, mode, seq)
            if len(os.listdir(seq_dir)) == 0:
                os.rmdir(seq_dir)
                continue
            img_fps = glob.glob(os.path.join(seq_dir, "*.png"))
            if len(img_fps) == 0:
                continue
            imgs_dir = os.path.join(seq_dir, "images")
            os.makedirs(imgs_dir, exist_ok=True)
            for img_fp in img_fps:
                if os.path.exists(os.path.join(imgs_dir, os.path.basename(img_fp))):
                    os.remove(img_fp)
                else:
                    os.rename(img_fp, os.path.join(imgs_dir, os.path.basename(img_fp)))
            count += 1
        print(f"Reorganized {count} sequences in {mode} dataset")


def gen_transforms(base_dir):
    """
    Generate transforms.json for each sequence.
    Meta_file convention: https://google.github.io/realestate10k/download.html
    Each line contains:
    - 1: timestamp (int: microseconds since start of video)
    - 2-6: camera intrinsics (float: focal_length_x, focal_length_y, principal_point_x, principal_point_y)
    - 7-19: camera pose (floats forming 3x4 matrix in row-major order)
    """

    for mode in ["train", "test"]:
        count = 0
        seqs = os.listdir(os.path.join(base_dir, mode))
        for seq in tqdm(seqs, desc=f"Generating transforms for {mode} dataset"):
            infos = {"frames": []}
            seq_dir = os.path.join(base_dir, mode, seq)
            imgs_dir = os.path.join(seq_dir, "images")
            assert os.path.exists(
                imgs_dir
            ), f"Images directory {imgs_dir} does not exist"
            if len(os.listdir(imgs_dir)) == 0:
                continue
            output_fp = os.path.join(seq_dir, "transforms.json")
            # if os.path.exists(output_fp):
            #     continue
            img0_path = os.path.join(imgs_dir, os.listdir(imgs_dir)[0])
            img0 = np.array(Image.open(img0_path))
            h, w = img0.shape[:2]
            infos.update({"w": w, "h": h})

            meta_file = os.path.join(base_dir, "RealEstate10K", mode, seq + ".txt")
            assert os.path.exists(meta_file), f"Meta file {meta_file} does not exist"
            with open(meta_file, "r") as f:
                lines = f.readlines()

            fx, fy, cx, cy = map(float, lines[1].split()[1:5])
            infos.update({"fl_x": fx * w, "fl_y": fy * h, "cx": cx * w, "cy": cy * h})

            for idx, line in enumerate(lines):
                if idx == 0:
                    continue  # line 0 is the youtube url
                timestamp, _fx, _fy, _cx, _cy, _tx, _ty, *pose = map(
                    float, line.split()
                )
                img_path = os.path.join(imgs_dir, f"{int(timestamp):06d}.png")
                if not os.path.exists(img_path):
                    continue
                assert (
                    _fx == fx and _fy == fy and _cx == cx and _cy == cy
                ), f"Camera intrinsics mismatch: {_fx} {_fy} {_cx} {_cy} {fx} {fy} {cx} {cy}"

                w2c = np.array(pose).reshape(3, 4)
                w2c_4x4 = np.eye(4)
                w2c_4x4[:3, :4] = w2c
                c2w = np.linalg.inv(w2c_4x4)
                infos["frames"].append(
                    {
                        "file_path": f"images/{int(timestamp):06d}.png",  # save relative path
                        "transform_matrix": c2w.tolist(),
                    }
                )
            with open(output_fp, "w") as f:
                json.dump(infos, f)
            count += 1
        print(f"Generated {count} transforms for {mode} dataset")


if __name__ == "__main__":
    base_dir = "/scratch/partial_datasets/realestate10k"
    reorg_imgs(base_dir)
    gen_transforms(base_dir)
