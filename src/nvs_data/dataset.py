"""
Dataset classes for novel view synthesis.

This module provides training and evaluation dataset classes
for loading and processing multiview data.
"""

import glob
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    load_frames_from_custom_format,
    normalize_poses,
    normalize_poses_identity_unit_distance,
)


class TrainDataset(Dataset):
    """
    Training dataset for novel view synthesis.
    
    This dataset loads multiview sequences and samples reference
    and target views for training the NVS model.
    """
    
    def __init__(
        self,
        json_files: List[str],
        patch_size: int = 256,
        zoom_factor: float = 1.0,
        random_zoom: bool = False,
        input_views: int = 2,
        supervise_views: int = 6,
        verbose: bool = False,
    ):
        """
        Initialize training dataset.
        
        Args:
            json_files: List of JSON metadata file paths
            patch_size: Target patch size for images
            zoom_factor: Zoom factor for images
            random_zoom: Whether to apply random zoom
            input_views: Number of input reference views
            supervise_views: Number of target views to supervise
            verbose: Whether to print verbose information
        """
        super().__init__()
        self.json_files = np.array(json_files).astype(np.bytes_)
        self.patch_size = patch_size
        self.zoom_factor = zoom_factor
        self.random_zoom = random_zoom
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.verbose = verbose
        
        if self.verbose:
            print(f"[TrainDataset] Initialized with {len(self.json_files)} scenes.")

    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        return len(self.json_files)

    def _select_views(
        self, 
        num_frames: int, 
        min_frame_dist: int = 25, 
        max_frame_dist: int = 100 
    ) -> Optional[List[int]]:
        """
        Select reference and target views from a sequence.
        
        Args:
            num_frames: Total number of frames in the sequence
            min_frame_dist: Minimum frame distance between views
            max_frame_dist: Maximum frame distance between views
            
        Returns:
            List of selected frame indices or None if selection fails
        """
        if num_frames < self.input_views + self.supervise_views:
            return None
            
        max_frame_dist = min(num_frames - 1, max_frame_dist)
        if max_frame_dist <= min_frame_dist:
            min_frame_dist = max_frame_dist - 1
            if min_frame_dist < 1: 
                return None

        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        if num_frames <= frame_dist:
            return None
            
        start_index = random.randint(0, num_frames - frame_dist - 1)
        end_index = start_index + frame_dist
        
        # Ensure enough intermediate frames for sampling
        if end_index - start_index - 1 < self.supervise_views:
            return None
        
        supervise_indices = random.sample(
            range(start_index + 1, end_index), self.supervise_views
        )
        indices = [start_index, end_index] + supervise_indices
        return indices

    def __getitem__(self, _: Any) -> Dict[str, Any]:
        """
        Get a training sample.
        
        Args:
            _: Unused index parameter
            
        Returns:
            Dictionary containing training data
        """
        # Randomly select a JSON file
        json_path = str(np.random.choice(self.json_files), encoding="utf-8")
        
        # Load JSON to get frame count
        try:
            with open(json_path, "r") as f:
                num_frames = len(json.load(f)["frames"])
        except (IOError, json.JSONDecodeError):
            # Retry if file is corrupted
            return self.__getitem__(None)

        # Select views
        frame_ids = self._select_views(num_frames)
        if frame_ids is None:
            # Retry if selection fails
            return self.__getitem__(None)

        # Reorder to ensure input views are first and last
        frame_ids = sorted(frame_ids)
        frame_ids = [frame_ids[0], frame_ids[-1]] + frame_ids[1:-1]

        # Load data using preprocessing function
        loaded = load_frames_from_custom_format(
            json_path,
            frame_ids,
            patch_size=self.patch_size,
            zoom_factor=self.zoom_factor,
            random_zoom=self.random_zoom,
        )

        # Preprocess poses
        camtoworld = torch.from_numpy(loaded["camtoworld"]).float()
        camtoworld = normalize_poses_identity_unit_distance(
            camtoworld, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        intrinsics = torch.from_numpy(loaded["K"]).float()
        images = torch.from_numpy(loaded["image"]).float()
        image_paths = loaded["image_path"]

        return {
            "camtoworld": camtoworld,
            "K": intrinsics,
            "image": images,
            "image_path": image_paths,
        }


class EvalDataset(Dataset):
    """
    Evaluation dataset for novel view synthesis.
    
    This dataset loads test sequences with predefined view configurations
    for evaluating the NVS model.
    """
    
    def __init__(
        self,
        folder: str,
        patch_size: int = 256,
        zoom_factor: float = 1.0,
        verbose: bool = False,
        first_n: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        input_views: int = 2,
        supervise_views: int = 3,
        render_video: bool = False,
        test_index_fp: str = "./assets/evaluation_index_re10k.json",
    ):
        """
        Initialize evaluation dataset.
        
        Args:
            folder: Path to dataset folder
            patch_size: Target patch size for images
            zoom_factor: Zoom factor for images
            verbose: Whether to print verbose information
            first_n: Use only first N scenes
            rank: Process rank for distributed evaluation
            world_size: Total number of processes
            input_views: Number of input reference views
            supervise_views: Number of target views to supervise
            render_video: Whether to render video sequences
            test_index_fp: Path to test index file
        """
        super().__init__()
        self.patch_size = patch_size
        self.zoom_factor = zoom_factor
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.render_video = render_video

        # Load and parse JSON index file
        if test_index_fp is None:
            raise ValueError("`test_index_fp` must be provided for evaluation.")
            
        assert os.path.exists(test_index_fp), f"Index file not found: {test_index_fp}"
        with open(test_index_fp, "r") as f:
            index_info = json.load(f)
        index_info = {key: value for key, value in index_info.items() if value is not None}

        # Get valid scenes (intersection of index file and disk)
        scenes_in_index = set(index_info.keys())
        scenes_on_disk = set(
            os.path.splitext(os.path.basename(p))[0] 
            for p in glob.glob(os.path.join(folder, "metadata", "*.json"))
        )
        scenes = sorted(list(scenes_in_index & scenes_on_disk))

        if verbose:
            print(f"[EvalDataset] Found {len(scenes_in_index)} scenes in the index file.")
            print(f"[EvalDataset] Found {len(scenes_on_disk)} scenes on disk.")
            print(f"[EvalDataset] Using {len(scenes)} valid scenes for evaluation.")

        if first_n is not None:
            scenes = scenes[:first_n]

        if rank is not None and world_size is not None:
            scenes = scenes[rank::world_size]

        # Store metadata paths and view indices
        self.json_files = [os.path.join(folder, "metadata", f"{s}.json") for s in scenes]
        self.contexts = [index_info[s]["context"] for s in scenes]
        self.targets = [index_info[s]["target"] for s in scenes]
        self.filename = [s for s in scenes]

    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        return len(self.json_files)

    def __getitem__(self, scene_id: int) -> Dict[str, Any]:
        """
        Get an evaluation sample.
        
        Args:
            scene_id: Scene index
            
        Returns:
            Dictionary containing evaluation data
        """
        json_path = self.json_files[scene_id]
        
        # Get view IDs from preloaded lists
        context_view_ids = self.contexts[scene_id][:self.input_views]
        
        if self.render_video:
            # Use all target frames for video rendering
            target_view_ids = self.targets[scene_id]
        else:
            # Use specified number of target frames for evaluation
            target_view_ids = self.targets[scene_id][:self.supervise_views]
            
        frame_ids = context_view_ids + target_view_ids

        try:
            loaded = load_frames_from_custom_format(
                json_path,
                frame_ids,
                patch_size=self.patch_size,
                zoom_factor=self.zoom_factor,
            )
        except Exception as e:
            print(f"Error loading frames from {json_path} with frame_ids {frame_ids}: {e}")
            return None 

        # Preprocess poses (different normalization for evaluation)
        camtoworld = torch.from_numpy(loaded["camtoworld"]).float()
        camtoworld = normalize_poses(camtoworld)
        intrinsics = torch.from_numpy(loaded["K"]).float()
        images = torch.from_numpy(loaded["image"]).float()
        image_paths = loaded["image_path"]

        return {
            "camtoworld": camtoworld,
            "K": intrinsics,
            "image": images,
            "image_path": image_paths,
            "scene": os.path.basename(json_path),
            "filename": self.filename,
        }


if __name__ == "__main__":
    """Test the dataset classes."""
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    # Test TrainDataset
    train_json_files = sorted(glob.glob("/root/sfs_test/for-hzu/for-hzu/dataset/DATA/test/metadata/*.json"))
    print(f"Found {len(train_json_files)} training JSON files.")
    
    dataset = TrainDataset(train_json_files, verbose=True)
    if len(dataset) > 0:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4)
        data = next(iter(dataloader))
        print("TrainDataset output shapes:")
        print(data["image"].shape, data["K"].shape, data["camtoworld"].shape)
    else:
        print("TrainDataset is empty.")

    # Test EvalDataset
    testset = EvalDataset(folder="/data/re10k/test", verbose=True)
    if len(testset) > 0:
        data = testset[0]
        print("EvalDataset output shapes:")
        print(data["image"].shape)
    else:
        print("EvalDataset is empty.")
