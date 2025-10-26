"""
Data preprocessing utilities for novel view synthesis.

This module provides functions for camera pose normalization,
image preprocessing, and data loading from custom formats.
"""

import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


def normalize_poses(
    camtoworld_matrices: torch.Tensor,
    scene_scale_factor: float = 1.35,
) -> torch.Tensor:
    """
    Normalize camera poses for consistent scene representation.
    
    From: https://github.com/Haian-Jin/LVSM/blob/ebeff4989a3e1ec38fcd51ae24919d0eadf38c8f/data/dataset_scene.py#L54-L95
    
    Preprocesses the poses to:
    1. Translate and rotate the scene to align the average camera direction and position
    2. Rescale the whole scene to a fixed scale
    
    Args:
        camtoworld_matrices: Camera-to-world transformation matrices [N, 4, 4]
        scene_scale_factor: Factor for scene scaling
        
    Returns:
        Normalized camera-to-world matrices [N, 4, 4]
    """
    # Translation and Rotation
    center = camtoworld_matrices[:, :3, 3].mean(0)
    avg_forward = F.normalize(camtoworld_matrices[:, :3, 2].mean(0), dim=-1)
    avg_down = camtoworld_matrices[:, :3, 1].mean(0)
    avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
    avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

    avg_pose = torch.eye(4, device=camtoworld_matrices.device)
    avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
    avg_pose[:3, 3] = center
    avg_pose = torch.linalg.inv(avg_pose)
    camtoworld_matrices = avg_pose @ camtoworld_matrices

    # Rescale the whole scene to a fixed scale
    scene_scale = torch.max(torch.abs(camtoworld_matrices[:, :3, 3]))
    scene_scale = scene_scale_factor * scene_scale

    camtoworld_matrices[:, :3, 3] /= scene_scale
    return camtoworld_matrices


def normalize_poses_identity_unit_distance(
    camtoworld_matrices: torch.Tensor,
    ref0_idx: int,
    ref1_idx: int,
) -> torch.Tensor:
    """
    Normalize poses such that ref0 camera is identity and ref1 camera is unit distance.
    
    Args:
        camtoworld_matrices: Camera-to-world transformation matrices [N, 4, 4]
        ref0_idx: Index of reference camera 0
        ref1_idx: Index of reference camera 1
        
    Returns:
        Normalized camera-to-world matrices [N, 4, 4]
    """
    ref0_c2w = camtoworld_matrices[ref0_idx]
    c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(ref0_c2w), camtoworld_matrices)

    ref1_c2w = c2ws[ref1_idx]
    # The original ref0 is now at origin, so we compute distance to origin
    dist = torch.linalg.norm(ref1_c2w[:3, 3])
    if dist > 1e-2:  # numerically stable
        c2ws[:, :3, 3] /= dist

    return c2ws


def resize_crop_with_subpixel_accuracy(
    image: np.ndarray, 
    intrinsics: np.ndarray, 
    patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize and crop the image to have the smallest side equal to patch_size,
    while maintaining sub-pixel accuracy using a single warpAffine transformation.
    
    Args:
        image: Input image [H, W, C]
        intrinsics: Camera intrinsics matrix [3, 3]
        patch_size: Target patch size
        
    Returns:
        Tuple of (processed_image, updated_intrinsics)
    """
    h, w = image.shape[:2]
    if h == patch_size and w == patch_size:
        return image, intrinsics
        
    scale = patch_size / min(h, w)

    new_w, new_h = w * scale, h * scale
    crop_x = (new_w - patch_size) / 2
    crop_y = (new_h - patch_size) / 2

    M = np.array([[scale, 0, -crop_x], [0, scale, -crop_y]], dtype=np.float32)

    is_downsampling = min(h, w) > patch_size
    interpolation = cv2.INTER_AREA if is_downsampling else cv2.INTER_CUBIC
    cropped_resized_image = cv2.warpAffine(
        image, M, (patch_size, patch_size), flags=interpolation
    )

    intrinsics_scaled = intrinsics.copy()
    intrinsics_scaled[:2, :] *= scale
    intrinsics_scaled[0, 2] -= crop_x
    intrinsics_scaled[1, 2] -= crop_y

    return cropped_resized_image, intrinsics_scaled


def center_zoom_in_with_subpixel_accuracy(
    image: np.ndarray, 
    intrinsics: np.ndarray, 
    scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zoom into the center of the image while maintaining sub-pixel accuracy.
    
    Args:
        image: Input image [H, W, C]
        intrinsics: Camera intrinsics matrix [3, 3]
        scale: Zoom scale factor
        
    Returns:
        Tuple of (zoomed_image, updated_intrinsics)
    """
    if scale == 1.0:
        return image, intrinsics

    h, w = image.shape[:2]
    center_x, center_y = w / 2, h / 2

    M = np.array(
        [[scale, 0, (1 - scale) * center_x], [0, scale, (1 - scale) * center_y]],
        dtype=np.float32,
    )

    zoomed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_AREA)

    intrinsics_zoomed = intrinsics.copy()
    intrinsics_zoomed[:2, :] *= scale
    intrinsics_zoomed[0, 2] += (1 - scale) * center_x
    intrinsics_zoomed[1, 2] += (1 - scale) * center_y

    return zoomed_image, intrinsics_zoomed


def load_frames_from_custom_format(
    json_path: str,
    frame_ids: List[int],
    patch_size: int = 256,
    zoom_factor: float = 1.0,
    random_zoom: bool = False,
) -> Dict[str, Any]:
    """
    Load and process frames from the custom JSON format.
    
    Args:
        json_path: Path to JSON metadata file
        frame_ids: List of frame indices to load
        patch_size: Target patch size for images
        zoom_factor: Zoom factor for images
        random_zoom: Whether to apply random zoom
        
    Returns:
        Dictionary containing loaded data
    """
    with open(json_path, "r") as f:
        meta_info = json.load(f)
    
    json_frames = meta_info["frames"]

    images, intrinsics, camtoworlds, image_paths = [], [], [], []

    for frame_id in frame_ids:
        frame_data = json_frames[frame_id]

        # Load image
        image_path = frame_data["image_path"]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
        image = imageio.imread(image_path)[..., :3]

        # Build intrinsics matrix
        fx, fy, cx, cy = frame_data["fxfycxcy"]
        intrinsics_raw = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=np.float32,
        )

        # Apply zoom and resize/crop
        per_image_zoom_factor = (
            np.random.uniform(1.0, zoom_factor) if random_zoom else zoom_factor
        )
        image_zoomed, intrinsics_zoomed = center_zoom_in_with_subpixel_accuracy(
            image, intrinsics_raw, per_image_zoom_factor
        )
        image_processed, intrinsics_processed = resize_crop_with_subpixel_accuracy(
            image_zoomed, intrinsics_zoomed, patch_size
        )

        # Load extrinsics matrix w2c and invert to get c2w
        world_to_camera = np.array(frame_data["w2c"], dtype=np.float32)
        camera_to_world = np.linalg.inv(world_to_camera)

        images.append(image_processed)
        intrinsics.append(intrinsics_processed)
        camtoworlds.append(camera_to_world)
        image_paths.append(image_path)

    return {
        "image": np.stack(images),
        "K": np.stack(intrinsics),
        "camtoworld": np.stack(camtoworlds),
        "image_path": image_paths,
    }
