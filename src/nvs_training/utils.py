"""
Training utilities for novel view synthesis.

This module provides utility functions for training, including
image saving, visualization, and data preprocessing.
"""

import os
from typing import Tuple

import cv2
import numpy as np
import torch
from einops import rearrange
from torch import Tensor


def write_tensor_to_image(
    tensor: Tensor,
    path: str,
    downscale: int = 1,
    sqrt: bool = False,
    point: Tuple[int, int] = None,
) -> None:
    """
    Write a tensor to an image file.
    
    Args:
        tensor: Image tensor [H, W, C] in range [0, 1]
        path: Output file path
        downscale: Downscale factor for the output image
        sqrt: Whether to reshape image to square layout
        point: Optional point to mark with a circle
    """
    assert tensor.ndim == 3, f"Expected 3D tensor, got {tensor.shape}"
    
    # Convert single channel to RGB if needed
    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)
    
    if sqrt:
        # Reshape image to square layout
        h, w = tensor.shape[:2]
        if h > w:
            n_images = h // w
            n_sqrt = int(np.sqrt(n_images))
            tensor = rearrange(tensor, "(n1 n2 h) w c -> (n1 h) (n2 w) c", n1=n_sqrt, n2=n_sqrt)
        elif h < w:
            n_images = w // h
            n_sqrt = int(np.sqrt(n_images))
            tensor = rearrange(tensor, "h (n1 n2 w) c -> (n1 h) (n2 w) c", n1=n_sqrt, n2=n_sqrt)
    
    # Create output directory
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert to numpy and scale to [0, 255]
    image = (tensor * 255).to(torch.uint8).detach().cpu().numpy()
    
    # Apply downscaling if needed
    if downscale > 1:
        image = cv2.resize(image, (0, 0), fx=1.0 / downscale, fy=1.0 / downscale)
    
    # Mark point if specified
    if point is not None:
        cv2.circle(image, point, 5, (255, 0, 0), -1)
    
    # Save image
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def create_visualization_grid(
    ref_images: Tensor,
    target_images: Tensor, 
    pred_images: Tensor,
    max_images: int = 10
) -> Tensor:
    """
    Create a visualization grid showing reference, target, and predicted images.
    
    Args:
        ref_images: Reference images [B, V_ref, H, W, C]
        target_images: Target images [B, V_tar, H, W, C]
        pred_images: Predicted images [B, V_tar, H, W, C]
        max_images: Maximum number of images to include in grid
        
    Returns:
        Visualization grid tensor
    """
    batch_size = min(ref_images.shape[0], max_images)
    
    # Select first batch item and limit views
    ref_vis = ref_images[:batch_size, :1]  # Use first reference view
    target_vis = target_images[:batch_size]
    pred_vis = pred_images[:batch_size]
    
    # Create left side: reference images
    ref_grid = rearrange(ref_vis, "b v h w c -> (b h) (v w) c")
    
    # Create right side: target and predicted images
    target_pred = torch.cat([target_vis, pred_vis], dim=3)  # Concatenate along width
    target_pred_grid = rearrange(target_pred, "b v h w c -> (b h) (v w) c")
    
    # Add separator
    separator = torch.ones(ref_grid.shape[0], 20, 3, device=ref_grid.device)
    
    # Combine grids
    grid = torch.cat([ref_grid, separator, target_pred_grid], dim=1)
    
    return grid


def compute_metrics(
    pred_images: Tensor,
    target_images: Tensor,
    psnr_fn,
    ssim_fn, 
    lpips_fn
) -> dict:
    """
    Compute evaluation metrics for predicted images.
    
    Args:
        pred_images: Predicted images [B, V, H, W, C]
        target_images: Target images [B, V, H, W, C]
        psnr_fn: PSNR function
        ssim_fn: SSIM function
        lpips_fn: LPIPS function
        
    Returns:
        Dictionary containing computed metrics
    """
    # Reshape for metric computation
    pred_flat = rearrange(pred_images, "b v h w c -> (b v) c h w")
    target_flat = rearrange(target_images, "b v h w c -> (b v) c h w")
    
    # Compute metrics
    psnr = psnr_fn(pred_flat, target_flat)
    ssim = ssim_fn(pred_flat, target_flat)
    lpips = lpips_fn(pred_flat, target_flat)
    
    return {
        "psnr": psnr.item(),
        "ssim": ssim.item(), 
        "lpips": lpips.item()
    }


def save_training_visualization(
    ref_images: Tensor,
    target_images: Tensor,
    pred_images: Tensor,
    output_dir: str,
    step: int
) -> None:
    """
    Save training visualization images.
    
    Args:
        ref_images: Reference images
        target_images: Target images
        pred_images: Predicted images
        output_dir: Output directory
        step: Training step
    """
    # Create visualization grid
    grid = create_visualization_grid(ref_images, target_images, pred_images)
    
    # Save images
    write_tensor_to_image(
        rearrange(pred_images, "b v h w c -> (b h) (v w) c"),
        f"{output_dir}/outputs_step_{step}.png"
    )
    write_tensor_to_image(
        rearrange(target_images, "b v h w c -> (b h) (v w) c"),
        f"{output_dir}/targets_step_{step}.png"
    )
    write_tensor_to_image(
        rearrange(ref_images, "b v h w c -> (b h) (v w) c"),
        f"{output_dir}/inputs_step_{step}.png"
    )
    write_tensor_to_image(
        grid,
        f"{output_dir}/comparison_step_{step}.png"
    )
