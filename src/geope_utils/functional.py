"""
Functional utilities for GeoPE attention mechanism.

This module provides core utility functions for camera operations,
ray processing, and geometric transformations used in novel view synthesis.
"""

import time
from collections import namedtuple
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# Camera data structure
Camera = namedtuple("Camera", ["K", "camtoworld", "width", "height"])


def random_SO3(batch_size: Tuple[int], device="cpu") -> Tensor:
    """
    Generate random SO(3) rotation matrices.
    
    Args:
        batch_size: Shape of the batch
        device: Device to place tensors on
        
    Returns:
        Random rotation matrices with shape (*batch_size, 3, 3)
    """
    # Generate random matrices
    random_matrices = torch.randn((*batch_size, 3, 3), device=device)
    random_matrices = random_matrices.reshape(-1, 3, 3)

    # Apply QR decomposition
    q, r = torch.linalg.qr(random_matrices)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[..., None, :]

    # Ensure positive determinant
    det_q = torch.det(q)
    negative_det_indices = det_q < 0
    q[negative_det_indices, :, 2] *= -1
    q = q.reshape(*batch_size, 3, 3)

    return q


def random_SE3(batch_size: Tuple[int], device="cpu") -> Tensor:
    """
    Generate random SE(3) transformation matrices.
    
    Args:
        batch_size: Shape of the batch
        device: Device to place tensors on
        
    Returns:
        Random transformation matrices with shape (*batch_size, 4, 4)
    """
    random_matrices = torch.eye(4, device=device).repeat(*batch_size, 1, 1)
    random_matrices[..., :3, :3] = random_SO3(batch_size, device)
    random_matrices[..., :3, 3] = torch.randn(*batch_size, 3, device=device)
    return random_matrices


def patchify(x: Tensor, patch_size: int) -> Tensor:
    """
    Split an image tensor into patches.
    
    Args:
        x: Input image tensor with shape (..., H * P, W * P, C)
        patch_size: Size of each patch
        
    Returns:
        Output tensor with shape (..., H * W, P * P * C)
    """
    assert (
        x.shape[-3] % patch_size == 0
    ), f"Expected height to be divisible by patch_size, got {x.shape[-3]} % {patch_size}"
    assert (
        x.shape[-2] % patch_size == 0
    ), f"Expected width to be divisible by patch_size, got {x.shape[-2]} % {patch_size}"

    x = rearrange(
        x, "... (h ph) (w pw) c -> ... (h w) (ph pw c)", ph=patch_size, pw=patch_size
    )
    return x


def unpatchify(x: Tensor, height: int, width: int, patch_size: int) -> Tensor:
    """
    Combine patches into an image tensor.
    
    Args:
        x: Input tensor with shape (..., H * W, P * P * C)
        height: Height of the original image
        width: Width of the original image
        patch_size: Size of each patch
        
    Returns:
        Output tensor with shape (..., H * P, W * P, C)
    """
    assert height % patch_size == 0, f"Expected height to be divisible by patch_size, got {height} % {patch_size}"
    assert width % patch_size == 0, f"Expected width to be divisible by patch_size, got {width} % {patch_size}"

    x = rearrange(
        x,
        "... (h w) (ph pw c) -> ... (h ph) (w pw) c",
        h=height // patch_size,
        w=width // patch_size,
        ph=patch_size,
        pw=patch_size,
    )
    return x


def camera_to_raymap(
    Ks: Tensor,
    camtoworlds: Tensor,
    height: int,
    width: int,
    downscale: int = 1,
    include_ups: bool = False,
) -> Tensor:
    """
    Construct raymap from camera intrinsics and extrinsics.
    
    Note: This function expects OpenCV camera coordinates.
    
    Args:
        Ks: Camera intrinsics tensor with shape (..., 3, 3)
        camtoworlds: Camera extrinsics tensor with shape (..., 4, 4)
        height: Height of original image corresponding to intrinsics
        width: Width of original image corresponding to intrinsics
        downscale: Downscale factor for the raymap
        include_ups: Whether to include the up direction in the raymap
        
    Returns:
        Raymap tensor with shape (..., H, W, 6) or (..., H, W, 9) if include_ups
    """
    assert Ks.shape[-2:] == (3, 3), f"Expected Ks to have shape (..., 3, 3), got {Ks.shape}"
    assert camtoworlds.shape[-2:] == (4, 4), f"Expected camtoworlds to have shape (..., 4, 4), got {camtoworlds.shape}"
    assert width % downscale == 0, f"Expected width to be divisible by downscale, got {width} % {downscale}"
    assert height % downscale == 0, f"Expected height to be divisible by downscale, got {height} % {downscale}"

    # Downscale the intrinsics
    Ks = torch.stack(
        [
            Ks[..., 0, :] / downscale,
            Ks[..., 1, :] / downscale,
            Ks[..., 2, :],
        ],
        dim=-2,
    )  # [..., 3, 3]
    width //= downscale
    height //= downscale

    # Construct pixel coordinates
    x, y = torch.meshgrid(
        torch.arange(width, device=Ks.device),
        torch.arange(height, device=Ks.device),
        indexing="xy",
    )  # [H, W]
    coords = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1)  # [H, W, 3]

    # To camera coordinates [..., H, W, 3]
    dirs = torch.einsum("...ij,...hwj->...hwi", Ks.inverse(), coords)

    # To world coordinates [..., H, W, 3]
    dirs = torch.einsum("...ij,...hwj->...hwi", camtoworlds[..., :3, :3], dirs)
    dirs = F.normalize(dirs, p=2, dim=-1)

    # Camera origin in world coordinates [..., H, W, 3]
    origins = torch.broadcast_to(camtoworlds[..., None, None, :3, -1], dirs.shape)

    if include_ups:
        # Extract the up direction (second column)
        ups = torch.broadcast_to(camtoworlds[..., None, None, :3, 1], dirs.shape)
        ups = F.normalize(ups, p=2, dim=-1)
        return torch.cat([origins, dirs, ups], dim=-1)
    else:
        return torch.cat([origins, dirs], dim=-1)  # [..., H, W, 6]


def raymap_to_plucker(raymap: Tensor) -> Tensor:
    """
    Convert raymap to Plücker coordinates.
    
    Args:
        raymap: Raymap tensor with shape (..., H, W, 6)
        
    Returns:
        Plücker coordinates tensor with shape (..., H, W, 6)
    """
    assert raymap.shape[-1] == 6, f"Expected raymap to have shape (..., H, W, 6), got {raymap.shape}"
    ray_origins, ray_directions = torch.split(raymap, [3, 3], dim=-1)
    
    # Normalize ray directions to unit vectors
    ray_directions = F.normalize(ray_directions, p=2, dim=-1)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    return torch.cat([ray_directions, plucker_normal], dim=-1)


def time_function(repeats: int, func: Callable, *args, **kwargs) -> Tuple[float, float, any]:
    """
    Time a function execution and measure memory usage.
    
    Args:
        repeats: Number of times to repeat the function
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (average_time, memory_usage, function_result)
    """
    torch.cuda.reset_peak_memory_stats()
    mem_tic = torch.cuda.max_memory_allocated() / 1024**3

    # Warmup
    for _ in range(5):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Time the function
    start = time.time()
    for _ in range(repeats):
        results = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    mem = torch.cuda.max_memory_allocated() / 1024**3 - mem_tic
    return (end - start) / repeats, mem, results
