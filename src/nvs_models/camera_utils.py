"""
Camera utilities for novel view synthesis.

This module provides camera-related data structures and utilities
for handling camera intrinsics and extrinsics.
"""

from collections import namedtuple
from typing import Tuple

import torch
from torch import Tensor

# Camera data structure
Camera = namedtuple("Camera", ["K", "camtoworld", "width", "height"])


def create_camera(
    K: Tensor,
    camtoworld: Tensor, 
    width: int,
    height: int
) -> Camera:
    """
    Create a Camera object with validation.
    
    Args:
        K: Camera intrinsics matrix [3, 3]
        camtoworld: Camera-to-world transformation [4, 4]
        width: Image width
        height: Image height
        
    Returns:
        Camera object
    """
    assert K.shape == (3, 3), f"Expected K to have shape (3, 3), got {K.shape}"
    assert camtoworld.shape == (4, 4), f"Expected camtoworld to have shape (4, 4), got {camtoworld.shape}"
    assert width > 0, f"Width must be positive, got {width}"
    assert height > 0, f"Height must be positive, got {height}"
    
    return Camera(K=K, camtoworld=camtoworld, width=width, height=height)


def validate_camera_batch(cameras: Camera) -> bool:
    """
    Validate a batch of cameras.
    
    Args:
        cameras: Camera object with batched tensors
        
    Returns:
        True if valid, False otherwise
    """
    try:
        batch_size = cameras.K.shape[0]
        assert cameras.camtoworld.shape[0] == batch_size
        assert cameras.K.shape[1:] == (3, 3)
        assert cameras.camtoworld.shape[1:] == (4, 4)
        return True
    except (AssertionError, IndexError):
        return False
