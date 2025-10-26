"""
GeoPE Utilities Package

This package contains utility functions for GeoPE attention mechanism,
including camera operations, ray processing, and transformer components.
"""

from .functional import (
    Camera,
    camera_to_raymap,
    patchify,
    raymap_to_plucker,
    unpatchify,
    random_SO3,
    random_SE3,
)
from .transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
    TransformerEncoder,
    TransformerEncoderLayer,
)

__all__ = [
    "Camera",
    "camera_to_raymap",
    "patchify", 
    "raymap_to_plucker",
    "unpatchify",
    "random_SO3",
    "random_SE3",
    "TransformerEncoderConfig",
    "TransformerEncoderLayerConfig", 
    "TransformerEncoder",
    "TransformerEncoderLayer",
]
