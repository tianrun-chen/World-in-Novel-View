"""
GeoPE (Geometric Positional Encoding) Attention Package

This package contains the implementation of GeoPE attention mechanism
for multiview transformers in novel view synthesis.
"""

from .geope_attention import GeoPEDotProductAttention, geope_dot_product_attention

__all__ = [
    "GeoPEDotProductAttention",
    "geope_dot_product_attention",
]
