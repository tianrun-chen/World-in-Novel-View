"""
Multi-head attention implementation for GeoPE transformers.

This module provides a custom multi-head attention implementation
that supports custom scaled-dot-product-attention functions.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiheadAttention(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        sdpa_fn: Optional[Callable] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with optional custom SDPA function.
        
        Args:
            query: Query tensor
            key: Key tensor  
            value: Value tensor
            key_padding_mask: Optional padding mask
            need_weights: Whether to return attention weights
            attn_mask: Optional attention mask
            average_attn_weights: Whether to average attention weights
            is_causal: Whether to use causal attention
            sdpa_fn: Custom scaled-dot-product-attention function
            
        Returns:
            Tuple of (output, attention_weights)
        """
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = query.transpose(1, 0), key.transpose(1, 0), value.transpose(1, 0)

        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # Get dimensions
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.size(0)
        assert embed_dim == self.embed_dim, f"Expected embed_dim to be {self.embed_dim}, got {embed_dim}"

        # Compute Q, K, V
        if self._qkv_same_embed_dim:
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
        else:
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)

        # Reshape for multi-head attention
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Apply custom SDPA function if provided
        if sdpa_fn is not None:
            attn_output = sdpa_fn(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        else:
            # Use default scaled dot product attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, 
                is_causal=is_causal
            )
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        if not is_batched:
            attn_output = attn_output.squeeze(1)

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, None
