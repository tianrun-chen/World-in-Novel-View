"""
Transformer components for GeoPE attention mechanism.

This module provides modified transformer implementations that support
customized scaled-dot-product-attention functions and other enhancements.
"""

import copy
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList, Sequential

from .config import InstantiateConfig
from .mha import MultiheadAttention

__all__ = [
    "TransformerEncoderConfig",
    "TransformerEncoder",
    "TransformerDecoderConfig", 
    "TransformerDecoder",
    "TransformerEncoderLayerConfig",
    "TransformerEncoderLayer",
    "TransformerDecoderLayerConfig",
    "TransformerDecoderLayer",
]


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply modulation to input tensor."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Default SDPA function
default_sdpa_fn = F.scaled_dot_product_attention


@dataclass
class TransformerLayerConfig:
    """Base configuration for transformer layers."""
    d_model: int = 512
    nhead: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: Callable[[Tensor], Tensor] = F.relu
    layer_norm_eps: float = 1e-5
    batch_first: bool = False
    norm_first: bool = False
    bias: bool = True
    elementwise_affine: bool = True
    qk_norm: bool = False


@dataclass
class TransformerEncoderLayerConfig(TransformerLayerConfig, InstantiateConfig):
    """Configuration for transformer encoder layers."""
    _target: Type = field(default_factory=lambda: TransformerEncoderLayer)

    norm_type: Literal["layer_norm", "adaLN-Zero"] = "layer_norm"
    modulation_activation: Optional[Callable[[Tensor], Tensor]] = None


@dataclass
class TransformerDecoderLayerConfig(TransformerLayerConfig, InstantiateConfig):
    """Configuration for transformer decoder layers."""
    _target: Type = field(default_factory=lambda: TransformerDecoderLayer)


@dataclass
class TransformerEncoderConfig(InstantiateConfig):
    """Configuration for transformer encoder."""
    _target: Type = field(default_factory=lambda: TransformerEncoder)
    layer: TransformerEncoderLayerConfig = field(
        default_factory=TransformerEncoderLayerConfig
    )
    num_layers: int = 24
    input_norm: bool = False
    output_norm: bool = False
    checkpointing: bool = False


@dataclass
class TransformerDecoderConfig(InstantiateConfig):
    """Configuration for transformer decoder."""
    _target: Type = field(default_factory=lambda: TransformerDecoder)
    layer: TransformerDecoderLayerConfig = field(
        default_factory=TransformerDecoderLayerConfig
    )
    num_layers: int = 24
    input_norm: bool = False
    output_norm: bool = False
    checkpointing: bool = False


class TransformerEncoder(Module):
    """Transformer encoder with support for custom SDPA functions."""

    def __init__(self, cfg: TransformerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        encoder_layer = cfg.layer.setup()
        self.layers = _get_clones(encoder_layer, cfg.num_layers)
        self.num_layers = cfg.num_layers
        
        # Input normalization
        self.in_norm = (
            LayerNorm(
                cfg.layer.d_model,
                eps=cfg.layer.layer_norm_eps,
                elementwise_affine=cfg.layer.elementwise_affine,
                bias=cfg.layer.bias,
            )
            if cfg.input_norm
            else None
        )
        
        # Output normalization
        self.out_norm = (
            LayerNorm(
                cfg.layer.d_model,
                eps=cfg.layer.layer_norm_eps,
                elementwise_affine=cfg.layer.elementwise_affine,
                bias=cfg.layer.bias,
            )
            if cfg.output_norm
            else None
        )
        
        # AdaLN-Zero specific components
        if cfg.layer.norm_type == "adaLN-Zero":
            assert (
                cfg.layer.modulation_activation is not None
            ), "modulation_activation must be provided for adaLN-Zero"
            assert (
                cfg.layer.norm_first
            ), "only norm_first=True is supported for adaLN-Zero"
            self.final_modulation_mlp = Sequential(
                cfg.layer.modulation_activation,
                Linear(cfg.layer.d_model, 2 * cfg.layer.d_model, bias=cfg.layer.bias),
            )
            self.final_norm = LayerNorm(
                cfg.layer.d_model,
                eps=cfg.layer.layer_norm_eps,
                elementwise_affine=cfg.layer.elementwise_affine,
                bias=cfg.layer.bias,
            )

        self.checkpointing = cfg.checkpointing

    def forward(
        self,
        src: Tensor,
        sdpa_fn: Callable = default_sdpa_fn,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the encoder."""
        output = src

        if self.in_norm is not None:
            output = self.in_norm(output)

        for mod in self.layers:
            if self.checkpointing:
                output = torch.utils.checkpoint.checkpoint(
                    mod,
                    output,
                    sdpa_fn=sdpa_fn,
                    cond=cond,
                    use_reentrant=False,
                )
            else:
                output = mod(output, sdpa_fn=sdpa_fn, cond=cond)

        if self.out_norm is not None:
            # Ensure dtype compatibility
            output = self.out_norm(output.to(self.out_norm.weight.dtype))

        if self.cfg.layer.norm_type == "adaLN-Zero":
            shift, scale = self.final_modulation_mlp(cond).chunk(2, dim=-1)
            output = modulate(self.final_norm(output), shift, scale)

        return output


class TransformerDecoder(Module):
    """Transformer decoder with support for custom SDPA functions."""

    def __init__(self, cfg: TransformerDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        decoder_layer = cfg.layer.setup()
        self.layers = _get_clones(decoder_layer, cfg.num_layers)
        self.num_layers = cfg.num_layers
        
        # Input normalization
        self.in_norm = (
            LayerNorm(
                cfg.layer.d_model,
                eps=cfg.layer.layer_norm_eps,
                elementwise_affine=cfg.layer.elementwise_affine,
                bias=cfg.layer.bias,
            )
            if cfg.input_norm
            else None
        )
        
        # Output normalization
        self.out_norm = (
            LayerNorm(
                cfg.layer.d_model,
                eps=cfg.layer.layer_norm_eps,
                elementwise_affine=cfg.layer.elementwise_affine,
                bias=cfg.layer.bias,
            )
            if cfg.output_norm
            else None
        )
        self.checkpointing = cfg.checkpointing

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        sdpa_fn: Callable = default_sdpa_fn,
    ) -> Tensor:
        """Forward pass through the decoder."""
        output = tgt

        if self.in_norm is not None:
            output = self.in_norm(output)

        for mod in self.layers:
            if self.checkpointing:
                output = torch.utils.checkpoint.checkpoint(
                    mod, output, memory, sdpa_fn=sdpa_fn, use_reentrant=False
                )
            else:
                output = mod(output, memory, sdpa_fn=sdpa_fn)

        if self.out_norm is not None:
            output = self.out_norm(output)

        return output


class TransformerEncoderLayer(Module):
    """Transformer encoder layer with support for custom SDPA functions."""

    def __init__(self, cfg: TransformerEncoderLayerConfig) -> None:
        self.cfg = cfg
        super().__init__()
        
        if cfg.norm_type == "adaLN-Zero":
            assert (
                cfg.modulation_activation is not None
            ), "modulation_activation must be provided for adaLN-Zero"
            assert cfg.norm_first, "only norm_first=True is supported for adaLN-Zero"

        # Self-attention
        self.self_attn = MultiheadAttention(
            cfg.d_model,
            cfg.nhead,
            dropout=cfg.dropout,
            bias=cfg.bias,
            qk_norm=cfg.qk_norm,
        )
        
        # Feedforward network
        self.linear1 = Linear(cfg.d_model, cfg.dim_feedforward, bias=cfg.bias)
        self.dropout = Dropout(cfg.dropout)
        self.linear2 = Linear(cfg.dim_feedforward, cfg.d_model, bias=cfg.bias)

        # Layer normalization
        self.norm_first = cfg.norm_first
        self.norm1 = LayerNorm(
            cfg.d_model,
            eps=cfg.layer_norm_eps,
            bias=cfg.bias,
            elementwise_affine=cfg.elementwise_affine,
        )
        self.norm2 = LayerNorm(
            cfg.d_model,
            eps=cfg.layer_norm_eps,
            bias=cfg.bias,
            elementwise_affine=cfg.elementwise_affine,
        )
        self.dropout1 = Dropout(cfg.dropout)
        self.dropout2 = Dropout(cfg.dropout)

        self.activation = cfg.activation
        self.norm_type = cfg.norm_type
        
        # AdaLN-Zero specific components
        if cfg.norm_type == "adaLN-Zero":
            self.modulation_mlp = Sequential(
                cfg.modulation_activation,
                Linear(cfg.d_model, 6 * cfg.d_model, bias=cfg.bias),
            )

    def forward(
        self,
        src: Tensor,
        sdpa_fn: Callable = default_sdpa_fn,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the encoder layer."""
        assert (cond is None and self.norm_type == "layer_norm") or (
            cond is not None and self.norm_type == "adaLN-Zero"
        ), "cond must be None for layer_norm, and not None for adaLN-Zero"

        if self.norm_type == "adaLN-Zero":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation_mlp(cond).chunk(6, dim=1)
            )

        # Transformer block with norm-first or norm-after
        x = src
        if self.norm_first:
            if self.norm_type == "adaLN-Zero":
                y = modulate(self.norm1(x), shift_msa, scale_msa)
                x = x + gate_msa.unsqueeze(1) * self._self_attention_block(y, sdpa_fn=sdpa_fn)
                x = x + gate_mlp.unsqueeze(1) * self._feedforward_block(
                    modulate(self.norm2(x), shift_mlp, scale_mlp)
                )
            else:
                # Ensure dtype compatibility
                x = x + self._self_attention_block(
                    self.norm1(x.to(self.norm1.weight.dtype)), sdpa_fn=sdpa_fn
                )
                x = x + self._feedforward_block(
                    self.norm2(x.to(self.norm2.weight.dtype))
                )
        else:
            x = self.norm1(x + self._self_attention_block(x, sdpa_fn=sdpa_fn))
            x = self.norm2(x + self._feedforward_block(x))

        return x

    def _self_attention_block(self, x: Tensor, sdpa_fn: Callable = default_sdpa_fn) -> Tensor:
        """Self-attention block."""
        x = self.self_attn(x, x, x, sdpa_fn=sdpa_fn)
        return self.dropout1(x)

    def _feedforward_block(self, x: Tensor) -> Tensor:
        """Feedforward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(Module):
    """Transformer decoder layer with support for custom SDPA functions."""

    def __init__(self, cfg: TransformerDecoderLayerConfig) -> None:
        self.cfg = cfg
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiheadAttention(
            cfg.d_model,
            cfg.nhead,
            dropout=cfg.dropout,
            bias=cfg.bias,
            qk_norm=cfg.qk_norm,
        )
        
        # Cross-attention
        self.multihead_attn = MultiheadAttention(
            cfg.d_model,
            cfg.nhead,
            dropout=cfg.dropout,
            bias=cfg.bias,
            qk_norm=cfg.qk_norm,
        )
        
        # Feedforward network
        self.linear1 = Linear(cfg.d_model, cfg.dim_feedforward, bias=cfg.bias)
        self.dropout = Dropout(cfg.dropout)
        self.linear2 = Linear(cfg.dim_feedforward, cfg.d_model, bias=cfg.bias)

        # Layer normalization
        self.norm_first = cfg.norm_first
        self.norm1 = LayerNorm(
            cfg.d_model,
            eps=cfg.layer_norm_eps,
            bias=cfg.bias,
            elementwise_affine=cfg.elementwise_affine,
        )
        self.norm2 = LayerNorm(
            cfg.d_model,
            eps=cfg.layer_norm_eps,
            bias=cfg.bias,
            elementwise_affine=cfg.elementwise_affine,
        )
        self.norm3 = LayerNorm(
            cfg.d_model,
            eps=cfg.layer_norm_eps,
            bias=cfg.bias,
            elementwise_affine=cfg.elementwise_affine,
        )
        self.dropout1 = Dropout(cfg.dropout)
        self.dropout2 = Dropout(cfg.dropout)
        self.dropout3 = Dropout(cfg.dropout)

        self.activation = cfg.activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        sdpa_fn: Callable = default_sdpa_fn,
    ) -> Tensor:
        """Forward pass through the decoder layer."""
        x = tgt
        if self.norm_first:
            x = x + self._self_attention_block(self.norm1(x), sdpa_fn)
            x = x + self._cross_attention_block(self.norm2(x), memory, sdpa_fn)
            x = x + self._feedforward_block(self.norm3(x))
        else:
            x = self.norm1(x + self._self_attention_block(x, sdpa_fn))
            x = self.norm2(x + self._cross_attention_block(x, memory, sdpa_fn))
            x = self.norm3(x + self._feedforward_block(x))

        return x

    def _self_attention_block(self, x: Tensor, sdpa_fn: Callable = default_sdpa_fn) -> Tensor:
        """Self-attention block."""
        x = self.self_attn(x, x, x, sdpa_fn=sdpa_fn)
        return self.dropout1(x)

    def _cross_attention_block(
        self, x: Tensor, mem: Tensor, sdpa_fn: Callable = default_sdpa_fn
    ) -> Tensor:
        """Cross-attention block."""
        x = self.multihead_attn(x, mem, mem, sdpa_fn=sdpa_fn)
        return self.dropout2(x)

    def _feedforward_block(self, x: Tensor) -> Tensor:
        """Feedforward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module: Module, N: int) -> ModuleList:
    """Create N copies of a module."""
    return ModuleList([copy.deepcopy(module) for i in range(N)])


if __name__ == "__main__":
    """Test the transformer components."""
    torch.manual_seed(42)
    device = "cuda:0"

    # Test encoder
    encoder = TransformerEncoder(
        TransformerEncoderConfig(
            layer=TransformerEncoderLayerConfig(
                d_model=128,
                nhead=8,
                dropout=0.1,
                bias=True,
                qk_norm=True,
            ),
            num_layers=2,
        )
    ).to(device)

    src = torch.randn(10, 16, 128).to(device)
    output = encoder(src)
    print(f"Encoder output shape: {output.shape}, sum: {output.sum()}")

    # Test decoder
    decoder = TransformerDecoder(
        TransformerDecoderConfig(
            layer=TransformerDecoderLayerConfig(
                d_model=128,
                nhead=8,
                dropout=0.1,
                bias=True,
                qk_norm=True,
            ),
            num_layers=2,
        )
    ).to(device)

    tgt = torch.randn(10, 16, 128).to(device)
    memory = torch.randn(10, 16, 128).to(device)
    output = decoder(tgt, memory)
    print(f"Decoder output shape: {output.shape}, sum: {output.sum()}")
