"""
KoNet (Knowledge Network) implementation.

This module contains the main KoNet model for novel view synthesis,
including the decoder-only architecture and configuration.
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from ..geope_attention import GeoPEDotProductAttention
from ..geope_utils.functional import (
    camera_to_raymap,
    patchify,
    raymap_to_plucker,
    unpatchify,
)
from ..geope_utils.transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
)
from .camera_utils import Camera


@dataclass
class KoNetDecoderOnlyModelConfig:
    """Configuration for KoNet Decoder-Only Model."""
    
    ref_views: int
    tar_views: int = 1

    encoder: TransformerEncoderConfig = field(
        default_factory=lambda: TransformerEncoderConfig(
            layer=TransformerEncoderLayerConfig(
                d_model=768,
                nhead=16,
                dim_feedforward=3072,
                dropout=0.0,
                activation=F.relu,
                layer_norm_eps=1e-5,
                batch_first=True,
                norm_first=True,
                bias=False,
                elementwise_affine=True,
                norm_type="layer_norm",
                modulation_activation=None,
                qk_norm=False,
            ),
            num_layers=6,
            input_norm=True,
            output_norm=True,
            checkpointing=False,
        ),
    )

    img_shape: Tuple[int, ...] = (256, 256, 3)
    cam_shape: Tuple[int, ...] = (256, 256, 6)
    patch_size: int = 8

    # Ray encoding options
    ray_encoding: Literal["plucker", "camray", "none", "raymap"] = "plucker"
    
    # Positional encoding options
    pos_enc: Literal["geope", "gta", "none"] = "geope"


class KoNetDecoderOnlyModel(nn.Module):
    """
    Knowledge Network with Decoder-Only Architecture.
    
    This model performs novel view synthesis using a transformer-based
    decoder-only architecture with geometric positional encoding (GeoPE).
    """
    
    def __init__(self, config: KoNetDecoderOnlyModelConfig):
        super().__init__()
        self.config = config

        # Initialize attention mechanism
        self.attention = GeoPEDotProductAttention(
            head_dim=config.encoder.layer.d_model // config.encoder.layer.nhead,
            patches_x=config.img_shape[1] // config.patch_size,
            patches_y=config.img_shape[0] // config.patch_size,
            image_width=config.img_shape[1],
            image_height=config.img_shape[0],
        )

        # Validate camera and image shape compatibility
        assert (
            config.cam_shape[:2] == config.img_shape[:2]
        ), f"Camera shape {config.cam_shape[:2]} must match image shape {config.img_shape[:2]}"

        # Initialize shared rays for 'none' ray encoding
        if config.ray_encoding == "none":
            shared_rays = torch.randn(config.cam_shape)
            self.shared_rays = nn.Parameter(shared_rays, requires_grad=False)

        # Query tokenizer for target camera
        self.query_tokenizer = nn.Linear(
            config.cam_shape[-1] * config.patch_size**2,
            config.encoder.layer.d_model,
            bias=config.encoder.layer.bias,
        )
        
        # Input tokenizer for reference images and cameras
        self.input_tokenizer = nn.Linear(
            (
                config.img_shape[-1] * config.patch_size**2
                + config.cam_shape[-1] * config.patch_size**2
            ),
            config.encoder.layer.d_model,
            bias=config.encoder.layer.bias,
        )

        # Initialize transformer encoder
        self.encoder = self.config.encoder.setup()

        # Output layer
        self.output_layer = nn.Linear(
            config.encoder.layer.d_model,
            config.img_shape[-1] * config.patch_size**2,
            bias=config.encoder.layer.bias,
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights with proper scaling."""
        for idx, layer in enumerate(self.encoder.layers):
            layer.apply(self._init_layer_weights(idx))

    def _init_layer_weights(self, layer_idx: int):
        """Initialize weights for a specific layer."""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight, 
                    mean=0, 
                    std=0.02 / (2 * (layer_idx + 1)) ** 0.5
                )
        return _init_weights

    def create_rays(self, cameras: Camera) -> Tensor:
        """
        Convert cameras to raymaps.
        
        Args:
            cameras: Camera object containing intrinsics and extrinsics
            
        Returns:
            Ray tensor with shape [B, V, H, W, C]
        """
        config = self.config
        batch_size, num_views = cameras.camtoworld.shape[:2]
        cam_dtype = cameras.camtoworld.dtype
        device = cameras.camtoworld.device

        if config.ray_encoding == "none":
            rays = repeat(
                self.shared_rays, 
                "h w c -> b v h w c", 
                b=batch_size, 
                v=num_views
            )
        else:
            # Preprocess cameras into rays
            downscale = config.img_shape[0] // config.cam_shape[0]
            rays = camera_to_raymap(
                Ks=cameras.K,
                camtoworlds=(
                    torch.eye(4, dtype=cam_dtype, device=device).broadcast_to(
                        cameras.camtoworld.shape
                    )
                    if config.ray_encoding == "camray"
                    else cameras.camtoworld
                ),
                height=cameras.height,
                width=cameras.width,
                downscale=downscale,
            )
            
            if config.ray_encoding in ["plucker", "camray"]:
                rays = raymap_to_plucker(rays)
            else:
                assert config.ray_encoding == "raymap"
                
        return rays

    def forward(
        self,
        ref_images: Tensor,
        ref_cameras: Camera,
        tar_cameras: Camera,
    ) -> Tensor:
        """
        Forward pass of the LVSM model.
        
        Args:
            ref_images: Reference images [B, V_ref, H, W, C]
            ref_cameras: Reference cameras
            tar_cameras: Target cameras
            
        Returns:
            Generated target images [B, V_tar, H, W, C]
        """
        batch_size, num_target_views = tar_cameras.camtoworld.shape[:2]
        config = self.config

        # Create rays with contiguous memory layout
        ref_rays = self.create_rays(ref_cameras).contiguous()
        tar_rays = self.create_rays(tar_cameras).contiguous()

        # Patchify operations with contiguous memory
        ref_images = patchify(ref_images, config.patch_size).contiguous()
        ref_rays = patchify(ref_rays, config.patch_size).contiguous()
        tar_rays = patchify(tar_rays, config.patch_size).contiguous()

        # Tokenize inputs
        x = self.input_tokenizer(
            torch.cat([ref_images, ref_rays], dim=-1)
        ).contiguous()
        x = repeat(x, "b v1 n d -> (b v2) (v1 n) d", v2=num_target_views).contiguous()
        
        q = self.query_tokenizer(tar_rays).contiguous()
        q = rearrange(q, "b v2 n d -> (b v2) n d").contiguous()
        query_tokens = q.shape[1]

        # Process camera parameters with contiguous memory
        ref_c2ws = repeat(
            ref_cameras.camtoworld, 
            "b v1 x y -> (b v2) v1 x y", 
            v2=num_target_views
        ).contiguous()
        ref_Ks = repeat(
            ref_cameras.K, 
            "b v1 x y -> (b v2) v1 x y", 
            v2=num_target_views
        ).contiguous()
        tar_c2ws = rearrange(
            tar_cameras.camtoworld, 
            "b v2 x y -> (b v2) 1 x y", 
            v2=num_target_views
        ).contiguous()
        tar_Ks = rearrange(
            tar_cameras.K, 
            "b v2 x y -> (b v2) 1 x y"
        ).contiguous()
        
        # Combine camera parameters
        c2ws = torch.cat([ref_c2ws, tar_c2ws], dim=1).contiguous()
        Ks = torch.cat([ref_Ks, tar_Ks], dim=1).contiguous()
        viewmats = torch.inverse(c2ws).contiguous()

        def scaled_dot_product_attention_fn(q, k, v, **kwargs):
            """Custom SDPA function with GeoPE encoding."""
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            
            if config.pos_enc == "geope":
                return self.attention(
                    q, k, v, 
                    viewmats=viewmats.contiguous(), 
                    Ks=Ks.contiguous(), 
                    **kwargs
                )
            elif config.pos_enc == "gta":
                return self.attention(
                    q, k, v, 
                    viewmats=viewmats.contiguous(), 
                    Ks=None, 
                    **kwargs
                )
            elif config.pos_enc == "none":
                return F.scaled_dot_product_attention(q, k, v, **kwargs)
            else:
                raise ValueError(f"Invalid positional encoding: {config.pos_enc}")

        # Process through encoder
        xq = torch.cat([x, q], dim=1).contiguous()
        xq = self.encoder(xq, sdpa_fn=scaled_dot_product_attention_fn).contiguous()
        q = xq[:, -query_tokens:, :].contiguous()
        q = rearrange(q, "(b v) n d -> b v n d", b=batch_size, v=num_target_views).contiguous()

        # Generate output
        output = self.output_layer(q).contiguous()
        output = unpatchify(
            output,
            height=config.img_shape[0],
            width=config.img_shape[1],
            patch_size=config.patch_size,
        )
        return output.contiguous()


if __name__ == "__main__":
    """Test the model."""
    import tqdm

    device = "npu:0"
    ref_views = 2
    tar_views = 4
    batch_size = 1
    height = 256
    width = 256

    # Create test data
    ref_images = torch.randn(batch_size, ref_views, height, width, 3).to(device)
    ref_cameras = Camera(
        K=torch.randn(1, ref_views, 3, 3).to(device),
        camtoworld=torch.randn(1, ref_views, 4, 4).to(device),
        height=height,
        width=width,
    )
    tar_cameras = Camera(
        K=torch.randn(1, tar_views, 3, 3).to(device),
        camtoworld=torch.randn(1, tar_views, 4, 4).to(device),
        height=height,
        width=width,
    )

    # Initialize model
    config = KoNetDecoderOnlyModelConfig(ref_views=2)
    model = KoNetDecoderOnlyModel(config).to(device)
    
    # Test forward pass
    with torch.npu.amp.autocast():
        for _ in tqdm.trange(100):
            output = model(ref_images, ref_cameras, tar_cameras)
        assert output.shape == (batch_size, tar_views, height, width, 3)
        print("Model test passed!")
