"""
Loss functions for novel view synthesis training.

This module provides perceptual loss functions and other
loss components used in NVS model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PerceptualLoss(nn.Module):
    """
    Perceptual loss for novel view synthesis.
    
    This loss function uses a pre-trained network to compute
    perceptual similarity between generated and target images.
    """
    
    def __init__(self, net_type: str = "alex", normalize: bool = True):
        """
        Initialize perceptual loss.
        
        Args:
            net_type: Type of pre-trained network ("alex" or "vgg")
            normalize: Whether to normalize inputs
        """
        super().__init__()
        self.net_type = net_type
        self.normalize = normalize
        
        # Import here to avoid circular imports
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        
        self.lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type, 
            normalize=normalize
        )
    
    def forward(self, pred_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target images.
        
        Args:
            pred_images: Predicted images [B, C, H, W]
            target_images: Target images [B, C, H, W]
            
        Returns:
            Perceptual loss scalar
        """
        # Ensure inputs are in the correct format
        if pred_images.dim() == 5:  # [B, V, H, W, C] -> [B*V, C, H, W]
            pred_images = rearrange(pred_images, "b v h w c -> (b v) c h w")
        if target_images.dim() == 5:  # [B, V, H, W, C] -> [B*V, C, H, W]
            target_images = rearrange(target_images, "b v h w c -> (b v) c h w")
        
        # Move to same device as the loss function
        pred_images = pred_images.to(self.lpips_fn.device)
        target_images = target_images.to(self.lpips_fn.device)
        
        return self.lpips_fn(pred_images, target_images)


class CombinedLoss(nn.Module):
    """
    Combined loss function for NVS training.
    
    This loss combines MSE loss with perceptual loss for
    better training stability and visual quality.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        perceptual_weight: float = 0.5,
        perceptual_net_type: str = "alex",
    ):
        """
        Initialize combined loss.
        
        Args:
            mse_weight: Weight for MSE loss
            perceptual_weight: Weight for perceptual loss
            perceptual_net_type: Type of perceptual network
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss(
            net_type=perceptual_net_type,
            normalize=True
        )
    
    def forward(
        self, 
        pred_images: torch.Tensor, 
        target_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred_images: Predicted images
            target_images: Target images
            
        Returns:
            Combined loss scalar
        """
        # MSE loss
        mse = self.mse_loss(pred_images, target_images)
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            perceptual = self.perceptual_loss(pred_images, target_images)
            total_loss = self.mse_weight * mse + self.perceptual_weight * perceptual
        else:
            total_loss = self.mse_weight * mse
        
        return total_loss
