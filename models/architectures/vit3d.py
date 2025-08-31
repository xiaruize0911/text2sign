"""
3D Vision Transformer model for video diffusion
This module implements a ViT3D architecture specifically designed for video generation, following the input/output format of UNet3D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Optional, Tuple

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        self.register_buffer('emb', emb)
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        emb = self.emb.to(device)
        time_expanded = time.reshape(-1, 1).float()
        emb_expanded = emb.reshape(1, -1)
        time_emb = time_expanded * emb_expanded
        time_emb = torch.cat([torch.sin(time_emb), torch.cos(time_emb)], dim=-1)
        return time_emb

class ViT3D(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, frames: int = 28, height: int = 128, width: int = 128, time_dim: int = 128, text_dim: Optional[int] = None, embed_dim: Optional[int] = None, freeze_backbone: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frames = frames
        self.height = height
        self.width = width
        self.time_embedding = TimeEmbedding(time_dim)
        self.text_proj = nn.Linear(text_dim, time_dim) if text_dim is not None else None
        # Project time embedding to match output channels
        self.time_to_channels = nn.Linear(time_dim, out_channels)
        # Use torchvision's ViT-B/16 backbone
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.conv_proj = nn.Conv2d(in_channels, self.vit.conv_proj.out_channels, kernel_size=16, stride=16)
        self.feature_dim = self.vit.heads.head.in_features
        # Remove classification head
        self.vit.heads = nn.Identity()
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        # Calculate patch grid size for 224x224 input with patch size 16
        self.patch_size = 16
        self.vit_height = 224
        self.vit_width = 224
        self.patch_grid_h = self.vit_height // self.patch_size  # 14
        self.patch_grid_w = self.vit_width // self.patch_size   # 14
        self.num_patches = self.patch_grid_h * self.patch_grid_w  # 196
        
        # Project features back to pixel space (calculated dynamically)
        self.feature_proj = nn.Linear(self.feature_dim, out_channels * self.num_patches)
        # Final conv to match output shape
        self.final_conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x: torch.Tensor, time: torch.Tensor, text_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, channels, frames, height, width)
        batch_size = x.shape[0]
        time_emb = self.time_embedding(time)
        if text_emb is not None and self.text_proj is not None:
            text_emb_proj = self.text_proj(text_emb)
            time_emb = time_emb + text_emb_proj
        # Flatten frames into batch for feature extraction
        x_flat = x.permute(0,2,1,3,4).reshape(-1, self.in_channels, self.height, self.width) # (batch*frames, channels, height, width)
        # Resize to ViT expected size 224x224
        x_flat = F.interpolate(x_flat, size=(224, 224), mode='bilinear', align_corners=False)
        # Extract features for each frame
        feats = self.vit(x_flat) # (batch*frames, num_patches, 768)
        
        # Project features to pixel patches
        pixel_patches = self.feature_proj(feats) # (batch*frames, out_channels*num_patches)
        pixel_patches = pixel_patches.view(batch_size, self.frames, self.out_channels, self.patch_grid_h, self.patch_grid_w)
        
        # Upsample patches to full frame size
        upsampled = F.interpolate(
            pixel_patches.view(batch_size*self.frames, self.out_channels, self.patch_grid_h, self.patch_grid_w), 
            size=(self.height, self.width), 
            mode='bilinear', 
            align_corners=False
        )
        upsampled = upsampled.view(batch_size, self.frames, self.out_channels, self.height, self.width)
        # Permute to (batch, out_channels, frames, height, width)
        out = upsampled.permute(0,2,1,3,4)
        # Add time embedding (broadcast) - project to match output channels
        time_emb_proj = self.time_to_channels(time_emb)  # [batch, out_channels]
        time_emb_proj = time_emb_proj[:, :, None, None, None]  # [batch, out_channels, 1, 1, 1]
        out = out + time_emb_proj
        # Final conv for smoothing (no activation for ε-parameterization)
        out = self.final_conv(out)
        return out

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_vit3d():
    print("Testing ViT3D model...")
    batch_size = 2
    channels, frames, height, width = 3, 28, 128, 128
    model = ViT3D(in_channels=channels, out_channels=channels, frames=frames, height=height, width=width, time_dim=128)
    x = torch.randn(batch_size, channels, frames, height, width)
    time = torch.randint(0, 1000, (batch_size,))
    with torch.no_grad():
        output = model(x, time)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape doesn't match input shape"
    print("ViT3D test completed successfully!")

if __name__ == "__main__":
    test_vit3d()
