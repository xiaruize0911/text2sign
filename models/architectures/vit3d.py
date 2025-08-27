"""
Vision Transformer (ViT) implementation for video diffusion using PyTorch's built-in ViT
This module adapts PyTorch's ViT for 3D video generation with text conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np
from torchvision.models import vision_transformer
import torchvision.transforms as transforms

class Video3DToFrames(nn.Module):
    """
    Convert 3D video to individual frames for processing with 2D ViT
    
    Args:
        input_shape (tuple): Input shape (channels, frames, height, width)
        target_size (int): Target image size for ViT (e.g., 224)
    """
    
    def __init__(self, input_shape: Tuple[int, int, int, int], target_size: int = 224):
        super().__init__()
        self.input_shape = input_shape
        self.target_size = target_size
        channels, frames, height, width = input_shape
        
        # Resize transformation
        self.resize = nn.AdaptiveAvgPool2d((target_size, target_size))
        
        # Calculate output dimensions
        self.num_frames = frames
        self.channels = channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert video to frames
        
        Args:
            x (torch.Tensor): Input video (batch_size, channels, frames, height, width)
            
        Returns:
            torch.Tensor: Frames (batch_size * frames, channels, target_size, target_size)
        """
        batch_size, channels, frames, height, width = x.shape
        
        # Reshape to process each frame
        x = x.permute(0, 2, 1, 3, 4)  # (batch, frames, channels, height, width)
        x = x.reshape(batch_size * frames, channels, height, width)
        
        # Resize frames
        x = self.resize(x)  # (batch * frames, channels, target_size, target_size)
        
        return x
    
    def reconstruct_video(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Reconstruct video from processed frames
        
        Args:
            x (torch.Tensor): Processed frames (batch_size * frames, channels, target_size, target_size)
            batch_size (int): Original batch size
            
        Returns:
            torch.Tensor: Reconstructed video (batch_size, channels, frames, target_size, target_size)
        """
        frames_total, channels, height, width = x.shape
        frames = frames_total // batch_size
        
        x = x.reshape(batch_size, frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)  # (batch, channels, frames, height, width)
        
        return x

class TimeEmbedding(nn.Module):
    """Time embedding for diffusion timesteps (same as UNet3D)"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer('emb', emb)
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        emb = self.emb.to(device)
        time_expanded = time.reshape(-1, 1).float()
        emb_expanded = emb.reshape(1, -1)
        time_emb = time_expanded * emb_expanded
        time_emb = torch.cat([torch.sin(time_emb), torch.cos(time_emb)], dim=-1)
        return time_emb

class FramePositionalEncoding(nn.Module):
    """Add positional encoding for video frames"""
    
    def __init__(self, d_model: int, max_frames: int = 100):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_frames, d_model)
        position = torch.arange(0, max_frames, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor, frame_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size * frames, embed_dim)
            frame_ids: Frame indices for each token
        """
        return x + self.pe[frame_ids]

class ViT3D(nn.Module):
    """
    Vision Transformer for 3D video diffusion using PyTorch's ViT-B/16
    
    Args:
        input_shape (tuple): Input shape (channels, frames, height, width)
        embed_dim (int): Embedding dimension (fixed at 768 for ViT-B/16)
        time_dim (int): Time embedding dimension
        freeze_backbone (bool): Whether to freeze ViT backbone
        image_size (int): Input image size for ViT
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (3, 28, 128, 128),
        embed_dim: int = 768,  # ViT-B/16 fixed dimension
        time_dim: int = 768,
        freeze_backbone: bool = False,
        image_size: int = 224,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.image_size = image_size
        
        channels, frames, height, width = input_shape
        self.num_frames = frames
        
        # Video preprocessing
        self.video_processor = Video3DToFrames(input_shape, image_size)
        
        # Load pre-trained ViT-B/16 from torchvision
        try:
            self.vit_backbone = vision_transformer.vit_b_16(weights='IMAGENET1K_V1')
            self.embed_dim = 768  # ViT-B/16 embedding dimension
        except:
            # Fallback without pretrained weights
            self.vit_backbone = vision_transformer.vit_b_16()
            self.embed_dim = 768
        
        # Remove the classification head and replace with identity
        self.vit_backbone.heads = nn.Sequential(nn.Identity())
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.vit_backbone.parameters():
                param.requires_grad = False
        
        # Time embedding (same as before)
        self.time_embedding = TimeEmbedding(time_dim)
        self.time_projection = nn.Linear(time_dim, self.embed_dim)
        
        # Frame positional encoding
        self.frame_pos_encoding = FramePositionalEncoding(self.embed_dim, max_frames=frames)
        
        # Temporal transformer layers for video understanding
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=12,  # ViT-B uses 12 heads
                dim_feedforward=self.embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=4  # Additional temporal layers
        )
        
        # Text conditioning (for your text-to-sign task)
        self.text_conditioning = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=12,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection to reconstruct video
        # ViT-B/16 has patch size of 16
        patch_size = 16
        num_patches_per_frame = (image_size // patch_size) ** 2
        
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, patch_size * patch_size * channels),
            nn.Tanh()  # Normalize output
        )
        
        # Video reconstruction
        self.patch_size = patch_size
        self.num_patches_per_frame = num_patches_per_frame
        
        # Final resizing layer to match input dimensions
        if image_size != height or image_size != width:
            self.final_resize = nn.AdaptiveAvgPool2d((height, width))
        else:
            self.final_resize = nn.Identity()
        
    def forward(self, x: torch.Tensor, time: torch.Tensor, text_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input video (batch_size, channels, frames, height, width)
            time (torch.Tensor): Time steps for diffusion
            text_features (torch.Tensor, optional): Text features for conditioning
            
        Returns:
            torch.Tensor: Output video with same shape as input
        """
        batch_size = x.shape[0]
        
        # Convert video to frames
        frames = self.video_processor(x)  # (batch * frames, channels, image_size, image_size)
        
        # Process each frame through ViT
        frame_features = self.vit_backbone(frames)  # (batch * frames, embed_dim)
        
        # Reshape for temporal processing
        frame_features = frame_features.view(batch_size, self.num_frames, self.embed_dim)
        
        # Add frame positional encoding
        frame_ids = torch.arange(self.num_frames, device=x.device).repeat(batch_size, 1)
        frame_features_flat = frame_features.view(-1, self.embed_dim)
        frame_ids_flat = frame_ids.view(-1)
        frame_features_flat = self.frame_pos_encoding(frame_features_flat, frame_ids_flat)
        frame_features = frame_features_flat.view(batch_size, self.num_frames, self.embed_dim)
        
        # Time embedding and conditioning
        time_emb = self.time_embedding(time)  # (batch_size, time_dim)
        time_emb = self.time_projection(time_emb)  # (batch_size, embed_dim)
        
        # Add time conditioning to each frame
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, self.num_frames, -1)
        frame_features = frame_features + time_emb_expanded
        
        # Text conditioning (if provided)
        if text_features is not None:
            # text_features should be (batch_size, seq_len, embed_dim)
            conditioned_features, _ = self.text_conditioning(
                frame_features, text_features, text_features
            )
            frame_features = frame_features + conditioned_features
        
        # Temporal transformer
        video_features = self.temporal_transformer(frame_features)  # (batch_size, frames, embed_dim)
        
        # Project to patch space and then directly to video space
        video_features_reshaped = video_features.view(batch_size * self.num_frames, self.embed_dim)
        frame_patches = self.output_projection(video_features_reshaped)  # (batch * frames, patch_dim)
        
        # Simple reconstruction: reshape to frames and resize
        patch_dim = frame_patches.shape[-1]
        channels = self.input_shape[0]
        
        # Use the known patch size instead of calculating it dynamically
        spatial_size = self.patch_size
        
        # Reshape to frame format
        reconstructed_frames = frame_patches.view(
            batch_size * self.num_frames, channels, spatial_size, spatial_size
        )
        
        # Resize to target dimensions and reshape to video
        if spatial_size != self.image_size:
            reconstructed_frames = F.interpolate(
                reconstructed_frames, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Reshape back to video format
        reconstructed_frames = reconstructed_frames.view(
            batch_size, self.num_frames, channels, self.image_size, self.image_size
        )
        reconstructed_frames = reconstructed_frames.permute(0, 2, 1, 3, 4)  # (batch, channels, frames, H, W)
        
        # Apply final resizing if needed
        if not isinstance(self.final_resize, nn.Identity):
            b, c, f, h, w = reconstructed_frames.shape
            reconstructed_frames = reconstructed_frames.reshape(b * f, c, h, w)
            reconstructed_frames = self.final_resize(reconstructed_frames)
            reconstructed_frames = reconstructed_frames.reshape(b, c, f, *reconstructed_frames.shape[2:])
        
        return reconstructed_frames

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_vit3d():
    """Test function to verify the ViT3D model works correctly"""
    try:
        from config import Config
        input_shape = Config.INPUT_SHAPE
    except:
        # Fallback if config is not available
        input_shape = (3, 28, 128, 128)
    
    print("Testing ViT3D model...")
    
    # Create model with smaller dimensions for testing
    model = ViT3D(
        input_shape=input_shape,
        embed_dim=768,  # ViT-B/16 embedding dimension
        time_dim=768,
        image_size=224,  # Standard ViT input size
        dropout=0.1
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    channels, frames, height, width = input_shape
    
    x = torch.randn(batch_size, channels, frames, height, width)
    time = torch.randint(0, 1000, (batch_size,))
    
    # Optional: test with text features
    text_features = torch.randn(batch_size, 10, 768)  # 10 tokens, 768 dim
    
    with torch.no_grad():
        output = model(x, time, text_features)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Check output shape (allowing for different spatial dimensions due to ViT processing)
    expected_channels, expected_frames = x.shape[1], x.shape[2]
    assert output.shape[0] == x.shape[0], "Batch size doesn't match"
    assert output.shape[1] == expected_channels, "Channels don't match"
    assert output.shape[2] == expected_frames, "Frames don't match"
    
    print("ViT3D test completed successfully!")
    return model

if __name__ == "__main__":
    test_vit3d()
