"""
3D Vision Transformer model for video diffusion
This module implements a ViT3D architecture specifically designed for video generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class TimeEmbedding(nn.Module):
    """
    Time embedding layer for diffusion timesteps

    Args:
        dim (int): Dimension of the time embedding
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time embedding

        Args:
            time (torch.Tensor): Time steps tensor

        Returns:
            torch.Tensor: Time embeddings
        """
        device = time.device
        emb = self.emb.to(device)
        # Expand dimensions for broadcasting
        time_expanded = time.reshape(-1, 1).float()
        emb_expanded = emb.reshape(1, -1)
        time_emb = time_expanded * emb_expanded
        time_emb = torch.cat([torch.sin(time_emb), torch.cos(time_emb)], dim=-1)
        return time_emb

class PatchEmbedding3D(nn.Module):
    """
    3D Patch Embedding layer

    Args:
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
        patch_size (Tuple[int, int, int]): Size of each patch (frames, height, width)
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: Tuple[int, int, int] = (2, 16, 16)):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Calculate number of patches
        self.patches_per_frame = (128 // patch_size[1]) * (128 // patch_size[2])
        self.num_frames = 28 // patch_size[0]
        self.num_patches = self.num_frames * self.patches_per_frame

        # Patch embedding layer
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, frames, height, width)

        Returns:
            torch.Tensor: Patch embeddings (batch_size, num_patches, embed_dim)
        """
        batch_size = x.shape[0]

        # Create patches: (batch_size, embed_dim, num_frames, num_patches_per_frame_h, num_patches_per_frame_w)
        x = self.proj(x)

        # Reshape to (batch_size, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # Add position embedding
        x = x + self.pos_embed

        return x

class Attention(nn.Module):
    """
    Multi-head self-attention mechanism

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)

        return output

class TransformerBlock(nn.Module):
    """
    Transformer encoder block

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mlp_dim (int): MLP hidden dimension
        dropout (float): Dropout rate
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, embed_dim)
        """
        # Self-attention
        x = x + self.attn(self.norm1(x))

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x

class ViT3D(nn.Module):
    """
    3D Vision Transformer for video diffusion

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        embed_dim (int): Embedding dimension
        time_dim (int): Time embedding dimension
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        mlp_dim (int): MLP hidden dimension
        dropout (float): Dropout rate
        patch_size (Tuple[int, int, int]): Size of each patch
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 768,
        time_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        patch_size: Tuple[int, int, int] = (2, 16, 16)
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Calculate number of patches
        self.patches_per_frame = (128 // patch_size[1]) * (128 // patch_size[2])
        self.num_frames = 28 // patch_size[0]
        self.num_patches = self.num_frames * self.patches_per_frame

        # Patch embedding
        self.patch_embed = PatchEmbedding3D(in_channels, embed_dim, patch_size)

        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, embed_dim)

        # Transformer layers
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, out_channels * patch_size[0] * patch_size[1] * patch_size[2])

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ViT3D

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, frames, height, width)
            time (torch.Tensor): Time steps tensor

        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        batch_size, channels, frames, height, width = x.shape

        # Create patch embeddings
        x_patched = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)

        # Time embedding
        time_emb = self.time_embedding(time)  # (batch_size, time_dim)
        time_emb = self.time_proj(time_emb)  # (batch_size, embed_dim)
        time_emb = time_emb.unsqueeze(1)  # (batch_size, 1, embed_dim)

        # Add time embedding to all patches
        x_patched = x_patched + time_emb

        # Apply transformer layers
        for layer in self.transformer:
            x_patched = layer(x_patched)

        # Final normalization
        x_patched = self.norm(x_patched)

        # Project to output space
        output_patches = self.output_proj(x_patched)  # (batch_size, num_patches, channels * patch_size^3)

        # Reshape patches back to spatial dimensions
        patch_channels = channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        output_patches = output_patches.view(batch_size, self.num_patches, patch_channels)

        # Reshape to spatial tensor
        # First, reshape to (batch_size, num_frames, patches_per_frame, channels * patch_size_t * patch_size_h * patch_size_w)
        output_patches = output_patches.view(
            batch_size,
            self.num_frames,
            self.patches_per_frame,
            channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        )

        # Then reshape patches_per_frame to spatial dimensions
        patches_h = 128 // self.patch_size[1]
        patches_w = 128 // self.patch_size[2]

        output = output_patches.view(
            batch_size,
            self.num_frames,
            patches_h,
            patches_w,
            channels,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2]
        )

        # Permute to correct order and reshape
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7)  # (batch, channels, frames, patch_t, patch_h, patch_w, h_patches, w_patches)
        output = output.reshape(batch_size, channels, frames, height, width)

        return output

    def _initialize_weights(self):
        """Initialize model weights deterministically"""
        torch.manual_seed(42)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_vit3d():
    """Test function to verify the ViT3D model works correctly"""
    from config import Config

    print("Testing ViT3D model...")

    # Create model with smaller config for testing
    model = ViT3D(
        in_channels=3,
        out_channels=3,
        embed_dim=384,  # Smaller for testing
        time_dim=384,
        num_layers=6,   # Fewer layers for testing
        num_heads=6,
        mlp_dim=1536,
        dropout=0.1
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 2
    channels, frames, height, width = (3, 28, 128, 128)

    x = torch.randn(batch_size, channels, frames, height, width)
    time = torch.randint(0, 1000, (batch_size,))

    with torch.no_grad():
        output = model(x, time)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    assert output.shape == x.shape, "Output shape doesn't match input shape"
    print("ViT3D test completed successfully!")

if __name__ == "__main__":
    test_vit3d()
