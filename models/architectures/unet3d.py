"""
3D UNet model for video diffusion
This module implements a UNet3D architecture specifically designed for video generation
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

class ResBlock3D(nn.Module):
    """
    3D Residual block with time conditioning
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        time_emb_dim (int): Dimension of time embedding
        groups (int): Number of groups for group normalization
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        
        # Adjust groups to be compatible with out_channels
        groups = min(groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        
        # First convolution block
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Second convolution block
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
            
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ResBlock3D
        
        Args:
            x (torch.Tensor): Input tensor
            time_emb (torch.Tensor): Time embedding
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Store input for residual connection
        residual = x
        
        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        # Reshape time embedding to match spatial dimensions
        while len(time_emb.shape) < len(x.shape):
            time_emb = time_emb.unsqueeze(-1)
        x = x + time_emb
        
        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Add residual connection
        x = x + self.residual_conv(residual)
        x = self.activation(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize model weights deterministically"""
        torch.manual_seed(42)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

class AttentionBlock3D(nn.Module):
    """
    3D Self-attention block
    
    Args:
        channels (int): Number of channels
        groups (int): Number of groups for group normalization
    """
    
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.channels = channels
        
        # Adjust groups to be compatible with channels
        groups = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
            
        self.norm = nn.GroupNorm(groups, channels)
        self.q = nn.Conv3d(channels, channels, kernel_size=1)
        self.k = nn.Conv3d(channels, channels, kernel_size=1)
        self.v = nn.Conv3d(channels, channels, kernel_size=1)
        self.out = nn.Conv3d(channels, channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AttentionBlock3D
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with self-attention applied
        """
        batch_size, channels, frames, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        
        # Compute query, key, value
        q = self.q(x).reshape(batch_size, channels, -1)
        k = self.k(x).reshape(batch_size, channels, -1)
        v = self.v(x).reshape(batch_size, channels, -1)
        
        # Compute attention weights
        attention = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(channels), dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attention.transpose(1, 2))
        out = out.reshape(batch_size, channels, frames, height, width)
        
        # Project output
        out = self.out(out)
        
        return out + residual

class UNet3D(nn.Module):
    """
    3D UNet for video diffusion
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        dim (int): Base dimension
        dim_mults (tuple): Dimension multipliers for each level
        time_dim (int): Time embedding dimension
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 3, 
        dim: int = 32,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        time_dim: int = 128
    ):
        super().__init__()
        
        # Calculate dimensions for each level
        dims = [dim * mult for mult in dim_mults]
        # For encoder: start with base dim, then apply multipliers
        encoder_dims = [(dim, dims[0])] + list(zip(dims[:-1], dims[1:]))
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels, dim, kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.encoder_resblocks1 = nn.ModuleList()
        self.encoder_resblocks2 = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        
        for i, (dim_in, dim_out) in enumerate(encoder_dims):
            is_last = i >= len(encoder_dims) - 1
            
            self.encoder_resblocks1.append(ResBlock3D(dim_in, dim_out, time_dim))
            self.encoder_resblocks2.append(ResBlock3D(dim_out, dim_out, time_dim))
            self.encoder_attentions.append(nn.Identity())  # Disable attention for MacBook M4
            self.encoder_downsamples.append(
                nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=2, padding=1) if not is_last else nn.Identity()
            )
        
        # Middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock3D(mid_dim, mid_dim, time_dim)
        self.mid_attention = nn.Identity()  # Disable attention for MacBook M4
        self.mid_block2 = ResBlock3D(mid_dim, mid_dim, time_dim)
        
        # Decoder (upsampling path)
        self.decoder_resblocks1 = nn.ModuleList()
        self.decoder_resblocks2 = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        for i, (dim_in, dim_out) in enumerate(reversed(encoder_dims)):
            is_last = i >= len(encoder_dims) - 1
            
            self.decoder_resblocks1.append(ResBlock3D(dim_out * 2, dim_in, time_dim))  # *2 for skip connections
            self.decoder_resblocks2.append(ResBlock3D(dim_in, dim_in, time_dim))
            self.decoder_attentions.append(nn.Identity())  # Disable attention for MacBook M4
            self.decoder_upsamples.append(
                nn.ConvTranspose3d(dim_in, dim_in, kernel_size=4, stride=2, padding=1) if not is_last else nn.Identity()
            )
        
        # Final convolution
        final_groups = min(8, dim)
        while dim % final_groups != 0 and final_groups > 1:
            final_groups -= 1
            
        
        # Initialize weights deterministically
        self._initialize_weights()
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(final_groups, dim),
            nn.SiLU(),
            nn.Conv3d(dim, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for UNet3D
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, frames, height, width)
            time (torch.Tensor): Time steps tensor
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        # Time embedding
        time_emb = self.time_embedding(time)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i in range(len(self.encoder_resblocks1)):
            x = self.encoder_resblocks1[i](x, time_emb)
            x = self.encoder_resblocks2[i](x, time_emb)
            x = self.encoder_attentions[i](x)
            skip_connections.append(x)
            x = self.encoder_downsamples[i](x)
        
        # Middle blocks
        x = self.mid_block1(x, time_emb)
        x = self.mid_attention(x)
        x = self.mid_block2(x, time_emb)
        
        # Decoder path
        for i in range(len(self.decoder_resblocks1)):
            # Add skip connection
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            
            x = self.decoder_resblocks1[i](x, time_emb)
            x = self.decoder_resblocks2[i](x, time_emb)
            x = self.decoder_attentions[i](x)
            x = self.decoder_upsamples[i](x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize model weights deterministically"""
        torch.manual_seed(42)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_unet3d():
    """Test function to verify the UNet3D model works correctly"""
    from config import Config
    
    print("Testing UNet3D model...")
    
    # Create model
    model = UNet3D(
        in_channels=Config.UNET_CHANNELS,
        out_channels=Config.UNET_CHANNELS,
        dim=Config.UNET_DIM,
        dim_mults=Config.UNET_DIM_MULTS,
        time_dim=Config.UNET_TIME_DIM
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    channels, frames, height, width = Config.INPUT_SHAPE
    
    x = torch.randn(batch_size, channels, frames, height, width)
    time = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        output = model(x, time)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    assert output.shape == x.shape, "Output shape doesn't match input shape"
    print("UNet3D test completed successfully!")

if __name__ == "__main__":
    test_unet3d()
