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

class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        groups = min(groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
        self.activation = nn.SiLU()
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        time_emb_proj = self.time_mlp(time_emb)
        time_emb_proj = time_emb_proj[:, :, None, None, None]
        assert x.dim() == 5, f"Input x to ResBlock3D must be 5D, got {x.shape}"
        assert x.shape[1] == time_emb_proj.shape[1], f"Channel mismatch: x.shape={x.shape}, time_emb_proj.shape={time_emb_proj.shape}"
        x = x + time_emb_proj
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + self.residual_conv(residual)
        x = self.activation(x)
        return x

class AttentionBlock3D(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        groups = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.q = nn.Conv3d(channels, channels, kernel_size=1)
        self.k = nn.Conv3d(channels, channels, kernel_size=1)
        self.v = nn.Conv3d(channels, channels, kernel_size=1)
        self.out = nn.Conv3d(channels, channels, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frames, height, width = x.shape
        residual = x
        x = self.norm(x)
        q = self.q(x).reshape(batch_size, channels, -1)
        k = self.k(x).reshape(batch_size, channels, -1)
        v = self.v(x).reshape(batch_size, channels, -1)
        attention = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(channels), dim=-1)
        out = torch.bmm(v, attention.transpose(1, 2))
        out = out.reshape(batch_size, channels, frames, height, width)
        out = self.out(out)
        return out + residual

class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 4, dim: int = 32, dim_mults: Tuple[int, ...] = (1, 2, 4), time_dim: int = 128, text_dim: Optional[int] = None):
        super().__init__()
        dims = [dim * mult for mult in dim_mults]
        encoder_dims = [(dim, dims[0])] + list(zip(dims[:-1], dims[1:]))
        self.time_embedding = TimeEmbedding(time_dim)
        self.text_proj = nn.Linear(text_dim, time_dim) if text_dim is not None else None
        self.init_conv = nn.Conv3d(in_channels, dim, kernel_size=3, padding=1)
        self.encoder_resblocks1 = nn.ModuleList()
        self.encoder_resblocks2 = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(encoder_dims):
            is_last = i >= len(encoder_dims) - 1
            self.encoder_resblocks1.append(ResBlock3D(dim_in, dim_out, time_dim))
            self.encoder_resblocks2.append(ResBlock3D(dim_out, dim_out, time_dim))
            self.encoder_attentions.append(nn.Identity())
            self.encoder_downsamples.append(
                nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=2, padding=1) if not is_last else nn.Identity()
            )
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock3D(mid_dim, mid_dim, time_dim)
        self.mid_attention = nn.Identity()
        self.mid_block2 = ResBlock3D(mid_dim, mid_dim, time_dim)
        self.decoder_resblocks1 = nn.ModuleList()
        self.decoder_resblocks2 = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(encoder_dims)):
            is_last = i >= len(encoder_dims) - 1
            self.decoder_resblocks1.append(ResBlock3D(dim_out * 2, dim_in, time_dim))
            self.decoder_resblocks2.append(ResBlock3D(dim_in, dim_in, time_dim))
            self.decoder_attentions.append(nn.Identity())
            self.decoder_upsamples.append(
                nn.ConvTranspose3d(dim_in, dim_in, kernel_size=4, stride=2, padding=1) if not is_last else nn.Identity()
            )
        final_groups = min(8, dim)
        while dim % final_groups != 0 and final_groups > 1:
            final_groups -= 1
        self.final_conv = nn.Sequential(
            nn.GroupNorm(final_groups, dim),
            nn.SiLU(),
            nn.Conv3d(dim, out_channels, kernel_size=3, padding=1)
            # No final activation for ε-parameterization - model predicts raw noise
        )
    def forward(self, x: torch.Tensor, time: torch.Tensor, text_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        time_emb = self.time_embedding(time)
        if text_emb is not None and self.text_proj is not None:
            text_emb_proj = self.text_proj(text_emb)
            time_emb = time_emb + text_emb_proj
        x = self.init_conv(x)
        skip_connections = []
        for i in range(len(self.encoder_resblocks1)):
            x = self.encoder_resblocks1[i](x, time_emb)
            x = self.encoder_resblocks2[i](x, time_emb)
            x = self.encoder_attentions[i](x)
            skip_connections.append(x)
            x = self.encoder_downsamples[i](x)
        x = self.mid_block1(x, time_emb)
        x = self.mid_attention(x)
        x = self.mid_block2(x, time_emb)
        for i in range(len(self.decoder_resblocks1)):
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_resblocks1[i](x, time_emb)
            x = self.decoder_resblocks2[i](x, time_emb)
            x = self.decoder_attentions[i](x)
            x = self.decoder_upsamples[i](x)
        x = self.final_conv(x)
        return x

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_unet3d():
    print("Testing UNet3D model...")
    batch_size = 2
    channels, frames, height, width = 4, 28, 128, 128
    model = UNet3D(in_channels=channels, out_channels=channels, dim=32, dim_mults=(1,2,4), time_dim=128)
    x = torch.randn(batch_size, channels, frames, height, width)
    time = torch.randint(0, 1000, (batch_size,))
    with torch.no_grad():
        output = model(x, time)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape doesn't match input shape"
    print("UNet3D test completed successfully!")

if __name__ == "__main__":
    test_unet3d()
