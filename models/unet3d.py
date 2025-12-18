"""
3D UNet architecture for video diffusion with text conditioning
Enhanced with Transformer (DiT-style) blocks for better temporal modeling

Based on:
- Diffusion Transformers (DiT) - Peebles & Xie 2023
- Video diffusion models with temporal attention
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    """
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    
    return emb


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Generate 3D sinusoidal positional embeddings for video (T, H, W).
    """
    t, h, w = grid_size
    
    grid_t = torch.arange(t, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    
    grid = torch.meshgrid(grid_t, grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)  # (3, T, H, W)
    grid = grid.reshape(3, -1).T  # (T*H*W, 3)
    
    # Split embedding dim across 3 dimensions
    dim_t = embed_dim // 3
    dim_h = embed_dim // 3
    dim_w = embed_dim - dim_t - dim_h
    
    def get_1d_sincos(positions, dim):
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (dim // 2)))
        out = positions[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    
    emb_t = get_1d_sincos(grid[:, 0], dim_t)
    emb_h = get_1d_sincos(grid[:, 1], dim_h)
    emb_w = get_1d_sincos(grid[:, 2], dim_w)
    
    return torch.cat([emb_t, emb_h, emb_w], dim=1)  # (T*H*W, embed_dim)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 computation for stability"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep (DiT-style)"""
    def __init__(self, dim: int, time_embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(time_embed_dim, dim * 2)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: (B, time_embed_dim)
        scale_shift = self.proj(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Handle different input shapes
        if x.dim() == 3:  # (B, N, C)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        elif x.dim() == 5:  # (B, C, T, H, W)
            scale = scale[:, :, None, None, None]
            shift = shift[:, :, None, None, None]
        
        return self.norm(x) * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization with zero-init (DiT-style)"""
    def __init__(self, dim: int, time_embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(time_embed_dim, dim * 6)  # scale, shift, gate for both attn and ff
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        params = self.proj(t_emb)
        return self.norm(x), params.chunk(6, dim=-1)


class Upsample3D(nn.Module):
    """3D Upsampling with convolution"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        return self.conv(x)


class Downsample3D(nn.Module):
    """3D Downsampling with convolution"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=(1, 2, 2), padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlock3D(nn.Module):
    """3D Residual block with time and context conditioning"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
        )
        
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
        )
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.in_layers(x)
        
        # Add time embedding
        time_emb = self.time_emb_proj(time_emb)
        h = h + time_emb[:, :, None, None, None]
        
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class SpatialAttention(nn.Module):
    """Self-attention over spatial dimensions"""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = GroupNorm32(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        
        # Reshape to (B*T, C, H*W)
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h * w)
        
        # Normalize
        x_norm = self.norm(x_flat.view(b * t, c, h, w)).view(b * t, c, h * w)
        
        # QKV projection
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(b * t, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = k.view(b * t, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        v = v.view(b * t, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b * t, c, h * w)
        
        out = self.proj(out)
        out = out.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        
        return x + out


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning"""
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = head_dim * num_heads
        
        self.norm = GroupNorm32(32, query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        b, c, t, h, w = x.shape
        
        # Reshape to (B, T*H*W, C)
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        
        # Normalize
        x_norm = self.norm(x.view(b, c, -1)).permute(0, 2, 1)
        
        # QKV
        q = self.to_q(x_norm)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head
        q = q.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, t * h * w, -1)
        out = self.to_out(out)
        
        out = out.view(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        
        return x + out


class TemporalAttention(nn.Module):
    """Self-attention over temporal dimension"""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = GroupNorm32(32, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        
        # Reshape to (B*H*W, T, C)
        x_flat = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        
        # Normalize
        x_norm = self.norm(x.view(b, c, -1)).view(b, c, t, h, w)
        x_norm = x_norm.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        
        # QKV
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(b * h * w, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b * h * w, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b * h * w, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b * h * w, t, c)
        out = self.proj(out)
        
        out = out.view(b, h, w, t, c).permute(0, 4, 3, 1, 2)
        
        return x + out


# ============================================================================
# Transformer Components (DiT-style)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional flash attention and rotary embeddings.
    Supports both self-attention and cross-attention.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        is_cross_attention: bool = False,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.is_cross_attention = is_cross_attention
        
        if is_cross_attention:
            self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
            self.to_kv = nn.Linear(context_dim or dim, dim * 2, bias=qkv_bias)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        if self.is_cross_attention and context is not None:
            q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.to_kv(context).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block (DiT-style).
    Uses adaptive layer norm for timestep conditioning.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        
        # Self-attention with adaptive norm
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MultiHeadAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        
        # Cross-attention for text conditioning
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = MultiHeadAttention(
            dim, num_heads, 
            attn_drop=dropout, 
            proj_drop=dropout,
            is_cross_attention=True,
            context_dim=context_dim,
        )
        
        # Feed-forward with adaptive norm
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim, int(dim * mlp_ratio), dropout)
        
        # Adaptive parameters (DiT-style)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, dim * 9),  # 3 params each for 3 blocks
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get adaptive parameters
        params = self.adaLN_modulation(t_emb)
        (
            scale1, shift1, gate1,
            scale2, shift2, gate2,
            scale3, shift3, gate3,
        ) = params.unsqueeze(1).chunk(9, dim=-1)
        
        # Self-attention
        x_norm = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(x_norm)
        
        # Cross-attention
        if context is not None:
            x_norm = self.norm2(x) * (1 + scale2) + shift2
            x = x + gate2 * self.cross_attn(x_norm, context)
        
        # Feed-forward
        x_norm = self.norm3(x) * (1 + scale3) + shift3
        x = x + gate3 * self.ff(x_norm)
        
        return x


class TemporalTransformerBlock(nn.Module):
    """
    Transformer block specifically for temporal attention.
    Processes video frames attending to other frames.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MultiHeadAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        
        # Adaptive parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, dim * 3),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) temporal sequence
            t_emb: (B, time_embed_dim) timestep embedding
        """
        params = self.adaLN_modulation(t_emb)
        scale, shift, gate = params.unsqueeze(1).chunk(3, dim=-1)
        
        x_norm = self.norm(x) * (1 + scale) + shift
        x = x + gate * self.attn(x_norm)
        
        return x


class SpatioTemporalTransformer(nn.Module):
    """
    Combined spatial and temporal transformer for video understanding.
    First applies spatial attention within each frame, then temporal attention across frames.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        time_embed_dim: int,
        context_dim: int,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.spatial_blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, time_embed_dim, dropout=dropout, context_dim=context_dim)
            for _ in range(depth)
        ])
        
        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(dim, num_heads, time_embed_dim, dropout)
            for _ in range(depth)
        ])
    
    def forward(
        self,
        x: torch.Tensor,  # (B, C, T, H, W)
        t_emb: torch.Tensor,  # (B, time_embed_dim)
        context: torch.Tensor,  # (B, seq_len, context_dim)
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape
        
        # Spatial attention: process each frame
        # Reshape to (B*T, H*W, C)
        x_spatial = rearrange(x, 'b c t h w -> (b t) (h w) c')
        t_emb_spatial = repeat(t_emb, 'b d -> (b t) d', t=T)
        context_spatial = repeat(context, 'b n d -> (b t) n d', t=T)
        
        for block in self.spatial_blocks:
            x_spatial = block(x_spatial, t_emb_spatial, context_spatial)
        
        # Reshape back: (B, T, H*W, C)
        x_spatial = rearrange(x_spatial, '(b t) n c -> b t n c', b=B, t=T)
        
        # Temporal attention: process each spatial location
        # Reshape to (B*H*W, T, C)
        x_temporal = rearrange(x_spatial, 'b t n c -> (b n) t c', n=H*W)
        t_emb_temporal = repeat(t_emb, 'b d -> (b n) d', n=H*W)
        
        for block in self.temporal_blocks:
            x_temporal = block(x_temporal, t_emb_temporal)
        
        # Reshape back to (B, C, T, H, W)
        x_out = rearrange(x_temporal, '(b h w) t c -> b c t h w', b=B, h=H, w=W)
        
        return x_out


class TransformerBlock3D(nn.Module):
    """
    Enhanced Transformer block with spatial, temporal, and cross attention.
    Uses DiT-style adaptive layer norm for better timestep conditioning.
    """
    def __init__(
        self,
        channels: int,
        context_dim: int,
        time_embed_dim: int,
        num_heads: int = 8,
        transformer_depth: int = 1,
        use_spatio_temporal: bool = True,
    ):
        super().__init__()
        
        self.use_spatio_temporal = use_spatio_temporal
        
        if use_spatio_temporal:
            # Use the new SpatioTemporalTransformer
            self.transformer = SpatioTemporalTransformer(
                dim=channels,
                num_heads=num_heads,
                time_embed_dim=time_embed_dim,
                context_dim=context_dim,
                depth=transformer_depth,
            )
        else:
            # Fallback to simpler attention
            self.spatial_attn = SpatialAttention(channels, num_heads)
            self.temporal_attn = TemporalAttention(channels, num_heads)
            self.cross_attn = CrossAttention(
                query_dim=channels,
                context_dim=context_dim,
                num_heads=num_heads,
            )
        
        # Feed-forward (used in both cases)
        self.ff = nn.Sequential(
            GroupNorm32(32, channels),
            nn.Conv3d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv3d(channels * 4, channels, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_spatio_temporal and t_emb is not None:
            x = self.transformer(x, t_emb, context)
        else:
            x = self.spatial_attn(x)
            x = self.temporal_attn(x)
            x = self.cross_attn(x, context)
        
        x = x + self.ff(x)
        return x


class TemporalAttention(nn.Module):
    """Self-attention over temporal dimension (legacy, for backward compatibility)"""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = GroupNorm32(32, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        
        # Reshape to (B*H*W, T, C)
        x_flat = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        
        # Normalize
        x_norm = self.norm(x.view(b, c, -1)).view(b, c, t, h, w)
        x_norm = x_norm.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        
        # QKV
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(b * h * w, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b * h * w, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b * h * w, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b * h * w, t, c)
        out = self.proj(out)
        
        out = out.view(b, h, w, t, c).permute(0, 4, 3, 1, 2)
        
        return x + out


class UNet3D(nn.Module):
    """
    3D UNet for video diffusion with text conditioning.
    Enhanced with DiT-style transformer blocks for better temporal modeling.
    """
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (8, 16),
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_heads: int = 8,
        context_dim: int = 512,
        dropout: float = 0.1,
        use_transformer: bool = True,  # Use enhanced transformer blocks
        transformer_depth: int = 1,  # Depth of transformer blocks
        use_gradient_checkpointing: bool = False,  # Enable gradient checkpointing for memory
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_transformer = use_transformer
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input convolution
        self.input_blocks = nn.ModuleList([
            nn.Conv3d(in_channels, model_channels, 3, padding=1)
        ])
        
        # Downsampling
        ch = model_channels
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock3D(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(
                        TransformerBlock3D(
                            channels=ch,
                            context_dim=context_dim,
                            time_embed_dim=time_embed_dim,
                            num_heads=num_heads,
                            transformer_depth=transformer_depth,
                            use_spatio_temporal=use_transformer,
                        )
                    )
                
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([Downsample3D(ch)]))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle
        self.middle_block = nn.ModuleList([
            ResBlock3D(ch, ch, time_embed_dim, dropout),
            TransformerBlock3D(
                channels=ch,
                context_dim=context_dim,
                time_embed_dim=time_embed_dim,
                num_heads=num_heads,
                transformer_depth=transformer_depth,
                use_spatio_temporal=use_transformer,
            ),
            ResBlock3D(ch, ch, time_embed_dim, dropout),
        ])
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock3D(ch + ich, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(
                        TransformerBlock3D(
                            channels=ch,
                            context_dim=context_dim,
                            time_embed_dim=time_embed_dim,
                            num_heads=num_heads,
                            transformer_depth=transformer_depth,
                            use_spatio_temporal=use_transformer,
                        )
                    )
                
                if level and i == num_res_blocks:
                    layers.append(Upsample3D(ch))
                    ds //= 2
                
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, 3, padding=1),
        )
    
    def _checkpoint_forward(self, layer, h, t_emb, context=None):
        """Helper for gradient checkpointing"""
        if isinstance(layer, ResBlock3D):
            return layer(h, t_emb)
        elif isinstance(layer, TransformerBlock3D):
            return layer(h, context, t_emb)
        elif isinstance(layer, (Downsample3D, Upsample3D)):
            return layer(h)
        return h
    
    def forward(
        self,
        x: torch.Tensor,  # (B, C, T, H, W)
        timesteps: torch.Tensor,  # (B,)
        context: torch.Tensor,  # (B, seq_len, context_dim)
    ) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Noisy video tensor (B, C, T, H, W)
            timesteps: Diffusion timesteps (B,)
            context: Text embeddings (B, seq_len, context_dim)
        Returns:
            Predicted noise (B, C, T, H, W)
        """
        from torch.utils.checkpoint import checkpoint
        
        # Time embedding
        t_emb = get_timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Downsampling path
        hs = []
        h = x
        
        for module in self.input_blocks:
            if isinstance(module, nn.Conv3d):
                h = module(h)
            elif isinstance(module, nn.ModuleList):
                for layer in module:
                    if self.use_gradient_checkpointing and self.training:
                        h = checkpoint(self._checkpoint_forward, layer, h, t_emb, context, use_reentrant=False)
                    else:
                        h = self._checkpoint_forward(layer, h, t_emb, context)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(self._checkpoint_forward, layer, h, t_emb, context, use_reentrant=False)
            else:
                h = self._checkpoint_forward(layer, h, t_emb, context)
        
        # Upsampling path
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if self.use_gradient_checkpointing and self.training:
                    h = checkpoint(self._checkpoint_forward, layer, h, t_emb, context, use_reentrant=False)
                else:
                    h = self._checkpoint_forward(layer, h, t_emb, context)
        
        return self.out(h)


def create_unet(config) -> UNet3D:
    """Create UNet model from config"""
    return UNet3D(
        in_channels=config.in_channels,
        model_channels=config.model_channels,
        out_channels=config.in_channels,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        channel_mult=config.channel_mult,
        num_heads=config.num_heads,
        context_dim=config.context_dim,
        use_transformer=getattr(config, 'use_transformer', True),
        transformer_depth=getattr(config, 'transformer_depth', 1),
        use_gradient_checkpointing=getattr(config, 'use_gradient_checkpointing', False),
    )


if __name__ == "__main__":
    # Test the enhanced model with transformer blocks
    print("Testing UNet3D with DiT-style Transformer blocks...")
    
    model = UNet3D(
        in_channels=3,
        model_channels=64,
        channel_mult=(1, 2, 4),
        attention_resolutions=(8, 16),
        num_heads=4,
        context_dim=256,
        use_transformer=True,
        transformer_depth=1,
    )
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 16, 64, 64)  # (B, C, T, H, W)
    t = torch.randint(0, 1000, (batch_size,))
    context = torch.randn(batch_size, 77, 256)  # (B, seq_len, context_dim)
    
    # Forward pass
    out = model(x, t, context)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test backward pass
    loss = out.sum()
    loss.backward()
    print("Backward pass successful!")
    
    # Test without transformer (legacy mode)
    print("\nTesting UNet3D without transformer (legacy mode)...")
    model_legacy = UNet3D(
        in_channels=3,
        model_channels=64,
        channel_mult=(1, 2, 4),
        attention_resolutions=(8, 16),
        num_heads=4,
        context_dim=256,
        use_transformer=False,
    )
    
    out_legacy = model_legacy(x, t, context)
    print(f"Legacy output shape: {out_legacy.shape}")
    print(f"Legacy parameters: {sum(p.numel() for p in model_legacy.parameters()):,}")
