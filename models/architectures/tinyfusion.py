"""TinyFusion video diffusion architecture wrapper.

This module adapts the TinyFusion 2D diffusion backbone (https://github.com/VainF/TinyFusion)
so it can be plugged into the Text2Sign training pipeline. The wrapper handles
loading the pretrained TinyFusion checkpoints (either via torch.hub or a local
checkpoint path) and runs the network frame-by-frame while keeping the overall
video tensor shape compatible with the rest of the codebase.
"""

from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# Try to import from external TinyFusion if available
try:
    # Add the external TinyFusion path to sys.path
    external_tinyfusion_path = os.path.join(os.path.dirname(__file__), '../../external/TinyFusion')
    if os.path.exists(external_tinyfusion_path) and external_tinyfusion_path not in sys.path:
        sys.path.insert(0, external_tinyfusion_path)
    
    from models import DiT_models as DiTConfigs
    from models import DiT
    print("Successfully imported TinyFusion models from external directory")
    
except ImportError:
    print("Could not import TinyFusion models, using fallback implementation")
    
    # Fallback DiT implementation
    class DiT(nn.Module):
        """Fallback DiT implementation for TinyFusion compatibility"""
        def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            num_classes=1000,
            learn_sigma=True,
        ):
            super().__init__()
            self.input_size = input_size
            self.patch_size = patch_size
            self.in_channels = in_channels
            self.out_channels = in_channels * 2 if learn_sigma else in_channels
            self.num_heads = num_heads

            # Patch embedding
            self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
            
            # Positional embedding (fixed size, will be interpolated as needed)
            self.base_input_size = input_size
            num_patches = (input_size // patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
            
            # Time embedding
            self.t_embedder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            
            # Class embedding
            self.y_embedder = nn.Embedding(num_classes, hidden_size)
            
            # Transformer blocks
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=int(hidden_size * mlp_ratio),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                ) for _ in range(depth)
            ])
            
            # Final layer
            self.final_layer = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)
            )
            
            self.initialize_weights()

        def initialize_weights(self):
            """Initialize weights"""
            # Initialize positional embedding
            torch.nn.init.normal_(self.pos_embed, std=0.02)
            
            # Initialize patch embedding like nn.Linear (instead of nn.Conv2d)
            w = self.x_embedder.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.constant_(self.x_embedder.bias, 0)

        def get_positional_embedding(self, H, W):
            """Get positional embedding interpolated to match input size"""
            target_patches = (H // self.patch_size) * (W // self.patch_size)
            
            if target_patches == self.pos_embed.shape[1]:
                return self.pos_embed
            
            # Need to interpolate positional embedding
            pos_embed = self.pos_embed
            
            # Reshape to 2D grid
            base_size = int(self.pos_embed.shape[1] ** 0.5)
            pos_embed_2d = pos_embed.reshape(1, base_size, base_size, -1).permute(0, 3, 1, 2)
            
            # Interpolate to target size
            target_h = H // self.patch_size
            target_w = W // self.patch_size
            pos_embed_resized = F.interpolate(
                pos_embed_2d, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Reshape back to sequence format
            pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1).reshape(1, target_patches, -1)
            
            return pos_embed_resized

        def forward(self, x, t, y):
            """
            Forward pass of DiT.
            x: (N, C, H, W) tensor of spatial inputs (images or latents)
            t: (N,) tensor of diffusion timesteps
            y: (N,) tensor of class labels
            """
            H_in, W_in = x.shape[-2:]
            # Patch embedding
            x = self.x_embedder(x)  # (N, hidden_size, H/patch_size, W/patch_size)
            N, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (N, num_patches, hidden_size)
            
            # Add positional embedding (dynamically sized)
            pos_embed = self.get_positional_embedding(H_in, W_in)
            x = x + pos_embed
            
            # Get hidden_size from the module shapes
            hidden_size = self.x_embedder.weight.shape[0]  # This is equivalent to hidden_size
            
            # Time embedding
            t_emb = self.t_embedder(timestep_embedding(t, hidden_size))
            t_emb = t_emb.unsqueeze(1)  # (N, 1, hidden_size)
            
            # Class embedding
            y_emb = self.y_embedder(y).unsqueeze(1)  # (N, 1, hidden_size)
            
            # Add conditioning
            x = x + t_emb + y_emb
            
            # Transformer blocks
            for block in self.blocks:
                x = block(x)
            
            # Final layer
            x = self.final_layer(x)  # (N, num_patches, patch_size^2 * out_channels)
            
            # Reshape to image format
            p = self.patch_size
            x = x.reshape(x.shape[0], H, W, p, p, self.out_channels)
            x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
            x = x.reshape(x.shape[0], self.out_channels, H * p, W * p)
            
            return x

    def timestep_embedding(timesteps, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    # DiT model configurations as a function that returns the appropriate DiT model
    DiTConfigs = {
        "DiT-XL/2": lambda **kwargs: DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs),
        "DiT-XL/4": lambda **kwargs: DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs),
        "DiT-XL/8": lambda **kwargs: DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs),
        "DiT-L/2": lambda **kwargs: DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs),
        "DiT-L/4": lambda **kwargs: DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs),
        "DiT-L/8": lambda **kwargs: DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs),
        "DiT-B/2": lambda **kwargs: DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs),
        "DiT-B/4": lambda **kwargs: DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs),
        "DiT-B/8": lambda **kwargs: DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs),
        "DiT-S/2": lambda **kwargs: DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs),
        "DiT-S/4": lambda **kwargs: DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs),
        "DiT-S/8": lambda **kwargs: DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs),
        "DiT-D14/2": lambda **kwargs: DiT(depth=14, hidden_size=384, patch_size=2, num_heads=6, **kwargs),
        "tinyfusion_mini": lambda **kwargs: DiT(depth=8, hidden_size=256, patch_size=4, num_heads=4, **kwargs),
    }


@dataclass
class TinyFusionConfig:
    """Configuration parameters for the TinyFusion video wrapper."""

    video_size: Tuple[int, int, int] = (28, 128, 128)  # (frames, height, width)
    in_channels: int = 3
    out_channels: int = 3
    variant: str = "tinyfusion_mini"
    checkpoint_path: Optional[str] = None
    freeze_backbone: bool = True
    enable_temporal_post: bool = True
    temporal_kernel: int = 3


class IdentityConditioner(nn.Module):
    """Fallback layer when the backbone is unconditional."""

    def __init__(self, cond_dim: int):
        super().__init__()
        self.cond_dim = cond_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TemporalPostProcessor(nn.Module):
    """Simple temporal smoothing module applied after per-frame inference."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            bias=False,
        )
        nn.init.dirac_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TinyFusionVideoWrapper(nn.Module):
    """Video diffusion wrapper around TinyFusion 2D UNet backbones."""

    def __init__(
        self,
        video_size: Tuple[int, int, int] = (28, 128, 128),
        in_channels: int = 3,
        out_channels: int = 3,
        text_dim: Optional[int] = None,
        variant: str = "tinyfusion_mini",
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
        enable_temporal_post: bool = True,
        temporal_kernel: int = 3,
        frame_chunk_size: int = 8,  # Add chunking parameter
    ) -> None:
        super().__init__()
        self.video_size = video_size
        self.frames, self.height, self.width = video_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_dim = text_dim
        self.variant = variant
        self.checkpoint_path = checkpoint_path
        self.frame_chunk_size = frame_chunk_size

        self.backbone = self._load_pretrained_backbone(variant, checkpoint_path)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # TinyFusion is unconditional; keep interface but ignore text embeddings.
        self.text_conditioner = IdentityConditioner(text_dim or 0)

        self.temporal_post = (
            TemporalPostProcessor(out_channels, temporal_kernel)
            if enable_temporal_post
            else nn.Identity()
        )

    def _load_pretrained_backbone(self, variant: str, checkpoint_path: str) -> nn.Module:
        """Load pretrained TinyFusion backbone with careful state dict handling"""
        print(f"Loading TinyFusion backbone: {variant}")
        
        # Create model with our target configuration
        if variant in DiTConfigs:
            # Call the lambda function to create the model
            backbone = DiTConfigs[variant](
                input_size=self.height,  # Use actual input size
                in_channels=self.in_channels,
                num_classes=1000,  # Standard ImageNet classes
            )
        else:
            print(f"Unknown variant {variant}, using default DiT-B/4")
            backbone = DiTConfigs["DiT-B/4"](
                input_size=self.height,
                in_channels=self.in_channels,
                num_classes=1000,
            )

        def forward(self, x, t, y):
            """
            Forward pass of DiT.
            x: (N, C, H, W) tensor of spatial inputs (images or latents)
            t: (N,) tensor of diffusion timesteps
            y: (N,) tensor of class labels
            """
            H_in, W_in = x.shape[-2:]
            # Patch embedding
            x = self.x_embedder(x)  # (N, hidden_size, H/patch_size, W/patch_size)
            N, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (N, num_patches, hidden_size)
            
            # Add positional embedding (dynamically sized)
            pos_embed = self.get_positional_embedding(H_in, W_in)
            x = x + pos_embed
            
            # Get hidden_size from the module shapes
            hidden_size = self.x_embedder.weight.shape[0]  # This is equivalent to hidden_size
            
            # Time embedding
            t_emb = self.t_embedder(timestep_embedding(t, hidden_size))
            t_emb = t_emb.unsqueeze(1)  # (N, 1, hidden_size)
            
            # Class embedding
            y_emb = self.y_embedder(y).unsqueeze(1)  # (N, 1, hidden_size)
            
            # Add conditioning
            x = x + t_emb + y_emb
            
            # Transformer blocks
            for block in self.blocks:
                x = block(x)
            
            # Final layer
            x = self.final_layer(x)  # (N, num_patches, patch_size^2 * out_channels)
            
            # Reshape to image format
            p = self.patch_size
            x = x.reshape(x.shape[0], H, W, p, p, self.out_channels)
            x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
            x = x.reshape(x.shape[0], self.out_channels, H * p, W * p)
            
            return x

    def timestep_embedding(timesteps, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    # DiT model configurations as a function that returns the appropriate DiT model
    DiTConfigs = {
        "DiT-XL/2": lambda **kwargs: DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs),
        "DiT-XL/4": lambda **kwargs: DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs),
        "DiT-XL/8": lambda **kwargs: DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs),
        "DiT-L/2": lambda **kwargs: DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs),
        "DiT-L/4": lambda **kwargs: DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs),
        "DiT-L/8": lambda **kwargs: DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs),
        "DiT-B/2": lambda **kwargs: DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs),
        "DiT-B/4": lambda **kwargs: DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs),
        "DiT-B/8": lambda **kwargs: DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs),
        "DiT-S/2": lambda **kwargs: DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs),
        "DiT-S/4": lambda **kwargs: DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs),
        "DiT-S/8": lambda **kwargs: DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs),
        "DiT-D14/2": lambda **kwargs: DiT(depth=14, hidden_size=384, patch_size=2, num_heads=6, **kwargs),
        "tinyfusion_mini": lambda **kwargs: DiT(depth=8, hidden_size=256, patch_size=4, num_heads=4, **kwargs),
    }


@dataclass
class TinyFusionConfig:
    """Configuration parameters for the TinyFusion video wrapper."""

    video_size: Tuple[int, int, int] = (28, 128, 128)  # (frames, height, width)
    in_channels: int = 3
    out_channels: int = 3
    variant: str = "tinyfusion_mini"
    checkpoint_path: Optional[str] = None
    freeze_backbone: bool = True
    enable_temporal_post: bool = True
    temporal_kernel: int = 3


class IdentityConditioner(nn.Module):
    """Fallback layer when the backbone is unconditional."""

    def __init__(self, cond_dim: int):
        super().__init__()
        self.cond_dim = cond_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TemporalPostProcessor(nn.Module):
    """Simple temporal smoothing module applied after per-frame inference."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            bias=False,
        )
        nn.init.dirac_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TinyFusionVideoWrapper(nn.Module):
    """Video diffusion wrapper around TinyFusion 2D UNet backbones."""

    def __init__(
        self,
        video_size: Tuple[int, int, int] = (28, 128, 128),
        in_channels: int = 3,
        out_channels: int = 3,
        text_dim: Optional[int] = None,
        variant: str = "tinyfusion_mini",
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
        enable_temporal_post: bool = True,
        temporal_kernel: int = 3,
        frame_chunk_size: int = 8,  # Add chunking parameter
    ) -> None:
        super().__init__()
        self.video_size = video_size
        self.frames, self.height, self.width = video_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_dim = text_dim
        self.variant = variant
        self.checkpoint_path = checkpoint_path
        self.frame_chunk_size = frame_chunk_size

        self.backbone = self._load_pretrained_backbone(variant, checkpoint_path)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # TinyFusion is unconditional; keep interface but ignore text embeddings.
        self.text_conditioner = IdentityConditioner(text_dim or 0)

        self.temporal_post = (
            TemporalPostProcessor(out_channels, temporal_kernel)
            if enable_temporal_post
            else nn.Identity()
        )

    def _load_pretrained_backbone(self, variant: str, checkpoint_path: str) -> nn.Module:
        """Load pretrained TinyFusion backbone with careful state dict handling"""
        print(f"Loading TinyFusion backbone: {variant}")
        
        # Create model with our target configuration
        if variant in DiTConfigs:
            # Call the lambda function to create the model
            backbone = DiTConfigs[variant](
                input_size=self.height,  # Use actual input size
                in_channels=self.in_channels,
                num_classes=1000,  # Standard ImageNet classes
            )
        else:
            print(f"Unknown variant {variant}, using default DiT-B/4")
            backbone = DiTConfigs["DiT-B/4"](
                input_size=self.height,
                in_channels=self.in_channels,
                num_classes=1000,
            )
        
        if checkpoint_path and checkpoint_path != "none":
            try:
                print(f"Loading checkpoint from: {checkpoint_path}")
                if not os.path.exists(checkpoint_path):
                    print(f"Warning: Checkpoint not found at {checkpoint_path}")
                    print("Continuing with randomly initialized model...")
                    return backbone
                
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Extract state dict (handle different checkpoint formats)
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'ema' in checkpoint:
                    state_dict = checkpoint['ema']
                else:
                    state_dict = checkpoint
                
                # Get current model state dict for comparison
                model_state = backbone.state_dict()
                
                # Filter and adapt state dict
                adapted_state = {}
                skipped_keys = []
                
                for key, value in state_dict.items():
                    if key in model_state:
                        model_shape = model_state[key].shape
                        checkpoint_shape = value.shape
                        
                        if model_shape == checkpoint_shape:
                            # Direct copy for matching shapes
                            adapted_state[key] = value
                        else:
                            # Handle specific mismatches
                            if key == 'x_embedder.weight':
                                # Input channel mismatch (4 -> 3 channels)
                                if checkpoint_shape[1] == 4 and model_shape[1] == 3:
                                    # Take first 3 channels
                                    adapted_state[key] = value[:, :3, :, :].clone()
                                    print(f"Adapted {key}: {checkpoint_shape} -> {model_shape} (channel reduction)")
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")
                            
                            elif key == 'pos_embed':
                                # Positional embedding size mismatch
                                if len(checkpoint_shape) == 3 and len(model_shape) == 3:
                                    checkpoint_seq_len = checkpoint_shape[1]
                                    model_seq_len = model_shape[1]
                                    
                                    if checkpoint_seq_len < model_seq_len:
                                        # Pad with zeros
                                        pad_size = model_seq_len - checkpoint_seq_len
                                        padded = torch.cat([
                                            value,
                                            torch.zeros(checkpoint_shape[0], pad_size, checkpoint_shape[2])
                                        ], dim=1)
                                        adapted_state[key] = padded
                                        print(f"Adapted {key}: {checkpoint_shape} -> {model_shape} (padded)")
                                    elif checkpoint_seq_len > model_seq_len:
                                        # Truncate
                                        adapted_state[key] = value[:, :model_seq_len, :].clone()
                                        print(f"Adapted {key}: {checkpoint_shape} -> {model_shape} (truncated)")
                                    else:
                                        skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")
                            
                            elif key == 'y_embedder.weight':
                                # Class embedding mismatch (1001 vs 1000 classes)
                                if checkpoint_shape[0] > model_shape[0]:
                                    # Truncate extra classes
                                    adapted_state[key] = value[:model_shape[0], :].clone()
                                    print(f"Adapted {key}: {checkpoint_shape} -> {model_shape} (truncated classes)")
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")
                            
                            elif key.startswith('final_layer.'):
                                # Output layer mismatch - skip these as they're task-specific
                                skipped_keys.append(f"{key} (output layer - will be randomly initialized)")
                            
                            else:
                                skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")
                    else:
                        skipped_keys.append(f"{key} (not in model)")
                
                # Load the adapted state dict
                missing_keys, unexpected_keys = backbone.load_state_dict(adapted_state, strict=False)
                
                print(f"Successfully loaded {len(adapted_state)} parameters from checkpoint")
                if missing_keys:
                    print(f"Missing keys (will be randomly initialized): {len(missing_keys)}")
                    for key in missing_keys[:5]:  # Show first 5
                        print(f"  - {key}")
                    if len(missing_keys) > 5:
                        print(f"  ... and {len(missing_keys) - 5} more")
                
                if skipped_keys:
                    print(f"Skipped keys due to incompatibility: {len(skipped_keys)}")
                    for key in skipped_keys[:5]:  # Show first 5
                        print(f"  - {key}")
                    if len(skipped_keys) > 5:
                        print(f"  ... and {len(skipped_keys) - 5} more")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing with randomly initialized model...")
        
        return backbone

    def _process_frame_chunk(self, x_chunk, time_chunk, dummy_labels_chunk):
        """Process a chunk of frames through the backbone with gradient checkpointing"""
        if self.training and hasattr(self.backbone, 'training'):
            # Use gradient checkpointing during training to save memory
            return checkpoint.checkpoint(
                self.backbone,
                x_chunk,
                time_chunk,
                dummy_labels_chunk,
                use_reentrant=False
            )
        else:
            return self.backbone(x_chunk, time_chunk, dummy_labels_chunk)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise for a batch of videos using TinyFusion with memory-efficient processing."""

        batch, channels, frames, height, width = x.shape
        assert (
            channels == self.in_channels
        ), f"Expected {self.in_channels} channels, got {channels}"

        if height != self.height or width != self.width:
            x = F.interpolate(
                x,
                size=(frames, self.height, self.width),
                mode="trilinear",
                align_corners=False,
            )
            height, width = self.height, self.width

        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        time_per_frame = time.view(batch, 1).repeat(1, frames).reshape(-1)

        # TinyFusion ignores text embeddings; keep shape compatibility.
        if text_emb is not None and text_emb.numel() > 0:
            _ = self.text_conditioner(text_emb)

        # TinyFusion DiT models require class labels (y parameter)
        # Create dummy class labels for unconditional generation
        dummy_labels = torch.zeros(batch * frames, dtype=torch.long, device=x.device)
        
        # Process frames in chunks to reduce memory usage
        total_frames = batch * frames
        predictions = []
        
        for i in range(0, total_frames, self.frame_chunk_size):
            end_idx = min(i + self.frame_chunk_size, total_frames)
            
            x_chunk = x_reshaped[i:end_idx]
            time_chunk = time_per_frame[i:end_idx]
            dummy_labels_chunk = dummy_labels[i:end_idx]
            
            try:
                pred_chunk = self._process_frame_chunk(x_chunk, time_chunk, dummy_labels_chunk)
            except TypeError as e:
                # Fallback for different model signatures
                try:
                    if self.training and hasattr(self.backbone, 'training'):
                        pred_chunk = checkpoint.checkpoint(
                            self.backbone,
                            x_chunk,
                            time_chunk,
                            use_reentrant=False
                        )
                    else:
                        pred_chunk = self.backbone(x_chunk, time_chunk)
                except Exception:
                    raise RuntimeError(f"Failed to call TinyFusion backbone: {e}") from e
            
            predictions.append(pred_chunk)
            
            # Clear cache to free memory between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all predictions
        pred = torch.cat(predictions, dim=0)

        if pred.shape[1] != self.out_channels:
            pred = pred[:, : self.out_channels]

        pred_video = pred.view(batch, frames, self.out_channels, height, width)
        pred_video = pred_video.permute(0, 2, 1, 3, 4)

        pred_video = self.temporal_post(pred_video)

        return pred_video


def create_tinyfusion_model(**kwargs) -> TinyFusionVideoWrapper:
    """Factory helper to build TinyFusion video wrapper with keyword overrides."""

    return TinyFusionVideoWrapper(**kwargs)


__all__ = ["TinyFusionConfig", "TinyFusionVideoWrapper", "create_tinyfusion_model"]
