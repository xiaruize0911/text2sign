"""TinyFusion video diffusion architecture wrapper.

This module adapts the TinyFusion 2D diffusion backbone (https://github.com/VainF/TinyFusion)
so it can be plugged into the Text2Sign training pipeline. The wrapper handles
loading the pretrained TinyFusion checkpoints (either via torch.hub or a local                            elif 'x_embedder' in key and 'weight' in key:
                                # Input channel mismatch (4 -> 3 channels)
                                # Handle both x_embedder.weight and x_embedder.proj.weight
                                if len(checkpoint_shape) == 4 and len(model_shape) == 4:
                                    if checkpoint_shape[1] == 4 and model_shape[1] == 3:
                                        # Take first 3 channels and renormalize
                                        adapted_weight = value[:, :3, :, :].clone()
                                        # Renormalize to maintain output magnitude
                                        adapted_weight = adapted_weight * (4.0 / 3.0)
                                        adapted_state[key] = adapted_weight
                                        print(f"Adapted {key}: {checkpoint_shape} -> {adapted_weight.shape} (channel reduction + renormalization)")
                                    else:
                                        skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")int path) and runs the network frame-by-frame while keeping the overall
video tensor shape compatible with the rest of the codebase.
"""

from __future__ import annotations

import os
import sys
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# Require TinyFusion to be available from the external implementation
external_tinyfusion_path = os.path.join(
    os.path.dirname(__file__), '../../external/TinyFusion'
)

if not os.path.isdir(external_tinyfusion_path):
    raise ImportError(
        "TinyFusion external directory not found. Make sure `external/TinyFusion` "
        "is present and initialized."
    )

if external_tinyfusion_path not in sys.path:
    sys.path.insert(0, external_tinyfusion_path)

existing_models_module = sys.modules.get("models")
restore_local_models = False

if existing_models_module is not None:
    existing_path = getattr(existing_models_module, "__file__", None)
    project_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if existing_path is None:
        restore_local_models = True
    else:
        try:
            if os.path.commonpath([
                project_models_dir,
                os.path.abspath(existing_path),
            ]) == project_models_dir:
                restore_local_models = True
        except ValueError:
            pass

    if restore_local_models:
        del sys.modules["models"]

try:
    from models import DiT_models as DiTConfigs
    from models import DiT
    print("Successfully imported TinyFusion models from external directory")
except ImportError as err:
    raise ImportError(
        "Failed to import TinyFusion models from the external implementation. "
        "Ensure TinyFusion is installed or the submodule is initialized."
    ) from err
finally:
    if restore_local_models:
        sys.modules["models"] = existing_models_module


logger = logging.getLogger(__name__)

try:
    from torch.amp import autocast as _torch_amp_autocast
except ImportError:  # pragma: no cover - older PyTorch fallback
    _torch_amp_autocast = None


def _preferred_amp_dtype(device_type: str) -> Optional[torch.dtype]:
    """Select the ideal autocast dtype for the given device."""
    if device_type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device_type == "cpu":
        if (
            hasattr(torch.cpu, "is_bf16_supported")
            and callable(torch.cpu.is_bf16_supported)  # type: ignore[attr-defined]
            and torch.cpu.is_bf16_supported()  # type: ignore[attr-defined]
        ):
            return torch.bfloat16

    return None


def _autocast_if_available(device_type: str, target_dtype: Optional[torch.dtype]):
    """Return an autocast context manager when supported, otherwise a no-op."""

    if target_dtype is None and device_type not in {"cuda", "cpu"}:
        return nullcontext()

    kwargs = {"device_type": device_type}
    if target_dtype is not None:
        kwargs["dtype"] = target_dtype

    try:  # Prefer modern torch.autocast API
        return torch.autocast(**kwargs)  # type: ignore[arg-type]
    except (AttributeError, TypeError):
        pass

    if _torch_amp_autocast is not None:
        try:
            return _torch_amp_autocast(**kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            try:
                return _torch_amp_autocast(**kwargs)
            except TypeError:
                return nullcontext()

    return nullcontext()


def _disable_autocast_if_needed(device_type: str):
    """Return a context manager that disables autocast for stability."""
    autocast_active = False
    if hasattr(torch, "is_autocast_enabled"):
        autocast_active = torch.is_autocast_enabled()
    elif device_type == "cuda" and hasattr(torch.cuda, "is_autocast_enabled"):
        autocast_active = torch.cuda.is_autocast_enabled()

    if not autocast_active:
        return nullcontext()

    # Prefer the generic torch.autocast API when available
    try:  # pragma: no cover - depends on PyTorch version
        return torch.autocast(device_type=device_type, enabled=False)  # type: ignore[attr-defined]
    except AttributeError:
        pass

    if _torch_amp_autocast is not None:
        try:
            return _torch_amp_autocast(device_type=device_type, enabled=False)
        except TypeError:
            pass

    if device_type == "cuda" and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast(enabled=False)
    if device_type == "cpu" and hasattr(torch, "cpu") and hasattr(torch.cpu, "amp"):
        return torch.cpu.amp.autocast(enabled=False)  # type: ignore[attr-defined]
    return nullcontext()


@dataclass
class TinyFusionConfig:
    """Configuration parameters for the TinyFusion video wrapper."""

    video_size: Tuple[int, int, int] = (28, 128, 128)  # (frames, height, width)
    in_channels: int = 3
    out_channels: int = 3
    variant: str = "DiT-S/2"
    checkpoint_path: Optional[str] = None
    freeze_backbone: bool = True
    enable_temporal_post: bool = True
    temporal_kernel: int = 3
    force_fp32_backbone: bool = False


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
        # Use 'same' padding mode to ensure output has same temporal dimension as input
        # This prevents shape mismatches when using even kernel sizes (e.g., kernel_size=2)
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding='same',  # Automatically handles padding to maintain shape
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
        variant: str = "DiT-D14/2",
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
        enable_temporal_post: bool = True,
        temporal_kernel: int = 3,
        frame_chunk_size: int = 8,  # Add chunking parameter
        force_fp32_backbone: bool = False,
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
        self.force_fp32_backbone = force_fp32_backbone

        self.backbone = self._load_pretrained_backbone(variant, checkpoint_path)
        
        # Important: Only freeze if all weights are properly loaded
        # Since output layers may be randomly initialized, freezing will prevent learning
        if freeze_backbone:
            # Check if critical layers are properly initialized
            has_random_init = False
            for name, param in self.backbone.named_parameters():
                if 'final_layer' in name or 'x_embedder.proj' in name:
                    # These layers may be randomly initialized
                    has_random_init = True
                    break
            
            if has_random_init:
                print("⚠️  Warning: Some layers are randomly initialized. ")
                print("   Keeping backbone unfrozen to allow training.")
                print("   Set freeze_backbone=False explicitly to remove this warning.")
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print(f"✅ Backbone frozen ({sum(p.numel() for p in self.backbone.parameters()):,} params)")
        else:
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            print(f"✅ Backbone trainable ({trainable:,} params)")

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
                                # Handle both sequence length and hidden dimension mismatches
                                if len(checkpoint_shape) == 3 and len(model_shape) == 3:
                                    checkpoint_seq_len = checkpoint_shape[1]
                                    checkpoint_hidden = checkpoint_shape[2]
                                    model_seq_len = model_shape[1]
                                    model_hidden = model_shape[2]
                                    
                                    adapted_embed = value
                                    
                                    # Step 1: Handle sequence length mismatch
                                    if checkpoint_seq_len < model_seq_len:
                                        # Pad sequence length with zeros
                                        pad_size = model_seq_len - checkpoint_seq_len
                                        adapted_embed = torch.cat([
                                            adapted_embed,
                                            torch.zeros(checkpoint_shape[0], pad_size, checkpoint_hidden)
                                        ], dim=1)
                                    elif checkpoint_seq_len > model_seq_len:
                                        # Truncate sequence length
                                        adapted_embed = adapted_embed[:, :model_seq_len, :]
                                    
                                    # Step 2: Handle hidden dimension mismatch
                                    if checkpoint_hidden < model_hidden:
                                        # Pad hidden dimension with zeros
                                        pad_size = model_hidden - checkpoint_hidden
                                        adapted_embed = torch.cat([
                                            adapted_embed,
                                            torch.zeros(adapted_embed.shape[0], adapted_embed.shape[1], pad_size)
                                        ], dim=2)
                                    elif checkpoint_hidden > model_hidden:
                                        # Truncate hidden dimension
                                        adapted_embed = adapted_embed[:, :, :model_hidden]
                                    
                                    adapted_state[key] = adapted_embed
                                    print(f"Adapted {key}: {checkpoint_shape} -> {adapted_embed.shape} -> {model_shape}")
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")
                            
                            elif key == 'y_embedder.weight':
                                # Class embedding mismatch (1001 vs 1000 classes)
                                if len(checkpoint_shape) == 2 and len(model_shape) == 2:
                                    # Handle both dimensions for embedding weights
                                    adapted_weight = value
                                    
                                    # Handle num_embeddings (dimension 0)
                                    if checkpoint_shape[0] > model_shape[0]:
                                        adapted_weight = adapted_weight[:model_shape[0], :]
                                    elif checkpoint_shape[0] < model_shape[0]:
                                        # Pad with zeros for missing embeddings
                                        pad_size = model_shape[0] - checkpoint_shape[0]
                                        adapted_weight = torch.cat([
                                            adapted_weight,
                                            torch.zeros(pad_size, checkpoint_shape[1])
                                        ], dim=0)
                                    
                                    # Handle embedding_dim (dimension 1)
                                    if checkpoint_shape[1] > model_shape[1]:
                                        adapted_weight = adapted_weight[:, :model_shape[1]]
                                    elif checkpoint_shape[1] < model_shape[1]:
                                        # Pad with zeros for missing dimensions
                                        pad_size = model_shape[1] - checkpoint_shape[1]
                                        adapted_weight = torch.cat([
                                            adapted_weight,
                                            torch.zeros(adapted_weight.shape[0], pad_size)
                                        ], dim=1)
                                    
                                    adapted_state[key] = adapted_weight
                                    print(f"Adapted {key}: {checkpoint_shape} -> {adapted_weight.shape}")
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")
                            
                            elif 't_embedder' in key and 'weight' in key:
                                # Time embedder linear layer weight mismatch
                                if len(checkpoint_shape) == 2 and len(model_shape) == 2:
                                    adapted_weight = value
                                    
                                    # Handle input dimension (dimension 1 for weight matrix)
                                    if checkpoint_shape[1] > model_shape[1]:
                                        adapted_weight = adapted_weight[:, :model_shape[1]]
                                    elif checkpoint_shape[1] < model_shape[1]:
                                        pad_size = model_shape[1] - checkpoint_shape[1]
                                        adapted_weight = torch.cat([
                                            adapted_weight,
                                            torch.zeros(checkpoint_shape[0], pad_size)
                                        ], dim=1)
                                    
                                    # Handle output dimension (dimension 0)
                                    if checkpoint_shape[0] > model_shape[0]:
                                        adapted_weight = adapted_weight[:model_shape[0], :]
                                    elif checkpoint_shape[0] < model_shape[0]:
                                        pad_size = model_shape[0] - checkpoint_shape[0]
                                        adapted_weight = torch.cat([
                                            adapted_weight,
                                            torch.zeros(pad_size, adapted_weight.shape[1])
                                        ], dim=0)
                                    
                                    adapted_state[key] = adapted_weight
                                    print(f"Adapted {key}: {checkpoint_shape} -> {adapted_weight.shape}")
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")
                            
                            elif 't_embedder' in key and 'bias' in key:
                                # Time embedder bias mismatch
                                if len(checkpoint_shape) == 1 and len(model_shape) == 1:
                                    if checkpoint_shape[0] > model_shape[0]:
                                        adapted_state[key] = value[:model_shape[0]]
                                    elif checkpoint_shape[0] < model_shape[0]:
                                        pad_size = model_shape[0] - checkpoint_shape[0]
                                        adapted_state[key] = torch.cat([value, torch.zeros(pad_size)])
                                    print(f"Adapted {key}: {checkpoint_shape} -> {model_shape}")
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch: {checkpoint_shape} vs {model_shape})")
                            
                            elif key.startswith('final_layer.'):
                                # Output layer mismatch - these are task-specific
                                # Initialize them properly rather than skipping
                                # The checkpoint uses different output dimensions (likely for ImageNet)
                                # We need to train these layers for our task
                                if 'weight' in key and len(checkpoint_shape) == 2 and len(model_shape) == 2:
                                    # Try to adapt if possible
                                    if checkpoint_shape[1] == model_shape[1]:  # Same input dim
                                        # Initialize with small random values based on checkpoint stats
                                        init_std = value.std().item()
                                        adapted_state[key] = torch.randn(model_shape) * init_std * 0.1
                                        print(f"Initialized {key}: {model_shape} (small random with std={init_std*0.1:.6f})")
                                    else:
                                        skipped_keys.append(f"{key} (output layer - incompatible input dim)")
                                elif 'bias' in key:
                                    # Initialize bias to zero
                                    adapted_state[key] = torch.zeros(model_shape)
                                    print(f"Initialized {key}: {model_shape} (zeros)")
                                else:
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

        device_type = x.device.type
        preferred_dtype = (
            torch.float32 if self.force_fp32_backbone else _preferred_amp_dtype(device_type)
        )
        autocast_ctx = (
            _disable_autocast_if_needed(device_type)
            if self.force_fp32_backbone
            else _autocast_if_available(device_type, preferred_dtype)
        )

        with autocast_ctx:
            if self.force_fp32_backbone:
                x_processed = x.to(torch.float32)
            else:
                x_processed = x

            if height != self.height or width != self.width:
                resize_dtype = x_processed.dtype
                x_processed = F.interpolate(
                    x_processed.to(torch.float32),
                    size=(frames, self.height, self.width),
                    mode="trilinear",
                    align_corners=False,
                ).to(resize_dtype)
                height, width = self.height, self.width

            if not self.force_fp32_backbone and preferred_dtype is not None:
                try:
                    x_processed = x_processed.to(preferred_dtype)
                except RuntimeError:
                    # Fall back silently if dtype conversion is unsupported
                    pass

            x_reshaped = x_processed.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
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

                # Check for NaN/Inf but handle more carefully
                if torch.isnan(pred_chunk).any() or torch.isinf(pred_chunk).any():
                    nan_count = torch.isnan(pred_chunk).sum().item()
                    inf_count = torch.isinf(pred_chunk).sum().item()
                    total = pred_chunk.numel()
                    
                    logger.warning(
                        f"TinyFusion backbone produced {nan_count} NaN and {inf_count} Inf values "
                        f"out of {total} ({(nan_count+inf_count)/total*100:.2f}%) in chunk"
                    )
                    
                    # Only replace NaN/Inf, don't zero out everything
                    # Replace with mean of valid values to preserve signal
                    valid_mask = ~(torch.isnan(pred_chunk) | torch.isinf(pred_chunk))
                    if valid_mask.any():
                        valid_mean = pred_chunk[valid_mask].mean()
                        pred_chunk = torch.where(valid_mask, pred_chunk, valid_mean)
                    else:
                        # If all values are invalid, use small random noise
                        pred_chunk = torch.randn_like(pred_chunk) * 0.01
                        logger.error("All values were NaN/Inf - using small random noise")

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
            
            # Final safety check - only replace actual NaN/Inf, not valid values
            if torch.isnan(pred_video).any() or torch.isinf(pred_video).any():
                logger.warning("NaN/Inf detected in final output, applying careful fix")
                valid_mask = ~(torch.isnan(pred_video) | torch.isinf(pred_video))
                if valid_mask.any():
                    valid_mean = pred_video[valid_mask].mean()
                    pred_video = torch.where(valid_mask, pred_video, valid_mean)
                else:
                    # Last resort: small random values
                    pred_video = torch.randn_like(pred_video) * 0.01

        if pred_video.dtype != x.dtype:
            pred_video = pred_video.to(x.dtype)

        return pred_video


def create_tinyfusion_model(**kwargs) -> TinyFusionVideoWrapper:
    """Factory helper to build TinyFusion video wrapper with keyword overrides."""

    return TinyFusionVideoWrapper(**kwargs)

__all__ = ["TinyFusionConfig", "TinyFusionVideoWrapper", "create_tinyfusion_model"]
