"""
Configuration for Text-to-Sign Language DDIM Diffusion Model
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Image/Video dimensions
    image_size: int = 64  # Reset to 64x64 for significantly faster training
    num_frames: int = 24  # Better balance between temporal quality and memory usage
    in_channels: int = 3  # RGB channels
    
    # UNet architecture (optimized for practical single-GPU training)
    model_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (8,)
    num_heads: int = 4
    
    # Transformer settings (DiT-style)
    use_transformer: bool = False  # Use enhanced DiT-style transformer blocks
    transformer_depth: int = 1  # Reduced from 2 for significant speed boost
    use_gradient_checkpointing: bool = True  # Enabled for 32 frames to save VRAM
    
    # Text encoder
    use_clip_text_encoder: bool = True  # Default to frozen pretrained CLIP text encoder
    text_embed_dim: int = 512  # Standard CLIP-base dimension
    max_text_length: int = 77
    vocab_size: int = 49408  # CLIP vocab size
    
    # Cross attention
    context_dim: int = 512  # Match text_embed_dim

    # Training conditioning
    use_length_prefix: bool = False  # Prepends [WORD_COUNT] to text for density control
    text_conditioning_mode: str = "normal"  # normal | none | random
    clip_trainable_layers: int = 0  # 0 keeps CLIP frozen, >0 unfreezes last N encoder layers


@dataclass
class DDIMConfig:
    """DDIM scheduler configuration"""
    num_train_timesteps: int = 1000  # Increased from 50 for much better convergence and quality
    num_inference_steps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "cosine"  # "linear" or "cosine" - cosine is better for quality
    clip_sample: bool = True
    prediction_type: str = "epsilon"  # "epsilon" or "v_prediction"


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    data_dir: str = "/teamspace/studios/this_studio/text_to_sign/training_data"
    batch_size: int = 2  # Safer micro-batch for 24-frame video diffusion on a single GPU
    num_workers: int = 4
    train_ratio: float = 0.9
    split_mode: str = "signer_disjoint"  # signer_disjoint | random
    split_seed: int = 42
    
    # Training
    num_epochs: int = 100  
    learning_rate: float = 5e-5  # Reduced from 1e-4 for better stability
    weight_decay: float = 0.01
    warmup_steps: int = 1000  # Reduced warmup for 64x64 stability
    gradient_accumulation_steps: int = 8  # Keep effective batch size healthy without VRAM spikes
    max_grad_norm: float = 0.5  # Reduced from 1.0 for better stability
    
    # EMA (Exponential Moving Average) - Critical for quality!
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_every: int = 10  # Reduced frequency to avoid CPU-GPU sync bottleneck
    
    # Mixed precision
    use_amp: bool = True
    precision: str = "auto"  # auto | fp16 | bf16 | fp32

    # Runtime optimizations
    enable_compile: bool = True
    compile_mode: str = "reduce-overhead"  # default | reduce-overhead | max-autotune
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    allow_tf32: bool = True
    channels_last_3d: bool = True

    # Dataloader throughput
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    
    # Checkpointing
    checkpoint_dir: str = "text_to_sign/checkpoints"
    save_every: int = 5  # Save every N epochs
    log_every: int = 100  # Log every N steps
    sample_every: int = 2048  # Generating samples is expensive; do it less frequently by default
    
    # TensorBoard
    log_dir: str = "text_to_sign/logs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GenerationConfig:
    """Generation/Inference configuration"""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0  # 0 for DDIM, 1 for DDPM
    output_dir: str = "text_to_sign/generated"
    fps: int = 8  # Output GIF frame rate


MODEL_SIZE_PRESETS: Dict[str, Dict[str, object]] = {
    "small": {
        "description": "Safest single-GPU baseline for first full training runs.",
        "model": {
            "image_size": 64,
            "num_frames": 16,
            "model_channels": 64,
            "channel_mult": (1, 2, 4),
            "attention_resolutions": (8,),
            "num_heads": 4,
            "use_gradient_checkpointing": True,
        },
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_workers": 4,
        },
    },
    "base": {
        "description": "Balanced quality/speed preset for a serious main training run.",
        "model": {
            "image_size": 64,
            "num_frames": 24,
            "model_channels": 64,
            "channel_mult": (1, 2, 4),
            "attention_resolutions": (8,),
            "num_heads": 4,
            "use_gradient_checkpointing": True,
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_workers": 4,
        },
    },
    "large": {
        "description": "Higher-capacity preset for larger GPUs after the baseline is stable.",
        "model": {
            "image_size": 64,
            "num_frames": 32,
            "model_channels": 96,
            "channel_mult": (1, 2, 4, 8),
            "attention_resolutions": (4, 8),
            "num_heads": 6,
            "use_gradient_checkpointing": True,
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_workers": 6,
        },
    },
}


def list_model_size_presets() -> Tuple[str, ...]:
    """Return supported model-size preset names."""
    return tuple(MODEL_SIZE_PRESETS.keys())


def apply_model_size_preset(
    model_config: ModelConfig,
    train_config: TrainingConfig,
    preset_name: str,
) -> Tuple[ModelConfig, TrainingConfig, str]:
    """Apply a named model/training size preset without mutating the inputs."""
    if preset_name not in MODEL_SIZE_PRESETS:
        available = ", ".join(MODEL_SIZE_PRESETS.keys())
        raise ValueError(f"Unknown model size preset '{preset_name}'. Available: {available}")

    preset = MODEL_SIZE_PRESETS[preset_name]
    updated_model = replace(model_config, **preset["model"])
    updated_training = replace(train_config, **preset["training"])
    return updated_model, updated_training, preset["description"]


def get_config():
    """Get all configurations"""
    return {
        "model": ModelConfig(),
        "ddim": DDIMConfig(),
        "training": TrainingConfig(),
        "generation": GenerationConfig(),
    }
