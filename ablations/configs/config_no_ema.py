"""
Ablation Study Configuration - EMA Disabled

Modification from baseline: Disable Exponential Moving Average (EMA)
Expected: -2-5% quality degradation, no training time difference
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration - same as baseline"""
    # Image/Video dimensions
    image_size: int = 64
    num_frames: int = 16
    in_channels: int = 3
    
    # UNet architecture
    model_channels: int = 96
    channel_mult: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (8, 16)
    num_heads: int = 6
    
    # Transformer settings
    use_transformer: bool = True
    transformer_depth: int = 2
    use_gradient_checkpointing: bool = True
    
    # Text encoder
    use_clip_text_encoder: bool = True
    text_embed_dim: int = 384
    max_text_length: int = 77
    vocab_size: int = 49408
    freeze_text_encoder: bool = True  # Same as baseline
    
    # Cross attention
    context_dim: int = 384


@dataclass
class DDIMConfig:
    """DDIM scheduler configuration"""
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    clip_sample: bool = True
    prediction_type: str = "epsilon"


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    data_dir: str = "text2sign/training_data"
    batch_size: int = 2
    num_workers: int = 4
    
    # Training
    num_epochs: int = 150
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # EMA settings - DISABLED (ablation)
    use_ema: bool = False  # KEY CHANGE: Disable EMA
    ema_decay: float = 0.9999
    ema_update_every: int = 10
    
    # Checkpointing
    checkpoint_dir: str = "text_to_sign/checkpoints"
    save_every: int = 5
    log_every: int = 100
    sample_every: int = 1000
    
    # TensorBoard
    log_dir: str = "text_to_sign/logs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GenerationConfig:
    """Generation/Inference configuration"""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    output_dir: str = "text_to_sign/generated"
    fps: int = 8


def get_config():
    """Get all configurations for EMA DISABLED variant"""
    return {
        "model": ModelConfig(),
        "ddim": DDIMConfig(),
        "training": TrainingConfig(),
        "generation": GenerationConfig(),
    }
