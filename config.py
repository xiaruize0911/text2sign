"""
Configuration for Text-to-Sign Language DDIM Diffusion Model
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Image/Video dimensions
    image_size: int = 128  # Increased to 128x128 for clearer hand gestures
    num_frames: int = 16  # Number of frames per video
    in_channels: int = 3  # RGB channels
    
    # UNet architecture (scaled for 128x128)
    model_channels: int = 128  # Increased from 96 for better capacity
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)  # Added level for 128x128 resolution
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (4, 8)  # Matches 32x32 and 16x16 feature maps at 128px
    num_heads: int = 8  # Increased to match 128 channels (16px per head)
    
    # Transformer settings (DiT-style)
    use_transformer: bool = True  # Use enhanced DiT-style transformer blocks
    transformer_depth: int = 2  # Increased from 1 for deeper transformers
    use_gradient_checkpointing: bool = True  # Enable gradient checkpointing for memory savings
    
    # Text encoder
    use_clip_text_encoder: bool = True  # Default to frozen pretrained CLIP text encoder
    text_embed_dim: int = 384  # Increased from 256 for richer text embeddings
    max_text_length: int = 77
    vocab_size: int = 49408  # CLIP vocab size
    
    # Cross attention
    context_dim: int = 384  # Increased from 256 for better cross-attention


@dataclass
class DDIMConfig:
    """DDIM scheduler configuration"""
    num_train_timesteps: int = 1000  # Increased from 50 for much better convergence and quality
    num_inference_steps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "cosine"  # "linear" or "cosine" - cosine is better for quality
    clip_sample: bool = True
    prediction_type: str = "epsilon"  # "epsilon" or "v_prediction"


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    data_dir: str = "/teamspace/studios/this_studio/text2sign/training_data"
    batch_size: int = 2  # Reduced for 128x128 VRAM constraints
    num_workers: int = 4
    
    # Training
    num_epochs: int = 150  # Increased for larger model convergence
    learning_rate: float = 1e-4  
    weight_decay: float = 0.01
    warmup_steps: int = 5000  # Increased warmup for larger architecture stability
    gradient_accumulation_steps: int = 8  # Effective batch = 16 to maintain stability
    max_grad_norm: float = 1.0
    
    # EMA (Exponential Moving Average) - Critical for quality!
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_every: int = 1  # Fixed: Increasing frequency to match decay (10 was too slow)
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "text_to_sign/checkpoints"
    save_every: int = 5  # Save every N epochs
    log_every: int = 100  # Log every N steps
    sample_every: int = 1024  # Generate samples every N steps
    
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


def get_config():
    """Get all configurations"""
    return {
        "model": ModelConfig(),
        "ddim": DDIMConfig(),
        "training": TrainingConfig(),
        "generation": GenerationConfig(),
    }
