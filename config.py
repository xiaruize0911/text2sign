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
    image_size: int = 64  # Resize GIFs to 64x64
    num_frames: int = 16  # Number of frames per video
    in_channels: int = 3  # RGB channels
    
    # UNet architecture (increased capacity for better quality)
    model_channels: int = 96  # Increased from 64 for better quality
    channel_mult: Tuple[int, ...] = (1, 2, 4)  # Depth levels
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (8, 16)
    num_heads: int = 6  # Increased from 4 for better attention
    
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
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear" or "cosine"
    clip_sample: bool = True
    prediction_type: str = "epsilon"  # "epsilon" or "v_prediction"


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    data_dir: str = "text2sign/training_data"
    batch_size: int = 2  # Reduced from 4 for memory
    num_workers: int = 4
    
    # Training
    num_epochs: int = 150  # Increased for more training
    learning_rate: float = 5e-5  # Reduced from 1e-4 for fine-tuning stability
    weight_decay: float = 0.01
    warmup_steps: int = 500  # Reduced warmup for fine-tuning
    gradient_accumulation_steps: int = 8  # Effective batch size = 16
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "text_to_sign/checkpoints"
    save_every: int = 5  # Save every N epochs
    log_every: int = 100  # Log every N steps
    sample_every: int = 1000  # Generate samples every N steps
    
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
