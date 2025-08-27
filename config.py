"""
Configuration file for the Text2Sign Diffusion Model
This file contains all hyperparameters and settings for easy modification
"""

import torch

class Config:
    """Configuration class containing all hyperparameters"""
    
    # Data settings
    DATA_ROOT = "training_data"
    BATCH_SIZE = 4
    NUM_WORKERS = 4

    # Model dimensions
    INPUT_SHAPE = (3, 28, 128, 128)  # (channels, frames, height, width)
    
    # Model architecture selection
    MODEL_ARCHITECTURE = "vit3d"  # Options: "unet3d", "vit3d"
    
    # UNet3D architecture settings (smaller for MacBook M4)
    UNET_DIM = 16  # Base dimension (reduced further for MacBook M4)
    UNET_DIM_MULTS = (1, 2)  # Dimension multipliers (reduced from (1, 2, 4))
    UNET_CHANNELS = 3  # RGB channels
    UNET_TIME_DIM = 16  # Time embedding dimension (reduced)
    
    # ViT architecture settings (ViT-B/16 only)
    VIT_EMBED_DIM = 768  # Embedding dimension (ViT-B/16 fixed)
    VIT_TIME_DIM = 768  # Time embedding dimension
    VIT_IMAGE_SIZE = 224  # Input image size for ViT
    VIT_FREEZE_BACKBONE = False  # Whether to freeze ViT backbone
    VIT_DROPOUT = 0.1  # Dropout rate
    
    # Diffusion process settings
    TIMESTEPS = 300  # Number of diffusion timesteps
    BETA_START = 0.0001  # Start of noise schedule
    BETA_END = 0.02  # End of noise schedule
    
    # Training settings
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    GRADIENT_CLIP = 1.0
    
    # AMP (Automatic Mixed Precision) settings
    USE_AMP = True  # Enable mixed precision training
    AMP_DTYPE = torch.float16  # Use float16 for AMP (can also be torch.bfloat16)
    
    # Device settings
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging and checkpointing
    LOG_DIR = "logs"
    CHECKPOINT_DIR = "checkpoints"
    SAMPLES_DIR = "generated_samples"  # Directory to save generated GIF samples
    SAMPLE_EVERY = 1000  # Sample and log every N steps
    LOG_EVERY = 10  # Log loss every N steps (reduced from 10 for more frequent updates)
    SAVE_EVERY = 1000  # Save checkpoint every N steps
    LOG_MODEL_GRAPH = False  # Disable model graph logging to avoid tracing issues
    
    # Sampling settings
    NUM_SAMPLES = 4  # Number of samples to generate for logging
    
    @classmethod
    def print_config(cls):
        """Print all configuration settings"""
        print("=" * 50)
        print("Configuration Settings:")
        print("=" * 50)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")
        print("=" * 50)
        
    @classmethod
    def get_model_config(cls):
        """Get model-specific configuration based on MODEL_ARCHITECTURE"""
        if cls.MODEL_ARCHITECTURE == "unet3d":
            return {
                'in_channels': cls.UNET_CHANNELS,
                'out_channels': cls.UNET_CHANNELS,
                'dim': cls.UNET_DIM,
                'dim_mults': cls.UNET_DIM_MULTS,
                'time_dim': cls.UNET_TIME_DIM
            }
        elif cls.MODEL_ARCHITECTURE == "vit3d":
            return {
                'input_shape': cls.INPUT_SHAPE,
                'embed_dim': cls.VIT_EMBED_DIM,
                'time_dim': cls.VIT_TIME_DIM,
                'image_size': cls.VIT_IMAGE_SIZE,
                'freeze_backbone': cls.VIT_FREEZE_BACKBONE,
                'dropout': cls.VIT_DROPOUT
            }
        else:
            raise ValueError(f"Unknown model architecture: {cls.MODEL_ARCHITECTURE}")
    
    @classmethod
    def set_model_architecture(cls, architecture: str):
        """Set the model architecture"""
        if architecture not in ["unet3d", "vit3d"]:
            raise ValueError(f"Unsupported architecture: {architecture}")
        cls.MODEL_ARCHITECTURE = architecture
        print(f"Model architecture set to: {architecture}")
