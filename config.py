"""
Configuration file for the Text2Sign Diffusion Model
This file contains all hyperparameters and settings for easy modification
"""

import torch

class Config:
    """Configuration class containing all hyperparameters"""
    
    # Data settings
    DATA_ROOT = "training_data"
    BATCH_SIZE = 2  # Small batch size for MacBook M4
    NUM_WORKERS = 2  # Reduced for MacBook
    
    # Model dimensions
    INPUT_SHAPE = (3, 28, 128, 128)  # (channels, frames, height, width)
    
    # UNet3D architecture settings (smaller for MacBook M4)
    UNET_DIM = 16  # Base dimension (reduced further for MacBook M4)
    UNET_DIM_MULTS = (1, 2)  # Dimension multipliers (reduced from (1, 2, 4))
    UNET_CHANNELS = 3  # RGB channels
    UNET_TIME_DIM = 16  # Time embedding dimension (reduced)
    
    # Diffusion process settings
    TIMESTEPS = 100  # Number of diffusion timesteps
    BETA_START = 0.0001  # Start of noise schedule
    BETA_END = 0.02  # End of noise schedule
    
    # Training settings
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    GRADIENT_CLIP = 1.0
    
    # Device settings
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging and checkpointing
    LOG_DIR = "logs"
    CHECKPOINT_DIR = "checkpoints"
    SAMPLES_DIR = "generated_samples"  # Directory to save generated GIF samples
    SAMPLE_EVERY = 1000  # Sample and log every N steps
    LOG_EVERY = 10  # Log loss every N steps
    SAVE_EVERY = 100  # Save checkpoint every N steps
    
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
