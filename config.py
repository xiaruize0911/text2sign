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
    MODEL_ARCHITECTURE = "unet3d"  # Options: "unet3d", "vit3d"
    
    # UNet3D architecture settings (smaller for MacBook M4)
    UNET_DIM = 16  # Base dimension (reduced further for MacBook M4)
    UNET_DIM_MULTS = (1, 2)  # Dimension multipliers (reduced from (1, 2, 4))
    UNET_CHANNELS = 3  # RGB channels
    UNET_TIME_DIM = 16  # Time embedding dimension (reduced)
    
    # ViT architecture settings (ViT-B/16 only)
    VIT_EMBED_DIM = 768  # Embedding dimension (ViT-B/16 fixed)
    VIT_TIME_DIM = 768  # Time embedding dimension
    VIT_IMAGE_SIZE = 224  # Keep at 224 for pre-trained ViT compatibility
    VIT_FREEZE_BACKBONE = False  # Whether to freeze ViT backbone
    VIT_DROPOUT = 0.1  # Dropout rate
    
    # Text conditioning settings
    TEXT_ENCODER_MODEL = "distilbert-base-uncased"  # Pre-trained text encoder
    TEXT_EMBED_DIM = 768  # Text embedding dimension
    TEXT_MAX_LENGTH = 77  # Maximum text sequence length
    TEXT_FREEZE_BACKBONE = True  # Whether to freeze text encoder backbone
    
    # Diffusion process settings
    TIMESTEPS = 1000  # Number of diffusion timesteps
    BETA_START = 0.01  # Start of noise schedule
    BETA_END = 0.02  # End of noise schedule
    
    # Noise scheduler settings
    NOISE_SCHEDULER = "linear"  # Options: "linear", "cosine", "quadratic", "sigmoid"
    COSINE_S = 0.008  # Small offset for cosine scheduler to prevent β from being too small near t=0
    COSINE_MAX_BETA = 0.999  # Maximum beta value for cosine scheduler
    
    # Training settings
    LEARNING_RATE = 0.0001  # Higher learning rate for ViT (was 0.00001 for UNet)
    NUM_EPOCHS = 200
    GRADIENT_CLIP = 1.0  # Enable gradient clipping for training stability
    
    # Optimizer settings for different architectures
    OPTIMIZER_TYPE = "adamw"  # Options: "adam", "adamw"
    WEIGHT_DECAY = 0.01  # Weight decay for AdamW (good for ViT)
    ADAM_BETAS = (0.9, 0.999)  # Beta values for Adam/AdamW
    
    
    # Device settings
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging and checkpointing
    EXPERIMENT_NAME = "text2sign_experiment_unet1"  # Name for this experiment
    LOG_DIR = f"logs/{EXPERIMENT_NAME}"  # Directory for TensorBoard logs under logs/
    CHECKPOINT_DIR = f"checkpoints/{EXPERIMENT_NAME}"
    SAMPLES_DIR = f"generated_samples/{EXPERIMENT_NAME}"  # Directory to save generated GIF samples
    SAMPLE_EVERY = 2040  # Sample and log every N steps (once per epoch, ~4082 samples / 4 batch size)
    LOG_EVERY = 2040  # Log loss every N steps (once per epoch)
    SAVE_EVERY = 2040  # Save checkpoint every N steps (every two epochs)
    LOG_MODEL_GRAPH = True  # Enable model graph logging to aid debugging

    # Sampling settings
    NUM_SAMPLES = 2  # Number of samples to generate for logging
    
    @classmethod
    def get_learning_rate(cls):
        """Get architecture-specific learning rate"""
        if cls.MODEL_ARCHITECTURE == "vit3d":
            return 0.0001  # Higher LR for ViT
        elif cls.MODEL_ARCHITECTURE == "unet3d":
            return 0.00001  # Lower LR for UNet
        else:
            return cls.LEARNING_RATE
    
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
                'time_dim': cls.UNET_TIME_DIM,
                'text_dim': cls.TEXT_EMBED_DIM
            }
        elif cls.MODEL_ARCHITECTURE == "vit3d":
            return {
                'in_channels': cls.UNET_CHANNELS,
                'out_channels': cls.UNET_CHANNELS,
                'embed_dim': cls.VIT_EMBED_DIM,
                'time_dim': cls.VIT_TIME_DIM,
                'text_dim': cls.TEXT_EMBED_DIM,  # Add text_dim for ViT3D
                'freeze_backbone': cls.VIT_FREEZE_BACKBONE
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
