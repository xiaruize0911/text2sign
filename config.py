"""
Configuration file for the Text2Sign Diffusion Model
This file contains all hyperparameters and settings for easy modification
"""

import torch

class Config:
    """Configuration class containing all hyperparameters"""
    
    # Data settings
    DATA_ROOT = "training_data"
    BATCH_SIZE = 2
    NUM_WORKERS = 2

    # Model dimensions
    INPUT_SHAPE = (3, 28, 128, 128)  # (channels, frames, height, width)
    NUM_FRAMES = 28  # Number of frames per video
    IMAGE_SIZE = 128  # Height and width of each frame
    
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
    VIT_IMAGE_SIZE = 224  # Keep at 224 for pre-trained ViT compatibility
    VIT_FREEZE_BACKBONE = True  # Whether to freeze ViT backbone
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
    NOISE_SCHEDULER = "cosine"  # Options: "linear", "cosine", "quadratic", "sigmoid"
    COSINE_S = 0.008  # Small offset for cosine scheduler to prevent β from being too small near t=0
    COSINE_MAX_BETA = 0.999  # Maximum beta value for cosine scheduler
    
    # Training settings
    LEARNING_RATE = 0.01  # Higher learning rate for ViT (was 0.00001 for UNet)
    NUM_EPOCHS = 200
    GRADIENT_CLIP = 1.0  # Enable gradient clipping for training stability
    GRADIENT_ACCUMULATION_STEPS = 4  # Number of steps to accumulate gradients before optimizer step
    
    # Reproducibility settings
    RANDOM_SEED = 42  # Random seed for reproducibility
    DETERMINISTIC = True  # Use deterministic algorithms when possible
    
    # Optimizer settings for different architectures
    OPTIMIZER_TYPE = "adamw"  # Options: "adam", "adamw"
    WEIGHT_DECAY = 0.01  # Weight decay for AdamW (good for ViT)
    ADAM_BETAS = (0.9, 0.999)  # Beta values for Adam/AdamW
    
    
    # Device settings
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging and checkpointing
    EXPERIMENT_NAME = "text2sign_experiment_vit1"  # Name for this experiment
    LOG_DIR = f"logs/{EXPERIMENT_NAME}"  # Directory for TensorBoard logs under logs/
    CHECKPOINT_DIR = f"checkpoints/{EXPERIMENT_NAME}"
    SAMPLES_DIR = f"generated_samples/{EXPERIMENT_NAME}"  # Directory to save generated GIF samples
    
    # Epoch-based logging frequencies
    SAMPLE_EVERY_EPOCHS = 5  # Generate samples every N epochs
    LOG_EVERY_EPOCHS = 1  # Log loss every N epochs
    SAVE_EVERY_EPOCHS = 10  # Save checkpoint every N epochs
    LOG_MODEL_GRAPH = True  # Enable model graph logging to aid debugging
    
    # Step-based diagnostic logging intervals (for within-epoch diagnostics)
    NOISE_DISPLAY_EVERY_STEPS = 500  # Save noise display GIFs every N steps
    DIAGNOSTIC_LOG_EVERY_STEPS = 500  # Log detailed diagnostics every N steps
    TENSORBOARD_FLUSH_EVERY_STEPS = 500  # Flush TensorBoard every N steps
    
    # Epoch-based logging intervals  
    PARAM_LOG_EVERY_EPOCHS = 10  # Log parameter histograms every N epochs
    SUMMARY_LOG_EVERY_EPOCHS = 10  # Log comprehensive training summary every N epochs

    # Sampling settings
    NUM_SAMPLES = 2  # Number of samples to generate for logging
    
    @classmethod
    def get_learning_rate(cls):
        """Get architecture-specific learning rate"""
        if cls.MODEL_ARCHITECTURE == "vit3d":
            return 0.001  # Higher LR for ViT
        elif cls.MODEL_ARCHITECTURE == "unet3d":
            return 0.00001  # Lower LR for UNet
        else:
            return cls.LEARNING_RATE
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        warnings = []
        
        # Validate required directories
        if not isinstance(cls.DATA_ROOT, str) or not cls.DATA_ROOT.strip():
            errors.append("DATA_ROOT must be a non-empty string")
        
        # Validate numerical parameters
        if not isinstance(cls.BATCH_SIZE, int) or cls.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be a positive integer")
        
        if not isinstance(cls.NUM_EPOCHS, int) or cls.NUM_EPOCHS <= 0:
            errors.append("NUM_EPOCHS must be a positive integer")
        
        if not isinstance(cls.LEARNING_RATE, (int, float)) or cls.LEARNING_RATE <= 0:
            errors.append("LEARNING_RATE must be a positive number")
        
        if not isinstance(cls.TIMESTEPS, int) or cls.TIMESTEPS <= 0:
            errors.append("TIMESTEPS must be a positive integer")
        
        if not isinstance(cls.GRADIENT_ACCUMULATION_STEPS, int) or cls.GRADIENT_ACCUMULATION_STEPS <= 0:
            errors.append("GRADIENT_ACCUMULATION_STEPS must be a positive integer")
        
        # Validate model architecture
        if cls.MODEL_ARCHITECTURE not in ["unet3d", "vit3d"]:
            errors.append(f"Unknown MODEL_ARCHITECTURE: {cls.MODEL_ARCHITECTURE}")
        
        # Validate noise scheduler
        valid_schedulers = ["linear", "cosine", "quadratic", "sigmoid"]
        if cls.NOISE_SCHEDULER not in valid_schedulers:
            errors.append(f"NOISE_SCHEDULER must be one of {valid_schedulers}")
        
        # Validate logging and checkpoint frequencies
        if not isinstance(cls.SAMPLE_EVERY_EPOCHS, int) or cls.SAMPLE_EVERY_EPOCHS <= 0:
            errors.append("SAMPLE_EVERY_EPOCHS must be a positive integer")
        
        if not isinstance(cls.LOG_EVERY_EPOCHS, int) or cls.LOG_EVERY_EPOCHS <= 0:
            errors.append("LOG_EVERY_EPOCHS must be a positive integer")
        
        if not isinstance(cls.SAVE_EVERY_EPOCHS, int) or cls.SAVE_EVERY_EPOCHS <= 0:
            errors.append("SAVE_EVERY_EPOCHS must be a positive integer")
        
        if not isinstance(cls.PARAM_LOG_EVERY_EPOCHS, int) or cls.PARAM_LOG_EVERY_EPOCHS <= 0:
            errors.append("PARAM_LOG_EVERY_EPOCHS must be a positive integer")
        
        if not isinstance(cls.SUMMARY_LOG_EVERY_EPOCHS, int) or cls.SUMMARY_LOG_EVERY_EPOCHS <= 0:
            errors.append("SUMMARY_LOG_EVERY_EPOCHS must be a positive integer")
        
        # Validate step-based diagnostic frequencies
        if not isinstance(cls.NOISE_DISPLAY_EVERY_STEPS, int) or cls.NOISE_DISPLAY_EVERY_STEPS <= 0:
            errors.append("NOISE_DISPLAY_EVERY_STEPS must be a positive integer")
        
        if not isinstance(cls.DIAGNOSTIC_LOG_EVERY_STEPS, int) or cls.DIAGNOSTIC_LOG_EVERY_STEPS <= 0:
            errors.append("DIAGNOSTIC_LOG_EVERY_STEPS must be a positive integer")
        
        if not isinstance(cls.TENSORBOARD_FLUSH_EVERY_STEPS, int) or cls.TENSORBOARD_FLUSH_EVERY_STEPS <= 0:
            errors.append("TENSORBOARD_FLUSH_EVERY_STEPS must be a positive integer")
        
        # Validate input shape
        if not isinstance(cls.INPUT_SHAPE, tuple) or len(cls.INPUT_SHAPE) != 4:
            errors.append("INPUT_SHAPE must be a tuple of length 4")
        
        # Warnings for potentially problematic settings
        if cls.BATCH_SIZE > 8:
            warnings.append(f"Large batch size ({cls.BATCH_SIZE}) may cause memory issues")
        
        if cls.LEARNING_RATE > 0.01:
            warnings.append(f"High learning rate ({cls.LEARNING_RATE}) may cause training instability")
        
        # Check effective batch size with gradient accumulation
        effective_batch_size = cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION_STEPS
        if effective_batch_size > 32:
            warnings.append(f"Large effective batch size ({effective_batch_size} = {cls.BATCH_SIZE} × {cls.GRADIENT_ACCUMULATION_STEPS}) may affect training dynamics")
        
        # Print results
        if errors:
            print("❌ Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            raise ValueError("Configuration validation failed")
        
        if warnings:
            print("⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        print("✅ Configuration validation passed")
    
    @classmethod
    def print_config(cls):
        """Print all configuration settings"""
        # Validate first
        cls.validate_config()
        
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
