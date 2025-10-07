"""
Configuration file for the Text2Sign Diffusion Model.
Contains all hyperparameters and settings for model training and inference.
"""

import torch

class Config:
    """Configuration class containing all hyperparameters for the diffusion model."""
    
    # Data settings
    DATA_ROOT = "training_data"
    BATCH_SIZE = 1
    NUM_WORKERS = 0

    # Model input/output dimensions  
    INPUT_SHAPE = (3, 28, 64, 64)  # (channels, frames, height, width) - Reduced to 64x64 for memory efficiency
    NUM_FRAMES = 28
    IMAGE_SIZE = 64  # Reduced from 128 to 64 for memory efficiency
    
    # Model architecture selection
    MODEL_ARCHITECTURE = "tinyfusion"  # Options: "unet3d", "vit3d", "dit3d", "vivit", "tinyfusion"
    
    # UNet3D architecture settings
    UNET_DIM = 16
    UNET_DIM_MULTS = (1, 2)
    UNET_CHANNELS = 3
    UNET_TIME_DIM = 16
    
    # ViT architecture settings
    VIT_EMBED_DIM = 768
    VIT_TIME_DIM = 768  # Time embedding dimension
    VIT_IMAGE_SIZE = 224  # Keep at 224 for pre-trained ViT compatibility
    VIT_FREEZE_BACKBONE = True  # Whether to freeze ViT backbone
    VIT_DROPOUT = 0.1  # Dropout rate
    
    # DiT3D architecture settings (Diffusion Transformer for Video)
    DIT_MODEL_SIZE = "DiT3D-XL-2"  # Fixed to HuggingFace pretrained model
    DIT_VIDEO_SIZE = (16, 64, 64)  # (frames, height, width) - matches INPUT_SHAPE - reduced for memory
    DIT_PATCH_SIZE = (4, 16, 16)  # (temporal_patch, spatial_patch_h, spatial_patch_w) - larger patches for efficiency
    DIT_LEARN_SIGMA = False  # Whether to predict noise variance - set to False for memory efficiency
    DIT_CLASS_DROPOUT_PROB = 0.1  # Dropout probability for classifier-free guidance
    DIT_NUM_CLASSES = 1000  # Number of classes (if using class conditioning instead of text)
    
    # ViViT architecture settings (Video Vision Transformer from HuggingFace)
    VIVIT_MODEL_NAME = "google/vivit-b-16x2-kinetics400"  # Use optimized config instead of pretrained
    VIVIT_VIDEO_SIZE = (28, 128, 128)  # (frames, height, width) - Match actual training data and INPUT_SHAPE
    VIVIT_TIME_DIM = 768  # Time embedding dimension
    VIVIT_FREEZE_BACKBONE = True  # Whether to freeze ViViT backbone
    VIVIT_NUM_TEMPORAL_LAYERS = 2  # Number of additional temporal attention layers
    VIVIT_NUM_HEADS = 8  # Number of attention heads
    VIVIT_DROPOUT = 0.1  # Dropout rate
    VIVIT_CLASS_DROPOUT_PROB = 0.1  # Dropout probability for classifier-free guidance

    # TinyFusion architecture settings (video wrapper around 2D TinyFusion backbone)
    TINYFUSION_VIDEO_SIZE = (28, 64, 64)  # Reduced from 128x128 to 64x64 for memory efficiency
    TINYFUSION_VARIANT = "DiT-D14/2"  # Use DiT-D14/2 which exactly matches the checkpoint architecture
    TINYFUSION_CHECKPOINT = "pretrained/TinyDiT-D14-MaskedKD-500K.pt"  # Pre-trained checkpoint
    TINYFUSION_FREEZE_BACKBONE = False  # Allow fine-tuning of the pre-trained model
    TINYFUSION_ENABLE_TEMPORAL_POST = True
    TINYFUSION_TEMPORAL_KERNEL = 2
    
    # Text conditioning settings
    TEXT_ENCODER_MODEL = "distilbert-base-uncased"  # Pre-trained text encoder
    TEXT_EMBED_DIM = 768  # Text embedding dimension
    TEXT_MAX_LENGTH = 77  # Maximum text sequence length
    TEXT_FREEZE_BACKBONE = True  # Whether to freeze text encoder backbone
    
    # Diffusion process settings
    TIMESTEPS = 50  # Number of diffusion timesteps for training
    INFERENCE_TIMESTEPS = 50  # Reduced timesteps for faster sampling (20x speedup)
    BETA_START = 0.01  # Start of noise schedule
    BETA_END = 0.02  # End of noise schedule
    
    # Noise scheduler settings
    NOISE_SCHEDULER = "cosine"  # Options: "linear", "cosine", "quadratic", "sigmoid"
    COSINE_S = 0.008  # Small offset for cosine scheduler to prevent β from being too small near t=0
    COSINE_MAX_BETA = 0.999  # Maximum beta value for cosine scheduler
    
    # Training settings
    LEARNING_RATE = 0.0001  # Higher learning rate for ViT (was 0.00001 for UNet)
    NUM_EPOCHS = 1000
    GRADIENT_CLIP = 1.0  # Enable gradient clipping for training stability
    GRADIENT_ACCUMULATION_STEPS = 4  # Increased for memory efficiency while maintaining effective batch size
    GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing to reduce memory usage
    
    # Memory optimization settings
    USE_MIXED_PRECISION = True  # Enable automatic mixed precision training
    USE_CPU_OFFLOAD = False  # Offload optimizer states to CPU (if needed)
    ENABLE_MEMORY_EFFICIENT_ATTENTION = True  # Use memory-efficient attention implementation
    CLEAR_CACHE_EVERY_STEPS = 1  # Clear CUDA cache every N steps to prevent fragmentation
    PREFETCH_FACTOR = 1  # DataLoader prefetch factor (reduced from default 2)
    PIN_MEMORY = False  # Disable pin memory to save GPU memory
    
    # TinyFusion memory optimization
    TINYFUSION_FRAME_CHUNK_SIZE = 4  # Process frames in smaller chunks (reduced from 8)
    TINYFUSION_USE_CHECKPOINTING = True  # Enable gradient checkpointing in TinyFusion
    
    # CUDA memory management settings
    PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"  # Disable expandable_segments due to compatibility issues
    CUDA_EMPTY_CACHE_STEPS = 1  # Empty CUDA cache every N steps
    
    # Dynamic Learning Rate Scheduler Settings
    USE_SCHEDULER = False  # Enable dynamic learning rate scheduling
    SCHEDULER_TYPE = "cosine_annealing_with_restarts"  # Options: "cosine_annealing", "cosine_annealing_with_restarts", "reduce_on_plateau", "exponential", "linear_warmup_cosine", "polynomial"
    
    # Cosine Annealing Settings
    COSINE_T_MAX = 50  # Period of cosine annealing (epochs)
    COSINE_ETA_MIN = 1e-7  # Minimum learning rate for cosine annealing
    
    # Cosine Annealing with Warm Restarts Settings
    COSINE_RESTARTS_T_0 = 25  # Initial restart period (epochs)
    COSINE_RESTARTS_T_MULT = 2  # Factor to increase restart period after each restart
    COSINE_RESTARTS_ETA_MIN = 1e-7  # Minimum learning rate
    
    # Reduce on Plateau Settings
    PLATEAU_FACTOR = 0.5  # Factor to reduce LR by when plateau is detected
    PLATEAU_PATIENCE = 15  # Number of epochs to wait before reducing LR
    PLATEAU_THRESHOLD = 1e-4  # Threshold for measuring improvement
    PLATEAU_MIN_LR = 1e-7  # Minimum learning rate for plateau scheduler
    
    # Exponential Decay Settings
    EXPONENTIAL_GAMMA = 0.95  # Multiplicative factor of learning rate decay
    
    # Linear Warmup + Cosine Settings
    WARMUP_EPOCHS = 10  # Number of epochs for linear warmup
    WARMUP_START_LR = 1e-6  # Starting learning rate for warmup
    
    # Polynomial Decay Settings
    POLYNOMIAL_POWER = 0.9  # Power for polynomial decay
    
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
    EXPERIMENT_NAME = "tinyfusion_test_1"  # Name for this experiment
    LOG_DIR = f"logs/{EXPERIMENT_NAME}"  # Directory for TensorBoard logs under logs/
    CHECKPOINT_DIR = f"checkpoints/{EXPERIMENT_NAME}"
    SAMPLES_DIR = f"generated_samples/{EXPERIMENT_NAME}"  # Directory to save generated GIF samples
    
    # Epoch-based logging frequencies
    SAMPLE_EVERY_EPOCHS = 5  # Generate samples every N epochs
    LOG_EVERY_EPOCHS = 1  # Log loss every N epochs
    SAVE_EVERY_EPOCHS = 10  # Save checkpoint every N epochs
    LOG_MODEL_GRAPH = True  # Enable model graph logging to aid debugging
    
    # Step-based diagnostic logging intervals (reduced for memory efficiency)
    NOISE_DISPLAY_EVERY_STEPS = 500   # Reduced frequency to save memory
    DIAGNOSTIC_LOG_EVERY_STEPS = 100    # Reduced frequency to save memory
    TENSORBOARD_FLUSH_EVERY_STEPS = 100 # Reduced frequency to save memory
    
    # Epoch-level flushing and comprehensive logging
    FLUSH_TENSORBOARD_EVERY_EPOCH = True  # Force TensorBoard flush at end of each epoch
    ENABLE_EPOCH_SUMMARY_LOGGING = True   # Enable comprehensive epoch-end summary logging
    ENABLE_REALTIME_METRICS = True        # Enable real-time metric tracking during training
    
    # Epoch-based logging intervals  
    PARAM_LOG_EVERY_EPOCHS = 10  # Log parameter histograms every N epochs
    SUMMARY_LOG_EVERY_EPOCHS = 10  # Log comprehensive training summary every N epochs
    
    # TensorBoard Logging Structure Settings
    TENSORBOARD_LOG_CATEGORIES = [
        "01_Training",           # Core training metrics (loss, LR, grad norm)
        "02_Loss_Components",    # Detailed loss breakdown
        "03_Epoch_Summary",      # Epoch-level aggregated metrics
        "04_Learning_Rate",      # LR scheduling and history
        "05_Performance",        # Training throughput metrics
        "06_Diffusion",         # Diffusion-specific metrics (noise MSE, SNR)
        "07_Noise_Analysis",    # Detailed noise prediction analysis
        "08_Model_Architecture", # Model parameters and structure
        "09_Parameter_Stats",   # Layer-wise parameter statistics
        "10_Parameter_Histograms", # Parameter distribution histograms
        "11_Gradient_Stats",    # Gradient statistics and norms
        "12_Generated_Samples", # Video samples and quality metrics
        "13_Noise_Visualization", # Noise prediction visualizations
        "14_System",           # GPU/MPS memory and system metrics
        "15_Configuration"     # Training configuration logging
    ]
    
    # Advanced logging features (reduced to save memory)
    ENABLE_GRADIENT_HISTOGRAMS = False   # Disabled - memory intensive
    ENABLE_PARAMETER_TRACKING = False    # Disabled - memory intensive
    ENABLE_VIDEO_LOGGING = True         # Keep enabled for essential monitoring
    ENABLE_NOISE_VISUALIZATION = False   # Disabled - memory intensive
    ENABLE_PERFORMANCE_PROFILING = False # Disabled - memory intensive
    ENABLE_LOSS_COMPONENT_TRACKING = True # Keep for loss monitoring
    ENABLE_GRADIENT_FLOW_ANALYSIS = True  # Temporarily enabled to check gradients
    ENABLE_MEMORY_TRACKING = True        # Keep for memory monitoring
    ENABLE_LEARNING_RATE_LOGGING = True  # Keep for LR monitoring

    # Sampling settings
    NUM_SAMPLES = 2  # Number of samples to generate for logging
    SAMPLE_GENERATION_TIMEOUT = 3000  # Timeout for sample generation (seconds)
    
    # TensorBoard writer settings
    TENSORBOARD_MAX_QUEUE = 100       # Maximum queue size for TensorBoard writer
    TENSORBOARD_FLUSH_SECS = 30       # Automatic flush interval (seconds)
    TENSORBOARD_FILENAME_SUFFIX = ""  # Optional suffix for TensorBoard files
    
    @classmethod
    def get_learning_rate(cls):
        """Get architecture-specific learning rate"""
        if cls.MODEL_ARCHITECTURE == "vit3d":
            return 0.00001  # Higher LR for ViT
        elif cls.MODEL_ARCHITECTURE == "unet3d":
            return 0.00001  # Lower LR for UNet
        elif cls.MODEL_ARCHITECTURE == "dit3d":
            return 0.0001   # DiT benefits from higher learning rates
        elif cls.MODEL_ARCHITECTURE == "vivit":
            return 0.00001  # ViViT moderate learning rate
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
        if cls.MODEL_ARCHITECTURE not in ["unet3d", "vit3d", "dit3d", "vivit", "tinyfusion"]:
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
        
        # Validate scheduler settings
        if hasattr(cls, 'USE_SCHEDULER') and cls.USE_SCHEDULER:
            valid_schedulers = [
                "cosine_annealing", "cosine_annealing_with_restarts", 
                "reduce_on_plateau", "exponential", "linear_warmup_cosine", 
                "polynomial", "adaptive"
            ]
            scheduler_type = getattr(cls, 'SCHEDULER_TYPE', '')
            if scheduler_type not in valid_schedulers:
                errors.append(f"SCHEDULER_TYPE must be one of {valid_schedulers}")
            
            # Validate scheduler-specific parameters
            if scheduler_type in ["cosine_annealing", "linear_warmup_cosine"]:
                if not hasattr(cls, 'COSINE_T_MAX') or cls.COSINE_T_MAX <= 0:
                    warnings.append("COSINE_T_MAX should be a positive integer")
            
            if scheduler_type == "cosine_annealing_with_restarts":
                if not hasattr(cls, 'COSINE_RESTARTS_T_0') or cls.COSINE_RESTARTS_T_0 <= 0:
                    warnings.append("COSINE_RESTARTS_T_0 should be a positive integer")
            
            if scheduler_type == "linear_warmup_cosine":
                if not hasattr(cls, 'WARMUP_EPOCHS') or cls.WARMUP_EPOCHS <= 0:
                    warnings.append("WARMUP_EPOCHS should be a positive integer")
                if cls.WARMUP_EPOCHS >= cls.NUM_EPOCHS:
                    warnings.append("WARMUP_EPOCHS should be less than NUM_EPOCHS")
        
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
        elif cls.MODEL_ARCHITECTURE == "dit3d":
            return {
                'video_size': cls.DIT_VIDEO_SIZE,
                'patch_size': cls.DIT_PATCH_SIZE,
                'in_channels': cls.UNET_CHANNELS,  # Input channels only
                'text_dim': cls.TEXT_EMBED_DIM,
                'learn_sigma': cls.DIT_LEARN_SIGMA,
                'class_dropout_prob': cls.DIT_CLASS_DROPOUT_PROB,
                'num_classes': cls.DIT_NUM_CLASSES
            }
        elif cls.MODEL_ARCHITECTURE == "vivit":
            return {
                'video_size': cls.VIVIT_VIDEO_SIZE,
                'in_channels': cls.UNET_CHANNELS,
                'out_channels': cls.UNET_CHANNELS,
                'time_dim': cls.VIVIT_TIME_DIM,
                'text_dim': cls.TEXT_EMBED_DIM,
                'model_name': cls.VIVIT_MODEL_NAME,
                'freeze_backbone': cls.VIVIT_FREEZE_BACKBONE,
                'num_temporal_layers': cls.VIVIT_NUM_TEMPORAL_LAYERS,
                'num_heads': cls.VIVIT_NUM_HEADS,
                'dropout': cls.VIVIT_DROPOUT,
                'class_dropout_prob': cls.VIVIT_CLASS_DROPOUT_PROB
            }
        elif cls.MODEL_ARCHITECTURE == "tinyfusion":
            return {
                'video_size': cls.TINYFUSION_VIDEO_SIZE,
                'in_channels': cls.UNET_CHANNELS,
                'out_channels': cls.UNET_CHANNELS,
                'text_dim': cls.TEXT_EMBED_DIM,
                'variant': cls.TINYFUSION_VARIANT,
                'checkpoint_path': cls.TINYFUSION_CHECKPOINT,
                'freeze_backbone': cls.TINYFUSION_FREEZE_BACKBONE,
                'enable_temporal_post': cls.TINYFUSION_ENABLE_TEMPORAL_POST,
                'temporal_kernel': cls.TINYFUSION_TEMPORAL_KERNEL,
                'frame_chunk_size': cls.TINYFUSION_FRAME_CHUNK_SIZE,
            }
        else:
            raise ValueError(f"Unknown model architecture: {cls.MODEL_ARCHITECTURE}")
    
    @classmethod
    def set_model_architecture(cls, architecture: str):
        """Set the model architecture"""
        if architecture not in ["unet3d", "vit3d", "dit3d", "vivit", "tinyfusion"]:
            raise ValueError(f"Unsupported architecture: {architecture}")
        cls.MODEL_ARCHITECTURE = architecture
        print(f"Model architecture set to: {architecture}")
    
    @classmethod
    def get_logging_config(cls):
        """Get comprehensive logging configuration dictionary"""
        return {
            # Basic logging settings
            'experiment_name': cls.EXPERIMENT_NAME,
            'log_dir': cls.LOG_DIR,
            'checkpoint_dir': cls.CHECKPOINT_DIR,
            'samples_dir': cls.SAMPLES_DIR,
            
            # Epoch-based frequencies
            'sample_every_epochs': cls.SAMPLE_EVERY_EPOCHS,
            'log_every_epochs': cls.LOG_EVERY_EPOCHS,
            'save_every_epochs': cls.SAVE_EVERY_EPOCHS,
            'param_log_every_epochs': cls.PARAM_LOG_EVERY_EPOCHS,
            'summary_log_every_epochs': cls.SUMMARY_LOG_EVERY_EPOCHS,
            
            # Step-based frequencies
            'noise_display_every_steps': cls.NOISE_DISPLAY_EVERY_STEPS,
            'diagnostic_log_every_steps': cls.DIAGNOSTIC_LOG_EVERY_STEPS,
            'tensorboard_flush_every_steps': cls.TENSORBOARD_FLUSH_EVERY_STEPS,
            
            # Epoch-level settings
            'flush_tensorboard_every_epoch': cls.FLUSH_TENSORBOARD_EVERY_EPOCH,
            'enable_epoch_summary_logging': cls.ENABLE_EPOCH_SUMMARY_LOGGING,
            'enable_realtime_metrics': cls.ENABLE_REALTIME_METRICS,
            
            # Feature flags
            'enable_gradient_histograms': cls.ENABLE_GRADIENT_HISTOGRAMS,
            'enable_parameter_tracking': cls.ENABLE_PARAMETER_TRACKING,
            'enable_video_logging': cls.ENABLE_VIDEO_LOGGING,
            'enable_noise_visualization': cls.ENABLE_NOISE_VISUALIZATION,
            'enable_performance_profiling': cls.ENABLE_PERFORMANCE_PROFILING,
            'enable_loss_component_tracking': cls.ENABLE_LOSS_COMPONENT_TRACKING,
            'enable_gradient_flow_analysis': cls.ENABLE_GRADIENT_FLOW_ANALYSIS,
            'enable_memory_tracking': cls.ENABLE_MEMORY_TRACKING,
            'enable_learning_rate_logging': cls.ENABLE_LEARNING_RATE_LOGGING,
            
            # TensorBoard settings
            'tensorboard_categories': cls.TENSORBOARD_LOG_CATEGORIES,
            'tensorboard_max_queue': cls.TENSORBOARD_MAX_QUEUE,
            'tensorboard_flush_secs': cls.TENSORBOARD_FLUSH_SECS,
            'tensorboard_filename_suffix': cls.TENSORBOARD_FILENAME_SUFFIX,
            
            # Sampling settings
            'num_samples': cls.NUM_SAMPLES,
            'sample_generation_timeout': cls.SAMPLE_GENERATION_TIMEOUT,
            
            # Advanced settings
            'log_model_graph': cls.LOG_MODEL_GRAPH
        }
    
    @classmethod
    def should_log_step(cls, step: int) -> dict:
        """Determine what logging actions should be performed at this step"""
        return {
            'log_diagnostics': step % cls.DIAGNOSTIC_LOG_EVERY_STEPS == 0,
            'flush_tensorboard': step % cls.TENSORBOARD_FLUSH_EVERY_STEPS == 0,
            'display_noise': step % cls.NOISE_DISPLAY_EVERY_STEPS == 0,
        }
    
    @classmethod
    def should_log_epoch(cls, epoch: int) -> dict:
        """Determine what logging actions should be performed at this epoch"""
        return {
            'log_loss': epoch % cls.LOG_EVERY_EPOCHS == 0,
            'generate_samples': epoch % cls.SAMPLE_EVERY_EPOCHS == 0,
            'save_checkpoint': epoch % cls.SAVE_EVERY_EPOCHS == 0,
            'log_parameters': epoch % cls.PARAM_LOG_EVERY_EPOCHS == 0,
            'log_summary': epoch % cls.SUMMARY_LOG_EVERY_EPOCHS == 0,
            'flush_tensorboard': cls.FLUSH_TENSORBOARD_EVERY_EPOCH,
            'comprehensive_logging': cls.ENABLE_EPOCH_SUMMARY_LOGGING
        }
    
    @classmethod
    def create_tensorboard_writer(cls):
        """Create and configure TensorBoard SummaryWriter with proper settings"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            import os
            
            # Ensure log directory exists
            os.makedirs(cls.LOG_DIR, exist_ok=True)
            
            # Configure writer with proper settings
            writer = SummaryWriter(
                log_dir=cls.LOG_DIR,
                max_queue=cls.TENSORBOARD_MAX_QUEUE,
                flush_secs=cls.TENSORBOARD_FLUSH_SECS,
                filename_suffix=cls.TENSORBOARD_FILENAME_SUFFIX
            )
            
            return writer
        except ImportError:
            print("⚠️ TensorBoard not available. Install with: pip install tensorboard")
            return None
        except Exception as e:
            print(f"❌ Failed to create TensorBoard writer: {e}")
            return None
    
    @classmethod
    def setup_logging_directories(cls):
        """Create all necessary logging directories"""
        import os
        
        directories = [
            cls.LOG_DIR,
            cls.CHECKPOINT_DIR,
            cls.SAMPLES_DIR,
        ]
        
        created_dirs = []
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                created_dirs.append(directory)
            except Exception as e:
                print(f"❌ Failed to create directory {directory}: {e}")
                return False
        
        print(f"✅ Created logging directories: {', '.join(created_dirs)}")
        return True
    
    @classmethod
    def print_logging_status(cls):
        """Print current logging configuration status"""
        print("=" * 60)
        print("LOGGING CONFIGURATION STATUS")
        print("=" * 60)
        
        config = cls.get_logging_config()
        
        print(f"📁 Directories:")
        print(f"   Log Dir: {config['log_dir']}")
        print(f"   Checkpoint Dir: {config['checkpoint_dir']}")
        print(f"   Samples Dir: {config['samples_dir']}")
        
        print(f"\n⏰ Epoch Frequencies:")
        print(f"   Sample Generation: Every {config['sample_every_epochs']} epochs")
        print(f"   Loss Logging: Every {config['log_every_epochs']} epochs")
        print(f"   Checkpoint Saving: Every {config['save_every_epochs']} epochs")
        print(f"   Parameter Logging: Every {config['param_log_every_epochs']} epochs")
        print(f"   Summary Logging: Every {config['summary_log_every_epochs']} epochs")
        
        print(f"\n🔄 Step Frequencies:")
        print(f"   Diagnostics: Every {config['diagnostic_log_every_steps']} steps")
        print(f"   TensorBoard Flush: Every {config['tensorboard_flush_every_steps']} steps")
        print(f"   Noise Display: Every {config['noise_display_every_steps']} steps")
        
        print(f"\n✨ Feature Flags:")
        features = [
            ('Gradient Histograms', config['enable_gradient_histograms']),
            ('Parameter Tracking', config['enable_parameter_tracking']),
            ('Video Logging', config['enable_video_logging']),
            ('Performance Profiling', config['enable_performance_profiling']),
            ('Loss Component Tracking', config['enable_loss_component_tracking']),
            ('Memory Tracking', config['enable_memory_tracking']),
            ('Epoch Summary Logging', config['enable_epoch_summary_logging']),
            ('Realtime Metrics', config['enable_realtime_metrics']),
        ]
        
        for feature_name, enabled in features:
            status = "✅ Enabled" if enabled else "❌ Disabled"
            print(f"   {feature_name}: {status}")
        
        print(f"\n📊 TensorBoard Categories: {len(config['tensorboard_categories'])} categories")
        for category in config['tensorboard_categories']:
            print(f"   • {category}")
        
        print("=" * 60)
