#!/usr/bin/env python3
"""
Main script for the Text2Sign diffusion model.
Provides a command-line interface for training, testing, sampling, and visualization.
"""

import argparse
import sys
import os
import torch
import logging
import shutil
from typing import Optional

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable HuggingFace progress bars for faster initialization
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Reduce HuggingFace logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from methods import setup_training
from dataset import test_dataloader
from models import test_unet3d
from models.architectures import test_vit3d, test_dit3d, test_vivit
from diffusion import test_diffusion
from utils import get_device_info, print_model_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(resume=False):
    """Train the diffusion model
    
    Args:
        resume (bool): Whether to resume training from the latest checkpoint
    """
    logger.info("Starting training...")
    
    # Clean log directory
    if Config.LOG_DIR and os.path.exists(Config.LOG_DIR):
        shutil.rmtree(Config.LOG_DIR)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Setup training
    trainer = setup_training(Config)
    
    # Resume from checkpoint if requested
    if resume:
        # Try to load the latest checkpoint
        latest_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "latest_checkpoint.pt")
        if os.path.exists(latest_checkpoint):
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
            try:
                trainer.load_checkpoint("latest_checkpoint.pt")
            except Exception as e:
                logger.error(f"Failed to load checkpoint securely: {e}")
                logger.warning("Starting fresh training.")
        else:
            logger.warning(f"No checkpoint found at {latest_checkpoint}. Starting fresh training.")
    
    # Start training
    trainer.train()

def list_checkpoints():
    """List available checkpoints with architecture detection"""
    checkpoint_base_dir = "checkpoints"
    
    if not os.path.exists(checkpoint_base_dir):
        logger.info(f"No checkpoint directory found at: {checkpoint_base_dir}")
        return
    
    # Function to detect architecture from checkpoint
    def detect_checkpoint_arch(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model_keys = list(checkpoint['model_state_dict'].keys())
            
            if any('init_conv' in key or 'encoder_resblocks' in key for key in model_keys):
                return 'unet3d'
            elif any('vit.embeddings' in key or 'temporal_layers' in key for key in model_keys):
                return 'vivit'
            elif any('patch_embed' in key or 'blocks' in key for key in model_keys):
                return 'vit3d'
            elif any('dit_blocks' in key or 'x_embedder' in key for key in model_keys):
                return 'dit3d'
            else:
                return 'unknown'
        except:
            return 'error'
    
    logger.info("Available checkpoints by architecture:")
    
    # Scan all experiment directories
    for exp_dir in os.listdir(checkpoint_base_dir):
        exp_path = os.path.join(checkpoint_base_dir, exp_dir)
        if os.path.isdir(exp_path):
            checkpoint_files = [f for f in os.listdir(exp_path) 
                              if f.endswith('.pt') and not f.startswith('._')]
            
            if checkpoint_files:
                # Detect architecture from first checkpoint
                first_checkpoint = os.path.join(exp_path, checkpoint_files[0])
                arch = detect_checkpoint_arch(first_checkpoint)
                
                logger.info(f"\n📁 {exp_dir} ({arch} architecture):")
                
                for checkpoint in sorted(checkpoint_files):
                    checkpoint_path = os.path.join(exp_path, checkpoint)
                    
                    # Get file size
                    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
                    
                    # Try to get epoch/step info
                    try:
                        checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                        epoch = checkpoint_data.get('epoch', 'unknown')
                        step = checkpoint_data.get('global_step', 'unknown')
                        logger.info(f"  • {checkpoint} ({file_size:.1f} MB) - Epoch: {epoch}, Step: {step}")
                    except Exception as e:
                        logger.info(f"  • {checkpoint} ({file_size:.1f} MB) - Could not read info: {e}")
    
    logger.info(f"\nCurrent config architecture: {Config.MODEL_ARCHITECTURE}")
    logger.info("To use a checkpoint with different architecture, either:")
    logger.info("1. Change Config.MODEL_ARCHITECTURE in config.py")
    logger.info("2. Use a checkpoint with matching architecture")

def fix_config_architecture(checkpoint_path):
    """Fix config architecture and dimensions to match checkpoint"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Detect architecture and dimensions from checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model_keys = list(checkpoint['model_state_dict'].keys())
        state_dict = checkpoint['model_state_dict']
        
        if any('init_conv' in key or 'encoder_resblocks' in key for key in model_keys):
            target_arch = 'unet3d'
        elif any('vit.embeddings' in key or 'temporal_layers' in key for key in model_keys):
            target_arch = 'vivit'
        elif any('patch_embed' in key or 'blocks' in key for key in model_keys):
            target_arch = 'vit3d'
        elif any('dit_blocks' in key or 'x_embedder' in key for key in model_keys):
            target_arch = 'dit3d'
        else:
            logger.error("Could not detect architecture from checkpoint")
            return
            
        # Try to detect dimensions from model parameters
        target_dims = None
        if target_arch == 'unet3d':
            # For UNet3D, look for final conv layer to infer output shape
            for key, tensor in state_dict.items():
                if 'final_conv' in key and 'weight' in key:
                    # final_conv usually has shape (out_channels, in_channels, ...)
                    out_channels = tensor.shape[0]
                    if out_channels == 3:  # RGB channels
                        logger.info(f"Detected UNet3D output channels: {out_channels}")
                    break
        
        current_arch = Config.MODEL_ARCHITECTURE
        current_shape = Config.INPUT_SHAPE
        
        if target_arch == current_arch and target_dims is None:
            logger.info(f"✅ Config architecture ({current_arch}) already matches checkpoint")
            return
        
        logger.info(f"Updating config architecture: {current_arch} → {target_arch}")
        
        # Update config file
        with open('config.py', 'r') as f:
            config_content = f.read()
        
        # Update architecture and dimensions
        import re
        updates_made = []
        
        # Replace MODEL_ARCHITECTURE
        arch_pattern = r'MODEL_ARCHITECTURE\s*=\s*["\'][^"\']*["\']'
        arch_replacement = f'MODEL_ARCHITECTURE = "{target_arch}"'
        
        if re.search(arch_pattern, config_content):
            config_content = re.sub(arch_pattern, arch_replacement, config_content)
            updates_made.append(f"MODEL_ARCHITECTURE = '{target_arch}'")
        
        # For UNet3D, update dimensions to match typical trained model
        if target_arch == 'unet3d':
            # Update INPUT_SHAPE
            shape_pattern = r'INPUT_SHAPE\s*=\s*\([^)]+\)'
            shape_replacement = 'INPUT_SHAPE = (3, 28, 128, 128)'
            if re.search(shape_pattern, config_content):
                config_content = re.sub(shape_pattern, shape_replacement, config_content)
                updates_made.append("INPUT_SHAPE = (3, 28, 128, 128)")
            
            # Update NUM_FRAMES
            frames_pattern = r'NUM_FRAMES\s*=\s*\d+'
            frames_replacement = 'NUM_FRAMES = 28'
            if re.search(frames_pattern, config_content):
                config_content = re.sub(frames_pattern, frames_replacement, config_content)
                updates_made.append("NUM_FRAMES = 28")
            
            # Update IMAGE_SIZE
            size_pattern = r'IMAGE_SIZE\s*=\s*\d+'
            size_replacement = 'IMAGE_SIZE = 128'
            if re.search(size_pattern, config_content):
                config_content = re.sub(size_pattern, size_replacement, config_content)
                updates_made.append("IMAGE_SIZE = 128")
        
        if updates_made:
            with open('config.py', 'w') as f:
                f.write(config_content)
                
            logger.info(f"✅ Config updated! Changes made:")
            for update in updates_made:
                logger.info(f"  • {update}")
            logger.info("You can now run sampling or training with the checkpoint")
        else:
            logger.error("Could not update config.py")
            
    except Exception as e:
        logger.error(f"Error fixing config: {e}")

def list_checkpoints_old():
    """List available checkpoints for resuming training"""
    checkpoint_dir = Config.CHECKPOINT_DIR
    
    if not os.path.exists(checkpoint_dir):
        logger.info(f"No checkpoint directory found at: {checkpoint_dir}")
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.endswith('.pt') and not f.startswith('._')]
    
    if not checkpoint_files:
        logger.info(f"No checkpoint files found in: {checkpoint_dir}")
        return
    
    logger.info(f"Available checkpoints in {checkpoint_dir}:")
    for checkpoint in sorted(checkpoint_files):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        file_size = os.path.getsize(checkpoint_path)
        file_size_mb = file_size / (1024 * 1024)
        # Try to load checkpoint info securely
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            epoch = checkpoint_data.get('epoch', 'unknown')
            step = checkpoint_data.get('global_step', 'unknown')
            logger.info(f"  • {checkpoint} ({file_size_mb:.1f} MB) - Epoch: {epoch}, Step: {step}")
        except Exception as e:
            logger.info(f"  • {checkpoint} ({file_size_mb:.1f} MB) - Could not read info: {e}")

def test_components():
    """Test all components of the system"""
    logger.info("Testing all components...")
    
    # Print device information
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Test dataloader
    logger.info("Testing dataloader...")
    test_dataloader()
    
    # Test model architectures
    logger.info("Testing model architectures...")
    
    # Test UNet3D model
    logger.info("Testing UNet3D model...")
    test_unet3d()
    
    # Test ViT3D model
    logger.info("Testing ViT3D model...")
    test_vit3d()
    
    # Test DiT3D model
    logger.info("Testing DiT3D model...")
    test_dit3d()
    
    # Test ViViT model
    logger.info("Testing ViViT model...")
    test_vivit()
    
    # Test diffusion model
    logger.info("Testing diffusion model...")
    test_diffusion()
    
    logger.info("All component tests completed successfully!")

def sample_videos(
    checkpoint_path: str,
    num_samples: int = 4,
    output_dir: str = "samples",
    text_prompt: str = "hello",
    eta: Optional[float] = None,
):
    """
    Generate sample videos using a trained model
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        num_samples (int): Number of videos to generate
        output_dir (str): Directory to save generated videos
        text_prompt (str): Text prompt for generation
        eta (float, optional): DDIM stochasticity coefficient. ``0`` for deterministic sampling,
            ``1`` for ancestral noise. Defaults to ``None`` (auto-select based on schedule).
    """
    logger.info(f"Generating {num_samples} sample videos with text: '{text_prompt}'")

    # --- Input Validation ---
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        logger.error("Checkpoint path must be a non-empty string")
        return
    if not isinstance(num_samples, int) or num_samples <= 0:
        logger.error("Number of samples must be a positive integer")
        return
    if not isinstance(output_dir, str) or not output_dir.strip():
        logger.error("Output directory must be a non-empty string")
        return
    if not text_prompt or not isinstance(text_prompt, str):
        logger.warning("Empty or invalid text prompt, using default 'hello'.")
        text_prompt = "hello"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect model architecture from checkpoint
    def detect_architecture_from_checkpoint(checkpoint_path):
        """Detect model architecture from checkpoint keys"""
        if not os.path.exists(checkpoint_path):
            return None
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model_keys = list(checkpoint['model_state_dict'].keys())
            
            # Check for UNet3D keys
            if any('init_conv' in key or 'encoder_resblocks' in key for key in model_keys):
                return 'unet3d'
            # Check for ViViT keys  
            elif any('vit.embeddings' in key or 'temporal_layers' in key for key in model_keys):
                return 'vivit'
            # Check for ViT3D keys
            elif any('patch_embed' in key or 'blocks' in key for key in model_keys):
                return 'vit3d'
            # Check for DiT3D keys
            elif any('dit_blocks' in key or 'x_embedder' in key for key in model_keys):
                return 'dit3d'
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not detect architecture from checkpoint: {e}")
            return None
    
    # Auto-detect and fix architecture mismatch
    detected_arch = detect_architecture_from_checkpoint(checkpoint_path)
    if detected_arch and detected_arch != Config.MODEL_ARCHITECTURE:
        logger.warning(f"Architecture mismatch detected!")
        logger.warning(f"Config has: {Config.MODEL_ARCHITECTURE}, Checkpoint has: {detected_arch}")
        logger.info(f"Automatically switching to {detected_arch} architecture for sampling")
        Config.MODEL_ARCHITECTURE = detected_arch
    
    # Auto-detect temporal layer count for ViViT models
    if detected_arch == 'vivit' or Config.MODEL_ARCHITECTURE == 'vivit':
        def detect_temporal_layers_from_checkpoint(checkpoint_path):
            """Detect required number of temporal layers from checkpoint"""
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                state_dict = checkpoint['model_state_dict']
                temporal_keys = [k for k in state_dict.keys() if 'temporal_layers.' in k or 'film_layers.' in k]
                
                max_temporal_idx = -1
                max_film_idx = -1
                for key in temporal_keys:
                    if 'temporal_layers.' in key:
                        idx = int(key.split('temporal_layers.')[1].split('.')[0])
                        max_temporal_idx = max(max_temporal_idx, idx)
                    elif 'film_layers.' in key:
                        idx = int(key.split('film_layers.')[1].split('.')[0])
                        max_film_idx = max(max_film_idx, idx)
                
                return max(max_temporal_idx, max_film_idx) + 1 if max(max_temporal_idx, max_film_idx) >= 0 else None
            except Exception as e:
                logger.warning(f"Could not detect temporal layers from checkpoint: {e}")
                return None
        
        detected_temporal_layers = detect_temporal_layers_from_checkpoint(checkpoint_path)
        if detected_temporal_layers and detected_temporal_layers != Config.VIVIT_NUM_TEMPORAL_LAYERS:
            logger.info(f"Adjusting temporal layers: {Config.VIVIT_NUM_TEMPORAL_LAYERS} → {detected_temporal_layers}")
            Config.VIVIT_NUM_TEMPORAL_LAYERS = detected_temporal_layers
    
    # Load model with correct architecture
    from diffusion import create_diffusion_model
    model = create_diffusion_model(Config)
    model.to(Config.DEVICE)
    
    # --- Load Checkpoint with Error Handling ---
    try:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            step = checkpoint.get('global_step', 'unknown')
            logger.info(f"Loaded checkpoint: {checkpoint_path} (Epoch: {epoch}, Step: {step})")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}. Using random weights.")
    except Exception as e:
        logger.error(f"❌ Critical error loading checkpoint: {e}")
        return # Exit if checkpoint fails to load
    
    # --- Generate Samples ---
    model.eval()
    with torch.no_grad():
        shape = (num_samples, *Config.INPUT_SHAPE)
        logger.info("🎬 Generating samples...")
        
        # Use text-conditioned sampling
        if hasattr(model, 'sample') and model.text_encoder is not None:
            samples = model.sample(text_prompt, batch_size=num_samples, eta=eta)
        else:
            # Fallback to unconditional sampling
            logger.warning("Text encoder not available, using unconditional sampling")
            samples = model.p_sample(shape, eta=eta)
        
        logger.info(f"✅ Generated {num_samples} samples")

        # --- Shape Validation ---
        expected_shape = (num_samples, *Config.INPUT_SHAPE)
        actual_shape = tuple(samples.shape)
        
        if actual_shape != expected_shape:
            logger.warning(f"Shape mismatch detected:")
            logger.warning(f"  Generated: {actual_shape}")
            logger.warning(f"  Expected:  {expected_shape}")
            logger.warning(f"This usually means config dimensions don't match the trained model")
            logger.warning(f"Consider running: python main.py fix-config --checkpoint {checkpoint_path}")
            
            # Update config dynamically for this session
            if len(actual_shape) == 5:  # (batch, channels, frames, height, width)
                Config.INPUT_SHAPE = actual_shape[1:]  # Remove batch dimension
                Config.NUM_FRAMES = actual_shape[2]
                Config.IMAGE_SIZE = actual_shape[3]  # Assume square images
                logger.info(f"Temporarily updated config to match model output: {Config.INPUT_SHAPE}")
        
        assert len(samples.shape) == 5, f"Expected 5D tensor (batch, channels, frames, height, width), got {samples.shape}"

    # --- Save Samples as GIFs ---
    import numpy as np
    import imageio
    from PIL import Image
    
    samples_np = samples.detach().cpu().numpy()
    frames = samples_np.shape[2]  # (batch, channels, frames, height, width)
    fps = 8
    duration_ms = int(1000 / fps)

    for i, sample in enumerate(samples_np):
        # Convert from CHW to HWC format and scale from [-1,1] to [0, 255]
        video_frames = []
        for frame_idx in range(frames):
            frame = sample[:, frame_idx]
            frame = np.transpose(frame, (1, 2, 0))
            frame = np.clip((frame + 1) * 127.5, 0, 255).astype(np.uint8)
            video_frames.append(frame)

        pil_frames = []
        palette = None
        for idx, frame in enumerate(video_frames):
            img = Image.fromarray(frame)
            if palette is None:
                img = img.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.NONE)
                palette = img.getpalette()
            else:
                img = img.convert("P", dither=Image.NONE)
                if palette is not None:
                    img.putpalette(palette)
            pil_frames.append(img)

        output_path = os.path.join(output_dir, f"sample_{i:03d}_{text_prompt.replace(' ', '_')}.gif")
        if pil_frames:
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                loop=0,
                duration=duration_ms,
                optimize=False,
                disposal=2,
            )
            file_size = os.path.getsize(output_path)
            logger.info(f"💾 Saved sample {i}: {output_path} ({file_size:,} bytes)")
        
    logger.info(f"✅ Sample generation completed. Saved to: {output_dir}")
    logger.info(f"🌐 You can view the GIFs in any web browser or image viewer")

def visualize_model():
    """Visualize the model architecture"""
    logger.info("Creating model visualization...")
    
    from diffusion import create_diffusion_model
    model = create_diffusion_model(Config)
    
    # Print model summary
    print_model_summary(model.model, Config.INPUT_SHAPE)
    
    logger.info("Model debugging completed - comprehensive TensorBoard logging will be available during training")
    logger.info(f"View training logs with: tensorboard --logdir {Config.LOG_DIR}")

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch",
        "torchvision",
        "numpy",
        "imageio",
        "matplotlib",
        "tensorboard",
        "tqdm"
    ]
    
    logger.info("Installing required packages...")
    
    import subprocess
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"Installed: {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Text2Sign Diffusion Model")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the diffusion model")
    train_parser.add_argument("--resume", action="store_true", 
                             help="Resume training from the latest checkpoint")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test all components")
    
    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Generate sample videos")
    sample_parser.add_argument("--checkpoint", type=str, default="checkpoints/text2sign_experiment_vivit_4/latest_checkpoint.pt",
                              help="Path to model checkpoint")
    sample_parser.add_argument("--num_samples", type=int, default=4,
                              help="Number of samples to generate")
    sample_parser.add_argument("--output_dir", type=str, default="samples",
                              help="Output directory for samples")
    sample_parser.add_argument("--text", type=str, default="hello",
                              help="Text prompt for generation")
    sample_parser.add_argument("--eta", type=float, default=None,
                              help="DDIM stochasticity (0=deter, 1=ancestral, default auto)")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize model architecture")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install required packages")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    
    # Checkpoints command
    checkpoints_parser = subparsers.add_parser("checkpoints", help="List available checkpoints")
    
    # Fix-config command
    fix_parser = subparsers.add_parser("fix-config", help="Fix architecture mismatch between config and checkpoint")
    fix_parser.add_argument("--checkpoint", type=str, required=True,
                           help="Path to checkpoint to match architecture with")

    args = parser.parse_args()
    
    if args.command == "train":
        train_model(resume=args.resume)
    elif args.command == "test":
        test_components()
    elif args.command == "sample":
        sample_videos(args.checkpoint, args.num_samples, args.output_dir, args.text, args.eta)
    elif args.command == "visualize":
        visualize_model()
    elif args.command == "install":
        install_requirements()
    elif args.command == "config":
        Config.print_config()
    elif args.command == "checkpoints":
        list_checkpoints()
    elif args.command == "fix-config":
        fix_config_architecture(args.checkpoint)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
