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

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from methods import setup_training
from dataset import test_dataloader
from models import test_unet3d
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
    
    # Test UNet3D model
    logger.info("Testing UNet3D model...")
    test_unet3d()
    
    # Test diffusion model
    logger.info("Testing diffusion model...")
    test_diffusion()
    
    logger.info("All component tests completed successfully!")

def sample_videos(checkpoint_path: str, num_samples: int = 4, output_dir: str = "samples", text_prompt: str = "hello"):
    """
    Generate sample videos using a trained model
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        num_samples (int): Number of videos to generate
        output_dir (str): Directory to save generated videos
        text_prompt (str): Text prompt for generation
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
    
    # Load model
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
            samples = model.sample(text_prompt, batch_size=num_samples)
        else:
            # Fallback to unconditional sampling
            logger.warning("Text encoder not available, using unconditional sampling")
            samples = model.p_sample(shape)
        
        # Clamp to [-1, 1] range (matching training data normalization)
        samples = torch.clamp(samples, -1, 1)
        logger.info(f"✅ Generated {num_samples} samples")

        # --- Shape Assertion ---
        expected_shape = (num_samples, *Config.INPUT_SHAPE)
        assert tuple(samples.shape) == expected_shape, f"Output shape {samples.shape} does not match expected {expected_shape}"

    # --- Save Samples as GIFs ---
    import numpy as np
    import imageio
    
    samples_np = samples.detach().cpu().numpy()
    frames = samples_np.shape[2]  # (batch, channels, frames, height, width)
    K = 10  # Frame interval for GIF

    for i, sample in enumerate(samples_np):
        # Convert from CHW to HWC format and scale from [-1,1] to [0, 255]
        video_frames = []
        for frame_idx in range(frames):
            # Only include frames at specified intervals for efficiency
            if frame_idx % K == 0 or frame_idx < 5 or frame_idx == frames - 1:
                frame = sample[:, frame_idx]  # (channels, height, width)
                frame = np.transpose(frame, (1, 2, 0))  # Convert to (height, width, channels)
                # Convert from [-1, 1] to [0, 255]
                frame = np.clip((frame + 1) * 127.5, 0, 255).astype(np.uint8)
                video_frames.append(frame)
        
        # Save as GIF
        output_path = os.path.join(output_dir, f"sample_{i:03d}_{text_prompt.replace(' ', '_')}.gif")
        imageio.mimsave(output_path, video_frames, fps=8, loop=0)
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
    sample_parser.add_argument("--checkpoint", type=str, default="checkpoints/text2sign_experiment_unet1/latest_checkpoint.pt",
                              help="Path to model checkpoint")
    sample_parser.add_argument("--num_samples", type=int, default=4,
                              help="Number of samples to generate")
    sample_parser.add_argument("--output_dir", type=str, default="samples",
                              help="Output directory for samples")
    sample_parser.add_argument("--text", type=str, default="hello",
                              help="Text prompt for generation")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize model architecture")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install required packages")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    
    # Checkpoints command
    checkpoints_parser = subparsers.add_parser("checkpoints", help="List available checkpoints")

    args = parser.parse_args()
    
    if args.command == "train":
        train_model(resume=args.resume)
    elif args.command == "test":
        test_components()
    elif args.command == "sample":
        sample_videos(args.checkpoint, args.num_samples, args.output_dir, args.text)
    elif args.command == "visualize":
        visualize_model()
    elif args.command == "install":
        install_requirements()
    elif args.command == "config":
        Config.print_config()
    elif args.command == "checkpoints":
        list_checkpoints()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
