#!/usr/bin/env python3
"""
Main script for the Text2Sign diffusion model
This script provides a command-line interface for training, testing, and sampling
"""

import argparse
import sys
import os
import torch
import logging

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
    
    # Setup training
    trainer = setup_training(Config)
    
    # Resume from checkpoint if requested
    if resume:
        # Try to load the latest checkpoint
        latest_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "latest_checkpoint.pt")
        if os.path.exists(latest_checkpoint):
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
            trainer.load_checkpoint("latest_checkpoint.pt")
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
        
        # Try to load checkpoint info
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
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

def sample_videos(checkpoint_path: str, num_samples: int = 4, output_dir: str = "samples"):
    """
    Generate sample videos using a trained model
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        num_samples (int): Number of videos to generate
        output_dir (str): Directory to save generated videos
    """
    logger.info(f"Generating {num_samples} sample videos...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    from diffusion import create_diffusion_model
    model = create_diffusion_model(Config)
    model.to(Config.DEVICE)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        step = checkpoint.get('global_step', 'unknown')
        logger.info(f"Loaded checkpoint: {checkpoint_path} (Epoch: {epoch}, Step: {step})")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}. Using random weights.")
    
    # Generate samples
    model.eval()
    with torch.no_grad():
        shape = (num_samples, *Config.INPUT_SHAPE)
        logger.info("🎬 Generating samples...")
        
        # Use AMP if available and enabled
        if getattr(Config, 'USE_AMP', False) and torch.cuda.is_available():
            from torch.amp import autocast
            with autocast('cuda'):
                samples = model.p_sample(shape)
        else:
            samples = model.p_sample(shape)
        
        # Clamp to [0, 1] range
        samples = torch.clamp(samples, 0, 1)
        logger.info(f"✅ Generated {num_samples} samples")
    
    # Save samples as GIFs
    import numpy as np
    import imageio
    
    samples_np = samples.detach().cpu().numpy()
    
    for i, sample in enumerate(samples_np):
        # Convert from CHW to HWC format and scale to [0, 255]
        video_frames = []
        for frame_idx in range(sample.shape[1]):  # frames dimension
            frame = sample[:, frame_idx]  # (channels, height, width)
            frame = np.transpose(frame, (1, 2, 0))  # Convert to (height, width, channels)
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255]
            video_frames.append(frame)
        
        # Save as GIF
        output_path = os.path.join(output_dir, f"sample_{i:03d}.gif")
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
    
    # Try to create a visual representation
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter("model_visualization")
        
        dummy_input = torch.randn(1, *Config.INPUT_SHAPE)
        dummy_time = torch.randint(0, Config.TIMESTEPS, (1,))
        
        writer.add_graph(model.model, (dummy_input, dummy_time))
        writer.close()
        
        logger.info("Model visualization saved to 'model_visualization' directory")
        logger.info("View with: tensorboard --logdir model_visualization")
    except Exception as e:
        logger.error(f"Failed to create model visualization: {e}")

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
    sample_parser.add_argument("--checkpoint", type=str, default="checkpoints/latest_checkpoint.pt",
                              help="Path to model checkpoint")
    sample_parser.add_argument("--num_samples", type=int, default=4,
                              help="Number of samples to generate")
    sample_parser.add_argument("--output_dir", type=str, default="samples",
                              help="Output directory for samples")
    
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
        sample_videos(args.checkpoint, args.num_samples, args.output_dir)
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
