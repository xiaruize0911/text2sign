#!/usr/bin/env python3
"""
Quick demo of the training system
This script runs a short training session to demonstrate functionality
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import create_dataloader
from diffusion import create_diffusion_model
import logging

# Import AMP for mixed precision
from torch.amp import GradScaler, autocast

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_training():
    """Run a short demo training session"""
    logger.info("Starting training demo...")
    
    # Print device info
    logger.info(f"Using device: {Config.DEVICE}")
    
    # Create small dataloader for demo
    dataloader = create_dataloader(
        data_root=Config.DATA_ROOT,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for demo
        shuffle=True
    )
    
    # Create model
    model = create_diffusion_model(Config)
    model.to(Config.DEVICE)
    
    # Count parameters
    from models import count_parameters
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Initialize AMP scaler if using AMP
    use_amp = getattr(Config, 'USE_AMP', False) and torch.cuda.is_available()
    scaler = GradScaler('cuda') if use_amp else None
    
    # Run a few training steps
    model.train()
    total_loss = 0.0
    num_steps = 3  # Just run 3 steps for demo
    
    logger.info(f"Running {num_steps} training steps...")
    
    for step, (videos, texts) in enumerate(dataloader):
        if step >= num_steps:
            break
            
        # Move data to device
        videos = videos.to(Config.DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with AMP
        if use_amp and scaler is not None:
            with autocast('cuda'):
                loss, pred_noise, actual_noise = model(videos)
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular forward pass
            loss, pred_noise, actual_noise = model(videos)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
        
        total_loss += loss.item()
        
        logger.info(f"Step {step + 1}/{num_steps}: Loss = {loss.item():.4f}")
        logger.info(f"  Predicted noise range: [{pred_noise.min():.3f}, {pred_noise.max():.3f}]")
        logger.info(f"  Actual noise range: [{actual_noise.min():.3f}, {actual_noise.max():.3f}]")
    
    avg_loss = total_loss / num_steps
    logger.info(f"Demo completed! Average loss: {avg_loss:.4f}")
    
    # Test sampling (generate one small sample)
    logger.info("Testing sample generation...")
    model.eval()
    with torch.no_grad():
        # Generate a single frame sequence for speed
        sample_shape = (1, Config.UNET_CHANNELS, 4, 64, 64)  # Smaller for demo
        logger.info(f"Generating sample with shape: {sample_shape}")
        
        # Just run a few denoising steps
        x = torch.randn(sample_shape, device=Config.DEVICE)
        
        # Run just 10 denoising steps for demo
        for i in range(min(10, Config.TIMESTEPS)):
            t = torch.full((1,), Config.TIMESTEPS - 1 - i, device=Config.DEVICE, dtype=torch.long)
            
            # Use AMP for inference if available
            if use_amp:
                with autocast('cuda'):
                    x = model.p_sample_step(x, t)
            else:
                x = model.p_sample_step(x, t)
                
            if i % 3 == 0:
                logger.info(f"  Denoising step {i+1}/10: output range [{x.min():.3f}, {x.max():.3f}]")
        
        logger.info(f"Sample generation completed! Final output range: [{x.min():.3f}, {x.max():.3f}]")
    
    logger.info("Training demo completed successfully!")
    logger.info("To start full training, run: python main.py train")

if __name__ == "__main__":
    demo_training()
