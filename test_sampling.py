
import torch
import os
import sys
from config import Config
from diffusion import create_diffusion_model

import logging

def test_sampling():
    print("Testing sampling procedure...")
    
    # Suppress verbose logs
    logging.getLogger('models').setLevel(logging.WARNING)
    
    # Setup minimal config for testing
    Config.DEVICE = torch.device("cpu") # Use CPU for testing to avoid CUDA OOM if any
    Config.INPUT_SHAPE = (4, 8, 64, 64) # Standard shape
    Config.NUM_FRAMES = 8
    Config.IMAGE_SIZE = 64
    Config.TINYFUSION_VIDEO_SIZE = (8, 64, 64)
    Config.TIMESTEPS = 10
    Config.INFERENCE_TIMESTEPS = 10
    
    # Create model
    print("Creating model...")
    try:
        model = create_diffusion_model(Config)
        model.eval()
    except Exception as e:
        print(f"Failed to create model: {e}")
        return

    # Test sampling
    print("Running sample()...")
    try:
        samples = model.sample(
            text="test prompt",
            batch_size=1,
            num_frames=8,
            height=64,
            width=64,
            num_inference_steps=5
        )
        print(f"Sample shape: {samples.shape}")
        print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
        
        if torch.isnan(samples).any():
            print("❌ NaN detected in samples!")
        else:
            print("✅ Sampling successful (no NaNs)")
            
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sampling()
