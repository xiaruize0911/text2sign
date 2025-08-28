#!/usr/bin/env python3
"""
Debug script to test model forward pass and identify NaN issues
"""

import torch
from config import Config
from diffusion import create_diffusion_model

def test_model():
    config = Config()
    print(f"Using device: {config.DEVICE}")
    print(f"Model architecture: {config.MODEL_ARCHITECTURE}")

    # Create model
    model = create_diffusion_model(config)
    model.to(config.DEVICE)
    print("Model created successfully")

    # Create dummy input
    x = torch.randn(1, 3, 28, 128, 128, device=config.DEVICE)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Input has NaN: {torch.isnan(x).any()}")
    print(f"Input has Inf: {torch.isinf(x).any()}")

    # Test forward pass
    try:
        loss, pred_noise, noise = model(x)
        print(f"Loss: {loss.item():.6f}")
        print(f"Loss is NaN: {torch.isnan(loss)}")
        print(f"Loss is Inf: {torch.isinf(loss)}")
        print(f"Pred noise range: [{pred_noise.min():.3f}, {pred_noise.max():.3f}]")
        print(f"Noise range: [{noise.min():.3f}, {noise.max():.3f}]")
        print("✅ Model forward pass successful!")
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
