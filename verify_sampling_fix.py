#!/usr/bin/env python3
"""
Verify that the sampling fix produces consistent, non-random results
"""

import torch
import numpy as np
from config import Config
from diffusion.text2sign import Text2SignDiffusion
from models.model_factory import create_model

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = Config()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and diffusion
    model = create_model(config)
    diffusion = Text2SignDiffusion(model, config, device)
    
    # Load latest checkpoint
    checkpoint_path = "checkpoints/latest_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    diffusion.model.load_state_dict(checkpoint['model_state_dict'])
    diffusion.model.eval()
    
    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    print(f"Device: {device}")
    
    # Test sampling with same seed multiple times
    print("\n=== Testing sampling consistency ===")
    
    text_prompt = "hello world"
    batch_size = 1
    
    samples = []
    for i in range(3):
        print(f"Sample {i+1}:")
        torch.manual_seed(42)  # Same seed each time
        
        # Sample from the model
        sample = diffusion.sample(
            text=text_prompt,
            batch_size=batch_size,
            num_frames=config.num_frames,
            height=config.image_height,
            width=config.image_width
        )
        
        # Basic statistics
        sample_mean = sample.mean().item()
        sample_std = sample.std().item()
        sample_min = sample.min().item()
        sample_max = sample.max().item()
        
        print(f"  Mean: {sample_mean:.6f}")
        print(f"  Std:  {sample_std:.6f}")
        print(f"  Min:  {sample_min:.6f}")
        print(f"  Max:  {sample_max:.6f}")
        
        samples.append(sample.cpu())
    
    # Check consistency between samples (should be identical with same seed)
    diff_1_2 = torch.abs(samples[0] - samples[1]).mean().item()
    diff_1_3 = torch.abs(samples[0] - samples[2]).mean().item()
    diff_2_3 = torch.abs(samples[1] - samples[2]).mean().item()
    
    print(f"\n=== Consistency check ===")
    print(f"Difference between sample 1 and 2: {diff_1_2:.8f}")
    print(f"Difference between sample 1 and 3: {diff_1_3:.8f}")
    print(f"Difference between sample 2 and 3: {diff_2_3:.8f}")
    
    if diff_1_2 < 1e-6 and diff_1_3 < 1e-6 and diff_2_3 < 1e-6:
        print("✅ PASS: Sampling is deterministic with same seed")
    else:
        print("❌ FAIL: Sampling is not deterministic")
    
    # Test with different seeds
    print(f"\n=== Testing sampling diversity ===")
    torch.manual_seed(123)
    sample_a = diffusion.sample(text=text_prompt, batch_size=1, 
                               num_frames=config.num_frames,
                               height=config.image_height, 
                               width=config.image_width)
    
    torch.manual_seed(456)
    sample_b = diffusion.sample(text=text_prompt, batch_size=1,
                               num_frames=config.num_frames,
                               height=config.image_height,
                               width=config.image_width)
    
    diff_different_seeds = torch.abs(sample_a - sample_b).mean().item()
    print(f"Difference between different seeds: {diff_different_seeds:.6f}")
    
    if diff_different_seeds > 0.01:
        print("✅ PASS: Different seeds produce different outputs")
    else:
        print("❌ FAIL: Different seeds produce similar outputs (possible issue)")
    
    print(f"\n=== Sampling verification complete ===")

if __name__ == "__main__":
    main()
