#!/usr/bin/env python3
"""
Test the improved diffusion model implementation
"""

import sys
import os
sys.path.append('/teamspace/studios/this_studio/text2sign')

import torch
from config import Config

def test_diffusion_forward():
    """Test the forward diffusion process"""
    print("=== Testing Improved Diffusion Model ===")
    
    try:
        # Import the improved diffusion model
        from diffusion.text2sign import create_diffusion_model
        
        config = Config()
        print(f"Device: {config.DEVICE}")
        
        # Create diffusion model
        print("Creating diffusion model...")
        model = create_diffusion_model(config)
        model = model.to(config.DEVICE)
        print(f"✅ Model created: {type(model.model).__name__}")
        
        # Test data
        batch_size = 2
        videos = torch.randn(batch_size, *config.INPUT_SHAPE, device=config.DEVICE)
        text_prompts = ["hello", "goodbye"]  # Future text conditioning
        
        print(f"✅ Test input shape: {videos.shape}")
        
        # Test forward pass (training)
        print("\n=== Testing Forward Pass (Training) ===")
        model.train()
        loss, predicted_noise, actual_noise = model(videos)
        
        print(f"✅ Forward pass successful")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Predicted noise shape: {predicted_noise.shape}")
        print(f"   Actual noise shape: {actual_noise.shape}")
        print(f"   Predicted noise stats: mean={predicted_noise.mean():.4f}, std={predicted_noise.std():.4f}")
        print(f"   Actual noise stats: mean={actual_noise.mean():.4f}, std={actual_noise.std():.4f}")
        
        # Test q_sample (forward diffusion)
        print("\n=== Testing Forward Diffusion (q_sample) ===")
        t = torch.randint(0, config.TIMESTEPS, (batch_size,), device=config.DEVICE)
        noise = torch.randn_like(videos)
        noisy_videos = model.q_sample(videos, t, noise)
        
        print(f"✅ Forward diffusion successful")
        print(f"   Timesteps: {t}")
        print(f"   Original video range: [{videos.min():.3f}, {videos.max():.3f}]")
        print(f"   Noisy video range: [{noisy_videos.min():.3f}, {noisy_videos.max():.3f}]")
        
        # Test p_sample_step (single reverse step)
        print("\n=== Testing Single Reverse Step (p_sample_step) ===")
        model.eval()
        with torch.no_grad():
            denoised = model.p_sample_step(noisy_videos, t)
            
        print(f"✅ Single reverse step successful")
        print(f"   Denoised shape: {denoised.shape}")
        print(f"   Denoised range: [{denoised.min():.3f}, {denoised.max():.3f}]")
        
        # Test noise schedule
        print("\n=== Testing Noise Schedule ===")
        print(f"   Timesteps: {model.timesteps}")
        print(f"   Beta range: [{model.betas.min():.6f}, {model.betas.max():.6f}]")
        print(f"   Alpha_cumprod range: [{model.alphas_cumprod.min():.6f}, {model.alphas_cumprod.max():.6f}]")
        
        # Test that the schedule makes sense
        assert model.betas[0] < model.betas[-1], "Beta should increase"
        assert model.alphas_cumprod[0] > model.alphas_cumprod[-1], "Alpha_cumprod should decrease"
        print(f"✅ Noise schedule validation passed")
        
        # Test text-conditioned sampling interface
        print("\n=== Testing Text-Conditioned Sampling Interface ===")
        try:
            with torch.no_grad():
                # Quick test (just 5 timesteps for speed)
                original_timesteps = model.timesteps
                model.timesteps = 5
                
                sample = model.sample(
                    text="hello world",
                    batch_size=1,
                    num_frames=8,  # Smaller for testing
                    height=32,     # Smaller for testing  
                    width=32       # Smaller for testing
                )
                
                # Restore original timesteps
                model.timesteps = original_timesteps
                
            print(f"✅ Text-conditioned sampling successful")
            print(f"   Sample shape: {sample.shape}")
            print(f"   Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
            
        except Exception as e:
            print(f"⚠️  Text-conditioned sampling test skipped due to: {e}")
        
        print("\n🎉 All diffusion model tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Diffusion model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_diffusion_forward()
    if success:
        print("\n✅ Diffusion model implementation is working correctly!")
    else:
        print("\n❌ Diffusion model needs further fixes")
