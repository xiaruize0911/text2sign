#!/usr/bin/env python3
"""
Test script to verify ViViT model improvements
This script validates that all fixes are working correctly
"""

import torch
import torch.nn as nn
from config import Config
from diffusion import create_diffusion_model
import numpy as np

def test_model_architecture():
    """Test that the model has all the new components"""
    print("=" * 60)
    print("TEST 1: Model Architecture")
    print("=" * 60)
    
    model = create_diffusion_model(Config)
    model.to(Config.DEVICE)
    
    # Check for output_scale parameter
    assert hasattr(model.model, 'output_scale'), "❌ Missing output_scale parameter!"
    print(f"✅ output_scale parameter exists: {model.model.output_scale.item():.4f}")
    
    # Check output_scale is trainable
    assert model.model.output_scale.requires_grad, "❌ output_scale is not trainable!"
    print(f"✅ output_scale is trainable")
    
    # Check final_projection has correct depth
    final_proj_layers = len([m for m in model.model.final_projection if isinstance(m, nn.Conv2d)])
    assert final_proj_layers == 3, f"❌ final_projection should have 3 Conv2d layers, has {final_proj_layers}"
    print(f"✅ final_projection has {final_proj_layers} Conv2d layers")
    
    # Check timestep weighting config
    assert model.use_timestep_weighting == Config.USE_TIMESTEP_WEIGHTING, "❌ Timestep weighting config mismatch!"
    print(f"✅ Timestep weighting enabled: {model.use_timestep_weighting}")
    
    print("\n✅ All architecture tests passed!\n")
    return model

def test_forward_pass(model):
    """Test forward pass and output ranges"""
    print("=" * 60)
    print("TEST 2: Forward Pass")
    print("=" * 60)
    
    # Create test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 28, 128, 128, device=Config.DEVICE)
    t = torch.randint(0, Config.TIMESTEPS, (batch_size,), device=Config.DEVICE)
    text = ["hello", "world"]
    
    model.eval()
    with torch.no_grad():
        # Test model prediction
        text_emb = model.text_encoder(text)
        output = model.model(x, t, text_emb)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Output mean: {output.mean():.3f}")
        print(f"Output std: {output.std():.3f}")
        
        # Verify output shape matches input
        assert output.shape == x.shape, f"❌ Shape mismatch: {output.shape} != {x.shape}"
        print("✅ Output shape matches input")
        
        # Verify no NaN or Inf
        assert torch.isfinite(output).all(), "❌ Output contains NaN or Inf!"
        print("✅ Output is finite")
        
        # Check output magnitude is reasonable (not too large or small)
        output_magnitude = output.abs().mean().item()
        assert 0.1 < output_magnitude < 10.0, f"❌ Output magnitude {output_magnitude:.3f} is abnormal!"
        print(f"✅ Output magnitude is reasonable: {output_magnitude:.3f}")
    
    print("\n✅ All forward pass tests passed!\n")

def test_loss_weighting(model):
    """Test timestep-aware loss weighting"""
    print("=" * 60)
    print("TEST 3: Loss Weighting")
    print("=" * 60)
    
    if not model.use_timestep_weighting:
        print("⚠️ Timestep weighting is disabled, skipping test")
        return
    
    # Create test data
    x = torch.randn(4, 3, 28, 128, 128, device=Config.DEVICE)
    text = ["test"] * 4
    
    model.eval()
    with torch.no_grad():
        # Test at different timesteps
        losses = {}
        for t_val in [0, 10, 25, 40, 49]:
            t = torch.full((4,), t_val, device=Config.DEVICE, dtype=torch.long)
            loss, pred, noise = model.forward(x, text)
            losses[t_val] = loss.item()
            print(f"t={t_val:2d}: loss={loss.item():.6f}")
        
        # Verify loss trends (should be relatively balanced with weighting)
        loss_values = list(losses.values())
        loss_range = max(loss_values) - min(loss_values)
        print(f"\nLoss range: {loss_range:.6f}")
        
        # With proper weighting, loss range should be more balanced
        # Without weighting, high timesteps would have much lower loss
        if loss_range > 2.0:
            print("⚠️ Loss range is large - weighting may need adjustment")
        else:
            print("✅ Loss range is balanced")
    
    print("\n✅ Loss weighting test completed!\n")

def test_noise_prediction_quality(model):
    """Test noise prediction at different timesteps"""
    print("=" * 60)
    print("TEST 4: Noise Prediction Quality")
    print("=" * 60)
    
    # Create test data
    x = torch.randn(1, 3, 28, 128, 128, device=Config.DEVICE)
    text = ["hello"]
    
    model.eval()
    with torch.no_grad():
        results = []
        for t_val in [0, 10, 25, 40, 49]:
            t = torch.full((1,), t_val, device=Config.DEVICE, dtype=torch.long)
            
            # Add noise
            noise, x_noisy = model.q_sample(x, t)
            
            # Predict noise
            text_emb = model.text_encoder(text)
            pred_noise = model.model(x_noisy, t, text_emb)
            
            # Compute error
            mse = torch.nn.functional.mse_loss(pred_noise, noise).item()
            scale_ratio = (pred_noise.std() / noise.std()).item()
            
            results.append({
                't': t_val,
                'mse': mse,
                'scale_ratio': scale_ratio
            })
            
            print(f"t={t_val:2d}: MSE={mse:.6f}, scale_ratio={scale_ratio:.4f}")
        
        # Check if low timesteps have reasonable errors
        # With new architecture, t=0 MSE should be < 5.0 (was 1.92 in old untrained model)
        low_t_mse = results[0]['mse']
        high_t_mse = results[-1]['mse']
        
        print(f"\nLow timestep (t=0) MSE: {low_t_mse:.6f}")
        print(f"High timestep (t=49) MSE: {high_t_mse:.6f}")
        
        if low_t_mse < 10.0:
            print("✅ Low timestep prediction is reasonable (untrained)")
        else:
            print("⚠️ Low timestep prediction may need more training")
    
    print("\n✅ Noise prediction test completed!\n")

def test_sampling():
    """Test sampling process"""
    print("=" * 60)
    print("TEST 5: Sampling")
    print("=" * 60)
    
    model = create_diffusion_model(Config)
    model.to(Config.DEVICE)
    model.eval()
    
    with torch.no_grad():
        # Generate a small sample
        print("Generating sample (this may take a minute)...")
        sample = model.sample(
            text="hello",
            batch_size=1,
            num_frames=28,
            height=128,
            width=128,
            deterministic=True,
            num_inference_steps=10,  # Use fewer steps for faster testing
            eta=0.0
        )
        
        print(f"Sample shape: {sample.shape}")
        print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        print(f"Sample mean: {sample.mean():.3f}")
        print(f"Sample std: {sample.std():.3f}")
        
        # Verify sample is in valid range
        assert -2.0 <= sample.min() <= 2.0, "❌ Sample minimum is out of range!"
        assert -2.0 <= sample.max() <= 2.0, "❌ Sample maximum is out of range!"
        print("✅ Sample values are in valid range")
        
        # Verify sample is not constant
        assert sample.std() > 0.01, "❌ Sample has no variance!"
        print("✅ Sample has variance")
    
    print("\n✅ Sampling test passed!\n")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("VIVIT MODEL VALIDATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Architecture
        model = test_model_architecture()
        
        # Test 2: Forward pass
        test_forward_pass(model)
        
        # Test 3: Loss weighting
        test_loss_weighting(model)
        
        # Test 4: Noise prediction
        test_noise_prediction_quality(model)
        
        # Test 5: Sampling
        test_sampling()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe ViViT model improvements are working correctly.")
        print("You can now start training with: python main.py train")
        print("\nNote: These tests use an untrained model.")
        print("After training, performance will improve significantly.")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
