#!/usr/bin/env python3
"""Debug script to check if TinyFusion checkpoint loading is working properly"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn.functional as F
from config import Config

def test_checkpoint_loading():
    """Test if the checkpoint loading works with the correct architecture"""
    print("=" * 60)
    print("DEBUGGING TINYFUSION CHECKPOINT LOADING")
    print("=" * 60)
    
    # Import here to avoid import issues
    from models.architectures.tinyfusion import create_tinyfusion_model
    
    print(f"Config variant: {Config.TINYFUSION_VARIANT}")
    print(f"Checkpoint path: {Config.TINYFUSION_CHECKPOINT}")
    print()
    
    # Create model config
    config = Config.get_model_config()
    print(f"Model config: {config}")
    print()
    
    # Create TinyFusion backbone
    print("Creating TinyFusion backbone...")
    backbone = create_tinyfusion_model(**config)
    
    print("\nBackbone created successfully!")
    print(f"Model device: {next(backbone.parameters()).device}")
    
    # Test a simple forward pass with small input
    print("\nTesting forward pass...")
    device = Config.DEVICE
    
    # Use smaller input to avoid memory issues
    test_input = torch.randn(1, 3, 32, 32).to(device)  # Small 2D input for TinyFusion
    test_time = torch.randint(0, 50, (1,)).to(device)
    test_labels = torch.zeros(1, dtype=torch.long).to(device)
    
    try:
        backbone.to(device)
        with torch.no_grad():
            output = backbone(test_input, test_time, test_labels)
        
        print(f"✅ Forward pass successful!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
        
        # Check if output has reasonable values (not all zeros/constants)
        output_var = output.var().item()
        if output_var < 1e-6:
            print("⚠️  Output has very low variance - model might not be properly loaded")
        elif output_var > 100:
            print("⚠️  Output has very high variance - might indicate random initialization")
        else:
            print(f"✅ Output variance looks reasonable: {output_var:.6f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_diffusion_model():
    """Test the full diffusion model creation"""
    print("\n" + "=" * 60)
    print("TESTING FULL DIFFUSION MODEL")
    print("=" * 60)
    
    try:
        from diffusion import create_diffusion_model
        
        print("Creating full diffusion model...")
        model = create_diffusion_model(Config)
        model.to(Config.DEVICE)
        
        print("✅ Diffusion model created successfully!")
        
        # Test a simple loss calculation
        print("Testing loss calculation...")
        device = Config.DEVICE
        
        # Small input for testing
        x = torch.randn(1, 3, 4, 64, 64).to(device)  # Smaller video
        text = ["hello"]
        
        with torch.no_grad():
            loss, pred_noise, actual_noise = model(x, text)
        
        print(f"✅ Loss calculation successful!")
        print(f"Loss: {loss.item():.6f}")
        print(f"Predicted noise variance: {pred_noise.var().item():.6f}")
        print(f"Actual noise variance: {actual_noise.var().item():.6f}")
        
        # Check if loss is reasonable (not exactly 1.0)
        if abs(loss.item() - 1.0) < 1e-6:
            print("⚠️  Loss is exactly 1.0 - possible issue")
        else:
            print("✅ Loss value looks reasonable")
            
        return True
        
    except Exception as e:
        print(f"❌ Diffusion model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_checkpoint_loading()
    success2 = test_full_diffusion_model()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Checkpoint loading test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Full diffusion model test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! The checkpoint loading issue should be fixed.")
        print("You can now resume training and the loss should improve.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")