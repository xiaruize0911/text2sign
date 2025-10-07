#!/usr/bin/env python3
"""
Test script to verify that the TinyFusion temporal padding fix resolves the shape mismatch issue.
"""

import torch
import sys
sys.path.insert(0, '/teamspace/studios/this_studio/text2sign')

from models.architectures.tinyfusion import TinyFusionVideoWrapper

def test_temporal_post_processor():
    """Test that TemporalPostProcessor maintains the temporal dimension."""
    print("=" * 60)
    print("Testing TemporalPostProcessor Shape Preservation")
    print("=" * 60)
    
    # Test with different kernel sizes
    kernel_sizes = [2, 3, 4, 5]
    
    for kernel_size in kernel_sizes:
        print(f"\n📏 Testing kernel_size={kernel_size}")
        
        # Create a TinyFusion model with the specified kernel size
        model = TinyFusionVideoWrapper(
            video_size=(28, 128, 128),
            in_channels=3,
            out_channels=3,
            text_dim=768,
            variant="tinyfusion_mini",
            checkpoint_path="none",
            freeze_backbone=True,
            enable_temporal_post=True,
            temporal_kernel=kernel_size,
        )
        
        # Create test input
        batch_size = 2
        x = torch.randn(batch_size, 3, 28, 128, 128)
        time = torch.randint(0, 50, (batch_size,))
        text_emb = torch.randn(batch_size, 768)
        
        # Forward pass
        try:
            output = model(x, time, text_emb)
            
            # Check shapes
            assert output.shape == x.shape, f"Shape mismatch! Input: {x.shape}, Output: {output.shape}"
            
            print(f"   ✅ Input shape:  {tuple(x.shape)}")
            print(f"   ✅ Output shape: {tuple(output.shape)}")
            print(f"   ✅ Temporal dimension preserved: {output.shape[2]} frames")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Temporal dimension is correctly preserved.")
    print("=" * 60)
    return True

def test_loss_calculation():
    """Test that loss calculation works without shape mismatch."""
    print("\n" + "=" * 60)
    print("Testing Loss Calculation (simulating training)")
    print("=" * 60)
    
    import torch.nn.functional as F
    
    # Create a TinyFusion model with kernel_size=2 (the problematic case)
    model = TinyFusionVideoWrapper(
        video_size=(28, 128, 128),
        in_channels=3,
        out_channels=3,
        text_dim=768,
        variant="tinyfusion_mini",
        checkpoint_path="none",
        freeze_backbone=True,
        enable_temporal_post=True,
        temporal_kernel=2,  # The problematic kernel size
    )
    
    # Simulate training
    batch_size = 2
    x = torch.randn(batch_size, 3, 28, 128, 128)
    time = torch.randint(0, 50, (batch_size,))
    text_emb = torch.randn(batch_size, 768)
    
    # Generate noise (what the model should predict)
    noise = torch.randn_like(x)
    
    # Forward pass to get predicted noise
    predicted_noise = model(x, time, text_emb)
    
    print(f"\n📊 Shapes:")
    print(f"   Input (x):         {tuple(x.shape)}")
    print(f"   Noise (target):    {tuple(noise.shape)}")
    print(f"   Predicted noise:   {tuple(predicted_noise.shape)}")
    
    # Try to calculate loss (this is where the error occurred before)
    try:
        loss = F.mse_loss(predicted_noise, noise)
        print(f"\n✅ Loss calculation successful!")
        print(f"   Loss value: {loss.item():.6f}")
        print("\n" + "=" * 60)
        print("✅ Loss calculation works correctly!")
        print("=" * 60)
        return True
    except RuntimeError as e:
        print(f"\n❌ Loss calculation failed!")
        print(f"   Error: {e}")
        print("\n" + "=" * 60)
        print("❌ Test failed!")
        print("=" * 60)
        return False

if __name__ == "__main__":
    print("\n🔧 TinyFusion Temporal Padding Fix - Verification Test\n")
    
    # Run tests
    test1_passed = test_temporal_post_processor()
    test2_passed = test_loss_calculation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Temporal dimension preservation: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Loss calculation:                {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The fix is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
