#!/usr/bin/env python3

import torch
import torch.nn as nn
from models.architectures.vit3d import ViT3D, count_parameters

def simple_test():
    print("🚀 Starting ViT3D simple test...")
    
    try:
        # Small test parameters
        batch_size = 1
        channels, frames, height, width = 3, 8, 64, 64  # Smaller for faster testing
        time_dim = 64
        
        print(f"📝 Test parameters: batch={batch_size}, frames={frames}, size={height}x{width}")
        
        # Create model
        print("🏗️  Creating ViT3D model...")
        model = ViT3D(
            in_channels=channels, 
            out_channels=channels, 
            frames=frames, 
            height=height, 
            width=width, 
            time_dim=time_dim,
            freeze_backbone=True  # Freeze to avoid downloading weights
        )
        
        print(f"✅ Model created with {count_parameters(model):,} parameters")
        
        # Create test inputs
        x = torch.randn(batch_size, channels, frames, height, width)
        time = torch.randint(0, 100, (batch_size,))
        
        print(f"📊 Input shape: {x.shape}")
        print(f"⏰ Time shape: {time.shape}")
        
        # Test forward pass
        print("🔄 Running forward pass...")
        with torch.no_grad():
            output = model(x, time)
        
        print(f"🎯 Output shape: {output.shape}")
        
        # Verify shapes match
        if output.shape == x.shape:
            print("✅ SUCCESS: Input and output shapes match!")
            return True
        else:
            print(f"❌ ERROR: Shape mismatch! Expected {x.shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Test failed!")
