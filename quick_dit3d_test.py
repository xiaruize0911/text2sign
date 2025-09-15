#!/usr/bin/env python3
"""
Quick test for DiT3D with pretrained backbone
"""

import torch
from models.architectures.dit3d import DiT3D, count_parameters

def quick_dit3d_test():
    """Quick test for DiT3D"""
    print("🔍 Quick DiT3D Test")
    print("=" * 40)
    
    try:
        # Create small model
        model = DiT3D(
            video_size=(8, 64, 64),
            dit_model_size="DiT-S/2",
            text_dim=256,
            freeze_dit_backbone=True
        )
        
        print(f"✅ Model created: {count_parameters(model):,} parameters")
        
        # Test forward pass
        x = torch.randn(1, 3, 8, 64, 64)
        t = torch.randint(0, 1000, (1,))
        text_emb = torch.randn(1, 256)
        
        model.eval()
        with torch.no_grad():
            output = model(x, t, text_emb)
            print(f"✅ Forward pass: {x.shape} → {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_dit3d_test()
    print(f"🎯 Result: {'✅ PASSED' if success else '❌ FAILED'}")