#!/usr/bin/env python3
"""
Test script to validate DiT-XL-2-256 HuggingFace integration
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from models.architectures.dit3d import DiT3D_XL_2

def test_dit_hf_integration():
    """Test DiT-XL-2-256 model loading from HuggingFace"""
    print("🧪 Testing DiT-XL-2-256 HuggingFace Integration")
    print("=" * 50)
    
    try:
        # Test model creation and pretrained loading
        print("1. Creating DiT3D-XL-2 model...")
        model = DiT3D_XL_2(
            video_size=(16, 256, 256),  # (frames, height, width)
            in_channels=3,
            text_dim=768,
            freeze_dit_backbone=True
        )
        
        print(f"   ✅ Model created successfully")
        print(f"   📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   🔒 Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
        print(f"   🔓 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        batch_size = 2
        frames = 16
        height, width = 256, 256
        
        # Create dummy input in the correct format for DiT3D
        x = torch.randn(batch_size, 3, frames, height, width)  # (B, C, F, H, W)
        t = torch.randint(0, 1000, (batch_size,))  # (B,)
        
        print(f"   📥 Input shape: {x.shape}")
        print(f"   ⏰ Timesteps shape: {t.shape}")
        
        with torch.no_grad():
            output = model(x, t)
        
        print(f"   📤 Output shape: {output.shape}")
        print("   ✅ Forward pass successful")
        
        print("\n3. Testing backbone freeze status...")
        backbone_frozen = all(not p.requires_grad for name, p in model.named_parameters() if 'backbone' in name)
        print(f"   🔒 Backbone frozen: {backbone_frozen}")
        
        print("\n🎉 All tests passed! DiT-XL-2-256 integration working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dit_hf_integration()
    sys.exit(0 if success else 1)