#!/usr/bin/env python3
"""
Minimal DiT3D test
"""

import torch
import sys
sys.path.append('/teamspace/studios/this_studio/text2sign')

def minimal_test():
    print("🔍 Minimal DiT3D Test")
    
    try:
        from models.architectures.dit3d import DiT3D
        print("✅ Import successful")
        
        # Very small model
        model = DiT3D(
            video_size=(4, 32, 32),
            dit_model_size="DiT-S/2",
            text_dim=128
        )
        print("✅ Model created")
        
        # Minimal inputs
        x = torch.randn(1, 3, 4, 32, 32)
        t = torch.tensor([100])
        text_emb = torch.randn(1, 128)
        
        print("✅ Inputs ready")
        
        # Forward pass
        with torch.no_grad():
            output = model(x, t, text_emb)
        
        print(f"✅ Output shape: {output.shape}")
        print("🎯 SUCCESS!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    minimal_test()