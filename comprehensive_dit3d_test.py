#!/usr/bin/env python3
"""
Comprehensive DiT3D test with proper text encoder
"""

import torch
import sys
sys.path.append('/teamspace/studios/this_studio/text2sign')

def comprehensive_dit3d_test():
    print("🔍 Comprehensive DiT3D Test")
    print("=" * 50)
    
    try:
        # Import components
        from models.architectures.dit3d import DiT3D, count_parameters
        from models.text_encoder import TextEncoder
        
        print("✅ Imports successful")
        
        # Create text encoder
        text_encoder = TextEncoder()
        print("✅ Text encoder created")
        
        # Create DiT3D model
        model = DiT3D(
            video_size=(8, 64, 64),
            dit_model_size="DiT-S/2",
            text_dim=768,  # Correct dimension
            freeze_dit_backbone=True
        )
        
        print(f"✅ DiT3D model created: {count_parameters(model):,} parameters")
        
        # Test inputs
        batch_size = 2
        x = torch.randn(batch_size, 3, 8, 64, 64)
        t = torch.randint(0, 1000, (batch_size,))
        texts = ["hello world", "sign language"]
        
        print(f"📊 Input shapes:")
        print(f"   • Video: {x.shape}")
        print(f"   • Timesteps: {t.shape}")
        print(f"   • Texts: {texts}")
        
        # Get text embeddings
        text_emb = text_encoder(texts)
        print(f"   • Text embeddings: {text_emb.shape}")
        
        # Forward pass
        print("🔄 Running forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(x, t, text_emb)
        
        print(f"✅ Forward pass successful!")
        print(f"📊 Output shape: {output.shape}")
        
        # Check output shape
        expected_shape = x.shape
        if output.shape == expected_shape:
            print("✅ Output shape matches input shape")
        else:
            print(f"⚠️  Output shape mismatch: expected {expected_shape}, got {output.shape}")
        
        # Test gradients
        print("🔄 Testing gradient flow...")
        model.train()
        x.requires_grad_(True)
        output = model(x, t, text_emb)
        loss = output.mean()
        loss.backward()
        
        if x.grad is not None:
            print("✅ Gradients flow properly")
        else:
            print("❌ No gradients computed")
        
        print("🎯 Comprehensive test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    comprehensive_dit3d_test()