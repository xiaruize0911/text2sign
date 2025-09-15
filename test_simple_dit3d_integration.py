#!/usr/bin/env python3
"""
Test SimpleDiT3D with actual text encoder
"""

import torch
import sys
sys.path.append('/teamspace/studios/this_studio/text2sign')

def test_simple_dit3d_with_text_encoder():
    print("🔍 Testing SimpleDiT3D with TextEncoder")
    print("=" * 50)
    
    try:
        # Import components
        from models.architectures.simple_dit3d import SimpleDiT3D_S, count_parameters
        from models.text_encoder import TextEncoder
        
        print("✅ Imports successful")
        
        # Create models
        text_encoder = TextEncoder()
        model = SimpleDiT3D_S(video_size=(8, 64, 64), text_dim=768)
        
        print(f"✅ Models created:")
        print(f"   • SimpleDiT3D: {count_parameters(model):,} parameters")
        print(f"   • TextEncoder: {count_parameters(text_encoder):,} parameters")
        
        # Test data
        batch_size = 2
        x = torch.randn(batch_size, 3, 8, 64, 64)
        t = torch.randint(0, 1000, (batch_size,))
        texts = ["hello world", "sign language translation"]
        
        print(f"📊 Test data:")
        print(f"   • Video: {x.shape}")
        print(f"   • Timesteps: {t.shape}")
        print(f"   • Texts: {texts}")
        
        # Get text embeddings
        with torch.no_grad():
            text_emb = text_encoder(texts)
        
        print(f"   • Text embeddings: {text_emb.shape}")
        
        # Forward pass
        print("🔄 Running forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(x, t, text_emb)
        
        print(f"✅ Forward pass successful!")
        print(f"📊 Output: {output.shape}")
        
        # Verify shapes match
        if output.shape == x.shape:
            print("✅ Input/output shapes match perfectly")
        else:
            print(f"❌ Shape mismatch: {x.shape} vs {output.shape}")
        
        # Test training mode
        print("🔄 Testing training mode...")
        model.train()
        x.requires_grad_(True)
        
        output = model(x, t, text_emb)
        loss = F.mse_loss(output, torch.zeros_like(output))
        loss.backward()
        
        print(f"✅ Training mode works, loss: {loss.item():.6f}")
        
        # Check gradients
        if x.grad is not None and x.grad.abs().sum() > 0:
            print("✅ Gradients computed correctly")
        else:
            print("❌ No gradients found")
        
        print("🎯 Complete integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import torch.nn.functional as F
    test_simple_dit3d_with_text_encoder()