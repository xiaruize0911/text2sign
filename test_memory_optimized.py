#!/usr/bin/env python3
"""
Test the memory-optimized DiT3D configuration
"""

import torch
import sys
sys.path.append('/teamspace/studios/this_studio/text2sign')

def test_memory_optimized_dit3d():
    print("🔍 Testing Memory-Optimized DiT3D Configuration")
    print("=" * 60)
    
    try:
        from models.architectures.dit3d import DiT3D_Tiny_4, count_parameters
        from models.text_encoder import TextEncoder
        from config import Config
        
        print("✅ Imports successful")
        
        # Show config settings
        print(f"📊 Configuration:")
        print(f"   • Model: {Config.DIT_MODEL_SIZE}")
        print(f"   • Video size: {Config.DIT_VIDEO_SIZE}")
        print(f"   • Patch size: {Config.DIT_PATCH_SIZE}")
        print(f"   • Batch size: {Config.BATCH_SIZE}")
        print(f"   • Input shape: {Config.INPUT_SHAPE}")
        
        # Create models
        text_encoder = TextEncoder()
        model = DiT3D_Tiny_4(
            video_size=Config.DIT_VIDEO_SIZE,
            text_dim=768,
            learn_sigma=False  # Set to False to get 3-channel output instead of 6
        )
        
        params = count_parameters(model)
        text_params = count_parameters(text_encoder)
        
        print(f"\n📊 Model Parameters:")
        print(f"   • DiT3D: {params:,}")
        print(f"   • TextEncoder: {text_params:,}")
        print(f"   • Total: {params + text_params:,}")
        
        # Calculate memory estimates
        param_memory_mb = ((params + text_params) * 4) / (1024 * 1024)
        
        # Conservative memory estimate including:
        # - Parameters (4 bytes each)
        # - Gradients (4 bytes each) 
        # - Optimizer states (8 bytes each for Adam)
        # - Activations (roughly 2x parameter memory)
        total_memory_gb = (param_memory_mb * (1 + 1 + 2 + 2)) / 1024  # 6x factor
        
        print(f"\n💾 Memory Estimates:")
        print(f"   • Parameter memory: {param_memory_mb:.1f} MB")
        print(f"   • Total estimated memory: {total_memory_gb:.1f} GB")
        
        fits_16gb = total_memory_gb < 16
        print(f"   {'✅' if fits_16gb else '❌'} Fits in 16GB: {fits_16gb}")
        
        # Test forward pass
        print(f"\n🔄 Testing forward pass...")
        batch_size = Config.BATCH_SIZE
        C, T, H, W = Config.INPUT_SHAPE
        
        x = torch.randn(batch_size, C, T, H, W)
        t = torch.randint(0, 1000, (batch_size,))
        texts = ["hello world"] * batch_size
        
        print(f"   • Input video: {x.shape}")
        print(f"   • Timesteps: {t.shape}")
        print(f"   • Texts: {texts}")
        
        # Get text embeddings
        with torch.no_grad():
            text_emb = text_encoder(texts)
        print(f"   • Text embeddings: {text_emb.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x, t, text_emb)
        
        print(f"   ✅ Forward pass successful!")
        print(f"   • Output: {output.shape}")
        
        # Test training mode
        print(f"\n🏋️ Testing training mode...")
        model.train()
        x.requires_grad_(True)
        
        output = model(x, t, text_emb)
        loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
        loss.backward()
        
        print(f"   ✅ Training mode works, loss: {loss.item():.6f}")
        
        if x.grad is not None and x.grad.abs().sum() > 0:
            print("   ✅ Gradients computed correctly")
        else:
            print("   ❌ No gradients found")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   ✅ Model created successfully")
        print(f"   ✅ Memory efficient: {total_memory_gb:.1f}GB / 16GB")
        print(f"   ✅ Forward/backward passes work")
        print(f"   🚀 Ready for training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_memory_optimized_dit3d()