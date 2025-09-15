#!/usr/bin/env python3
"""
Memory usage analysis for DiT3D models
"""

import torch
import sys
sys.path.append('/teamspace/studios/this_studio/text2sign')

def get_model_memory_usage(model, input_shapes):
    """Calculate approximate memory usage for a model"""
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate parameter memory (4 bytes per float32 parameter)
    param_memory_mb = (total_params * 4) / (1024 * 1024)
    
    # Estimate activation memory for forward pass
    x, t, text_emb = input_shapes
    
    # Create dummy inputs
    dummy_x = torch.randn(*x)
    dummy_t = torch.randint(0, 1000, t)
    dummy_text = torch.randn(*text_emb)
    
    # Memory before forward pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        memory_before = 0
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        try:
            output = model(dummy_x, dummy_t, dummy_text)
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / (1024 * 1024)
                activation_memory = memory_after - memory_before
            else:
                # Rough estimate for activation memory
                activation_memory = param_memory_mb * 2  # Conservative estimate
                
        except RuntimeError as e:
            print(f"❌ Forward pass failed: {e}")
            activation_memory = float('inf')
    
    # Total memory estimate (params + activations + gradients + optimizer states)
    # Optimizer states typically require 2x parameter memory for Adam
    total_memory_mb = param_memory_mb + activation_memory + (param_memory_mb * 2)
    
    return {
        'total_params': total_params,
        'param_memory_mb': param_memory_mb,
        'activation_memory_mb': activation_memory,
        'total_memory_mb': total_memory_mb
    }

def test_dit_model_sizes():
    """Test different DiT model sizes for memory usage"""
    print("🔍 DiT3D Memory Usage Analysis")
    print("=" * 60)
    
    try:
        from models.architectures.dit3d import DiT3D
        from config import Config
        
        # Test configurations
        configs = [
            ("DiT-S/2", (2, 3, 16, 64, 64)),   # Smaller video size
            ("DiT-S/4", (2, 3, 16, 64, 64)),   
            ("DiT-B/2", (2, 3, 16, 64, 64)),   
            ("DiT-B/4", (2, 3, 16, 64, 64)),   
        ]
        
        text_dim = 768
        input_shapes = [
            (2, 3, 16, 64, 64),  # x shape
            (2,),                # t shape  
            (2, text_dim)        # text_emb shape
        ]
        
        results = []
        
        for model_size, video_size in configs:
            print(f"\n📊 Testing {model_size} with video size {video_size}")
            
            try:
                model = DiT3D(
                    video_size=video_size[2:],  # (T, H, W)
                    dit_model_size=model_size,
                    text_dim=text_dim,
                    freeze_dit_backbone=True
                )
                
                memory_info = get_model_memory_usage(model, input_shapes)
                
                print(f"   ✅ Parameters: {memory_info['total_params']:,}")
                print(f"   📊 Parameter memory: {memory_info['param_memory_mb']:.1f} MB")
                print(f"   📊 Activation memory: {memory_info['activation_memory_mb']:.1f} MB")
                print(f"   📊 Total estimated memory: {memory_info['total_memory_mb']:.1f} MB")
                print(f"   📊 Total estimated memory: {memory_info['total_memory_mb']/1024:.1f} GB")
                
                # Check if it fits in 16GB
                fits_16gb = memory_info['total_memory_mb'] < 16 * 1024
                print(f"   {'✅' if fits_16gb else '❌'} Fits in 16GB: {fits_16gb}")
                
                results.append({
                    'model_size': model_size,
                    'video_size': video_size,
                    'memory_gb': memory_info['total_memory_mb'] / 1024,
                    'fits_16gb': fits_16gb,
                    'params': memory_info['total_params']
                })
                
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ❌ Failed to create {model_size}: {e}")
                results.append({
                    'model_size': model_size,
                    'video_size': video_size,
                    'memory_gb': float('inf'),
                    'fits_16gb': False,
                    'params': 0
                })
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 MEMORY USAGE SUMMARY")
        print("=" * 60)
        
        suitable_models = []
        for result in results:
            status = "✅ FITS" if result['fits_16gb'] else "❌ TOO BIG"
            print(f"{result['model_size']:8} | {result['memory_gb']:6.1f} GB | {status}")
            if result['fits_16gb']:
                suitable_models.append(result['model_size'])
        
        print("\n🎯 RECOMMENDATIONS:")
        if suitable_models:
            print(f"✅ Models that fit in 16GB: {', '.join(suitable_models)}")
            print(f"🏆 Recommended: {suitable_models[0]} (smallest that fits)")
        else:
            print("❌ No models fit in 16GB with current settings")
            print("💡 Try reducing video size or using even smaller model")
            
        return suitable_models
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    test_dit_model_sizes()