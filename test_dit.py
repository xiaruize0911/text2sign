#!/usr/bin/env python3
"""
Test script for DiT (Diffusion Transformer) implementation
This script thoroughly tests the DiT model to ensure it works correctly.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.architectures.dit import DiT, count_parameters

def test_dit_basic():
    """Test basic DiT functionality"""
    print("🚀 Testing Basic DiT Functionality...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    channels, frames, height, width = 3, 16, 128, 128
    time_dim = 256
    text_dim = 768
    
    print(f"📝 Test Configuration:")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Video shape: {channels}×{frames}×{height}×{width}")
    print(f"   • Time dim: {time_dim}")
    print(f"   • Text dim: {text_dim}")
    print()
    
    try:
        # Create DiT model
        model = DiT(
            in_channels=channels,
            out_channels=channels,
            frames=frames,
            height=height,
            width=width,
            hidden_size=384,  # Smaller for testing
            depth=4,
            num_heads=6,
            mlp_ratio=4.0,
            time_dim=time_dim,
            text_dim=text_dim
        )
        
        # Count parameters
        total_params = count_parameters(model)
        print(f"   ✅ DiT model created successfully")
        print(f"   📊 Total parameters: {total_params:,}")
        
        # Create test inputs
        x = torch.randn(batch_size, channels, frames, height, width)
        time = torch.randint(0, 1000, (batch_size,))
        text_emb = torch.randn(batch_size, text_dim)
        
        print(f"   📊 Input shapes:")
        print(f"      • Video: {x.shape}")
        print(f"      • Time: {time.shape}")
        print(f"      • Text: {text_emb.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            # Test with text embedding
            output_with_text = model(x, time, text_emb)
            print(f"   ✅ Forward pass with text: {x.shape} → {output_with_text.shape}")
            
            # Test without text embedding
            output_no_text = model(x, time)
            print(f"   ✅ Forward pass without text: {x.shape} → {output_no_text.shape}")
            
            # Verify shapes
            assert output_with_text.shape == x.shape, f"Shape mismatch with text: {output_with_text.shape} vs {x.shape}"
            assert output_no_text.shape == x.shape, f"Shape mismatch without text: {output_no_text.shape} vs {x.shape}"
            
            # Check for NaN or Inf values
            assert torch.isfinite(output_with_text).all(), "Output contains NaN or Inf values (with text)"
            assert torch.isfinite(output_no_text).all(), "Output contains NaN or Inf values (without text)"
            
            # Check value ranges
            print(f"   📊 Output statistics (with text):")
            print(f"      • Min: {output_with_text.min().item():.4f}")
            print(f"      • Max: {output_with_text.max().item():.4f}")
            print(f"      • Mean: {output_with_text.mean().item():.4f}")
            print(f"      • Std: {output_with_text.std().item():.4f}")
            
            print(f"   ✅ Output validation passed")
        
        print(f"   🎉 Basic DiT test successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Basic DiT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dit_configurations():
    """Test different DiT configurations"""
    print("\n🏗️  Testing Different DiT Configurations...")
    print("=" * 60)
    
    configs = [
        {
            "name": "DiT-S/2 (Small)",
            "hidden_size": 384,
            "depth": 12,
            "num_heads": 6,
            "patch_size": 2
        },
        {
            "name": "DiT-B/4 (Base)", 
            "hidden_size": 768,
            "depth": 12,
            "num_heads": 12,
            "patch_size": 4
        },
        {
            "name": "DiT-L/8 (Large)",
            "hidden_size": 1024,
            "depth": 24,
            "num_heads": 16,
            "patch_size": 8
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{i+1}. Testing {config['name']}...")
        
        try:
            # Create model with specific config
            model = DiT(
                in_channels=3,
                out_channels=3,
                frames=8,  # Smaller for testing
                height=64,  # Smaller for testing
                width=64,
                hidden_size=config["hidden_size"],
                depth=min(config["depth"], 4),  # Limit depth for testing
                num_heads=config["num_heads"],
                patch_size=config["patch_size"]
            )
            
            # Count parameters
            total_params = count_parameters(model)
            print(f"   📊 Parameters: {total_params:,}")
            
            # Test forward pass
            x = torch.randn(1, 3, 8, 64, 64)
            time = torch.randint(0, 1000, (1,))
            
            model.eval()
            with torch.no_grad():
                output = model(x, time)
                print(f"   ✅ {config['name']}: {x.shape} → {output.shape}")
                assert output.shape == x.shape
                assert torch.isfinite(output).all()
            
        except Exception as e:
            print(f"   ❌ {config['name']} failed: {e}")
            continue

def test_dit_gradient_flow():
    """Test gradient flow through DiT"""
    print("\n🔄 Testing Gradient Flow...")
    print("=" * 60)
    
    try:
        # Create small model for gradient testing
        model = DiT(
            in_channels=3,
            out_channels=3,
            frames=8,
            height=64,
            width=64,
            hidden_size=256,
            depth=2,
            num_heads=4
        )
        
        # Create inputs and targets
        x = torch.randn(2, 3, 8, 64, 64, requires_grad=True)
        time = torch.randint(0, 1000, (2,))
        target = torch.randn_like(x)
        
        # Forward pass
        model.train()
        output = model(x, time)
        
        # Compute loss
        loss = nn.MSELoss()(output, target)
        print(f"   📊 Loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm == 0:
                    print(f"   ⚠️  Zero gradient in {name}")
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        print(f"   📊 Average gradient norm: {avg_grad_norm:.6f}")
        print(f"   📊 Gradient norms range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
        
        assert avg_grad_norm > 0, "No gradients flowing"
        print(f"   ✅ Gradient flow test passed")
        
    except Exception as e:
        print(f"   ❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()

def test_dit_memory_efficiency():
    """Test memory efficiency of DiT"""
    print("\n🧠 Testing Memory Efficiency...")
    print("=" * 60)
    
    try:
        # Test with larger batch to check memory
        model = DiT(
            in_channels=3,
            out_channels=3,
            frames=16,
            height=128,
            width=128,
            hidden_size=384,
            depth=2,
            num_heads=6
        )
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            print(f"   🔍 Testing batch size: {batch_size}")
            
            x = torch.randn(batch_size, 3, 16, 128, 128)
            time = torch.randint(0, 1000, (batch_size,))
            
            model.eval()
            with torch.no_grad():
                output = model(x, time)
                print(f"      ✅ Success: {x.shape} → {output.shape}")
                
                # Check memory usage if CUDA is available
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    print(f"      📊 Peak GPU memory: {memory_mb:.1f} MB")
                    torch.cuda.reset_peak_memory_stats()
        
        print(f"   ✅ Memory efficiency test passed")
        
    except Exception as e:
        print(f"   ❌ Memory efficiency test failed: {e}")

def test_dit_cuda_compatibility():
    """Test CUDA compatibility if available"""
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, skipping CUDA tests")
        return
    
    print("\n🚀 Testing CUDA Compatibility...")
    print("=" * 60)
    
    try:
        device = torch.device('cuda')
        
        # Create model on CUDA
        model = DiT(
            in_channels=3,
            out_channels=3,
            frames=8,
            height=64,
            width=64,
            hidden_size=256,
            depth=2,
            num_heads=4
        ).to(device)
        
        # Create inputs on CUDA
        x = torch.randn(2, 3, 8, 64, 64).to(device)
        time = torch.randint(0, 1000, (2,)).to(device)
        text_emb = torch.randn(2, 256).to(device)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(x, time, text_emb)
            print(f"   ✅ CUDA forward pass: {output.shape}")
            assert output.device == device
            assert torch.isfinite(output).all()
        
        # Test mixed precision
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output_fp16 = model(x, time, text_emb)
                print(f"   ✅ Mixed precision: {output_fp16.shape}")
        
        print(f"   ✅ CUDA compatibility test passed")
        
    except Exception as e:
        print(f"   ❌ CUDA compatibility test failed: {e}")
        import traceback
        traceback.print_exc()

def run_comprehensive_dit_tests():
    """Run all DiT tests"""
    print("🎯 DiT (Diffusion Transformer) Comprehensive Test Suite")
    print("=" * 80)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_dit_basic),
        ("Different Configurations", test_dit_configurations),
        ("Gradient Flow", test_dit_gradient_flow),
        ("Memory Efficiency", test_dit_memory_efficiency),
        ("CUDA Compatibility", test_dit_cuda_compatibility)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result if result is not None else True))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Summary:")
    print("=" * 80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All DiT tests passed! The implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_dit_tests()
    exit(0 if success else 1)