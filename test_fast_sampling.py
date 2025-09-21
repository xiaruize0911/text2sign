#!/usr/bin/env python3
"""
Test script to compare DDPM vs DDIM sampling speeds
"""

import torch
import time
from config import Config
from diffusion.text2sign import DiffusionModel

def test_sampling_speed():
    """Test sampling speed with different timestep configurations"""
    
    print("🚀 Testing Fast Sampling Performance")
    print("="*50)
    
    # Initialize model using the main.py approach
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    from config import Config
    
    try:
        # Try to load a trained model if available
        from main import load_diffusion_model
        diffusion_model = load_diffusion_model(Config, device)
        print("✅ Loaded trained model")
    except Exception as e:
        print(f"⚠️  Could not load trained model ({e}), creating new model for testing...")
        
        # Create a minimal model for testing
        from models.architectures.vivit import ViViTDiffusion
        from models.text_encoder import TextEncoder
        
        # Create backbone model
        backbone_model = ViViTDiffusion(
            video_size=Config.VIVIT_VIDEO_SIZE,
            patch_size=Config.VIVIT_PATCH_SIZE,
            embed_dim=Config.VIVIT_EMBED_DIM,
            num_heads=Config.VIVIT_NUM_HEADS,
            num_layers=Config.VIVIT_NUM_LAYERS,
            in_channels=Config.VIVIT_IN_CHANNELS,
            num_classes=None,  # Not needed for diffusion
            freeze_backbone=Config.VIVIT_FREEZE_BACKBONE
        ).to(device)
        
        # Create text encoder
        text_encoder = TextEncoder(Config.TEXT_ENCODER_TYPE).to(device)
        
        # Create diffusion model
        diffusion_model = DiffusionModel(
            model=backbone_model,
            timesteps=Config.TIMESTEPS,
            noise_scheduler=Config.NOISE_SCHEDULER,
            device=device,
            text_encoder=text_encoder
        )
        print("✅ Created new model for testing")
    
    # Test dimensions (use actual config dimensions but smaller for speed)
    batch_size = 1
    channels = Config.VIVIT_IN_CHANNELS
    frames = 8  # Reduced from Config.VIVIT_VIDEO_SIZE[0] for speed
    height = 32  # Reduced from Config.VIVIT_VIDEO_SIZE[1] for speed
    width = 32   # Reduced from Config.VIVIT_VIDEO_SIZE[2] for speed
    shape = (batch_size, channels, frames, height, width)
    
    test_text = "a person waving hello"
    
    print(f"Test shape: {shape}")
    print(f"Model device: {diffusion_model.device}")
    print()
    
    # Test configurations
    test_configs = [
        {"steps": 1000, "deterministic": False, "name": "DDPM Full (1000 steps)"},
        {"steps": 100, "deterministic": False, "name": "DDPM Fast (100 steps)"},
        {"steps": 50, "deterministic": True, "name": "DDIM Fast (50 steps)"},
        {"steps": 25, "deterministic": True, "name": "DDIM Ultra-Fast (25 steps)"},
        {"steps": 10, "deterministic": True, "name": "DDIM Lightning (10 steps)"},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"Testing {config['name']}...")
        
        try:
            # Measure sampling time
            start_time = time.time()
            
            with torch.no_grad():
                sample = diffusion_model.p_sample(
                    shape=shape,
                    device=device,
                    text=test_text,
                    deterministic=config['deterministic'],
                    num_inference_steps=config['steps']
                )
                
                # Ensure computation is complete
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Calculate statistics
            speed_ratio = test_configs[0]["steps"] / config["steps"] if config["steps"] > 0 else 1
            theoretical_speedup = f"{speed_ratio:.1f}x" if speed_ratio > 1 else "1.0x"
            
            result = {
                "name": config["name"],
                "steps": config["steps"],
                "time": elapsed,
                "theoretical_speedup": theoretical_speedup,
                "sample_range": (sample.min().item(), sample.max().item()),
                "sample_mean": sample.mean().item()
            }
            results.append(result)
            
            print(f"  ✅ Completed in {elapsed:.2f}s")
            print(f"  📊 Sample range: [{sample.min().item():.3f}, {sample.max().item():.3f}]")
            print(f"  ⚡ Theoretical speedup: {theoretical_speedup}")
            print()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            print()
            continue
    
    # Print summary
    print("📈 SAMPLING SPEED COMPARISON")
    print("="*60)
    print(f"{'Method':<25} {'Steps':<6} {'Time':<8} {'Speedup':<10} {'Quality'}")
    print("-"*60)
    
    baseline_time = results[0]["time"] if results else 0
    
    for result in results:
        actual_speedup = f"{baseline_time / result['time']:.1f}x" if result['time'] > 0 else "N/A"
        quality_indicator = "✓" if abs(result['sample_mean']) < 1.0 else "?"
        
        print(f"{result['name']:<25} {result['steps']:<6} {result['time']:<8.2f} {actual_speedup:<10} {quality_indicator}")
    
    print()
    print("🎯 RECOMMENDATIONS:")
    
    if len(results) >= 3:
        fastest_method = min(results[1:], key=lambda x: x['time'])
        print(f"  • For production: {fastest_method['name']}")
        print(f"    - {fastest_method['time']:.1f}s sampling time")
        print(f"    - {baseline_time / fastest_method['time']:.1f}x faster than full DDPM")
        
    print(f"  • DDIM with 50 steps typically provides good quality/speed balance")
    print(f"  • Consider 25 steps for interactive applications")
    print(f"  • 10 steps for real-time scenarios (may sacrifice quality)")

if __name__ == "__main__":
    test_sampling_speed()