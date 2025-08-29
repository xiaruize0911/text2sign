#!/usr/bin/env python3
"""
Test the cosine noise scheduler implementation
"""

import sys
import os
sys.path.append('/teamspace/studios/this_studio/text2sign')

import torch
import matplotlib.pyplot as plt
import numpy as np

def test_cosine_scheduler():
    """Test the cosine noise scheduler"""
    print("=== Testing Cosine Noise Scheduler ===")
    
    try:
        from schedulers.noise_schedulers import create_noise_scheduler, compare_schedulers
        from config import Config
        
        config = Config()
        timesteps = config.TIMESTEPS
        
        print(f"Testing with {timesteps} timesteps...")
        
        # Test cosine scheduler creation
        print("\n1. Testing scheduler creation...")
        cosine_scheduler = create_noise_scheduler('cosine', timesteps, s=0.008, max_beta=0.999)
        linear_scheduler = create_noise_scheduler('linear', timesteps, beta_start=0.0001, beta_end=0.02)
        
        print("✅ Schedulers created successfully")
        
        # Get schedules
        cosine_betas = cosine_scheduler.get_schedule()
        linear_betas = linear_scheduler.get_schedule()
        
        print(f"\n2. Comparing schedules...")
        print(f"Cosine beta range: [{cosine_betas.min():.6f}, {cosine_betas.max():.6f}]")
        print(f"Linear beta range: [{linear_betas.min():.6f}, {linear_betas.max():.6f}]")
        
        # Compute alpha schedules
        _, cosine_alpha_cumprod, _ = cosine_scheduler.compute_alpha_schedule(cosine_betas)
        _, linear_alpha_cumprod, _ = linear_scheduler.compute_alpha_schedule(linear_betas)
        
        print(f"Cosine alpha_cumprod range: [{cosine_alpha_cumprod.min():.6f}, {cosine_alpha_cumprod.max():.6f}]")
        print(f"Linear alpha_cumprod range: [{linear_alpha_cumprod.min():.6f}, {linear_alpha_cumprod.max():.6f}]")
        
        # Validate monotonicity
        cosine_decreasing = torch.all(cosine_alpha_cumprod[1:] <= cosine_alpha_cumprod[:-1])
        linear_decreasing = torch.all(linear_alpha_cumprod[1:] <= linear_alpha_cumprod[:-1])
        
        print(f"Cosine alpha_cumprod decreasing: {cosine_decreasing}")
        print(f"Linear alpha_cumprod decreasing: {linear_decreasing}")
        
        # Test with diffusion model
        print(f"\n3. Testing with diffusion model...")
        from diffusion.text2sign import create_diffusion_model
        
        # Temporarily change config to use cosine scheduler
        original_scheduler = config.NOISE_SCHEDULER
        config.NOISE_SCHEDULER = "cosine"
        
        model = create_diffusion_model(config)
        print("✅ Diffusion model with cosine scheduler created successfully")
        
        # Test forward pass
        batch_size = 2
        test_videos = torch.randn(batch_size, *config.INPUT_SHAPE, device=config.DEVICE)
        
        model.eval()
        with torch.no_grad():
            loss, pred_noise, actual_noise = model(test_videos)
            
        print(f"✅ Forward pass successful")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Model uses {model.noise_scheduler_type} scheduler")
        
        # Restore original scheduler
        config.NOISE_SCHEDULER = original_scheduler
        
        # Create comparison plot
        print(f"\n4. Creating comparison plot...")
        create_comparison_plot(timesteps)
        
        print(f"\n🎉 Cosine scheduler test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Cosine scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comparison_plot(timesteps: int):
    """Create a comparison plot of different schedulers"""
    try:
        from schedulers.noise_schedulers import create_noise_scheduler
        
        # Create schedulers
        schedulers = {
            'Linear': create_noise_scheduler('linear', timesteps),
            'Cosine': create_noise_scheduler('cosine', timesteps),
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot beta schedules
        ax = axes[0, 0]
        for name, scheduler in schedulers.items():
            betas = scheduler.get_schedule()
            ax.plot(betas.numpy(), label=f'{name}', linewidth=2)
        ax.set_title('Beta Schedules', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Beta')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot alpha_cumprod schedules
        ax = axes[0, 1]
        for name, scheduler in schedulers.items():
            betas = scheduler.get_schedule()
            _, alphas_cumprod, _ = scheduler.compute_alpha_schedule(betas)
            ax.plot(alphas_cumprod.numpy(), label=f'{name}', linewidth=2)
        ax.set_title('Alpha Cumprod Schedules', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Alpha Cumprod')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot noise levels
        ax = axes[1, 0]
        for name, scheduler in schedulers.items():
            betas = scheduler.get_schedule()
            _, alphas_cumprod, _ = scheduler.compute_alpha_schedule(betas)
            noise_levels = torch.sqrt(1 - alphas_cumprod)
            ax.plot(noise_levels.numpy(), label=f'{name}', linewidth=2)
        ax.set_title('Noise Levels', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Sqrt(1 - Alpha Cumprod)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot signal-to-noise ratio
        ax = axes[1, 1]
        for name, scheduler in schedulers.items():
            betas = scheduler.get_schedule()
            _, alphas_cumprod, _ = scheduler.compute_alpha_schedule(betas)
            snr = alphas_cumprod / (1 - alphas_cumprod)
            ax.plot(torch.log10(snr).numpy(), label=f'{name} (log10)', linewidth=2)
        ax.set_title('Signal-to-Noise Ratio (log scale)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Log10(SNR)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/teamspace/studios/this_studio/text2sign/cosine_scheduler_comparison.png', 
                   dpi=150, bbox_inches='tight')
        print("✅ Comparison plot saved as 'cosine_scheduler_comparison.png'")
        
        # Print key differences
        print("\n📊 Key Scheduler Differences:")
        for name, scheduler in schedulers.items():
            betas = scheduler.get_schedule()
            _, alphas_cumprod, _ = scheduler.compute_alpha_schedule(betas)
            
            # Early vs late timestep behavior
            early_alpha = alphas_cumprod[timesteps//10].item()  # 10% through
            late_alpha = alphas_cumprod[9*timesteps//10].item()  # 90% through
            
            print(f"{name} scheduler:")
            print(f"  Early timestep alpha_cumprod (t={timesteps//10}): {early_alpha:.4f}")
            print(f"  Late timestep alpha_cumprod (t={9*timesteps//10}): {late_alpha:.4f}")
            print(f"  Beta standard deviation: {betas.std().item():.6f}")
        
    except ImportError:
        print("ℹ️  Matplotlib not available, skipping comparison plot")
    except Exception as e:
        print(f"⚠️  Error creating comparison plot: {e}")

if __name__ == "__main__":
    success = test_cosine_scheduler()
    if success:
        print("\n🎉 Cosine scheduler implementation is working correctly!")
        print("\nBenefits of cosine scheduler:")
        print("• More gradual noise addition in early timesteps")
        print("• Better preservation of image structure") 
        print("• Improved training stability")
        print("• Better sample quality in many cases")
    else:
        print("\n💥 Cosine scheduler implementation needs fixes")
