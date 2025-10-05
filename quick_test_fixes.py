"""Quick test to verify ViViT model fixes"""
import torch
import numpy as np
from diffusion import create_diffusion_model
from config import Config

def quick_test():
    print("="*60)
    print("QUICK VIVIT FIX VERIFICATION")
    print("="*60)
    
    # Initialize
    config = Config()
    config.model_type = "vivit"
    config.experiment_name = "test_fixes"
    config.num_frames = 16
    config.frame_height = 64
    config.frame_width = 64
    config.timesteps = 50  # Use 50 timesteps for faster testing
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Using device: {device}")
    
    # Create model
    print("\n" + "="*60)
    print("Loading ViViT model...")
    print("="*60)
    model = create_diffusion_model(config)
    model.to(device)
    model.eval()
    
    # Check output_scale parameter
    print("\n" + "="*60)
    print("TEST 1: Check output_scale parameter")
    print("="*60)
    if hasattr(model.model, 'output_scale'):
        scale = model.model.output_scale.item()
        print(f"✅ output_scale exists: {scale:.4f}")
        print(f"   Expected: ~1.0-2.0 for proper scaling")
        if 0.5 <= scale <= 3.0:
            print("   ✅ PASS: Scale is in reasonable range")
        else:
            print("   ⚠️  WARNING: Scale might need adjustment")
    else:
        print("❌ FAIL: output_scale parameter not found")
    
    # Test predictions at different timesteps
    print("\n" + "="*60)
    print("TEST 2: Prediction quality at different timesteps")
    print("="*60)
    
    batch_size = 2
    C, T, H, W = 3, config.num_frames, config.frame_height, config.frame_width
    
    # Create test data (batch, channels, frames, height, width)
    x_0 = torch.randn(batch_size, C, T, H, W).to(device)
    text_prompt = ["hello", "world"]
    
    # Encode text
    if model.text_encoder is not None:
        text_emb = model.text_encoder(text_prompt).to(device)
    else:
        text_emb = None
    
    # Test at high, medium, and low timesteps (using 50 timesteps)
    test_timesteps = [49, 25, 0]
    results = {}
    
    with torch.no_grad():
        for t_val in test_timesteps:
            print(f"\n  Testing at t={t_val}...")
            t = torch.tensor([t_val] * batch_size).to(device)
            
            # Add noise
            alpha_t = model.alphas_cumprod[t_val]
            noise = torch.randn_like(x_0)
            x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
            
            # Predict noise
            pred_noise = model.model(x_t, t, text_emb)
            
            # Calculate error
            mse = torch.mean((pred_noise - noise) ** 2).item()
            output_std = pred_noise.std().item()
            noise_std = noise.std().item()
            
            results[t_val] = {
                'mse': mse,
                'output_std': output_std,
                'noise_std': noise_std
            }
            
            print(f"    MSE: {mse:.6f}")
            print(f"    Output std: {output_std:.4f}")
            print(f"    Noise std: {noise_std:.4f}")
    
    # Analyze results
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    mse_high = results[49]['mse']
    mse_low = results[0]['mse']
    mse_ratio = mse_low / mse_high if mse_high > 0 else float('inf')
    
    print(f"\nMSE at t=49 (high noise): {mse_high:.6f}")
    print(f"MSE at t=0 (low noise):   {mse_low:.6f}")
    print(f"Ratio (low/high):         {mse_ratio:.2f}x")
    
    # Check if the issue is fixed
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if mse_ratio < 10:
        print("✅ EXCELLENT: Model performs well at all timesteps")
        print("   The low-timestep issue appears to be FIXED!")
    elif mse_ratio < 50:
        print("⚠️  IMPROVED: Model is better but still has some issues")
        print("   Consider further training or adjustments")
    else:
        print("❌ ISSUE REMAINS: Model still struggles at low timesteps")
        print("   Additional fixes needed")
    
    # Check output scaling
    avg_output_std = np.mean([results[t]['output_std'] for t in test_timesteps])
    avg_noise_std = np.mean([results[t]['noise_std'] for t in test_timesteps])
    
    print(f"\nAverage output std: {avg_output_std:.4f}")
    print(f"Average noise std:  {avg_noise_std:.4f}")
    print(f"Ratio: {avg_output_std/avg_noise_std:.4f}")
    
    if 0.8 <= avg_output_std/avg_noise_std <= 1.2:
        print("✅ Output scaling is good")
    else:
        print("⚠️  Output scaling may need adjustment")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    quick_test()
