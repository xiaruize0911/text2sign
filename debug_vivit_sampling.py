#!/usr/bin/env python3
"""
Debug script to analyze ViViT model training and sampling issues
"""

import torch
import numpy as np
from config import Config
from diffusion import create_diffusion_model
from dataset import create_dataloader
import matplotlib.pyplot as plt
import os

def analyze_model_output():
    """Analyze what the model is actually outputting"""
    print("=" * 60)
    print("ANALYZING VIVIT MODEL OUTPUT")
    print("=" * 60)
    
    # Load model
    model = create_diffusion_model(Config)
    model.to(Config.DEVICE)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/text2sign_vivit6/latest_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print("⚠️ No checkpoint found, using random weights")
    
    model.eval()
    
    # Get a sample from the dataset
    train_loader = create_dataloader(
        Config.DATA_ROOT,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS,
        shuffle=True,
        num_frames=Config.NUM_FRAMES
    )
    video_batch, text_batch = next(iter(train_loader))
    video_batch = video_batch.to(Config.DEVICE)
    
    print(f"\n📊 Input video statistics:")
    print(f"   Shape: {video_batch.shape}")
    print(f"   Range: [{video_batch.min():.3f}, {video_batch.max():.3f}]")
    print(f"   Mean: {video_batch.mean():.3f}")
    print(f"   Std: {video_batch.std():.3f}")
    
    # Test at different timesteps
    timesteps_to_test = [0, 10, 25, 49]
    
    with torch.no_grad():
        for t_val in timesteps_to_test:
            print(f"\n{'='*60}")
            print(f"Testing at timestep t={t_val}")
            print(f"{'='*60}")
            
            # Add noise at this timestep
            t = torch.full((video_batch.shape[0],), t_val, device=Config.DEVICE, dtype=torch.long)
            actual_noise, noisy_video = model.q_sample(video_batch, t)
            
            print(f"\n📊 Noisy video statistics:")
            print(f"   Range: [{noisy_video.min():.3f}, {noisy_video.max():.3f}]")
            print(f"   Mean: {noisy_video.mean():.3f}")
            print(f"   Std: {noisy_video.std():.3f}")
            
            print(f"\n📊 Actual noise statistics:")
            print(f"   Range: [{actual_noise.min():.3f}, {actual_noise.max():.3f}]")
            print(f"   Mean: {actual_noise.mean():.3f}")
            print(f"   Std: {actual_noise.std():.3f}")
            
            # Get model prediction
            text_emb = model.text_encoder(text_batch)
            predicted_noise = model.model(noisy_video, t, text_emb)
            
            print(f"\n📊 Predicted noise statistics:")
            print(f"   Range: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")
            print(f"   Mean: {predicted_noise.mean():.3f}")
            print(f"   Std: {predicted_noise.std():.3f}")
            
            # Compute error
            mse = torch.nn.functional.mse_loss(predicted_noise, actual_noise)
            mae = torch.nn.functional.l1_loss(predicted_noise, actual_noise)
            
            print(f"\n📉 Prediction error:")
            print(f"   MSE: {mse.item():.6f}")
            print(f"   MAE: {mae.item():.6f}")
            
            # Check correlation
            pred_flat = predicted_noise.flatten()
            actual_flat = actual_noise.flatten()
            correlation = torch.corrcoef(torch.stack([pred_flat, actual_flat]))[0, 1]
            print(f"   Correlation: {correlation.item():.4f}")
            
            # Check if model is just outputting zeros or constants
            pred_variance = predicted_noise.var()
            print(f"   Predicted noise variance: {pred_variance.item():.6f}")
            
            if pred_variance < 0.01:
                print(f"   ⚠️ WARNING: Model output has very low variance! Possibly outputting near-constant values")
            
            # Check output scale
            output_scale = predicted_noise.std() / actual_noise.std()
            print(f"   Output scale ratio (pred/actual std): {output_scale.item():.4f}")
            
            if output_scale < 0.5 or output_scale > 2.0:
                print(f"   ⚠️ WARNING: Output scale is off! Model is predicting with wrong magnitude")

def analyze_sampling_process():
    """Analyze the sampling process step by step"""
    print("\n" + "=" * 60)
    print("ANALYZING SAMPLING PROCESS")
    print("=" * 60)
    
    # Load model
    model = create_diffusion_model(Config)
    model.to(Config.DEVICE)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/text2sign_vivit6/latest_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Start with random noise
    shape = (1, 3, 28, 128, 128)
    x = torch.randn(shape, device=Config.DEVICE)
    
    print(f"\n📊 Initial noise (t=T):")
    print(f"   Range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"   Mean: {x.mean():.3f}")
    print(f"   Std: {x.std():.3f}")
    
    # Sample a few denoising steps
    text = "hello"
    timesteps_to_check = [49, 40, 30, 20, 10, 0]
    
    with torch.no_grad():
        for i, t_val in enumerate(timesteps_to_check):
            t = torch.full((1,), t_val, device=Config.DEVICE, dtype=torch.long)
            prev_t_val = timesteps_to_check[i+1] if i+1 < len(timesteps_to_check) else -1
            prev_t = torch.full((1,), prev_t_val, device=Config.DEVICE, dtype=torch.long)
            
            # Single denoising step
            x = model.p_sample_step(x, t, prev_t, text=text, deterministic=True, eta=0.0)
            
            print(f"\n📊 After denoising step t={t_val} -> {prev_t_val}:")
            print(f"   Range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"   Mean: {x.mean():.3f}")
            print(f"   Std: {x.std():.3f}")
            
            # Check if values are exploding or collapsing
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"   ❌ ERROR: NaN or Inf values detected!")
                break
            
            if x.std() < 0.01:
                print(f"   ⚠️ WARNING: Output collapsed to near-constant values!")
            
            if x.abs().max() > 10.0:
                print(f"   ⚠️ WARNING: Values are exploding!")
    
    print(f"\n📊 Final output:")
    print(f"   Range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"   Mean: {x.mean():.3f}")
    print(f"   Std: {x.std():.3f}")
    
    # Expected range for images is [-1, 1]
    if x.min() < -2.0 or x.max() > 2.0:
        print(f"   ⚠️ WARNING: Output is out of expected range [-1, 1]!")

def check_model_gradients():
    """Check if model has learned anything by looking at parameter distributions"""
    print("\n" + "=" * 60)
    print("CHECKING MODEL PARAMETERS")
    print("=" * 60)
    
    # Load model
    model = create_diffusion_model(Config)
    model.to(Config.DEVICE)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/text2sign_vivit6/latest_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        step = checkpoint.get('global_step', 'unknown')
        print(f"✅ Loaded checkpoint from epoch {epoch}, step {step}")
    
    # Check key components
    components_to_check = [
        ('temporal_layers', model.model.temporal_layers),
        ('film_layers', model.model.film_layers),
        ('final_projection', model.model.final_projection),
        ('temporal_conv', model.model.temporal_conv),
    ]
    
    for name, module in components_to_check:
        trainable_params = [p for p in module.parameters() if p.requires_grad]
        if trainable_params:
            print(f"\n📊 {name}:")
            param_stats = []
            for p in trainable_params:
                param_stats.append({
                    'mean': p.data.mean().item(),
                    'std': p.data.std().item(),
                    'min': p.data.min().item(),
                    'max': p.data.max().item(),
                })
            
            avg_mean = np.mean([s['mean'] for s in param_stats])
            avg_std = np.mean([s['std'] for s in param_stats])
            
            print(f"   Avg param mean: {avg_mean:.6f}")
            print(f"   Avg param std: {avg_std:.6f}")
            
            # Check for uninitialized or collapsed parameters
            if avg_std < 0.001:
                print(f"   ⚠️ WARNING: Parameters have very low variance! May not be learning.")
            elif avg_std > 10.0:
                print(f"   ⚠️ WARNING: Parameters have very high variance! May have exploded.")
    
    # Check output_scale parameter specifically (critical for noise prediction)
    if hasattr(model.model, 'upsampler') and hasattr(model.model.upsampler, 'output_scale'):
        output_scale = model.model.upsampler.output_scale.item()
        print(f"\n📊 Output scale parameter: {output_scale:.4f}")
        if output_scale < 0.1:
            print(f"   ⚠️ WARNING: Output scale is very small! Model outputs will be suppressed.")
        elif output_scale > 10.0:
            print(f"   ⚠️ WARNING: Output scale is very large! Model outputs may explode.")

if __name__ == "__main__":
    print("🔍 ViViT Model Debugging Script")
    print("=" * 60)
    
    # Run analyses
    analyze_model_output()
    analyze_sampling_process()
    check_model_gradients()
    
    print("\n" + "=" * 60)
    print("✅ Debugging complete!")
    print("=" * 60)
