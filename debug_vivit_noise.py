#!/usr/bin/env python3
"""
Debug script to investigate ViViT noise prediction issues.

This script creates a ViViT model and examines its output range and behavior
to understand why the predicted noise appears as grey color.
"""

import torch
import torch.nn.functional as F
import numpy as np
from config import Config
from models.architectures.vivit import ViViT
from models.text_encoder import create_text_encoder
import os
import imageio

def create_debug_model():
    """Create a ViViT model for debugging"""
    print("🔧 Creating ViViT model for debugging...")
    
    # Use the same config as the main training
    config = Config()
    
    # Create text encoder
    text_encoder = create_text_encoder(config)
    
    # Create ViViT model
    model = ViViT(
        video_size=(config.NUM_FRAMES, config.IMAGE_SIZE, config.IMAGE_SIZE),
        in_channels=3,
        out_channels=3,
        time_dim=768,
        text_dim=text_encoder.embed_dim if hasattr(text_encoder, 'embed_dim') else 768,
        model_name="google/vivit-b-16x2-kinetics400",
        freeze_backbone=True,
        num_temporal_layers=2,
        num_heads=8,
        dropout=0.1,
        class_dropout_prob=0.1
    )
    
    return model, text_encoder, config

def debug_model_output(model, text_encoder, config):
    """Debug the model output to understand the grey noise issue"""
    print("\n🔍 Debugging model output...")
    
    # Move model to appropriate device
    device = config.DEVICE
    model = model.to(device)
    text_encoder = text_encoder.to(device)
    model.eval()
    
    # Create test input
    batch_size = 2
    videos = torch.randn(
        batch_size, 3, config.NUM_FRAMES, config.IMAGE_SIZE, config.IMAGE_SIZE
    ).to(device)
    
    # Normalize videos to [-1, 1] range (as expected by the model)
    videos = torch.clamp(videos, -1, 1)
    
    # Create test text
    texts = ["hello", "goodbye"]
    
    # Create test timesteps
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    
    print(f"📊 Input statistics:")
    print(f"   Video shape: {videos.shape}")
    print(f"   Video range: [{videos.min().item():.3f}, {videos.max().item():.3f}]")
    print(f"   Video mean: {videos.mean().item():.3f}")
    print(f"   Video std: {videos.std().item():.3f}")
    print(f"   Timesteps: {timesteps}")
    print(f"   Texts: {texts}")
    
    with torch.no_grad():
        # Test text encoding
        text_emb = text_encoder(texts)
        print(f"\n📝 Text embedding statistics:")
        print(f"   Text embedding shape: {text_emb.shape}")
        print(f"   Text embedding range: [{text_emb.min().item():.3f}, {text_emb.max().item():.3f}]")
        print(f"   Text embedding mean: {text_emb.mean().item():.3f}")
        print(f"   Text embedding std: {text_emb.std().item():.3f}")
        
        # Test model prediction
        predicted_noise = model(videos, timesteps, text_emb)
        
        print(f"\n🎯 Model output statistics:")
        print(f"   Predicted noise shape: {predicted_noise.shape}")
        print(f"   Predicted noise range: [{predicted_noise.min().item():.3f}, {predicted_noise.max().item():.3f}]")
        print(f"   Predicted noise mean: {predicted_noise.mean().item():.3f}")
        print(f"   Predicted noise std: {predicted_noise.std().item():.3f}")
        
        # Check for unusual patterns
        print(f"\n🔍 Pattern analysis:")
        
        # Check if output is constant
        variance_per_sample = predicted_noise.view(batch_size, -1).var(dim=1)
        print(f"   Variance per sample: {variance_per_sample}")
        
        # Check if output has any spatial/temporal variation
        spatial_var = predicted_noise.var(dim=[2, 3, 4])  # Variance across spatial-temporal dimensions
        print(f"   Spatial-temporal variance per channel: {spatial_var}")
        
        # Check for NaN or infinite values
        nan_count = torch.isnan(predicted_noise).sum()
        inf_count = torch.isinf(predicted_noise).sum()
        print(f"   NaN values: {nan_count}")
        print(f"   Infinite values: {inf_count}")
        
        # Check gradients
        print(f"\n🔄 Gradient analysis:")
        videos.requires_grad_(True)
        predicted_noise_grad = model(videos, timesteps, text_emb)
        loss = predicted_noise_grad.mean()
        loss.backward()
        
        # Check if gradients are flowing
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm < 1e-8:
                    print(f"   WARNING: Very small gradient for {name}: {grad_norm}")
        
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        print(f"   Average gradient norm: {avg_grad_norm:.6f}")
        print(f"   Number of parameters with gradients: {len(grad_norms)}")
        
        # Test different timesteps to see if behavior changes
        print(f"\n⏱️  Timestep analysis:")
        for t_val in [0, 100, 500, 999]:
            t_test = torch.full((batch_size,), t_val, device=device)
            pred_t = model(videos, t_test, text_emb)
            print(f"   t={t_val}: range=[{pred_t.min().item():.3f}, {pred_t.max().item():.3f}], "
                  f"mean={pred_t.mean().item():.3f}, std={pred_t.std().item():.3f}")
    
    return predicted_noise

def save_debug_visualization(predicted_noise, config):
    """Save debug visualization of predicted noise"""
    print("\n💾 Saving debug visualization...")
    
    # Take first sample from batch
    noise_sample = predicted_noise[0]  # Shape: (3, frames, height, width)
    
    # Normalize for visualization (clamp to reasonable range)
    noise_viz = torch.clamp(noise_sample, -3, 3) / 3.0  # Normalize to [-1, 1]
    
    # Convert to numpy
    noise_np = noise_viz.detach().cpu().numpy()
    noise_np = np.transpose(noise_np, (1, 2, 3, 0))  # (frames, height, width, channels)
    
    # Convert to uint8 for saving
    noise_uint8 = np.clip((noise_np + 1) * 127.5, 0, 255).astype(np.uint8)
    
    # Save as GIF
    os.makedirs('./debug_output', exist_ok=True)
    imageio.mimsave('./debug_output/debug_predicted_noise.gif', noise_uint8, fps=10)
    print("   Saved debug visualization to ./debug_output/debug_predicted_noise.gif")
    
    # Also save some statistics
    with open('./debug_output/debug_stats.txt', 'w') as f:
        f.write(f"Predicted Noise Debug Statistics\n")
        f.write(f"================================\n")
        f.write(f"Shape: {predicted_noise.shape}\n")
        f.write(f"Range: [{predicted_noise.min().item():.6f}, {predicted_noise.max().item():.6f}]\n")
        f.write(f"Mean: {predicted_noise.mean().item():.6f}\n")
        f.write(f"Std: {predicted_noise.std().item():.6f}\n")
        f.write(f"Variance: {predicted_noise.var().item():.6f}\n")
        
        # Per-channel statistics
        for c in range(predicted_noise.shape[1]):
            channel_data = predicted_noise[:, c]
            f.write(f"\nChannel {c}:\n")
            f.write(f"  Range: [{channel_data.min().item():.6f}, {channel_data.max().item():.6f}]\n")
            f.write(f"  Mean: {channel_data.mean().item():.6f}\n")
            f.write(f"  Std: {channel_data.std().item():.6f}\n")
    
    print("   Saved debug statistics to ./debug_output/debug_stats.txt")

def investigate_layer_outputs(model, videos, timesteps, text_emb):
    """Investigate intermediate layer outputs"""
    print("\n🔬 Investigating intermediate layer outputs...")
    
    device = videos.device
    
    # Hook to capture intermediate outputs
    intermediate_outputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                intermediate_outputs[name] = {
                    'shape': output.shape,
                    'range': [output.min().item(), output.max().item()],
                    'mean': output.mean().item(),
                    'std': output.std().item()
                }
        return hook
    
    # Register hooks on key layers
    hooks = []
    for name, module in model.named_modules():
        if any(layer_type in name for layer_type in ['backbone', 'cond_proj', 'temporal_layers', 'upsampler', 'output_norm']):
            if len(name.split('.')) <= 3:  # Only top-level modules to avoid too much output
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
    
    with torch.no_grad():
        _ = model(videos, timesteps, text_emb)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print results
    print("   Intermediate layer statistics:")
    for name, stats in intermediate_outputs.items():
        print(f"   {name}: shape={stats['shape']}, range=[{stats['range'][0]:.3f}, {stats['range'][1]:.3f}], "
              f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")

def main():
    """Main debugging function"""
    print("🚀 ViViT Noise Prediction Debug Analysis")
    print("=" * 60)
    
    try:
        # Create model
        model, text_encoder, config = create_debug_model()
        
        # Debug model output
        predicted_noise = debug_model_output(model, text_encoder, config)
        
        # Save visualization
        save_debug_visualization(predicted_noise, config)
        
        # Investigate intermediate layers
        batch_size = 1
        videos = torch.randn(
            batch_size, 3, config.NUM_FRAMES, config.IMAGE_SIZE, config.IMAGE_SIZE
        ).to(config.DEVICE)
        videos = torch.clamp(videos, -1, 1)
        timesteps = torch.randint(0, 1000, (batch_size,), device=config.DEVICE)
        texts = ["test"]
        text_emb = text_encoder(texts)
        
        investigate_layer_outputs(model, videos, timesteps, text_emb)
        
        print("\n" + "=" * 60)
        print("🎯 Debug analysis complete!")
        print("   Check ./debug_output/ for visualizations and statistics")
        
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()