#!/usr/bin/env python3
"""
Script to analyze the ViViT model architecture and identify potential issues.
"""

import torch
import torch.nn as nn
from models.architectures.vivit import ViViT

def analyze_vivit_architecture():
    """Analyze the ViViT model architecture"""
    print("🔍 ViViT Model Architecture Analysis")
    print("=" * 60)
    
    # Create a ViViT model with the actual configuration
    model = ViViT(
        video_size=(16, 64, 64),  # Same as config
        in_channels=3,
        out_channels=3,
        time_dim=768,
        text_dim=768,
        model_name="google/vivit-b-16x2-kinetics400",
        freeze_backbone=True,
        num_temporal_layers=2,
        num_heads=8,
        dropout=0.1,
        class_dropout_prob=0.1
    )
    
    print("\n📊 Model Component Analysis:")
    print("-" * 40)
    
    # 1. Count parameters by component
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.1f}%")
    
    # 2. Analyze trainable components
    print("\n🔓 Trainable Components:")
    print("-" * 40)
    
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if module_trainable > 0:
            print(f"{name}: {module_trainable:,} params ({100 * module_trainable / trainable_params:.1f}% of trainable)")
    
    # 3. Analyze component structure
    print("\n🏗️  Component Structure:")
    print("-" * 40)
    
    print(f"ViViT Backbone:")
    print(f"  - Model: {model.backbone.model_name}")
    print(f"  - Embed dim: {model.backbone.embed_dim}")
    print(f"  - Num patches: {model.backbone.num_patches}")
    print(f"  - Tubelet size: {model.backbone.tubelet_size}")
    print(f"  - Frozen: {model.backbone.freeze_backbone}")
    
    print(f"\nTemporal Layers: {len(model.temporal_layers)} layers")
    print(f"FiLM Layers: {len(model.film_layers)} layers")
    
    print(f"\nFeature Upsampler:")
    print(f"  - Input dim: {model.upsampler.embed_dim}")
    print(f"  - Output channels: {model.upsampler.out_channels}")
    print(f"  - Video size: {model.upsampler.video_size}")
    print(f"  - Patch grid: {model.upsampler.patch_frames}x{model.upsampler.patch_height}x{model.upsampler.patch_width}")
    
    # 4. Analyze potential bottlenecks
    print("\n⚠️  Potential Issues:")
    print("-" * 40)
    
    issues = []
    
    # Check if backbone is completely frozen
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    if backbone_trainable == 0:
        issues.append("Backbone is completely frozen - may limit expressiveness")
    
    # Check upsampler complexity
    upsampler_params = sum(p.numel() for p in model.upsampler.parameters())
    if upsampler_params < 100000:  # Less than 100K params
        issues.append(f"Upsampler has only {upsampler_params:,} parameters - may be insufficient")
    
    # Check final normalization
    if hasattr(model, 'output_norm'):
        issues.append("Final GroupNorm may be over-constraining outputs")
    
    # Check conditioning strength
    cond_params = sum(p.numel() for p in model.cond_proj.parameters())
    film_params = sum(p.numel() for p in model.film_layers.parameters())
    total_cond_params = cond_params + film_params
    if total_cond_params < 10000:  # Less than 10K params
        issues.append(f"Conditioning has only {total_cond_params:,} parameters - may be weak")
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    if not issues:
        print("No obvious architectural issues detected")
    
    return model

def test_forward_pass():
    """Test a forward pass and examine outputs"""
    print("\n🚀 Forward Pass Analysis:")
    print("-" * 40)
    
    model = ViViT(
        video_size=(16, 64, 64),
        in_channels=3,
        out_channels=3,
        time_dim=768,
        text_dim=768,
        freeze_backbone=True
    )
    
    model.eval()
    
    # Create test inputs
    batch_size = 1
    video = torch.randn(batch_size, 3, 16, 64, 64)
    timestep = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 768)
    
    print(f"Input video shape: {video.shape}")
    print(f"Input video range: [{video.min():.3f}, {video.max():.3f}]")
    print(f"Input video mean: {video.mean():.3f}")
    print(f"Input video std: {video.std():.3f}")
    
    with torch.no_grad():
        # Forward pass
        predicted_noise = model(video, timestep, text_emb)
        
        print(f"\nOutput noise shape: {predicted_noise.shape}")
        print(f"Output noise range: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")
        print(f"Output noise mean: {predicted_noise.mean():.3f}")
        print(f"Output noise std: {predicted_noise.std():.3f}")
        
        # Check for degenerate outputs
        variance = predicted_noise.var()
        print(f"Output variance: {variance:.6f}")
        
        if variance < 1e-6:
            print("⚠️  WARNING: Output variance is very low - model may be predicting near-constant values")
        
        # Check channel-wise statistics
        for c in range(3):
            channel_data = predicted_noise[0, c]
            print(f"Channel {c}: mean={channel_data.mean():.3f}, std={channel_data.std():.3f}")

def main():
    """Main analysis function"""
    model = analyze_vivit_architecture()
    test_forward_pass()
    
    print("\n" + "=" * 60)
    print("Analysis complete!")

if __name__ == "__main__":
    main()