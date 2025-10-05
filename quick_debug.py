#!/usr/bin/env python3
"""Quick debug script to identify ViViT sampling issues"""

import torch
import numpy as np
from config import Config
from diffusion import create_diffusion_model
import os

print("🔍 Quick ViViT Debug")
print("=" * 60)

# Load model
model = create_diffusion_model(Config)
model.to(Config.DEVICE)

# Load checkpoint
checkpoint_path = "checkpoints/text2sign_vivit6/latest_checkpoint.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, step {checkpoint.get('global_step', 'unknown')}")

model.eval()

print("\n1. Testing model forward pass at different timesteps...")
# Create random input
x = torch.randn(1, 3, 28, 128, 128, device=Config.DEVICE)
text = ["hello"]

with torch.no_grad():
    for t_val in [0, 25, 49]:
        t = torch.full((1,), t_val, device=Config.DEVICE, dtype=torch.long)
        
        # Add noise
        noise, x_noisy = model.q_sample(x, t)
        
        # Get prediction
        text_emb = model.text_encoder(text)
        pred_noise = model.model(x_noisy, t, text_emb)
        
        # Compute error
        mse = torch.nn.functional.mse_loss(pred_noise, noise)
        
        print(f"\n   t={t_val}:")
        print(f"     Noise range: [{noise.min():.3f}, {noise.max():.3f}], std: {noise.std():.3f}")
        print(f"     Pred range: [{pred_noise.min():.3f}, {pred_noise.max():.3f}], std: {pred_noise.std():.3f}")
        print(f"     MSE: {mse.item():.6f}")
        print(f"     Scale ratio (pred/noise): {(pred_noise.std() / noise.std()).item():.4f}")

print("\n2. Testing sampling process...")
# Generate a sample
shape = (1, 3, 28, 128, 128)
with torch.no_grad():
    x_start = torch.randn(shape, device=Config.DEVICE)
    print(f"   Initial noise std: {x_start.std():.3f}")
    
    # Do a few denoising steps
    text = "hello"
    for step_idx, t_val in enumerate([49, 40, 30, 20, 10, 0]):
        t = torch.full((1,), t_val, device=Config.DEVICE, dtype=torch.long)
        prev_t_val = [40, 30, 20, 10, 0, -1][step_idx]
        prev_t = torch.full((1,), prev_t_val, device=Config.DEVICE, dtype=torch.long)
        
        x_start = model.p_sample_step(x_start, t, prev_t, text=text, deterministic=True, eta=0.0)
        
        if step_idx == 0 or step_idx == 5:
            print(f"   After t={t_val}: range [{x_start.min():.3f}, {x_start.max():.3f}], std: {x_start.std():.3f}")

print("\n3. Checking key model components...")
# Check output scaling
if hasattr(model.model, 'final_projection'):
    with torch.no_grad():
        test_input = torch.randn(1, 768, 14, 14, device=Config.DEVICE)
        # Note: final_projection expects different input shape
        print(f"   final_projection exists: ✅")

# Check temporal layers
num_temporal = len(model.model.temporal_layers) if hasattr(model.model, 'temporal_layers') else 0
print(f"   Number of temporal layers: {num_temporal}")

# Check if backbone is frozen
backbone_trainable = sum(p.numel() for p in model.model.backbone.parameters() if p.requires_grad)
backbone_total = sum(p.numel() for p in model.model.backbone.parameters())
print(f"   Backbone params: {backbone_total:,} (trainable: {backbone_trainable:,})")

# Check trainable head
head_params = []
for name, module in [('temporal_layers', getattr(model.model, 'temporal_layers', None)),
                      ('film_layers', getattr(model.model, 'film_layers', None)),
                      ('final_projection', getattr(model.model, 'final_projection', None)),
                      ('temporal_conv', getattr(model.model, 'temporal_conv', None))]:
    if module is not None:
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        print(f"   {name}: {total:,} ({trainable:,} trainable)")

print("\n=" * 60)
print("✅ Debug complete!")
