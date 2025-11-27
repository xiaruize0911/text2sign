import sys
import os

# Redirect stdout/stderr to file
log_file = open("debug_log_internal.txt", "w")
sys.stdout = log_file
sys.stderr = log_file

import torch
from config import Config
from models.architectures.tinyfusion import create_tinyfusion_model

print("Starting debug script...")
try:
    # Create backbone directly
    backbone = create_tinyfusion_model(**Config.get_model_config())
    print("Backbone created successfully!")
    
    device = torch.device("cpu")
    backbone.to(device)
    backbone.train()
    
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-4)
    
    # Create dummy inputs
    batch_size = 1
    frames = 4 # Config.NUM_FRAMES
    height = Config.IMAGE_SIZE
    width = Config.IMAGE_SIZE
    channels = 4 # RGBA
    
    print("Running training simulation for 1 steps...")
    
    for i in range(1):
        optimizer.zero_grad()
        
        x = torch.randn(batch_size, channels, frames, height, width).to(device)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        text_emb = torch.randn(batch_size, Config.TEXT_EMBED_DIM).to(device)
        target_noise = torch.randn_like(x)
        
        # Forward pass (WITH gradients)
        pred = backbone(x, t, text_emb)
        
        loss = torch.nn.functional.mse_loss(pred, target_noise)
        
        loss.backward()
        
        # Check gradients before clipping
        total_norm = 0.0
        for p in backbone.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        
        optimizer.step()
        
        print(f"Step {i+1}: Loss={loss.item():.6f}, Grad Norm={total_norm:.6f}")
        print(f"  Pred: mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")

    print("Debug script finished successfully!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
