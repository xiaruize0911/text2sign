import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from diffusion import create_diffusion_model
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def reproduce():
    # Override config for testing
    Config.TINYFUSION_VARIANT = "DiT-S/2"
    Config.TINYFUSION_CHECKPOINT = "none"
    
    # Force CPU for deterministic debugging if needed, or use Config.DEVICE
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Create model
    print("Loading checkpoint...")
    try:
        model = create_diffusion_model(Config)
        model.to(device)
        model.train()
        print("Model created successfully.")
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Create dummy batch
    batch_size = 2
    # Input shape: (channels, frames, height, width)
    # Config.INPUT_SHAPE is (4, 16, 64, 64)
    # Reduce frames for speed
    frames = 4
    input_shape = (4, frames, 64, 64)
    x = torch.randn(batch_size, *input_shape).to(device)
    # Normalize to [-1, 1]
    x = torch.clamp(x, -1, 1)
    
    text = ["test"] * batch_size
    
    print("Starting training loop...")
    for i in range(5):
        optimizer.zero_grad()
        
        # Forward
        loss, pred, noise = model(x, text)
        
        # Backward
        loss.backward()
        
        # Check grad norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        optimizer.step()
        
        print(f"Step {i}: Loss = {loss.item():.6f}, Grad Norm = {total_norm:.6f}, Pred Mean = {pred.mean().item():.6f}, Pred Std = {pred.std().item():.6f}")

if __name__ == "__main__":
    reproduce()
