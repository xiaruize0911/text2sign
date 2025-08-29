import torch
import os
import imageio
import numpy as np
from config import Config
from diffusion import create_diffusion_model

def debug_sampling():
    # Load latest checkpoint
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, 'latest_checkpoint.pt')
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)

    # Create model and load weights
    model = create_diffusion_model(Config)
    model.to(Config.DEVICE)
    model.eval()
    model.load_state_dict(ckpt['model_state_dict'])
    
    print(f"Loaded checkpoint from step {ckpt['global_step']}, epoch {ckpt['epoch']}")

    # Test single step to understand the process
    batch_size = 1
    shape = (batch_size, *Config.INPUT_SHAPE)
    
    # Start with pure noise
    x = torch.randn(shape, device=Config.DEVICE)
    print(f"Starting noise range: [{x.min():.3f}, {x.max():.3f}], mean: {x.mean():.3f}, std: {x.std():.3f}")
    
    # Test at different timesteps
    for timestep in [Config.TIMESTEPS-1, Config.TIMESTEPS//2, 10, 0]:
        t = torch.full((batch_size,), timestep, device=Config.DEVICE, dtype=torch.long)
        
        with torch.no_grad():
            # Get model prediction
            predicted_noise = model.model(x, t)
            print(f"Timestep {timestep}: predicted noise range [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}], mean: {predicted_noise.mean():.3f}, std: {predicted_noise.std():.3f}")
            
            # Apply one reverse step
            if timestep > 0:
                x_prev = model.p_sample_step(x, t)
                print(f"  After reverse step: range [{x_prev.min():.3f}, {x_prev.max():.3f}], mean: {x_prev.mean():.3f}, std: {x_prev.std():.3f}")

    # Test full sampling with fixed seed for reproducibility
    torch.manual_seed(42)
    num_samples = 2
    shape = (num_samples, *Config.INPUT_SHAPE)
    
    print("\nStarting full sampling process...")
    with torch.no_grad():
        samples = model.p_sample(shape, device=Config.DEVICE)
        samples = torch.clamp(samples, 0, 1)
    
    print(f"Final samples range: [{samples.min():.3f}, {samples.max():.3f}], mean: {samples.mean():.3f}, std: {samples.std():.3f}")

    # Save debug samples
    samples_np = samples.cpu().numpy()
    for sample_idx in range(num_samples):
        sample = samples_np[sample_idx]  # (channels, frames, height, width)
        video_frames = []
        for frame_idx in range(sample.shape[1]):
            frame = sample[:, frame_idx]  # (channels, height, width)
            frame = frame.transpose(1, 2, 0)  # (height, width, channels)
            frame = (frame * 255).clip(0, 255).astype('uint8')
            video_frames.append(frame)
        gif_path = os.path.join(Config.SAMPLES_DIR, f'debug_sample_{sample_idx}.gif')
        imageio.mimsave(gif_path, video_frames, fps=8, loop=0)
        print(f"Saved debug sample: {gif_path}")

if __name__ == "__main__":
    debug_sampling()
