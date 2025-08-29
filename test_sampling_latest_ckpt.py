import torch
import os
import imageio
from config import Config
from diffusion import create_diffusion_model

# Load latest checkpoint
ckpt_path = os.path.join(Config.CHECKPOINT_DIR, 'latest_checkpoint.pt')
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)

# Create model and load weights
model = create_diffusion_model(Config)
model.to(Config.DEVICE)
model.eval()
model.load_state_dict(ckpt['model_state_dict'])

# Generate samples
num_samples = 4
shape = (num_samples, *Config.INPUT_SHAPE)
with torch.no_grad():
    samples = model.p_sample(shape, device=Config.DEVICE)
    samples = torch.clamp(samples, 0, 1)

# Save as GIFs
samples_np = samples.cpu().numpy()
for sample_idx in range(num_samples):
    sample = samples_np[sample_idx]  # (channels, frames, height, width)
    video_frames = []
    for frame_idx in range(sample.shape[1]):
        frame = sample[:, frame_idx]  # (channels, height, width)
        frame = frame.transpose(1, 2, 0)  # (height, width, channels)
        frame = (frame * 255).clip(0, 255).astype('uint8')
        video_frames.append(frame)
    gif_path = os.path.join(Config.SAMPLES_DIR, f'test_latest_ckpt_idx_{sample_idx}.gif')
    imageio.mimsave(gif_path, video_frames, fps=8, loop=0)
    print(f"Saved: {gif_path}")
