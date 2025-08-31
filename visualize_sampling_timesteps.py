"""
Standalone script to visualize the first frame of generated videos at each timestep during diffusion sampling.
This script loads a trained diffusion model and generates samples while creating an animated GIF
showing the progression of the first frame through all timesteps.

Usage:
    python visualize_sampling_timesteps.py --checkpoint_path /path/to/checkpoint.pt --output_path ./sampling_progression.gif --text "hello world"
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

# Add the project root to the path so we can import modules
sys.path.append('/teamspace/studios/this_studio/text2sign')

# Import project modules
from config import Config
from diffusion.text2sign import DiffusionModel, create_diffusion_model
from models.text_encoder import create_text_encoder
from schedulers.noise_schedulers import create_noise_scheduler


class SamplingVisualizer:
    """
    Class to handle visualization of sampling process at each timestep
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE

    def denormalize_image(self, tensor):
        """
        Denormalize from [-1, 1] to [0, 255] for image saving (matches main.py logic)

        Args:
            tensor: Tensor of shape (C, H, W) with values in [-1, 1]

        Returns:
            np.ndarray: Denormalized array with values in [0, 255] and dtype uint8
        """
        # tensor shape: (C, H, W)
        array = tensor.cpu().numpy()
        array = np.clip((array + 1) * 127.5, 0, 255).astype(np.uint8)
        return array

    def tensor_to_pil(self, tensor):
        """
        Convert tensor to PIL Image

        Args:
            tensor: Tensor of shape (C, H, W) with values in [-1, 1]

        Returns:
            PIL.Image: PIL Image object
        """
        # Denormalize to [0, 255] and transpose to (H, W, C)
        img_array = self.denormalize_image(tensor)
        img_array = np.transpose(img_array, (1, 2, 0))
        return Image.fromarray(img_array)

    def create_gif(self, frames, output_path, duration=100):
        """
        Create animated GIF from frames

        Args:
            frames: List of PIL Image objects
            output_path: Path to save the GIF
            duration: Duration between frames in milliseconds
        """
        if frames:
            # Save as animated GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0  # Infinite loop
            )
            print(f"✅ GIF saved to {output_path}")
        else:
            print("⚠️ No frames to save")

    @torch.no_grad()
    def sample_with_visualization(self, shape, output_path, text=None, deterministic=False, duration=100):
        """
        Modified sampling that creates an animated GIF of the first frame at each timestep

        Args:
            shape: Shape of sample to generate (batch_size, channels, frames, height, width)
            output_path: Path to save the output GIF file
            text: Text conditioning
            deterministic: Whether to use deterministic sampling
            duration: Duration between frames in milliseconds

        Returns:
            torch.Tensor: Generated video sample
        """
        # --- Input Validation ---
        assert shape[0] > 0, "Batch size must be positive"
        if self.model.text_encoder is not None:
            assert text is not None, "Text prompt must be provided for a conditioned model"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Start with random noise
        x = torch.randn(shape, device=self.device)

        # Collect frames for GIF
        frames = []

        # Save initial noise frame
        first_frame = x[0, :, 0, :, :]  # (C, H, W)
        frames.append(self.tensor_to_pil(first_frame))

        # Reverse diffusion process
        K = 10  # Frame/log interval
        for i in tqdm(reversed(range(self.model.timesteps)), desc="Sampling with visualization"):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)

            # Single denoising step
            x = self.model.p_sample_step(x, t, text, deterministic)

            # Save first frame at interval
            if i % K == 0 or i < 5 or i == self.model.timesteps - 1:
                first_frame = x[0, :, 0, :, :]  # (C, H, W)
                frames.append(self.tensor_to_pil(first_frame))

            # Log progress every K steps and for the first 5 steps
            if i % K == 0 or i < 5:
                print(f"Timestep {i}: x range [{x.min().item():.3f}, {x.max().item():.3f}]")

        # Final clamp
        x = torch.clamp(x, -1.0, 1.0)

        # Save final frame
        first_frame = x[0, :, 0, :, :]
        frames.append(self.tensor_to_pil(first_frame))

        # Shape assertion
        expected_shape = shape
        assert tuple(x.shape) == expected_shape, f"Output shape {x.shape} does not match expected {expected_shape}"

        # Create animated GIF
        self.create_gif(frames, output_path, duration)

        return x


def load_checkpoint(model, checkpoint_path):
    """
    Load model weights from checkpoint

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file

    Returns:
        Model with loaded weights
    """
    try:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)

        print("✅ Checkpoint loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Critical error loading checkpoint: {e}")
        sys.exit(1) # Exit if checkpoint fails to load


def main():
    """
    Main function to run the timestep visualization
    """
    parser = argparse.ArgumentParser(description="Visualize diffusion sampling timesteps")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, default="./sampling_progression.gif",
                       help="Path to save the output GIF file")
    parser.add_argument("--text", type=str, default="hello world",
                       help="Text prompt for generation")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for generation")
    parser.add_argument("--num_frames", type=int, default=28,
                       help="Number of frames in video")
    parser.add_argument("--height", type=int, default=128,
                       help="Video height")
    parser.add_argument("--width", type=int, default=128,
                       help="Video width")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic sampling")
    parser.add_argument("--duration", type=int, default=100,
                       help="Duration between frames in milliseconds")

    args = parser.parse_args()

    # --- Input Validation ---
    if args.batch_size <= 0:
        print("Error: Batch size must be a positive integer.")
        sys.exit(1)
    if not args.text:
        print("Warning: Empty text prompt provided. Using a default.")
        args.text = "hello"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = device

    print("🚀 Starting timestep visualization...")
    print(f"Device: {device}")
    print(f"Text: {args.text}")
    print(f"Output path: {args.output_path}")

    # Create model
    model = create_diffusion_model(Config)
    model = model.to(device)

    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint_path)
    model.eval()

    # Create visualizer
    visualizer = SamplingVisualizer(model, Config)

    # Generate sample with visualization
    shape = (args.batch_size, 3, args.num_frames, args.height, args.width)
    sample = visualizer.sample_with_visualization(
        shape=shape,
        output_path=args.output_path,
        text=args.text,
        deterministic=args.deterministic,
        duration=args.duration
    )

    # The number of frames is not simply timesteps + 1, it depends on the sampling interval K
    # This message is now more accurate.
    print(f"✅ Visualization complete. GIF saved to {args.output_path}")


if __name__ == "__main__":
    main()
