import torch
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.models.pipeline import create_pipeline, Text2SignDiffusionPipeline


def tensor_to_pil_images(video_tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert video tensor to list of PIL images.
    
    Args:
        video_tensor: Video tensor of shape (C, T, H, W) or (B, C, T, H, W)
        
    Returns:
        List of PIL images
    """
    if video_tensor.dim() == 5:
        # Take first batch item
        video_tensor = video_tensor[0]
    
    # Convert from (C, T, H, W) to (T, H, W, C)
    video_tensor = video_tensor.permute(1, 2, 3, 0)
    
    # Convert to [0, 255] and uint8
    video_tensor = (video_tensor * 255).clamp(0, 255).to(torch.uint8)
    
    # Convert to numpy
    video_np = video_tensor.cpu().numpy()
    
    # Convert to PIL images
    images = []
    for frame in video_np:
        img = Image.fromarray(frame)
        images.append(img)
    
    return images


def save_video_as_gif(images: List[Image.Image], output_path: str, fps: int = 8):
    """
    Save list of images as animated GIF.
    
    Args:
        images: List of PIL images
        output_path: Output file path
        fps: Frames per second
    """
    duration = 1000 // fps  # Duration in milliseconds
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"Video saved as GIF: {output_path}")


def save_video_as_frames(images: List[Image.Image], output_dir: str):
    """
    Save images as individual frame files.
    
    Args:
        images: List of PIL images
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(images):
        img.save(output_path / f"frame_{i:04d}.png")
    
    print(f"Frames saved to: {output_dir}")


def create_video_grid(video_list: List[torch.Tensor], prompts: List[str]) -> List[Image.Image]:
    """
    Create a grid of videos for comparison.
    
    Args:
        video_list: List of video tensors
        prompts: List of text prompts
        
    Returns:
        List of grid images
    """
    if not video_list:
        return []
    
    # Get dimensions
    num_videos = len(video_list)
    num_frames = video_list[0].shape[1] if video_list[0].dim() == 4 else video_list[0].shape[2]
    
    # Create grid for each frame
    grid_images = []
    
    for frame_idx in range(num_frames):
        # Collect all frames at this timestep
        frame_images = []
        
        for video_tensor in video_list:
            if video_tensor.dim() == 5:
                video_tensor = video_tensor[0]  # Take first batch item
            
            # Get frame
            frame = video_tensor[:, frame_idx, :, :]  # (C, H, W)
            frame = frame.permute(1, 2, 0)  # (H, W, C)
            frame = (frame * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            frame_img = Image.fromarray(frame)
            frame_images.append(frame_img)
        
        # Create grid
        if len(frame_images) == 1:
            grid_img = frame_images[0]
        else:
            # Simple horizontal concatenation
            widths, heights = zip(*(img.size for img in frame_images))
            total_width = sum(widths)
            max_height = max(heights)
            
            grid_img = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in frame_images:
                grid_img.paste(img, (x_offset, 0))
                x_offset += img.width
        
        grid_images.append(grid_img)
    
    return grid_images


class Text2SignInference:
    """
    Inference class for text-to-sign language generation.
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the inference pipeline.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create pipeline (you may need to adjust these parameters based on your training)
        self.pipeline = create_pipeline(
            model_channels=32,
            text_encoder_type="simple",
            scheduler_type="ddim",  # Use DDIM for faster inference
            device=device
        )
        
        # Load model state
        self.pipeline.load_state_dict(checkpoint['model_state_dict'])
        self.pipeline.eval()
        
        print("Model loaded successfully!")
    
    def generate(
        self,
        prompts: List[str],
        num_frames: int = 16,
        height: int = 64,
        width: int = 64,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Generate sign language videos from text prompts.
        
        Args:
            prompts: List of text prompts
            num_frames: Number of frames to generate
            height: Frame height
            width: Frame width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            
        Returns:
            List of generated video tensors
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            # Build vocabulary if using simple text encoder
            if hasattr(self.pipeline.text_encoder, 'build_vocab'):
                self.pipeline.text_encoder.build_vocab(prompts)  # type: ignore
            
            # Generate videos
            results = self.pipeline(
                prompts=prompts,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                return_dict=True
            )
            
            # Extract videos from results
            if isinstance(results, dict) and 'videos' in results:
                videos = results['videos']
            else:
                videos = results  # Fallback if return_dict=False
            
            # Split into individual videos
            video_list: List[torch.Tensor] = []
            if isinstance(videos, torch.Tensor):
                for i in range(len(prompts)):
                    video_list.append(videos[i:i+1])
            
            return video_list
    
    def generate_and_save(
        self,
        prompts: List[str],
        output_dir: str,
        save_format: str = "gif",
        **kwargs
    ):
        """
        Generate videos and save them.
        
        Args:
            prompts: List of text prompts
            output_dir: Output directory
            save_format: Save format ("gif", "frames", "both")
            **kwargs: Additional arguments for generate()
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate videos
        print(f"Generating videos for {len(prompts)} prompts...")
        video_list = self.generate(prompts, **kwargs)
        
        # Save each video
        for i, (video, prompt) in enumerate(zip(video_list, prompts)):
            # Clean prompt for filename
            clean_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_prompt = clean_prompt.replace(' ', '_')[:50]  # Limit length
            
            # Convert to images
            images = tensor_to_pil_images(video)
            
            if save_format in ["gif", "both"]:
                gif_path = output_path / f"{i:03d}_{clean_prompt}.gif"
                save_video_as_gif(images, str(gif_path))
            
            if save_format in ["frames", "both"]:
                frames_dir = output_path / f"{i:03d}_{clean_prompt}_frames"
                save_video_as_frames(images, str(frames_dir))
        
        # Create comparison grid if multiple prompts
        if len(prompts) > 1:
            print("Creating comparison grid...")
            grid_images = create_video_grid(video_list, prompts)
            
            if save_format in ["gif", "both"]:
                grid_gif_path = output_path / "comparison_grid.gif"
                save_video_as_gif(grid_images, str(grid_gif_path))
            
            if save_format in ["frames", "both"]:
                grid_frames_dir = output_path / "comparison_grid_frames"
                save_video_as_frames(grid_images, str(grid_frames_dir))
        
        print(f"All outputs saved to: {output_dir}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Generate sign language videos from text")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompts", type=str, nargs="+", required=True, help="Text prompts")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=28, help="Number of frames")
    parser.add_argument("--height", type=int, default=128, help="Frame height")
    parser.add_argument("--width", type=int, default=128, help="Frame width")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--save_format", type=str, default="gif", 
                       choices=["gif", "frames", "both"], help="Save format")
    
    args = parser.parse_args()
    
    # Create inference pipeline
    inference = Text2SignInference(args.checkpoint)
    
    # Generate and save videos
    inference.generate_and_save(
        prompts=args.prompts,
        output_dir=args.output_dir,
        save_format=args.save_format,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
