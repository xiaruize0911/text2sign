#!/usr/bin/env python3
"""
Test script to sample from the dataloader and visualize using matplotlib.
This script helps verify that the dataloader is working correctly and shows
what the training data looks like.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

# Add src to path
sys.path.append('./src')

from src.data.dataset import create_dataloader, SignLanguageDataset


def tensor_to_numpy_frames(video_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert video tensor to numpy array for matplotlib visualization.
    
    Args:
        video_tensor: Tensor of shape (C, T, H, W)
        
    Returns:
        Numpy array of shape (T, H, W, C) with values in [0, 1]
    """
    # Move to CPU and convert to numpy
    video_np = video_tensor.detach().cpu().numpy()
    
    # Convert from (C, T, H, W) to (T, H, W, C)
    video_np = np.transpose(video_np, (1, 2, 3, 0))
    
    # Ensure values are in [0, 1] range
    video_np = np.clip(video_np, 0, 1)
    
    return video_np


def plot_video_frames(video_tensor: torch.Tensor, text: str, save_path: Optional[str] = None, max_frames: int = 16):
    """
    Plot video frames in a grid layout.
    
    Args:
        video_tensor: Video tensor of shape (C, T, H, W)
        text: Associated text description
        save_path: Optional path to save the plot
        max_frames: Maximum number of frames to display
    """
    # Convert tensor to numpy
    frames = tensor_to_numpy_frames(video_tensor)
    num_frames = min(frames.shape[0], max_frames)
    
    # Calculate grid dimensions
    cols = 4
    rows = (num_frames + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle(f"Video Frames: {text}", fontsize=14, fontweight='bold')
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot frames
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        
        if i < num_frames:
            # Show frame
            axes[row, col].imshow(frames[i])
            axes[row, col].set_title(f"Frame {i+1}")
            axes[row, col].axis('off')
        else:
            # Hide empty subplot
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def create_video_animation(video_tensor: torch.Tensor, text: str, save_path: Optional[str] = None):
    """
    Create an animated visualization of the video.
    
    Args:
        video_tensor: Video tensor of shape (C, T, H, W)
        text: Associated text description
        save_path: Optional path to save the animation
    """
    # Convert tensor to numpy
    frames = tensor_to_numpy_frames(video_tensor)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Video Animation: {text}", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Initialize image
    im = ax.imshow(frames[0], animated=True)
    
    # Animation function
    def animate(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f"Video Animation: {text}\nFrame {frame_idx + 1}/{len(frames)}")
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), 
        interval=200, blit=True, repeat=True
    )
    
    if save_path:
        # Save as GIF
        anim.save(save_path, writer='pillow', fps=5)
        print(f"Saved animation to: {save_path}")
    
    plt.show()
    return anim


def plot_batch_overview(batch: dict, save_dir: Optional[str] = None):
    """
    Plot an overview of a batch showing multiple samples.
    
    Args:
        batch: Batch dictionary from dataloader
        save_dir: Optional directory to save plots
    """
    videos = batch['videos']  # Shape: (B, C, T, H, W)
    texts = batch['texts']
    batch_size = videos.shape[0]
    
    print(f"Batch overview:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Video shape: {videos.shape}")
    print(f"  - Video dtype: {videos.dtype}")
    print(f"  - Video range: [{videos.min():.3f}, {videos.max():.3f}]")
    print(f"  - Texts: {len(texts)}")
    
    # Create a grid showing first frame of each video
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.suptitle("Batch Overview - First Frame of Each Video", fontsize=16, fontweight='bold')
    
    # Handle different subplot layouts
    if batch_size == 1:
        axes = [axes]
    elif rows == 1 and cols > 1:
        # axes is already a 1D array
        pass
    elif rows > 1 and cols > 1:
        # axes is a 2D array, flatten it for easier indexing
        axes = axes.flatten()
    
    for i in range(batch_size):
        # Get first frame of video i
        video = videos[i]  # Shape: (C, T, H, W)
        first_frame = tensor_to_numpy_frames(video)[0]  # Shape: (H, W, C)
        text = texts[i]
        
        # Get the axis for this sample
        ax = axes[i]
        
        # Show frame
        ax.imshow(first_frame)
        ax.set_title(f"Sample {i+1}\n{text[:30]}{'...' if len(text) > 30 else ''}", fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    total_subplots = rows * cols
    for i in range(batch_size, total_subplots):
        if i < len(axes):
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "batch_overview.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"Saved batch overview to: {save_path}")
    
    plt.show()


def analyze_dataset_statistics(dataloader):
    """
    Analyze and print statistics about the dataset.
    
    Args:
        dataloader: DataLoader to analyze
    """
    print("\n📊 Dataset Statistics")
    print("=" * 40)
    
    # Collect statistics from first few batches
    video_shapes = []
    video_ranges = []
    text_lengths = []
    sample_texts = []
    
    num_batches_to_analyze = min(5, len(dataloader))
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches_to_analyze:
            break
            
        videos = batch['videos']
        texts = batch['texts']
        
        video_shapes.append(videos.shape)
        video_ranges.append((videos.min().item(), videos.max().item()))
        
        for text in texts:
            text_lengths.append(len(text))
            if len(sample_texts) < 10:  # Keep first 10 texts as samples
                sample_texts.append(text)
    
    # Print statistics
    print(f"Analyzed {num_batches_to_analyze} batches")
    print(f"Video shapes: {set(video_shapes)}")
    print(f"Video value ranges: {video_ranges}")
    print(f"Text lengths - Min: {min(text_lengths)}, Max: {max(text_lengths)}, Avg: {np.mean(text_lengths):.1f}")
    
    print(f"\nSample texts:")
    for i, text in enumerate(sample_texts):
        print(f"  {i+1:2d}. {text[:80]}{'...' if len(text) > 80 else ''}")


def test_dataloader_visualization(data_dir: str, batch_size: int = 4, num_samples: int = 3):
    """
    Main function to test dataloader and create visualizations.
    
    Args:
        data_dir: Path to training data directory
        batch_size: Batch size for dataloader
        num_samples: Number of individual samples to visualize in detail
    """
    print("🔍 Testing Dataloader Visualization")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("./dataloader_test_outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create dataloader
    print(f"\n📦 Creating dataloader...")
    try:
        dataloader = create_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            shuffle=True,
            max_frames=28,
            frame_size=(128, 128)
        )
        print(f"✅ Dataloader created successfully")
        print(f"   - Dataset size: {len(dataloader.dataset)}")
        print(f"   - Number of batches: {len(dataloader)}")
        print(f"   - Batch size: {batch_size}")
        
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        return
    
    # Analyze dataset statistics
    analyze_dataset_statistics(dataloader)
    
    # Get a batch for visualization
    print(f"\n🎬 Sampling batch for visualization...")
    try:
        batch = next(iter(dataloader))
        print(f"✅ Successfully sampled batch")
        
        # Plot batch overview
        plot_batch_overview(batch, save_dir=str(output_dir))
        
        # Visualize individual samples in detail
        videos = batch['videos']
        texts = batch['texts']
        
        num_to_visualize = min(num_samples, videos.shape[0])
        print(f"\n🖼️ Creating detailed visualizations for {num_to_visualize} samples...")
        
        for i in range(num_to_visualize):
            video = videos[i]  # Shape: (C, T, H, W)
            text = texts[i]
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Text: {text}")
            print(f"Video shape: {video.shape}")
            print(f"Video range: [{video.min():.3f}, {video.max():.3f}]")
            
            # Create frame grid plot
            frame_plot_path = output_dir / f"sample_{i+1}_frames.png"
            plot_video_frames(video, text, save_path=str(frame_plot_path))
            
            # Create animation
            anim_path = output_dir / f"sample_{i+1}_animation.gif"
            try:
                anim = create_video_animation(video, text, save_path=str(anim_path))
                # Close the animation to free memory
                plt.close()
            except Exception as e:
                print(f"⚠️ Failed to create animation for sample {i+1}: {e}")
        
        print(f"\n🎉 Visualization completed!")
        print(f"📁 All outputs saved to: {output_dir.absolute()}")
        
        # List generated files
        png_files = list(output_dir.glob("*.png"))
        gif_files = list(output_dir.glob("*.gif"))
        
        print(f"\n📊 Generated files:")
        print(f"   - PNG plots: {len(png_files)} files")
        for png_file in sorted(png_files):
            file_size = png_file.stat().st_size / 1024  # KB
            print(f"     • {png_file.name} ({file_size:.1f} KB)")
        
        print(f"   - GIF animations: {len(gif_files)} files")
        for gif_file in sorted(gif_files):
            file_size = gif_file.stat().st_size / 1024  # KB
            print(f"     • {gif_file.name} ({file_size:.1f} KB)")
            
    except Exception as e:
        print(f"❌ Failed to sample from dataloader: {e}")
        return


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test dataloader and create visualizations")
    parser.add_argument("--data_dir", type=str, default="./training_data", help="Path to training data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for dataloader")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to visualize in detail")
    
    args = parser.parse_args()
    
    print("🚀 Dataloader Visualization Test")
    print("=" * 60)
    print(f"Purpose: Test the dataloader and visualize training data using matplotlib")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Samples to visualize: {args.num_samples}")
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Run the test
    test_dataloader_visualization(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    print(f"\n💡 Tips:")
    print(f"   - Check the generated PNG files to see frame grids")
    print(f"   - Check the generated GIF files to see video animations")
    print(f"   - Adjust --batch_size and --num_samples as needed")
    print(f"   - The batch overview shows first frames of all samples in a batch")


if __name__ == "__main__":
    main()
