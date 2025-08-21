#!/usr/bin/env python3
"""
Simple dataloader test script that prints statistics and saves sample frames as images.
This is a lightweight version that doesn't require matplotlib display.
"""

import torch
import numpy as np
import sys
from pathlib import Path
import argparse
from PIL import Image

# Add src to path
sys.path.append('./src')

from src.data.dataset import create_dataloader, SignLanguageDataset


def tensor_to_pil_frames(video_tensor: torch.Tensor) -> list:
    """
    Convert video tensor to list of PIL Images.
    
    Args:
        video_tensor: Tensor of shape (C, T, H, W)
        
    Returns:
        List of PIL Images
    """
    # Move to CPU and convert
    video_tensor = video_tensor.detach().cpu()
    
    # Convert from (C, T, H, W) to (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    
    # Clamp to [0, 1] and convert to [0, 255]
    video_tensor = torch.clamp(video_tensor, 0, 1)
    video_tensor = (video_tensor * 255).to(torch.uint8)
    
    frames = []
    for i in range(video_tensor.shape[0]):
        frame = video_tensor[i]  # (C, H, W)
        # Convert to (H, W, C)
        frame_np = frame.permute(1, 2, 0).numpy()
        pil_frame = Image.fromarray(frame_np, 'RGB')
        frames.append(pil_frame)
    
    return frames


def save_sample_frames(video_tensor: torch.Tensor, text: str, output_dir: Path, sample_idx: int):
    """
    Save individual frames and create a grid image.
    
    Args:
        video_tensor: Video tensor of shape (C, T, H, W)
        text: Associated text
        output_dir: Output directory
        sample_idx: Sample index for naming
    """
    frames = tensor_to_pil_frames(video_tensor)
    
    # Save individual frames
    frames_dir = output_dir / f"sample_{sample_idx}_frames"
    frames_dir.mkdir(exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame.save(frames_dir / f"frame_{i:03d}.png")
    
    # Create a grid of frames (4x7 for 28 frames)
    grid_cols = 7
    grid_rows = 4
    frame_width, frame_height = frames[0].size
    
    grid_image = Image.new('RGB', (grid_cols * frame_width, grid_rows * frame_height))
    
    for i, frame in enumerate(frames[:grid_rows * grid_cols]):
        row = i // grid_cols
        col = i % grid_cols
        x = col * frame_width
        y = row * frame_height
        grid_image.paste(frame, (x, y))
    
    # Save grid
    grid_path = output_dir / f"sample_{sample_idx}_grid.png"
    grid_image.save(grid_path)
    
    print(f"✅ Sample {sample_idx}:")
    print(f"   - Text: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"   - Frames saved to: {frames_dir}")
    print(f"   - Grid saved to: {grid_path}")


def test_dataloader_simple(data_dir: str, batch_size: int = 4, num_samples: int = 3):
    """
    Simple dataloader test with basic statistics and frame saving.
    
    Args:
        data_dir: Path to training data directory
        batch_size: Batch size for dataloader
        num_samples: Number of samples to save
    """
    print("🧪 Simple Dataloader Test")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("./simple_dataloader_test")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create dataloader
    print(f"\n📦 Creating dataloader...")
    try:
        dataloader = create_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=0,
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
    
    # Sample and analyze
    print(f"\n📊 Sampling and analyzing...")
    sample_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        videos = batch['videos']  # Shape: (B, C, T, H, W)
        texts = batch['texts']
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  - Video shape: {videos.shape}")
        print(f"  - Video dtype: {videos.dtype}")
        print(f"  - Video range: [{videos.min():.3f}, {videos.max():.3f}]")
        print(f"  - Mean: {videos.mean():.3f}, Std: {videos.std():.3f}")
        
        # Save samples from this batch
        for i in range(min(videos.shape[0], num_samples - sample_count)):
            video = videos[i]
            text = texts[i]
            
            save_sample_frames(video, text, output_dir, sample_count + 1)
            sample_count += 1
            
            if sample_count >= num_samples:
                break
        
        if sample_count >= num_samples:
            break
    
    print(f"\n🎉 Test completed!")
    print(f"📁 {sample_count} samples saved to: {output_dir.absolute()}")
    
    # List output files
    png_files = list(output_dir.glob("**/*.png"))
    print(f"\n📊 Generated {len(png_files)} PNG files:")
    
    total_size = 0
    for png_file in sorted(png_files)[:10]:  # Show first 10
        file_size = png_file.stat().st_size / 1024  # KB
        total_size += file_size
        print(f"   • {png_file.name} ({file_size:.1f} KB)")
    
    if len(png_files) > 10:
        print(f"   ... and {len(png_files) - 10} more files")
    
    print(f"Total size: {total_size:.1f} KB")


def quick_dataset_info(data_dir: str):
    """
    Quick dataset information without visualization.
    
    Args:
        data_dir: Path to training data directory
    """
    print("ℹ️ Quick Dataset Info")
    print("=" * 30)
    
    try:
        dataset = SignLanguageDataset(
            data_dir=data_dir,
            max_frames=28,
            frame_size=(128, 128)
        )
        
        print(f"✅ Dataset loaded successfully")
        print(f"   - Total samples: {len(dataset)}")
        
        # Sample a few examples
        print(f"\n📝 Sample texts:")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            text = sample['text']
            video_shape = sample['video'].shape
            print(f"   {i+1}. Shape: {video_shape}, Text: {text[:60]}{'...' if len(text) > 60 else ''}")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple dataloader test")
    parser.add_argument("--data_dir", type=str, default="./training_data", help="Path to training data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for dataloader")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to save")
    parser.add_argument("--info_only", action="store_true", help="Show dataset info only")
    
    args = parser.parse_args()
    
    print("🚀 Simple Dataloader Test")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    if args.info_only:
        quick_dataset_info(args.data_dir)
    else:
        test_dataloader_simple(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_samples=args.num_samples
        )
    
    print(f"\n💡 Tips:")
    print(f"   - Use --info_only for quick dataset overview")
    print(f"   - Check the generated PNG grid files for visual inspection")
    print(f"   - Individual frames are saved for detailed analysis")


if __name__ == "__main__":
    main()
