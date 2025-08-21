#!/usr/bin/env python3
"""
Test script to generate 8 samples before any training to show output shapes.
This script demonstrates what the untra    # Generation parameters
    generation_params = {
        'num_frames': 28,
        'height': 128,
        'width': 128,
        'num_inference_steps': 5,  # Very few steps for memory efficiency
        'guidance_scale': 1.0,  # No guidance for untrained model
    }el produces as a baseline.
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import sys
from typing import List, Dict, Any

# Add src to path
sys.path.append('./src')

from src.models.pipeline import create_pipeline, Text2SignDiffusionPipeline
from src.data.dataset import create_dataloader, SignLanguageDataset
from src.models.text_encoder import SimpleTextEncoder


def tensor_to_gif_frames(video_tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert video tensor to list of PIL Images for GIF creation.
    
    Args:
        video_tensor: Tensor of shape (C, T, H, W) or (T, C, H, W)
        
    Returns:
        List of PIL Images
    """
    # Ensure tensor is on CPU and detached
    video_tensor = video_tensor.detach().cpu()
    
    # Handle different tensor shapes
    if video_tensor.dim() == 4:
        if video_tensor.shape[0] == 3:  # (C, T, H, W)
            video_tensor = video_tensor.permute(1, 0, 2, 3)  # (T, C, H, W)
        # else assume it's already (T, C, H, W)
    
    # Clamp values to [0, 1] range
    video_tensor = torch.clamp(video_tensor, 0, 1)
    
    frames = []
    for frame_idx in range(video_tensor.shape[0]):
        frame = video_tensor[frame_idx]  # (C, H, W)
        
        # Convert to numpy and scale to [0, 255]
        frame_np = (frame.numpy() * 255).astype(np.uint8)
        
        # Convert from (C, H, W) to (H, W, C)
        frame_np = np.transpose(frame_np, (1, 2, 0))
        
        # Create PIL Image
        pil_frame = Image.fromarray(frame_np, 'RGB')
        frames.append(pil_frame)
    
    return frames


def save_video_as_gif(video_tensor: torch.Tensor, output_path: str, duration: int = 200):
    """
    Save video tensor as animated GIF.
    
    Args:
        video_tensor: Video tensor of shape (C, T, H, W)
        output_path: Path to save GIF
        duration: Duration per frame in milliseconds
    """
    frames = tensor_to_gif_frames(video_tensor)
    
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )


def analyze_tensor_shape(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    """Analyze tensor properties and return statistics."""
    stats = {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'min': float(tensor.min().item()),
        'max': float(tensor.max().item()),
        'mean': float(tensor.mean().item()),
        'std': float(tensor.std().item()),
        'num_params': tensor.numel()
    }
    return stats


def test_untrained_generation():
    """Test generation with untrained model to show baseline output shapes."""
    print("🎯 Testing Untrained Model Generation")
    print("=" * 50)
    
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create pipeline with minimal size for testing large output
    print("\n📦 Creating Pipeline...")
    pipeline = create_pipeline(
        model_channels=16,  # Much smaller for memory efficiency
        num_res_blocks=1,
        attention_resolutions=[32, 64],  # Adjusted for 128x128
        channel_mult=[1, 2, 3],  # Reduced multipliers
        num_heads=2,  # Fewer attention heads
        text_encoder_type="simple",
        scheduler_type="ddpm",
        device=device
    )
    
    print(f"✅ Pipeline created successfully")
    print(f"   - UNet parameters: {sum(p.numel() for p in pipeline.unet.parameters()):,}")
    print(f"   - Text encoder parameters: {sum(p.numel() for p in pipeline.text_encoder.parameters()):,}")
    
    # Test prompts from training data style
    test_prompts = [
        "The aileron is the control surface in the wing that is controlled by lateral movement right and left of the stick.",
        "By moving the stick, you cause pressure to increase or decrease the angle of attack on that particular raising or lowering the wing.",
        "The elevator is the part that moves with the stick forward and back, and that adjusts the angle of attack of the airplane in the air.",
        "Therefore, it's either going uphill, downhill, or flat and that adjusts the air speed that we talked about earlier.",
        "Hello",
        "How are you?",
        "Thank you",
        "Good morning"
    ]
    
    print(f"\n🎬 Generating 8 samples...")
    print(f"Prompts to generate:")
    for i, prompt in enumerate(test_prompts):
        print(f"   {i+1}. {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    
    # Generation parameters
    generation_params = {
        'num_frames': 28,
        'height': 128,
        'width': 128,
        'num_inference_steps': 5,  # Very few steps for memory efficiency
        'guidance_scale': 1.0,  # No guidance for untrained model
    }
    
    print(f"\nGeneration parameters:")
    for key, value in generation_params.items():
        print(f"   - {key}: {value}")
    
    # Create output directory
    output_dir = Path("./test_generation_outputs")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🔄 Starting generation...")
    
    # Generate samples
    with torch.no_grad():
        pipeline.eval()
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Sample {i+1}/8 ---")
            print(f"Prompt: {prompt}")
            
            try:
                # Generate single sample
                result = pipeline(
                    prompts=[prompt],
                    **generation_params
                )
                
                # Extract video tensor
                if isinstance(result, dict):
                    video = result['videos'][0]  # First (and only) sample
                    latents = result.get('latents', None)
                else:
                    video = result[0]
                    latents = None
                
                # Analyze generated video
                video_stats = analyze_tensor_shape(video, f"Generated_Video_{i+1}")
                print(f"✅ Generated video: {video_stats['shape']}")
                print(f"   - Range: [{video_stats['min']:.3f}, {video_stats['max']:.3f}]")
                print(f"   - Mean: {video_stats['mean']:.3f}, Std: {video_stats['std']:.3f}")
                
                # Save as GIF
                gif_path = output_dir / f"sample_{i+1:02d}.gif"
                save_video_as_gif(video, str(gif_path))
                print(f"💾 Saved: {gif_path}")
                
                # Save tensor for analysis
                tensor_path = output_dir / f"sample_{i+1:02d}.pt"
                torch.save({
                    'video': video,
                    'prompt': prompt,
                    'stats': video_stats,
                    'generation_params': generation_params
                }, tensor_path)
                
                # Analyze latents if available
                if latents is not None:
                    latent_stats = analyze_tensor_shape(latents[0], f"Latents_{i+1}")
                    print(f"📊 Latents: {latent_stats['shape']}")
                
            except Exception as e:
                print(f"❌ Error generating sample {i+1}: {e}")
                continue
    
    print(f"\n🎉 Generation completed!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📊 Generated files:")
    
    # List generated files
    gif_files = list(output_dir.glob("*.gif"))
    tensor_files = list(output_dir.glob("*.pt"))
    
    print(f"   - GIFs: {len(gif_files)} files")
    for gif_file in sorted(gif_files):
        file_size = gif_file.stat().st_size / 1024  # KB
        print(f"     • {gif_file.name} ({file_size:.1f} KB)")
    
    print(f"   - Tensors: {len(tensor_files)} files")
    for tensor_file in sorted(tensor_files):
        file_size = tensor_file.stat().st_size / 1024  # KB
        print(f"     • {tensor_file.name} ({file_size:.1f} KB)")
    
    return output_dir


def test_training_data_shapes(data_dir: str):
    """Test the shapes of training data for comparison."""
    print(f"\n📊 Testing Training Data Shapes")
    print("=" * 40)
    
    try:
        # Create dataset
        from src.data.dataset import SignLanguageDataset
        dataset = SignLanguageDataset(
            data_dir=data_dir,
            max_frames=28,
            frame_size=(128, 128)
        )
        
        print(f"Dataset size: {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Get first sample
            sample = dataset[0]
            
            video_stats = analyze_tensor_shape(sample['video'], "Training_Video")
            print(f"✅ Training video shape: {video_stats['shape']}")
            print(f"   - Range: [{video_stats['min']:.3f}, {video_stats['max']:.3f}]")
            print(f"   - Mean: {video_stats['mean']:.3f}, Std: {video_stats['std']:.3f}")
            print(f"   - Text: {sample['text'][:100]}{'...' if len(sample['text']) > 100 else ''}")
            
            return video_stats
        else:
            print("❌ No samples found in dataset")
            return None
            
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test untrained model generation")
    parser.add_argument("--data_dir", type=str, default="./training_data", help="Path to training data")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    print("🚀 Text2Sign Untrained Model Test")
    print("=" * 60)
    print(f"Purpose: Generate {args.num_samples} samples before training to show baseline output shapes")
    print(f"This helps understand what the model produces without any learning.")
    
    # Test training data shapes first
    if Path(args.data_dir).exists():
        training_stats = test_training_data_shapes(args.data_dir)
    else:
        print(f"⚠️ Training data directory not found: {args.data_dir}")
        training_stats = None
    
    # Test untrained generation
    output_dir = test_untrained_generation()
    
    # Summary
    print(f"\n📋 Summary")
    print("=" * 30)
    print(f"✅ Generated {args.num_samples} untrained samples")
    print(f"📁 Outputs saved to: {output_dir}")
    
    if training_stats:
        print(f"📊 Training data shape: {training_stats['shape']}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Compare untrained outputs with training data")
    print(f"   2. Start training: python train.py --data_dir {args.data_dir}")
    print(f"   3. Generate samples after training to see improvement")
    print(f"   4. Use TensorBoard to monitor training: python launch_tensorboard.py")
    
    print(f"\n🎯 Expected Behavior:")
    print(f"   - Untrained model outputs should be random noise-like")
    print(f"   - All samples should have the same shape: (3, 28, 128, 128)")
    print(f"   - Values should be in range [0, 1]")
    print(f"   - After training, outputs should resemble sign language movements")


if __name__ == "__main__":
    main()
