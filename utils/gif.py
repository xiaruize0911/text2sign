"""
Utility functions for the diffusion model project
"""

import torch
import numpy as np
import imageio
import os
from typing import List, Tuple
import matplotlib.pyplot as plt

def save_video_as_gif(video_tensor: torch.Tensor, filepath: str, fps: int = 10):
    """
    Save a video tensor as a GIF file
    
    Args:
        video_tensor (torch.Tensor): Video tensor with shape (channels, frames, height, width)
        filepath (str): Output filepath for the GIF
        fps (int): Frames per second
    """
    # Convert to numpy and rearrange dimensions
    video = video_tensor.detach().cpu().numpy()
    video = np.transpose(video, (1, 2, 3, 0))  # (frames, height, width, channels)
    
    # Convert to uint8
    # Convert from [-1, 1] to [0, 255] for saving
    video = np.clip((video + 1) * 127.5, 0, 255).astype(np.uint8)
    #create file if not exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Save as GIF
    imageio.mimsave(filepath, video, fps=fps)

def load_gif_as_tensor(filepath: str) -> torch.Tensor:
    """
    Load a GIF file as a tensor
    
    Args:
        filepath (str): Path to the GIF file
        
    Returns:
        torch.Tensor: Video tensor with shape (channels, frames, height, width)
    """
    # Load GIF
    frames = imageio.mimread(filepath)
    frames = np.array(frames)
    
    # Convert to torch tensor and normalize to [-1, 1] (matching dataset)
    frames = torch.from_numpy(frames).float() / 255.0
    frames = frames * 2 - 1  # [0, 1] → [-1, 1]
    
    # Rearrange dimensions
    frames = frames.permute(3, 0, 1, 2)  # (channels, frames, height, width)
    
    return frames

def create_video_grid(videos: torch.Tensor, nrow: int = 4) -> torch.Tensor:
    """
    Create a grid of videos for visualization
    
    Args:
        videos (torch.Tensor): Batch of videos with shape (batch, channels, frames, height, width)
        nrow (int): Number of videos per row
        
    Returns:
        torch.Tensor: Grid of videos
    """
    batch_size, channels, frames, height, width = videos.shape
    
    # Calculate grid dimensions
    ncol = (batch_size + nrow - 1) // nrow
    
    # Create empty grid
    grid_height = ncol * height
    grid_width = nrow * width
    grid = torch.zeros(channels, frames, grid_height, grid_width)
    
    # Fill grid
    for idx, video in enumerate(videos):
        row = idx // nrow
        col = idx % nrow
        
        start_h = row * height
        end_h = start_h + height
        start_w = col * width
        end_w = start_w + width
        
        grid[:, :, start_h:end_h, start_w:end_w] = video
    
    return grid

def interpolate_videos(video1: torch.Tensor, video2: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """
    Create interpolation between two videos
    
    Args:
        video1 (torch.Tensor): First video
        video2 (torch.Tensor): Second video
        steps (int): Number of interpolation steps
        
    Returns:
        torch.Tensor: Interpolated videos
    """
    interpolated = []
    
    for i in range(steps):
        alpha = i / (steps - 1)
        interpolated_video = (1 - alpha) * video1 + alpha * video2
        interpolated.append(interpolated_video)
    
    return torch.stack(interpolated, dim=0)

def compute_video_metrics(pred_videos: torch.Tensor, target_videos: torch.Tensor) -> dict:
    """
    Compute metrics between predicted and target videos
    
    Args:
        pred_videos (torch.Tensor): Predicted videos
        target_videos (torch.Tensor): Target videos
        
    Returns:
        dict: Computed metrics
    """
    # Mean Squared Error
    mse = torch.mean((pred_videos - target_videos) ** 2)
    
    # Peak Signal-to-Noise Ratio
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(pred_videos - target_videos))
    
    return {
        'mse': mse.item(),
        'psnr': psnr.item(),
        'mae': mae.item()
    }

def get_device_info():
    """Get information about available devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    
    return info

def setup_logging_dirs(base_dir: str = "experiments"):
    """
    Setup logging directories with timestamp
    
    Args:
        base_dir (str): Base directory for experiments
        
    Returns:
        tuple: (log_dir, checkpoint_dir)
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    
    log_dir = os.path.join(exp_dir, "logs")
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return log_dir, checkpoint_dir

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, loss: float) -> bool:
        """
        Check if training should stop
        
        Args:
            loss (float): Current validation loss
            
        Returns:
            bool: True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def print_model_summary(model: torch.nn.Module, input_shape: Tuple[int, ...]):
    """
    Print a summary of the model
    
    Args:
        model (torch.nn.Module): The model to summarize
        input_shape (tuple): Input shape (without batch dimension)
    """
    from models import count_parameters
    
    total_params = count_parameters(model)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    print(f"Input shape: {input_shape}")
    print("=" * 60)

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test device info
    device_info = get_device_info()
    print("Device info:", device_info)
    
    # Test logging dirs
    log_dir, checkpoint_dir = setup_logging_dirs()
    print(f"Created logging directories: {log_dir}, {checkpoint_dir}")
    
    print("Utility functions test completed!")
