"""
Dataset class for loading and preprocessing sign language video data
"""

import os
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Tuple, List
import glob
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class SignLanguageDataset(Dataset):
    """
    Dataset class for sign language video data
    
    Args:
        data_root (str): Root directory containing GIF and text files
        transform (callable, optional): Optional transform to be applied on a sample
    """
    
    def __init__(self, data_root: str, transform=None, num_frames: int = 16):
        self.data_root = data_root
        self.transform = transform
        self.num_frames = num_frames
        
        # Find all GIF files with progress bar
        print(f"Scanning for GIF files in {data_root}...")
        gif_pattern = os.path.join(data_root, "*.gif")
        self.gif_files = glob.glob(gif_pattern)
        self.gif_files.sort()  # Ensure consistent ordering
        
        # Validate that corresponding text files exist
        print("Validating dataset files...")
        valid_files = []
        for gif_file in tqdm(self.gif_files, desc="Validating files", unit="file"):
            text_file = gif_file.replace('.gif', '.txt')
            if os.path.exists(text_file):
                valid_files.append(gif_file)
            else:
                print(f"Warning: Missing text file for {gif_file}")
        
        self.gif_files = valid_files
        print(f"Found {len(self.gif_files)} valid GIF-text pairs in {data_root}")
        
    def __len__(self) -> int:
        return len(self.gif_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (video_tensor, text) where video_tensor has shape (frames, channels, height, width)
        """
        gif_path = self.gif_files[idx]
        
        # Load corresponding text file
        text_path = gif_path.replace('.gif', '.txt')
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Load GIF frames
        try:
            frames = imageio.mimread(gif_path)
            if not frames:
                raise ValueError("No frames found in GIF")
            frames = np.array(frames)  # Shape: (num_frames, height, width, channels)
            # Ensure RGBA channels
            if frames.ndim == 3:
                frames = np.expand_dims(frames, axis=-1)
            if frames.shape[-1] == 1:
                frames = np.repeat(frames, 4, axis=-1)
            elif frames.shape[-1] == 3:
                alpha_channel = np.full((*frames.shape[:-1], 1), 255, dtype=frames.dtype)
                frames = np.concatenate([frames, alpha_channel], axis=-1)
            elif frames.shape[-1] > 4:
                frames = frames[..., :4]
            # Convert to torch tensor and normalize to [0, 1]
            frames = torch.from_numpy(frames).float() / 255.0
            # Normalize to [-1, 1] for diffusion models
            frames = frames * 2 - 1
            
            # Ensure we have exactly the specified number of frames (pad or truncate)
            num_frames = frames.shape[0]
            if num_frames < self.num_frames:
                # Pad with the last frame
                padding = self.num_frames - num_frames
                last_frame = frames[-1:].repeat(padding, 1, 1, 1)
                frames = torch.cat([frames, last_frame], dim=0)
            elif num_frames > self.num_frames:
                # Truncate to first num_frames frames
                frames = frames[:self.num_frames]
            
            # Apply transforms if provided (expects frames, channels, height, width)
            if self.transform:
                frames = self.transform(frames)
            
            # Final rearrangement to (channels, frames, height, width) for the model
            frames = frames.permute(3, 0, 1, 2)
            # Sanity check: data should be in [-1, 1]
            if frames.numel() > 0:
                fmin, fmax = frames.min().item(), frames.max().item()
                assert -1.0001 <= fmin <= 1.0001 and -1.0001 <= fmax <= 1.0001, \
                    f"Frames not normalized to [-1,1]: min={fmin}, max={fmax}"
                
            return frames, text
            
        except Exception as e:
            logger.error(f"Error loading {gif_path}: {e}")
            # Return a dummy tensor if loading fails (channels, frames, height, width)
            from config import Config
            dummy_frames = torch.zeros(*Config.INPUT_SHAPE)
            return dummy_frames, "error"

class CenterCropTransform:
    """
    Center crop transform for video tensors
    
    Args:
        size (int): Target size for center crop
    """
    
    def __init__(self, size: int = 128):
        self.size = size
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply center crop to video frames
        
        Args:
            frames (torch.Tensor): Input frames with shape (num_frames, height, width, channels)
            
        Returns:
            torch.Tensor: Center cropped frames with shape (num_frames, size, size, channels)
        """
        # Get current dimensions
        _, h, w, _ = frames.shape
        
        # Calculate crop boundaries
        crop_h = min(h, self.size)
        crop_w = min(w, self.size)
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        # Apply center crop
        cropped_frames = frames[:, start_h:start_h + crop_h, start_w:start_w + crop_w, :]
        
        # If the cropped size is smaller than target, pad with zeros
        if crop_h < self.size or crop_w < self.size:
            pad_h = max(0, self.size - crop_h)
            pad_w = max(0, self.size - crop_w)
            
            # Calculate padding
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            cropped_frames = torch.nn.functional.pad(
                cropped_frames, 
                (0, 0, pad_left, pad_right, pad_top, pad_bottom),
                mode='constant', 
                value=0
            )
        
        return cropped_frames

def create_dataloader(data_root: str, batch_size: int, num_workers: int = 2, shuffle: bool = True, num_frames: int = 16) -> DataLoader:
    """
    Create a DataLoader for the sign language dataset
    
    Args:
        data_root (str): Root directory containing the data
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader instance
        
    Raises:
        ValueError: If data_root doesn't exist or batch_size is invalid
    """
    # Input validation
    if not isinstance(data_root, str) or not data_root.strip():
        raise ValueError("data_root must be a non-empty string")
    
    if not os.path.exists(data_root):
        raise ValueError(f"Data directory does not exist: {data_root}")
    
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    
    # Define transforms
    from config import Config
    transform = CenterCropTransform(size=Config.IMAGE_SIZE)
    
    # Create dataset
    dataset = SignLanguageDataset(data_root=data_root, transform=transform, num_frames=num_frames)
    
    # Create dataloader with deterministic behavior
    if shuffle:
        # Use a configurable seed for shuffling to ensure reproducibility
        torch.manual_seed(42)  # TODO: Make this configurable via Config
    # Memory-efficient dataloader settings
    pin_memory = getattr(Config, 'PIN_MEMORY', False)  # Configurable pin memory
    prefetch_factor = getattr(Config, 'PREFETCH_FACTOR', 1)  # Reduced prefetch factor
    
    # Configure DataLoader parameters based on multiprocessing usage
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': True,  # Drop last incomplete batch
    }
    
    # Only add multiprocessing-specific parameters when num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
        dataloader_kwargs['persistent_workers'] = True
    
    dataloader = DataLoader(**dataloader_kwargs)
    
    return dataloader

def test_dataloader():
    """Test function to verify the dataloader works correctly"""
    from config import Config
    
    print("Testing dataloader...")
    dataloader = create_dataloader(
        data_root=Config.DATA_ROOT,
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        shuffle=False
    )
    
    for batch_idx, (videos, texts) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Video shape: {videos.shape}")
        print(f"  Video range: [{videos.min():.3f}, {videos.max():.3f}]")
        print(f"  Texts: {texts}")
        
        if batch_idx >= 2:  # Only test first few batches
            break
    
    print("Dataloader test completed successfully!")

if __name__ == "__main__":
    test_dataloader()
