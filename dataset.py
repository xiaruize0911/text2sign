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

class SignLanguageDataset(Dataset):
    """
    Dataset class for sign language video data
    
    Args:
        data_root (str): Root directory containing GIF and text files
        transform (callable, optional): Optional transform to be applied on a sample
    """
    
    def __init__(self, data_root: str, transform=None):
        self.data_root = data_root
        self.transform = transform
        
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
            frames = np.array(frames)  # Shape: (num_frames, height, width, channels)
            
            # Convert to torch tensor and normalize to [0, 1]
            frames = torch.from_numpy(frames).float() / 255.0
            
            # Ensure we have exactly 28 frames (pad or truncate)
            num_frames = frames.shape[0]
            if num_frames < 28:
                # Pad with the last frame
                padding = 28 - num_frames
                last_frame = frames[-1:].repeat(padding, 1, 1, 1)
                frames = torch.cat([frames, last_frame], dim=0)
            elif num_frames > 28:
                # Truncate to first 28 frames
                frames = frames[:28]
            
            # Apply transforms if provided (expects frames, channels, height, width)
            if self.transform:
                frames = self.transform(frames)
            
            # Final rearrangement to (channels, frames, height, width) for the model
            frames = frames.permute(3, 0, 1, 2)
                
            return frames, text
            
        except Exception as e:
            print(f"Error loading {gif_path}: {e}")
            # Return a dummy tensor if loading fails (channels, frames, height, width)
            dummy_frames = torch.zeros(3, 28, 128, 128)
            return dummy_frames, ""

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

def create_dataloader(data_root: str, batch_size: int, num_workers: int = 2, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for the sign language dataset
    
    Args:
        data_root (str): Root directory containing the data
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader instance
    """
    # Define transforms
    transform = CenterCropTransform(size=128)
    
    # Create dataset
    dataset = SignLanguageDataset(data_root=data_root, transform=transform)
    
    # Create dataloader with deterministic behavior
    if shuffle:
        # Use a fixed seed for shuffling to ensure reproducibility
        torch.manual_seed(42)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch
    )
    
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
