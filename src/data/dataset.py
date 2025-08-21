import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
import torchvision.transforms as transforms
from pathlib import Path
import re


class SignLanguageDataset(Dataset):
    """
    Dataset class for sign language video-text pairs.
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_frames: int = 28,
        frame_size: Tuple[int, int] = (128, 128),
        text_encoder = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing GIF and TXT files
            transform: Image transformations
            max_frames: Maximum number of frames to extract from each GIF
            frame_size: Target size for frames (height, width)
            text_encoder: Text encoder to preprocess text
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.text_encoder = text_encoder
        
        # Find all GIF files and their corresponding text files
        self.samples = self._find_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_dir}")
        
        print(f"Found {len(self.samples)} samples in dataset")
    
    def _find_samples(self) -> List[Tuple[Path, Path]]:
        """Find all valid GIF-text pairs."""
        samples = []
        
        for gif_path in self.data_dir.glob("*.gif"):
            # Find corresponding text file
            if "._" in gif_path.name:
                continue
            txt_path = gif_path.with_suffix(".txt")
            
            if txt_path.exists():
                samples.append((gif_path, txt_path))
        
        return samples
    
    def _center_crop(self, image: Image.Image, crop_size: int) -> Image.Image:
        """
        Apply center cropping to make image square.
        
        Args:
            image: PIL Image to crop
            crop_size: Size of the square crop
            
        Returns:
            Center-cropped square image
        """
        width, height = image.size
        
        # Calculate crop box for center crop
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        return image.crop((left, top, right, bottom))
    
    def _load_gif(self, gif_path: Path) -> torch.Tensor:
        """
        Load GIF and convert to tensor with center cropping.
        
        Args:
            gif_path: Path to GIF file
            
        Returns:
            Video tensor of shape (channels, frames, height, width)
        """
        try:
            # Open GIF
            gif = Image.open(gif_path)
            
            frames = []
            frame_idx = 0
            
            # Extract frames
            while len(frames) < self.max_frames:
                try:
                    # Seek to the current frame
                    gif.seek(frame_idx)
                    
                    # Convert to RGB if needed
                    frame = gif.convert('RGB')
                    
                    # Apply center cropping first
                    frame = self._center_crop(frame, min(frame.size))
                    
                    # Resize frame to target size
                    frame = frame.resize(self.frame_size, Image.Resampling.LANCZOS)
                    
                    # Convert to tensor
                    frame_tensor = transforms.ToTensor()(frame)
                    
                    # Apply transforms if provided
                    if self.transform:
                        frame_tensor = self.transform(frame_tensor)
                    
                    frames.append(frame_tensor)
                    frame_idx += 1
                    
                except EOFError:
                    # End of GIF - no more frames
                    break
                    
                except EOFError:
                    # End of GIF
                    break
            
            # If we have fewer frames than max_frames, repeat the last frame
            while len(frames) < self.max_frames:
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    # Create a black frame if no frames were loaded
                    black_frame = torch.zeros(3, self.frame_size[0], self.frame_size[1])
                    frames.append(black_frame)
            
            # Stack frames: (frames, channels, height, width)
            video_tensor = torch.stack(frames[:self.max_frames])
            
            # Rearrange to (channels, frames, height, width) for 3D convolution
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            
            return video_tensor
            
        except Exception as e:
            print(f"Error loading GIF {gif_path}: {e}")
            # Return a black video
            return torch.zeros(3, self.max_frames, self.frame_size[0], self.frame_size[1])
    
    def _load_text(self, txt_path: Path) -> str:
        """
        Load text from file.
        
        Args:
            txt_path: Path to text file
            
        Returns:
            Text string
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text
        except Exception as e:
            print(f"Error loading text {txt_path}: {e}")
            return ""
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing 'video', 'text', and 'text_embedding'
        """
        gif_path, txt_path = self.samples[idx]
        
        # Load video
        video = self._load_gif(gif_path)
        
        # Load text
        text = self._load_text(txt_path)
        
        sample = {
            'video': video,
            'text': text,
            'gif_path': str(gif_path),
            'txt_path': str(txt_path)
        }
        
        return sample


def get_default_transforms():
    """Get default image transforms for the dataset."""
    return transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for batching samples.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data
    """
    videos = torch.stack([sample['video'] for sample in batch])
    texts = [sample['text'] for sample in batch]
    gif_paths = [sample['gif_path'] for sample in batch]
    txt_paths = [sample['txt_path'] for sample in batch]
    
    return {
        'videos': videos,
        'texts': texts,
        'gif_paths': gif_paths,
        'txt_paths': txt_paths
    }


def create_dataloader(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    max_frames: int = 28,
    frame_size: Tuple[int, int] = (128, 128),
    transform: Optional[transforms.Compose] = None
) -> DataLoader:
    """
    Create a DataLoader for the sign language dataset.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        max_frames: Maximum number of frames per video
        frame_size: Target frame size (height, width)
        transform: Image transforms
        
    Returns:
        DataLoader
    """
    if transform is None:
        transform = get_default_transforms()
    
    dataset = SignLanguageDataset(
        data_dir=data_dir,
        transform=transform,
        max_frames=max_frames,
        frame_size=frame_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def test_dataset(data_dir: str):
    """Test the dataset loading."""
    print(f"Testing dataset with data directory: {data_dir}")
    
    # Create dataset
    transform = get_default_transforms()
    dataset = SignLanguageDataset(
        data_dir=data_dir,
        transform=transform,
        max_frames=28, 
        frame_size=(128, 128) 
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Video shape: {sample['video'].shape}")
        print(f"Text: {sample['text']}")
        print(f"GIF path: {sample['gif_path']}")
        print(f"TXT path: {sample['txt_path']}")
        
        # Test dataloader
        dataloader = create_dataloader(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            max_frames=28, 
            frame_size=(128, 128) 
        )
        
        # Get a batch
        batch = next(iter(dataloader))
        print(f"Batch videos shape: {batch['videos'].shape}")
        print(f"Batch texts: {batch['texts']}")
        
        print("Dataset test passed!")
    else:
        print("No samples found in dataset!")


if __name__ == "__main__":
    # Test with the actual data directory
    data_dir = "../../training_data"
    test_dataset(data_dir)
