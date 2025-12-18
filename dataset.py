"""
Dataset for loading text-GIF pairs for sign language generation
"""

import os
import glob
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class SignLanguageDataset(Dataset):
    """Dataset for text-to-sign language video generation"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 64,
        num_frames: int = 16,
        train: bool = True,
        train_ratio: float = 0.9,
    ):
        """
        Args:
            data_dir: Directory containing .gif and .txt files
            image_size: Size to resize frames to
            num_frames: Number of frames to sample from each GIF
            train: Whether this is training set
            train_ratio: Ratio of data to use for training
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_frames = num_frames
        self.train = train
        
        # Find all pairs
        self.pairs = self._find_pairs()
        
        # Split into train/val
        random.seed(42)
        indices = list(range(len(self.pairs)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        
        if train:
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        print(f"Loaded {len(self.indices)} {'training' if train else 'validation'} samples")
    
    def _find_pairs(self) -> List[Tuple[str, str]]:
        """Find all GIF-text pairs in the data directory"""
        pairs = []
        
        # Find all GIF files
        gif_files = glob.glob(os.path.join(self.data_dir, "*.gif"))
        
        for gif_path in gif_files:
            # Find corresponding text file
            txt_path = gif_path.replace(".gif", ".txt")
            
            if os.path.exists(txt_path):
                pairs.append((gif_path, txt_path))
        
        return pairs
    
    def _load_gif(self, gif_path: str) -> torch.Tensor:
        """Load GIF and sample frames"""
        try:
            gif = Image.open(gif_path)
            
            # Get all frames
            frames = []
            try:
                while True:
                    # Convert to RGB
                    frame = gif.convert("RGB")
                    frame = self.transform(frame)
                    frames.append(frame)
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
            
            if len(frames) == 0:
                raise ValueError(f"No frames found in {gif_path}")
            
            # Sample or pad frames
            if len(frames) >= self.num_frames:
                # Uniform sampling
                indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
                frames = [frames[i] for i in indices]
            else:
                # Pad by repeating last frame
                while len(frames) < self.num_frames:
                    frames.append(frames[-1])
            
            # Stack frames: (num_frames, C, H, W)
            video = torch.stack(frames)
            
            return video
            
        except Exception as e:
            print(f"Error loading {gif_path}: {e}")
            # Return random noise as fallback
            return torch.randn(self.num_frames, 3, self.image_size, self.image_size)
    
    def _load_text(self, txt_path: str) -> str:
        """Load text from file"""
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            return text
        except Exception as e:
            print(f"Error loading {txt_path}: {e}")
            return ""
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        gif_path, txt_path = self.pairs[real_idx]
        
        video = self._load_gif(gif_path)  # (T, C, H, W)
        text = self._load_text(txt_path)
        
        return {
            "video": video,
            "text": text,
        }


class SimpleTokenizer:
    """Simple tokenizer for text encoding"""
    
    def __init__(self, vocab_size: int = 49408, max_length: int = 77):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Simple character-level tokenization with hash
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs"""
        # Simple hash-based encoding
        tokens = [self.bos_token_id]
        
        for char in text.lower():
            # Hash character to token ID
            token_id = (ord(char) % (self.vocab_size - 3)) + 3
            tokens.append(token_id)
            
            if len(tokens) >= self.max_length - 1:
                break
        
        tokens.append(self.eos_token_id)
        
        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.pad_token_id)
        
        return torch.tensor(tokens[:self.max_length], dtype=torch.long)
    
    def __call__(self, texts: List[str]) -> torch.Tensor:
        """Batch encode texts"""
        return torch.stack([self.encode(text) for text in texts])


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    tokenizer = SimpleTokenizer()
    
    videos = torch.stack([item["video"] for item in batch])
    texts = [item["text"] for item in batch]
    tokens = tokenizer(texts)
    
    return {
        "video": videos,  # (B, T, C, H, W)
        "tokens": tokens,  # (B, max_length)
        "text": texts,  # List of strings
    }


def get_dataloader(
    data_dir: str,
    batch_size: int = 4,
    image_size: int = 64,
    num_frames: int = 16,
    num_workers: int = 4,
    train: bool = True,
) -> DataLoader:
    """Create dataloader for training or validation"""
    
    dataset = SignLanguageDataset(
        data_dir=data_dir,
        image_size=image_size,
        num_frames=num_frames,
        train=train,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=train,
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset
    dataset = SignLanguageDataset(
        data_dir="text2sign/training_data",
        image_size=64,
        num_frames=16,
        train=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Video shape: {sample['video'].shape}")
    print(f"Text: {sample['text']}")
