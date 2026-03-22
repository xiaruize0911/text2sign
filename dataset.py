"""
Dataset for loading text-GIF pairs for sign language generation
"""

import os
import glob
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
        split_mode: str = "signer_disjoint",
        random_seed: int = 42,
        tokenizer: Optional[any] = None,
        use_length_prefix: bool = False,
        cache_size: int = 2000,
    ):
        """
        Args:
            data_dir: Directory containing .gif and .txt files
            image_size: Size to resize frames to
            num_frames: Number of frames to sample from each GIF
            train: Whether this is training set
            train_ratio: Ratio of data to use for training
            tokenizer: Optional tokenizer instance to pre-tokenize text
            use_length_prefix: Whether to prepend word count to text
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_frames = num_frames
        self.train = train
        self.train_ratio = train_ratio
        self.split_mode = split_mode
        self.random_seed = random_seed
        self.tokenizer = tokenizer
        self.use_length_prefix = use_length_prefix
        self.cache_size = cache_size
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Find all pairs
        self.pairs = self._find_pairs()
        
        # Split into train/val
        self.indices, self.split_stats = self._build_split_indices()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        overlap = self.split_stats["overlap_signers"]
        print(
            f"Loaded {len(self.indices)} {'training' if train else 'validation'} samples "
            f"using {self.split_mode} split "
            f"({self.split_stats['num_train_signers']} train signers, "
            f"{self.split_stats['num_val_signers']} val signers, overlap={overlap})"
        )

    @staticmethod
    def count_words(text: str) -> int:
        return len(text.split()) if text else 0
    
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

    def _extract_signer_id(self, gif_path: str) -> str:
        """Extract a signer/video identifier from the file name."""
        stem = os.path.splitext(os.path.basename(gif_path))[0]
        return stem.split("_")[0]

    def _build_split_indices(self) -> Tuple[List[int], Dict[str, int]]:
        """Create either a random or signer-disjoint split."""
        rng = random.Random(self.random_seed)

        if self.split_mode == "random":
            indices = list(range(len(self.pairs)))
            rng.shuffle(indices)
            split_idx = int(len(indices) * self.train_ratio)
            selected = indices[:split_idx] if self.train else indices[split_idx:]
            return selected, {
                "num_train_signers": -1,
                "num_val_signers": -1,
                "overlap_signers": -1,
            }

        signer_to_indices: Dict[str, List[int]] = {}
        for idx, (gif_path, _) in enumerate(self.pairs):
            signer_id = self._extract_signer_id(gif_path)
            signer_to_indices.setdefault(signer_id, []).append(idx)

        signer_ids = list(signer_to_indices.keys())
        rng.shuffle(signer_ids)

        target_train_samples = max(1, int(len(self.pairs) * self.train_ratio))
        train_indices: List[int] = []
        val_indices: List[int] = []
        train_signers = set()
        val_signers = set()

        running_samples = 0
        for signer_id in signer_ids:
            signer_indices = signer_to_indices[signer_id]
            if running_samples < target_train_samples:
                train_indices.extend(signer_indices)
                train_signers.add(signer_id)
                running_samples += len(signer_indices)
            else:
                val_indices.extend(signer_indices)
                val_signers.add(signer_id)

        if not val_indices and train_indices:
            # Ensure a non-empty validation split by moving the last signer group.
            last_train_signer = next(reversed(list(train_signers)))
            moved = signer_to_indices[last_train_signer]
            train_indices = [idx for idx in train_indices if idx not in moved]
            val_indices.extend(moved)
            train_signers.remove(last_train_signer)
            val_signers.add(last_train_signer)

        selected = sorted(train_indices if self.train else val_indices)
        stats = {
            "num_train_signers": len(train_signers),
            "num_val_signers": len(val_signers),
            "overlap_signers": len(train_signers.intersection(val_signers)),
        }
        return selected, stats
    
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
                if self.train:
                    # Random sampling for training data variability
                    # This demonstrates robustness to frame rate variations
                    start_idx = random.randint(0, len(frames) - self.num_frames)
                    indices = np.arange(start_idx, start_idx + self.num_frames)
                else:
                    # Uniform sampling for validation/testing
                    indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
                
                frames = [frames[i] for i in indices]
            else:
                # Better padding: Symmetric padding or repeating last frame
                # Symmetric padding looks more natural for sign language
                while len(frames) < self.num_frames:
                    # Alternate between padding start and end for centering
                    if len(frames) % 2 == 0:
                        frames.append(frames[-1])
                    else:
                        frames.insert(0, frames[0])
            
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
        
        if real_idx in self.cache:
            return self.cache[real_idx]
            
        gif_path, txt_path = self.pairs[real_idx]
        
        video = self._load_gif(gif_path)  # (T, C, H, W)
        text = self._load_text(txt_path)
        
        # Add length prefix if enabled
        if self.use_length_prefix:
            word_count = len(text.split())
            # Normalize word count to bins if it's too high? 
            # For now, just use the raw count or a capped one
            safe_count = min(word_count, 30)
            text = f"[LEN_{safe_count}] {text}"
        
        sample = {
            "video": video,
            "text": text,
        }
        
        # Pre-tokenize if tokenizer is provided (saves time in forward pass)
        if self.tokenizer is not None:
            # Check if it's our SimpleTokenizer or a HuggingFace one
            # HF tokenizers usually return a dict when called, and have special methods
            if hasattr(self.tokenizer, 'encode') and not hasattr(self.tokenizer, 'model_max_length'):
                # SimpleTokenizer
                sample["tokens"] = self.tokenizer.encode(text)
            else:
                # CLIP Tokenizer (HuggingFace)
                encoded = self.tokenizer(
                    text,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                )
                sample["tokens"] = encoded["input_ids"].squeeze(0)
        
        # Cache if possible (limit to 2000 samples to avoid OOM)
        if len(self.cache) < self.cache_size:
            self.cache[real_idx] = sample
            
        return sample


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
    videos = torch.stack([item["video"] for item in batch])
    texts = [item["text"] for item in batch]
    
    # Use pre-tokenized tokens if available
    if "tokens" in batch[0]:
        # Safety check: convert to tensor if needed (e.g. if loaded from list-based cache)
        token_tensors = []
        for item in batch:
            t = item["tokens"]
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.long)
            token_tensors.append(t)
        tokens = torch.stack(token_tensors)
    else:
        # Fallback to character-level tokenization
        tokenizer = SimpleTokenizer()
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
    train_ratio: float = 0.9,
    split_mode: str = "signer_disjoint",
    random_seed: int = 42,
    tokenizer: Optional[any] = None,
    use_length_prefix: bool = False,
    short_text_max_words: Optional[int] = None,
    short_text_oversample_factor: float = 1.0,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create dataloader for training or validation"""
    
    dataset = SignLanguageDataset(
        data_dir=data_dir,
        image_size=image_size,
        num_frames=num_frames,
        train=train,
        train_ratio=train_ratio,
        split_mode=split_mode,
        random_seed=random_seed,
        tokenizer=tokenizer,
        use_length_prefix=use_length_prefix,
    )

    sampler = None
    use_short_text_oversampling = (
        train
        and short_text_max_words is not None
        and short_text_oversample_factor is not None
        and short_text_oversample_factor > 1.0
    )

    if use_short_text_oversampling:
        weights: List[float] = []
        num_short = 0
        for real_idx in dataset.indices:
            _, txt_path = dataset.pairs[real_idx]
            word_count = dataset.count_words(dataset._load_text(txt_path))
            is_short = word_count <= short_text_max_words
            if is_short:
                num_short += 1
            weights.append(float(short_text_oversample_factor if is_short else 1.0))

        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        print(
            f"Enabled short-text oversampling: <= {short_text_max_words} words, "
            f"factor={short_text_oversample_factor:.2f}, short_samples={num_short}/{len(weights)}"
        )
    
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": train and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "drop_last": train,
        "persistent_workers": (num_workers > 0) if persistent_workers is None else (persistent_workers and num_workers > 0),
    }

    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(**dataloader_kwargs)
    
    return dataloader


if __name__ == "__main__":
    # Test dataset
    dataset = SignLanguageDataset(
        data_dir="text_to_sign/training_data",
        image_size=64,
        num_frames=16,
        train=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Video shape: {sample['video'].shape}")
    print(f"Text: {sample['text']}")
