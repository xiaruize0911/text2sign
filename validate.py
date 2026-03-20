"""
Validation and Evaluation Script for Text-to-Sign Diffusion Model

Metrics included:
1. FVD (Fréchet Video Distance) - Video quality and distribution similarity
2. Motion Pattern Analysis - Compares motion dynamics (signer-agnostic)
3. Temporal Consistency - Motion smoothness across frames
4. Generation Diversity - Variety in outputs for different prompts
5. Visual Quality Metrics - FID-style feature distance

Note on Sign Language Validation:
- SSIM/PSNR are NOT ideal for sign language because different signers
  look different but may sign correctly. A generated video of a different
  "virtual signer" will have low SSIM even if the sign is correct.
- Motion-based metrics are better as they focus on gesture patterns.
- The ideal validation would use a sign language recognition model.
"""

import os
import argparse
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
try:
    from torchvision.models.video import R3D_18_Weights
except ImportError:
    R3D_18_Weights = None

DEFAULT_FVD_BACKBONE = "videomae"
DEFAULT_VIDEOMAE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"

# Local imports
from config import ModelConfig, DDIMConfig, TrainingConfig
from dataset import SignLanguageDataset, get_dataloader, collate_fn
from models import UNet3D, TextEncoder
from schedulers import DDIMScheduler
from pipeline import Text2SignPipeline


# ============================================================================
# Motion-Based Metrics (Signer-Agnostic)
# ============================================================================

def extract_motion_features(video: torch.Tensor) -> torch.Tensor:
    """
    Extract motion features from video using optical flow approximation.
    This is signer-agnostic - focuses on how things move, not what they look like.
    
    Args:
        video: (B, C, T, H, W) or (C, T, H, W) tensor
    
    Returns:
        Motion feature tensor
    """
    if video.dim() == 4:
        video = video.unsqueeze(0)
    
    B, C, T, H, W = video.shape
    
    # Convert to grayscale for motion analysis
    if C == 3:
        gray = 0.299 * video[:, 0] + 0.587 * video[:, 1] + 0.114 * video[:, 2]
    else:
        gray = video[:, 0]
    
    # Calculate frame differences (temporal gradient)
    frame_diffs = gray[:, 1:] - gray[:, :-1]  # (B, T-1, H, W)
    
    # Motion magnitude per frame
    motion_magnitude = torch.abs(frame_diffs).mean(dim=(2, 3))  # (B, T-1)
    
    # Spatial gradient (where motion occurs)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=video.dtype, device=video.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=video.dtype, device=video.device).view(1, 1, 3, 3)
    
    # Apply to each frame diff
    spatial_features = []
    for t in range(frame_diffs.shape[1]):
        frame = frame_diffs[:, t:t+1]  # (B, 1, H, W)
        gx = F.conv2d(frame, sobel_x, padding=1)
        gy = F.conv2d(frame, sobel_y, padding=1)
        gradient_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
        spatial_features.append(gradient_mag.mean(dim=(2, 3)))
    
    spatial_grad = torch.cat(spatial_features, dim=1)  # (B, T-1)
    
    # Combine features
    motion_features = {
        'magnitude_mean': motion_magnitude.mean(dim=1),
        'magnitude_std': motion_magnitude.std(dim=1),
        'magnitude_max': motion_magnitude.max(dim=1)[0],
        'spatial_grad_mean': spatial_grad.mean(dim=1),
        'temporal_pattern': motion_magnitude,  # Full temporal pattern
    }
    
    return motion_features


def calculate_motion_similarity(video1: torch.Tensor, video2: torch.Tensor) -> Dict[str, float]:
    """
    Calculate motion pattern similarity between two videos.
    This is more appropriate for sign language than pixel-level metrics.
    
    Returns similarity scores (higher = more similar motion patterns)
    """
    feat1 = extract_motion_features(video1)
    feat2 = extract_motion_features(video2)
    
    # Compare motion magnitude patterns
    mag1, mag2 = feat1['temporal_pattern'], feat2['temporal_pattern']
    
    # Normalize to same length if needed
    min_len = min(mag1.shape[1], mag2.shape[1])
    mag1, mag2 = mag1[:, :min_len], mag2[:, :min_len]
    
    # Cosine similarity of temporal patterns
    mag1_norm = F.normalize(mag1, dim=1)
    mag2_norm = F.normalize(mag2, dim=1)
    temporal_similarity = (mag1_norm * mag2_norm).sum(dim=1).mean().item()
    
    # Compare motion statistics
    stat_diff = (
        abs(feat1['magnitude_mean'] - feat2['magnitude_mean']).mean().item() +
        abs(feat1['magnitude_std'] - feat2['magnitude_std']).mean().item()
    )
    stat_similarity = 1.0 / (1.0 + stat_diff)
    
    return {
        'temporal_similarity': temporal_similarity,
        'stat_similarity': stat_similarity,
        'combined': (temporal_similarity + stat_similarity) / 2
    }


def calculate_motion_realism(videos: torch.Tensor, reference_videos: torch.Tensor) -> Dict[str, float]:
    """
    Check if generated motion patterns are realistic compared to real sign language.
    
    This compares the distribution of motion characteristics, not individual videos.
    """
    gen_features = extract_motion_features(videos)
    real_features = extract_motion_features(reference_videos)
    
    # Compare distributions of motion statistics
    results = {}
    
    # Motion magnitude distribution
    gen_mag_mean = gen_features['magnitude_mean'].mean().item()
    real_mag_mean = real_features['magnitude_mean'].mean().item()
    gen_mag_std = gen_features['magnitude_std'].mean().item()
    real_mag_std = real_features['magnitude_std'].mean().item()
    
    results['magnitude_ratio'] = gen_mag_mean / (real_mag_mean + 1e-8)
    results['variance_ratio'] = gen_mag_std / (real_mag_std + 1e-8)
    
    # Ideal ratios should be close to 1.0
    results['motion_realism_score'] = 1.0 / (1.0 + abs(results['magnitude_ratio'] - 1.0) + abs(results['variance_ratio'] - 1.0))
    
    return results


# ============================================================================
# Original Metric Functions
# ============================================================================

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Higher is better (max 1.0).
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Ensure inputs are 4D (B, C, H, W)
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    # Ensure same device
    img2 = img2.to(img1.device)
    
    # Get number of channels from the actual input after dimension check
    b, c, h, w = img1.shape
    
    # Create gaussian window
    def gaussian_window(size, num_channels, device, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_1d = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
        return window_1d.expand(num_channels, 1, size, size).contiguous()
    
    window = gaussian_window(window_size, c, img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=c)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=c)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size//2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size//2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=c) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    Higher is better.
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)).item()


def calculate_temporal_consistency(video: torch.Tensor) -> float:
    """
    Calculate temporal consistency by measuring frame-to-frame differences.
    Lower difference = more consistent (smoother motion).
    Returns a score where higher is better.
    """
    # video: (C, T, H, W)
    if video.dim() == 4:
        video = video.unsqueeze(0)  # Add batch dim
    
    # Calculate optical flow proxy (frame differences)
    frame_diffs = []
    for t in range(video.shape[2] - 1):
        diff = torch.abs(video[:, :, t+1] - video[:, :, t]).mean()
        frame_diffs.append(diff.item())
    
    # Lower variance in frame differences = more consistent motion
    variance = np.var(frame_diffs)
    mean_diff = np.mean(frame_diffs)
    
    # Convert to a score (higher is better)
    consistency_score = 1.0 / (1.0 + variance)
    
    return consistency_score, mean_diff


class VideoFeatureExtractor(nn.Module):
    """Extract video-level features for FVD-style evaluation.

    Uses a VideoMAE backbone when available and falls back to R3D-18.
    """

    def __init__(
        self,
        backbone: str = DEFAULT_FVD_BACKBONE,
        videomae_model_name: str = DEFAULT_VIDEOMAE_MODEL,
        allow_fallback: bool = True,
    ):
        super().__init__()
        self.input_size = 224
        self.backend = backbone
        self.videomae_model_name = videomae_model_name

        if backbone == "videomae":
            try:
                from transformers import VideoMAEImageProcessor, VideoMAEModel

                self.processor = VideoMAEImageProcessor.from_pretrained(videomae_model_name)
                self.features = VideoMAEModel.from_pretrained(videomae_model_name).eval()

                processor_size = getattr(self.processor, "size", None)
                if isinstance(processor_size, dict):
                    self.input_size = int(
                        processor_size.get("shortest_edge")
                        or processor_size.get("height")
                        or processor_size.get("width")
                        or 224
                    )

                self.register_buffer(
                    "mean",
                    torch.tensor(self.processor.image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
                )
                self.register_buffer(
                    "std",
                    torch.tensor(self.processor.image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
                )
                self.backend = f"videomae:{videomae_model_name}"
            except Exception as exc:
                if not allow_fallback:
                    raise
                print(f"VideoMAE backbone unavailable ({exc}); falling back to R3D-18 for FVD.")
                self._init_r3d()
        elif backbone == "r3d_18":
            self._init_r3d()
        else:
            raise ValueError(f"Unsupported FVD backbone '{backbone}'")

        for param in self.parameters():
            param.requires_grad = False

    def _init_r3d(self):
        if R3D_18_Weights is not None:
            backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        else:
            backbone = r3d_18(pretrained=True)
        backbone.fc = nn.Identity()
        self.features = backbone.eval()
        self.register_buffer("mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1))
        self.backend = "r3d_18"
        self.input_size = 112

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        if self.backend.startswith("videomae"):
            # VideoMAE expects (B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            bsz, frames, channels, height, width = x.shape
            if (height, width) != (self.input_size, self.input_size):
                x = x.reshape(bsz * frames, channels, height, width)
                x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
                x = x.view(bsz, frames, channels, self.input_size, self.input_size)
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            outputs = self.features(pixel_values=x)
            if getattr(outputs, "pooler_output", None) is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state.mean(dim=1)

        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(x.shape[2], self.input_size, self.input_size), mode='trilinear', align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.features(x)


def calculate_fvd(
    real_videos: torch.Tensor,
    fake_videos: torch.Tensor,
    device: str = 'cuda',
    backbone: str = DEFAULT_FVD_BACKBONE,
    feature_extractor: Optional[nn.Module] = None,
) -> float:
    """
    Calculate Fréchet Video Distance (FVD).
    Lower is better.
    
    Args:
        real_videos: (N, C, T, H, W) real video tensor
        fake_videos: (N, C, T, H, W) generated video tensor
    """
    if feature_extractor is None:
        feature_extractor = VideoFeatureExtractor(backbone=backbone).to(device)
        feature_extractor.eval()
    
    def get_video_features(videos: torch.Tensor) -> np.ndarray:
        """Extract pooled video features in batches."""
        all_features = []
        batch_size = min(8, max(1, videos.shape[0]))
        
        with torch.no_grad():
            for start in range(0, videos.shape[0], batch_size):
                batch = videos[start:start + batch_size].to(device)
                feats = feature_extractor(batch)
                all_features.append(feats.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    # Extract features
    real_features = get_video_features(real_videos)
    fake_features = get_video_features(fake_videos)
    
    # Calculate FVD (Fréchet distance)
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Handle 1D case
    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
        sigma_fake = np.array([[sigma_fake]])
    
    # Calculate FVD
    diff = mu_real - mu_fake
    
    try:
        from scipy import linalg
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fvd = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    except:
        # Simplified calculation without scipy
        fvd = np.sum(diff ** 2) + np.trace(sigma_real) + np.trace(sigma_fake)
    
    return float(fvd)


# ============================================================================
# Validation Class
# ============================================================================

class ModelValidator:
    """Comprehensive validation for text-to-sign diffusion model"""
    
    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        device: str = 'cuda',
        num_samples: int = 50,
        benchmark_repeats: int = 5,
        enable_backtranslation: bool = True,
        fvd_backbone: str = DEFAULT_FVD_BACKBONE,
    ):
        self.device = device
        self.num_samples = num_samples
        self.data_dir = data_dir
        self.benchmark_repeats = benchmark_repeats
        self.enable_backtranslation = enable_backtranslation
        self.project_root = Path(__file__).resolve().parents[1]
        self.artifact_dir = Path(__file__).resolve().parent
        self.fvd_backbone = fvd_backbone
        
        # Load configs
        self.model_config = ModelConfig()
        self.ddim_config = DDIMConfig()
        self.train_config = TrainingConfig()
        
        # Load model
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.pipeline = self._load_model(checkpoint_path)
        
        # Store dataset params for creating fresh dataloaders
        print(f"Loading dataset from {data_dir}...")
        self._dataloader_params = {
            'data_dir': data_dir,
            'batch_size': 4,
            'image_size': self.model_config.image_size,
            'num_frames': self.model_config.num_frames,
            'train': False,
            'train_ratio': self.train_config.train_ratio,
            'split_mode': self.train_config.split_mode,
            'random_seed': self.train_config.split_seed,
        }
        
        # Create initial dataloader to verify dataset loads
        test_loader = self._get_fresh_dataloader()
        print(f"Loaded {len(test_loader.dataset)} validation samples")
        
        self.results = {}

    def _save_artifact(self, filename: str, payload: Dict) -> Path:
        """Persist JSON artifacts for benchmarking/back-translation and validation."""
        path = self.artifact_dir / filename
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def _hardware_info(self) -> Dict[str, Optional[str]]:
        device_name = None
        if isinstance(self.device, str) and self.device == "cuda" and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        elif isinstance(self.device, torch.device) and self.device.type == "cuda" and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(self.device)
        return {
            "device": str(self.device),
            "device_name": device_name,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
        }

    def _glofe_paths(self) -> Optional[Dict[str, Path]]:
        """Return default GloFE resource paths if they exist."""
        glofe_root = self.project_root / "GloFE"
        weights = glofe_root / "pretrained_weights/how2sign/vn_model/glofe_vn_how2sign_0224.pt"
        config = glofe_root / "pretrained_weights/how2sign/vn_model/exp_config.json"
        tokenizer = glofe_root / "notebooks/how2sign/how2sign-bpe25000-tokenizer-uncased"
        if all(path.exists() for path in (glofe_root, weights, config, tokenizer)):
            return {
                "root": glofe_root,
                "weights": weights,
                "config": config,
                "tokenizer": tokenizer,
            }
        return None

    def _tokenize_text(self, text: str) -> List[str]:
        return [token for token in text.lower().replace("_", " ").split() if token]

    def _sentence_bleu(self, reference: str, prediction: str, max_n: int = 4) -> float:
        """Compute a lightweight BLEU-style score without extra dependencies."""
        ref_tokens = self._tokenize_text(reference)
        pred_tokens = self._tokenize_text(prediction)
        if not pred_tokens:
            return 0.0

        clipped_precisions = []
        for n in range(1, max_n + 1):
            if len(pred_tokens) < n:
                clipped_precisions.append(1e-8)
                continue
            ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(max(len(ref_tokens) - n + 1, 0)))
            pred_ngrams = Counter(tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))
            overlap = sum(min(count, ref_ngrams[ngram]) for ngram, count in pred_ngrams.items())
            total = max(sum(pred_ngrams.values()), 1)
            clipped_precisions.append(max(overlap / total, 1e-8))

        ref_len = len(ref_tokens)
        pred_len = len(pred_tokens)
        brevity_penalty = 1.0 if pred_len > ref_len else np.exp(1 - ref_len / max(pred_len, 1))
        geo_mean = np.exp(np.mean(np.log(clipped_precisions)))
        return float(brevity_penalty * geo_mean)

    def _score_translation(self, reference: str, prediction: str) -> Dict[str, float]:
        ref_tokens = self._tokenize_text(reference)
        pred_tokens = self._tokenize_text(prediction)
        ref_set = set(ref_tokens)
        pred_set = set(pred_tokens)
        overlap = len(ref_set & pred_set)
        precision = overlap / max(len(pred_set), 1)
        recall = overlap / max(len(ref_set), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return {
            "exact_match": float(reference.strip().lower() == prediction.strip().lower()),
            "bleu": self._sentence_bleu(reference, prediction),
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
            "sequence_similarity": SequenceMatcher(None, reference.lower(), prediction.lower()).ratio(),
        }

    def _save_video_file(self, video: torch.Tensor, path: str, fps: int = 8):
        """Save a generated tensor video as MP4 for external translators."""
        import cv2

        video = video.detach().cpu().clamp(0, 1)
        _, _, height, width = video.shape
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for t in range(video.shape[1]):
            frame = video[:, t].permute(1, 2, 0).numpy()
            frame = (frame * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    def _run_glofe_translation(self, video_path: str) -> Optional[str]:
        """Call the existing GloFE CLI as a subprocess to avoid module-name conflicts."""
        paths = self._glofe_paths()
        if paths is None:
            return None

        cmd = [
            sys.executable,
            str(paths["root"] / "inference_glofe.py"),
            "--weights", str(paths["weights"]),
            "--config", str(paths["config"]),
            "--tokenizer", str(paths["tokenizer"]),
            "--pose_file", video_path,
            "--device", self.device,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(paths["root"]))
        if proc.returncode != 0:
            print(f"GloFE back-translation failed for {video_path}: {proc.stderr.strip()}")
            return None

        for line in proc.stdout.splitlines():
            if "📝 Translation:" in line:
                return line.split("📝 Translation:", 1)[1].strip()
        return None
    
    def _get_fresh_dataloader(self):
        """Create a fresh dataloader instance (DataLoaders are single-use iterators)"""
        return get_dataloader(**self._dataloader_params)
    
    def _load_model(self, checkpoint_path: str) -> Text2SignPipeline:
        """Load model from checkpoint using the pipeline's from_pretrained method"""
        # Load checkpoint first to get configs for logging
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Use config from checkpoint if available
        if 'model_config' in checkpoint:
            self.model_config = checkpoint['model_config']
            print(f"  Using model config from checkpoint")
        
        print(f"  Loaded epoch {checkpoint.get('epoch', 'unknown')}, step {checkpoint.get('global_step', 'unknown')}")
        del checkpoint
        if isinstance(self.device, str) and self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use the pipeline's from_pretrained method which handles all model creation
        # This ensures the model architecture matches the checkpoint
        pipeline = Text2SignPipeline.from_pretrained(checkpoint_path, device=self.device)
        
        return pipeline
    
    def evaluate_reconstruction(self) -> Dict[str, float]:
        """
        Evaluate how well the model can reconstruct training data.
        Uses SSIM and PSNR metrics.
        """
        print("\n" + "="*60)
        print("Evaluating Reconstruction Quality...")
        print("="*60)
        
        ssim_scores = []
        psnr_scores = []
        
        val_loader = self._get_fresh_dataloader()
        num_batches = max(1, min(self.num_samples // 4, len(val_loader)))
        
        for i, batch in enumerate(tqdm(val_loader, total=num_batches, desc="Reconstruction")):
            if i >= num_batches:
                break
            
            videos = batch['video'].to(self.device)  # (B, T, C, H, W) from dataset
            texts = batch['text']
            
            # Generate videos from the same text prompts (request tensor output)
            with torch.no_grad():
                generated = self.pipeline(
                    texts,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    output_type="tensor",
                )  # (B, C, T, H, W) from pipeline
            
            # Ensure generated is on the same device
            if isinstance(generated, torch.Tensor):
                generated = generated.to(self.device)
            
            # Calculate metrics for each video pair
            for j in range(min(len(videos), len(generated))):
                # Convert real video from (T, C, H, W) to (C, T, H, W) to match generated
                real_video = videos[j].permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
                # Denormalize real video from [-1, 1] to [0, 1]
                real_video = (real_video + 1) / 2
                fake_video = generated[j]  # (C, T, H, W) already in [0, 1]
                
                # Calculate per-frame metrics and average
                frame_ssims = []
                frame_psnrs = []
                
                for t in range(real_video.shape[1]):
                    real_frame = real_video[:, t].unsqueeze(0)
                    fake_frame = fake_video[:, t].unsqueeze(0)
                    
                    frame_ssims.append(calculate_ssim(real_frame, fake_frame))
                    frame_psnrs.append(calculate_psnr(real_frame, fake_frame))
                
                ssim_scores.append(np.mean(frame_ssims))
                psnr_scores.append(np.mean(frame_psnrs))
        
        # Handle empty results
        if ssim_scores:
            results = {
                'ssim_mean': np.mean(ssim_scores),
                'ssim_std': np.std(ssim_scores),
                'psnr_mean': np.mean(psnr_scores),
                'psnr_std': np.std(psnr_scores),
            }
        else:
            results = {
                'ssim_mean': 0.0,
                'ssim_std': 0.0,
                'psnr_mean': 0.0,
                'psnr_std': 0.0,
            }
            print("\nWarning: No reconstruction samples processed")
        
        print(f"\nReconstruction Results:")
        print(f"  SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
        print(f"  PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
        
        self.results['reconstruction'] = results
        return results
    
    def evaluate_temporal_consistency(self) -> Dict[str, float]:
        """
        Evaluate temporal consistency of generated videos.
        """
        print("\n" + "="*60)
        print("Evaluating Temporal Consistency...")
        print("="*60)
        
        consistency_scores = []
        motion_scores = []
        
        # Sample prompts for generation
        sample_prompts = ["Hello", "Thank you", "Please", "Sorry", "Help", 
                         "Yes", "No", "Good", "Bad", "Love"]
        
        for prompt in tqdm(sample_prompts, desc="Temporal Consistency"):
            with torch.no_grad():
                videos = self.pipeline(
                    [prompt],
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    output_type="tensor",
                )
            
            video = videos[0]  # (C, T, H, W)
            consistency, motion = calculate_temporal_consistency(video)
            consistency_scores.append(consistency)
            motion_scores.append(motion)
        
        results = {
            'consistency_mean': np.mean(consistency_scores),
            'consistency_std': np.std(consistency_scores),
            'motion_mean': np.mean(motion_scores),
            'motion_std': np.std(motion_scores),
        }
        
        print(f"\nTemporal Consistency Results:")
        print(f"  Consistency Score: {results['consistency_mean']:.4f} ± {results['consistency_std']:.4f}")
        print(f"  Motion Magnitude: {results['motion_mean']:.4f} ± {results['motion_std']:.4f}")
        
        self.results['temporal'] = results
        return results
    
    def evaluate_fvd(self) -> Dict[str, float]:
        """
        Calculate Fréchet Video Distance between real and generated videos.
        """
        print("\n" + "="*60)
        print("Calculating Fréchet Video Distance (FVD)...")
        print("="*60)
        
        real_videos = []
        fake_videos = []
        texts = []
        
        val_loader = self._get_fresh_dataloader()
        num_batches = max(1, min(self.num_samples // 4, len(val_loader)))
        
        for i, batch in enumerate(tqdm(val_loader, total=num_batches, desc="Collecting videos")):
            if i >= num_batches:
                break
            
            # Dataset returns (B, T, C, H, W), convert to (B, C, T, H, W)
            video = batch['video'].permute(0, 2, 1, 3, 4)
            # Denormalize from [-1, 1] to [0, 1]
            video = (video + 1) / 2
            real_videos.append(video)
            texts.extend(batch['text'])
        
        if not real_videos:
            print("\nWarning: No real videos collected for FVD calculation")
            self.results['fvd'] = {'fvd': float('inf')}
            return self.results['fvd']
            
        real_videos = torch.cat(real_videos, dim=0)[:self.num_samples]
        texts = texts[:self.num_samples]
        
        # Generate videos
        print("Generating videos...")
        for i in tqdm(range(0, len(texts), 4), desc="Generating"):
            batch_texts = texts[i:i+4]
            with torch.no_grad():
                generated = self.pipeline(
                    batch_texts,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    output_type="tensor",
                )
            fake_videos.append(generated.cpu())
        
        # Check if we have any videos
        if not fake_videos:
            print("\nWarning: No videos generated for FVD calculation")
            self.results['fvd'] = {'fvd': float('inf')}
            return self.results['fvd']
        
        fake_videos = torch.cat(fake_videos, dim=0)
        
        # Calculate FVD
        fvd = calculate_fvd(
            real_videos,
            fake_videos,
            self.device,
            backbone=self.fvd_backbone,
        )
        
        results = {
            'fvd': fvd,
            'backbone': self.fvd_backbone,
        }
        
        print(f"\nFVD Results:")
        print(f"  FVD Score: {fvd:.2f}")
        print(f"  (Lower is better, <100 is good, <50 is very good)")
        
        self.results['fvd'] = results
        return results
    
    def evaluate_diversity(self) -> Dict[str, float]:
        """
        Evaluate diversity of generated videos for the same prompt.
        """
        print("\n" + "="*60)
        print("Evaluating Generation Diversity...")
        print("="*60)
        
        prompt = "Hello"
        num_generations = 10
        
        videos = []
        for _ in tqdm(range(num_generations), desc="Generating diverse samples"):
            with torch.no_grad():
                video = self.pipeline(
                    [prompt],
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    output_type="tensor",
                )
            videos.append(video[0])
        
        videos = torch.stack(videos)  # (N, C, T, H, W)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(videos)):
            for j in range(i + 1, len(videos)):
                dist = F.mse_loss(videos[i], videos[j]).item()
                distances.append(dist)
        
        results = {
            'diversity_mean': np.mean(distances),
            'diversity_std': np.std(distances),
        }
        
        print(f"\nDiversity Results:")
        print(f"  Mean Pairwise Distance: {results['diversity_mean']:.4f} ± {results['diversity_std']:.4f}")
        print(f"  (Higher distance = more diverse generations)")
        
        self.results['diversity'] = results
        return results

    def benchmark_inference(self, prompt: str = "Hello world") -> Dict[str, float]:
        """Benchmark inference latency and throughput on the current hardware."""
        print("\n" + "="*60)
        print("Benchmarking Inference...")
        print("="*60)

        num_inference_steps = 50
        guidance_scale = 7.5
        results = self.pipeline.benchmark(
            prompt=prompt,
            repeats=self.benchmark_repeats,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        results.update({
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **self._hardware_info(),
        })

        print(f"\nInference Benchmark:")
        print(f"  Mean clip latency: {results['latency_mean_sec']:.3f} s ± {results['latency_std_sec']:.3f}")
        print(f"  Effective frame throughput: {results['frames_per_second']:.2f} FPS")
        if results['peak_memory_gb'] is not None:
            print(f"  Peak GPU memory: {results['peak_memory_gb']:.2f} GB")

        self.results['benchmark'] = results
        artifact_path = self._save_artifact('benchmark_results.json', results)
        print(f"  Saved benchmark artifact: {artifact_path}")
        return results

    def evaluate_backtranslation(self, max_samples: Optional[int] = None) -> Dict[str, float]:
        """Evaluate semantic faithfulness through GloFE back-translation."""
        print("\n" + "="*60)
        print("Evaluating Back-Translation Faithfulness...")
        print("="*60)

        if not self.enable_backtranslation:
            print("Back-translation disabled by configuration.")
            return {}

        if self._glofe_paths() is None:
            print("GloFE resources not found; skipping back-translation evaluation.")
            return {}

        max_samples = max_samples or min(12, self.num_samples)
        val_loader = self._get_fresh_dataloader()
        texts: List[str] = []
        for batch in val_loader:
            texts.extend(batch['text'])
            if len(texts) >= max_samples:
                break
        texts = texts[:max_samples]

        metrics = []
        predictions = []
        with tempfile.TemporaryDirectory(prefix="text2sign_bt_") as tmp_dir:
            for idx, text in enumerate(tqdm(texts, desc="Back-translation")):
                with torch.no_grad():
                    video = self.pipeline(
                        [text],
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        output_type="tensor",
                    )[0]
                video_path = os.path.join(tmp_dir, f"sample_{idx}.mp4")
                self._save_video_file(video, video_path)
                prediction = self._run_glofe_translation(video_path) or ""
                predictions.append(prediction)
                metrics.append(self._score_translation(text, prediction))

        if not metrics:
            return {}

        results = {
            key: float(np.mean([metric[key] for metric in metrics]))
            for key in metrics[0].keys()
        }
        results["num_samples"] = len(metrics)
        results["backbone"] = "GloFE"
        results["device"] = str(self.device)
        results["examples"] = [
            {"reference": ref, "prediction": pred}
            for ref, pred in list(zip(texts, predictions))[:5]
        ]

        print(f"\nBack-Translation Results:")
        print(f"  Exact match: {results['exact_match']:.4f}")
        print(f"  BLEU: {results['bleu']:.4f}")
        print(f"  Token F1: {results['token_f1']:.4f}")
        print(f"  Sequence similarity: {results['sequence_similarity']:.4f}")

        self.results['backtranslation'] = results
        artifact_path = self._save_artifact('backtranslation_results.json', results)
        print(f"  Saved back-translation artifact: {artifact_path}")
        return results
    
    def compare_with_training_data(self, output_dir: str = "validation_output"):
        """
        Generate side-by-side comparisons with training data.
        """
        print("\n" + "="*60)
        print("Generating Visual Comparisons...")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get some samples from validation set (use fresh dataloader)
        val_loader = self._get_fresh_dataloader()
        batch = next(iter(val_loader))
        real_videos = batch['video'][:4]
        texts = batch['text'][:4]
        
        # Generate videos
        with torch.no_grad():
            generated = self.pipeline(
                texts,
                num_inference_steps=50,
                guidance_scale=7.5,
                output_type="tensor",
            )
        
        # Create comparison figure
        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
        
        # Real videos from dataset: (B, T, C, H, W)
        # Generated videos from pipeline: (B, C, T, H, W)
        num_frames_real = real_videos.shape[1]  # T is dim 1 for real
        num_frames_gen = generated.shape[2]  # T is dim 2 for generated
        
        num_samples = min(4, real_videos.shape[0], generated.shape[0])
        
        for i in range(num_samples):
            # Real video frames (first 4 frames)
            # real_videos shape: (B, T, C, H, W)
            for j in range(4):
                frame_idx = min(j * max(1, num_frames_real // 4), num_frames_real - 1)
                # real_videos[i] is (T, C, H, W), get frame at time t: [t, :, :, :]
                frame = real_videos[i, frame_idx, :, :, :].permute(1, 2, 0).cpu().numpy()
                frame = (frame + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
                frame = np.clip(frame, 0, 1)
                axes[i, j].imshow(frame)
                axes[i, j].axis('off')
                if i == 0:
                    axes[i, j].set_title(f'Real F{frame_idx}')
            
            # Generated video frames
            # generated shape: (B, C, T, H, W)
            for j in range(4):
                frame_idx = min(j * max(1, num_frames_gen // 4), num_frames_gen - 1)
                # generated[i] is (C, T, H, W), get frame at time t: [:, t, :, :]
                frame = generated[i, :, frame_idx, :, :].permute(1, 2, 0).cpu().numpy()
                frame = np.clip(frame, 0, 1)
                axes[i, j+4].imshow(frame)
                axes[i, j+4].axis('off')
                if i == 0:
                    axes[i, j+4].set_title(f'Gen F{frame_idx}')
            
            # Add text label
            axes[i, 0].set_ylabel(texts[i][:20], fontsize=10)
        
        plt.suptitle('Real (Left) vs Generated (Right) Videos', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
        plt.close()
        
        print(f"Comparison saved to {output_dir}/comparison.png")
        
        # Save individual GIFs
        for i in range(min(num_samples, len(texts))):
            # Save real video - convert from (T, C, H, W) to (C, T, H, W)
            real_vid = real_videos[i].permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
            # Denormalize from [-1, 1] to [0, 1]
            real_vid = (real_vid + 1) / 2
            self._save_gif(
                real_vid,
                os.path.join(output_dir, f'{texts[i]}_real.gif')
            )
            # Save generated video - already (C, T, H, W)
            self._save_gif(
                generated[i],
                os.path.join(output_dir, f'{texts[i]}_generated.gif')
            )
        
        print(f"GIFs saved to {output_dir}/")
    
    def evaluate_motion_realism(self) -> Dict[str, float]:
        """
        Evaluate if generated motion patterns are realistic compared to real sign language.
        This is signer-agnostic - compares motion dynamics, not appearance.
        """
        print("\n" + "="*60)
        print("Evaluating Motion Realism (Signer-Agnostic)...")
        print("="*60)
        
        # Collect real videos
        val_loader = self._get_fresh_dataloader()
        real_videos = []
        texts = []
        
        num_batches = max(1, min(self.num_samples // 4, len(val_loader)))
        
        for i, batch in enumerate(tqdm(val_loader, total=num_batches, desc="Collecting real videos")):
            if i >= num_batches:
                break
            # Dataset returns (B, T, C, H, W), convert to (B, C, T, H, W)
            video = batch['video'].permute(0, 2, 1, 3, 4)
            video = (video + 1) / 2  # Denormalize
            real_videos.append(video)
            texts.extend(batch['text'])
        
        if not real_videos:
            print("Warning: No real videos collected")
            return {}
        
        real_videos = torch.cat(real_videos, dim=0)[:self.num_samples]
        texts = texts[:self.num_samples]
        
        # Generate videos for the same prompts
        print("Generating videos...")
        gen_videos = []
        for i in tqdm(range(0, len(texts), 4), desc="Generating"):
            batch_texts = texts[i:i+4]
            with torch.no_grad():
                generated = self.pipeline(
                    batch_texts,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    output_type="tensor",
                )
            gen_videos.append(generated.cpu())
        
        gen_videos = torch.cat(gen_videos, dim=0)
        
        # Calculate motion realism
        motion_results = calculate_motion_realism(gen_videos, real_videos)
        
        # Also calculate motion similarity for each pair
        similarities = []
        for i in range(min(len(gen_videos), len(real_videos))):
            sim = calculate_motion_similarity(
                gen_videos[i:i+1], 
                real_videos[i:i+1]
            )
            similarities.append(sim['combined'])
        
        results = {
            **motion_results,
            'motion_similarity_mean': np.mean(similarities),
            'motion_similarity_std': np.std(similarities),
        }
        
        print(f"\nMotion Realism Results:")
        print(f"  Motion Realism Score: {results['motion_realism_score']:.4f} (1.0 = perfect)")
        print(f"  Motion Magnitude Ratio: {results['magnitude_ratio']:.2f} (1.0 = same as real)")
        print(f"  Motion Variance Ratio: {results['variance_ratio']:.2f} (1.0 = same as real)")
        print(f"  Motion Similarity: {results['motion_similarity_mean']:.4f} ± {results['motion_similarity_std']:.4f}")
        print(f"\n  Note: These metrics compare motion patterns, not appearance.")
        print(f"  They are signer-agnostic and better suited for sign language.")
        
        self.results['motion_realism'] = results
        return results
    
    def _save_gif(self, video: torch.Tensor, path: str, fps: int = 8):
        """Save video tensor as GIF"""
        frames = []
        for t in range(video.shape[1]):
            frame = video[:, t].permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))
        
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0,
        )
    
    def run_full_validation(self, output_dir: str = "validation_output") -> Dict:
        """Run all validation metrics"""
        print("\n" + "="*60)
        print("RUNNING FULL MODEL VALIDATION")
        print("="*60)
        
        # Run all evaluations
        self.evaluate_reconstruction()
        self.evaluate_temporal_consistency()
        self.evaluate_diversity()
        self.benchmark_inference()
        
        # Motion-based evaluation (signer-agnostic, better for sign language)
        try:
            self.evaluate_motion_realism()
        except Exception as e:
            print(f"Motion realism evaluation failed: {e}")

        try:
            self.evaluate_backtranslation()
        except Exception as e:
            print(f"Back-translation evaluation failed: {e}")
        
        # FVD is computationally expensive, make it optional
        try:
            self.evaluate_fvd()
        except Exception as e:
            print(f"FVD calculation failed: {e}")
        
        # Generate visual comparisons
        self.compare_with_training_data(output_dir)
        
        # Save results
        results_path = os.path.join(output_dir, 'validation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        self._save_artifact('validation_results.json', self.results)
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        if 'reconstruction' in self.results:
            r = self.results['reconstruction']
            print(f"\n📊 Reconstruction Quality (Pixel-Based):")
            print(f"   SSIM: {r['ssim_mean']:.4f} (1.0 = perfect match)")
            print(f"   PSNR: {r['psnr_mean']:.2f} dB (>20 dB is good)")
            print(f"   ⚠️  Note: Low scores are expected when comparing different signers!")
        
        if 'motion_realism' in self.results:
            m = self.results['motion_realism']
            print(f"\n🤟 Motion Realism (Signer-Agnostic, RECOMMENDED):")
            print(f"   Realism Score: {m['motion_realism_score']:.4f} (1.0 = realistic motion)")
            print(f"   Motion Similarity: {m['motion_similarity_mean']:.4f}")
            print(f"   ✓ This metric focuses on gesture dynamics, not appearance")
        
        if 'temporal' in self.results:
            t = self.results['temporal']
            print(f"\n🎬 Temporal Consistency:")
            print(f"   Score: {t['consistency_mean']:.4f} (higher = smoother)")
        
        if 'diversity' in self.results:
            d = self.results['diversity']
            print(f"\n🎨 Generation Diversity:")
            print(f"   Variance: {d['diversity_mean']:.4f}")
        
        if 'fvd' in self.results:
            print(f"\n📈 Fréchet Video Distance:")
            print(f"   FVD: {self.results['fvd']['fvd']:.2f} (lower = better)")

        if 'benchmark' in self.results:
            b = self.results['benchmark']
            print(f"\n⚙️  Runtime on current hardware:")
            print(f"   Clip latency: {b['clip_latency_sec']:.3f} s")
            print(f"   Effective FPS: {b['frames_per_second']:.2f}")

        if 'backtranslation' in self.results:
            bt = self.results['backtranslation']
            print(f"\n🔁 Back-translation faithfulness:")
            print(f"   Exact match: {bt['exact_match']:.4f}")
            print(f"   BLEU: {bt['bleu']:.4f}")
            print(f"   Token F1: {bt['token_f1']:.4f}")
        
        print(f"\n" + "="*60)
        print("INTERPRETATION GUIDE")
        print("="*60)
        print("""
For sign language generation with multiple signers in training data:

✅ BEST METRICS (signer-agnostic):
   - Motion Realism Score: Compares motion patterns, not appearance
   - FVD: Compares video distributions using deep features
   - Temporal Consistency: Measures smoothness

⚠️  LIMITED METRICS (signer-dependent):  
   - SSIM/PSNR: Will be low even for correct signs if signer differs
   - These compare pixels, so different people = low scores

🎯 IDEAL VALIDATION (future work):
   - Use a Sign Language Recognition model to verify sign correctness
   - Human evaluation by sign language experts
""")
        
        print(f"\n✅ Results saved to {results_path}")
        print(f"📁 Visual comparisons in {output_dir}/")
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description="Validate Text-to-Sign Diffusion Model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='/teamspace/studios/this_studio/text_to_sign/training_data',
                       help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='validation_output',
                       help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--benchmark-repeats', type=int, default=5,
                       help='Number of repeated runs for latency benchmarking')
    parser.add_argument('--fvd-backbone', type=str, default=DEFAULT_FVD_BACKBONE,
                       choices=['videomae', 'r3d_18'],
                       help='Video backbone used for FVD feature extraction')
    parser.add_argument('--skip-backtranslation', action='store_true',
                       help='Skip GloFE back-translation evaluation')
    
    args = parser.parse_args()
    
    validator = ModelValidator(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
        num_samples=args.num_samples,
        benchmark_repeats=args.benchmark_repeats,
        enable_backtranslation=not args.skip_backtranslation,
        fvd_backbone=args.fvd_backbone,
    )
    
    validator.run_full_validation(args.output_dir)


if __name__ == "__main__":
    main()
