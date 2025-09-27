"""
DiT3D: Diffusion Transformer for 3D Video Generation
Based on Facebook Research's DiT architecture, adapted for video diffusion.

This module implements the DiT (Diffusion Transformers) architecture adapted for 3D video data.
Key adaptations include:
- Pretrained DiT backbone with video-specific adaptations
- 3D patch embedding for video tokens
- Temporal-aware attention mechanisms
- Video-specific positional embeddings
- Efficient video processing pipeline with pretrained components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple
import requests
import os
from pathlib import Path


def modulate(x, shift, scale):
    """
    Apply FiLM-style modulation to features
    Args:
        x: Input features
        shift: Shift parameter
        scale: Scale parameter
    Returns:
        Modulated features
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Helper to choose a valid group count for GroupNorm
def _get_group_count(channels: int) -> int:
    """Return the largest group count <=8 that divides the number of channels"""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


#################################################################################
#                            Pretrained DiT Backbone                           #
#################################################################################

class DiTBackboneExtractor(nn.Module):
    """
    Pretrained DiT-XL-2-256 backbone extractor for video processing
    Uses DiT-XL-2-256 from HuggingFace for high-quality features
    """
    def __init__(self, 
                 freeze_backbone: bool = True,
                 input_size: int = 256,
                 patch_size: int = 2):
        super().__init__()
        
        self.model_size = "DiT-XL-2-256"  # Fixed to HuggingFace pretrained model
        self.input_size = input_size
        self.patch_size = patch_size
        self.freeze_backbone = freeze_backbone
        
        # Create base DiT model (we'll load pretrained weights automatically)
        self.dit_config = self._get_dit_config("DiT-XL-2-256")
        self.backbone = self._create_dit_backbone()
        
        # Load pretrained weights automatically
        self._load_pretrained_weights()
        
        # Extract useful attributes
        self.embed_dim = self.dit_config['hidden_size']
        self.num_patches = (input_size // patch_size) ** 2
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"🔒 DiT-XL-2-256 backbone frozen ({sum(p.numel() for p in self.backbone.parameters()):,} parameters)")
        else:
            print(f"🔓 DiT-XL-2-256 backbone trainable ({sum(p.numel() for p in self.backbone.parameters() if p.requires_grad):,} parameters)")
                
    def _get_dit_config(self, model_size: str) -> dict:
        """Get configuration for DiT-XL-2-256 (pretrained model from HuggingFace)"""
        # Use DiT-XL-2-256 configuration from HuggingFace
        config = {"hidden_size": 1152, "depth": 28, "num_heads": 16, "patch_size": 2}
        return config
    
    def _create_dit_backbone(self):
        """Create DiT backbone without final layers"""
        config = self.dit_config
        
        # Create backbone components as a simple class
        class DiTBackbone(nn.Module):
            def __init__(self, config, input_size):
                super().__init__()
                # Patch embedding (2D for now, we'll extend to 3D)
                self.patch_embed = nn.Conv2d(3, config['hidden_size'], 
                                           kernel_size=config['patch_size'], 
                                           stride=config['patch_size'])
                
                # Positional embedding
                num_patches = (input_size // config['patch_size']) ** 2
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config['hidden_size']))
                
                # Transformer blocks
                self.blocks = nn.ModuleList([
                    DiTBlock(config['hidden_size'], config['num_heads'], mlp_ratio=4.0)
                    for _ in range(config['depth'])
                ])
                
                # Time embedder
                self.t_embedder = TimestepEmbedder(config['hidden_size'])
                
                # Initialize positional embeddings
                nn.init.normal_(self.pos_embed, std=0.02)
        
        return DiTBackbone(config, self.input_size)
    
    def _load_pretrained_weights(self):
        """Load pretrained DiT-XL-2-256 weights from HuggingFace"""
        try:
            
            
            # Try to import huggingface_hub for better downloading
            try:
                from huggingface_hub import hf_hub_download
                
                # Download DiT-XL-2-256 model file from HuggingFace
                model_path = hf_hub_download(
                    repo_id="facebook/DiT-XL-2-256",
                    filename="transformer/diffusion_pytorch_model.bin",
                    cache_dir=Path.home() / ".cache" / "huggingface" / "dit_models"
                )
                print(f"✅ Downloaded DiT-XL-2-256 from HuggingFace to {model_path}")
                
            except ImportError:
                print("⚠️  huggingface_hub not available, skipping pretrained weights")
                return
            
            except Exception as e:
                print(f"⚠️  HuggingFace download failed: {e}")
                return
            
            # Load safetensors format

                print(f"� Downloading DiT-S/2 weights from {model_url}")

            
            # Load pytorch model file
            checkpoint = torch.load(model_path, map_location='cpu')
            print("📦 Loaded model using torch.load")

            
            # Load compatible weights into our backbone
            model_state = self.backbone.state_dict()
            loaded_keys = []
            incompatible_keys = []
            
            for key in model_state.keys():
                if key in checkpoint:
                    if model_state[key].shape == checkpoint[key].shape:
                        model_state[key] = checkpoint[key]
                        loaded_keys.append(key)
                    else:
                        incompatible_keys.append(f"{key}: {model_state[key].shape} vs {checkpoint[key].shape}")
                else:
                    # Try to find similar keys (handle potential naming differences)
                    found_similar = False
                    for ckpt_key in checkpoint.keys():
                        if key.split('.')[-1] == ckpt_key.split('.')[-1]:  # Same parameter name
                            if model_state[key].shape == checkpoint[ckpt_key].shape:
                                model_state[key] = checkpoint[ckpt_key]
                                loaded_keys.append(f"{key} <- {ckpt_key}")
                                found_similar = True
                                break
                    
                    if not found_similar:
                        incompatible_keys.append(f"{key}: not found in checkpoint")
            
            self.backbone.load_state_dict(model_state, strict=False)
            
            print(f"✅ Loaded {len(loaded_keys)} pretrained weights into DiT backbone")
            if incompatible_keys:
                print(f"⚠️  {len(incompatible_keys)} incompatible keys (using random initialization)")
                for key in incompatible_keys[:5]:  # Show first 5 for brevity
                    print(f"   - {key}")
                if len(incompatible_keys) > 5:
                    print(f"   ... and {len(incompatible_keys) - 5} more")
            
            print(f"🔒 Freeze backbone: {self.freeze_backbone}")
            
        except Exception as e:
            print(f"⚠️  Failed to load pretrained weights: {e}")
            print("🔧 Continuing with random initialization")
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, return_features: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from DiT backbone
        Args:
            x: (batch*frames, channels, height, width) - Input frames
            t: (batch*frames,) - Timesteps for each frame
            return_features: Whether to return intermediate features
        Returns:
            global_features: (batch*frames, embed_dim) - Global features
            patch_features: (batch*frames, num_patches, embed_dim) - Patch features
        """
        # Resize to expected input size if needed
        if x.shape[-2:] != (self.input_size, self.input_size):
            target_size = (self.input_size, self.input_size)
            antialias = x.shape[-2] > target_size[0] or x.shape[-1] > target_size[1]
            x = F.interpolate(
                x,
                size=target_size,
                mode="bicubic",
                align_corners=False,
                antialias=antialias,
            )
        
        # Patch embedding
        x = self.backbone.patch_embed(x)  # (batch*frames, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  # (batch*frames, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.backbone.pos_embed
        
        # Time embedding
        t_emb = self.backbone.t_embedder(t)  # (batch*frames, embed_dim)
        
        # Apply transformer blocks
        for block in self.backbone.blocks:
            x = block(x, t_emb)  # (batch*frames, num_patches, embed_dim)
        
        # Extract global and patch features
        global_features = x.mean(dim=1)  # Global average pooling (batch*frames, embed_dim)
        patch_features = x if return_features else None  # (batch*frames, num_patches, embed_dim)
        
        return global_features, patch_features


#################################################################################
#                         Video-Specific Adapter Layers                        #
#################################################################################

class VideoTemporalProcessor(nn.Module):
    """Process temporal information across video frames"""
    def __init__(self, feature_dim: int, num_frames: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        # Temporal attention for cross-frame modeling
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )
        
        # Temporal positional encoding
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, feature_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal processing
        Args:
            x: (batch, frames, feature_dim)
        Returns:
            x: (batch, frames, feature_dim)
        """
        # Add temporal positional encoding
        x = x + self.temporal_pos_embed
        
        # Temporal self-attention
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class VideoFeatureUpsampler(nn.Module):
    """Upsample DiT features to video format"""
    def __init__(self, feature_dim: int, out_channels: int, target_size: Tuple[int, int]):
        super().__init__()
        self.target_size = target_size
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        
        # Progressive upsampling pathway
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, out_channels * 64)  # 8x8 base resolution
        )
        
        # Helper function to get safe group count
        def get_group_count(channels):
            for groups in [8, 4, 2, 1]:
                if channels % groups == 0:
                    return groups
            return 1
        
        # Convolutional upsampling layers
        self.conv_layers = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(self.out_channels, self.out_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(get_group_count(self.out_channels * 2), self.out_channels * 2),
            nn.GELU(),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(self.out_channels * 2, self.out_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(get_group_count(self.out_channels * 2), self.out_channels * 2),
            nn.GELU(),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(self.out_channels * 2, self.out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(get_group_count(self.out_channels), self.out_channels),
            nn.GELU(),

            # 64x64 -> target_size (adaptive)
            nn.ConvTranspose2d(self.out_channels, self.out_channels, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, features_flat: torch.Tensor) -> torch.Tensor:
        """
        Upsample flat features to spatial frames
        Args:
            features_flat: (batch*frames, feature_dim)
        Returns:
            x: (batch*frames, out_channels, target_h, target_w)
        """
        # Initial projection to low-res features (8x8)
        x = self.feature_proj(features_flat)  # (B*F, out_channels*64)
        x = x.view(-1, self.out_channels, 8, 8)
        
        # Progressive conv transpose upsampling
        x = self.conv_layers(x)
        
        # Ensure final size
        if x.shape[-2:] != self.target_size:
            antialias = x.shape[-2] > self.target_size[0] or x.shape[-1] > self.target_size[1]
            x = F.interpolate(
                x,
                size=self.target_size,
                mode="bicubic",
                align_corners=False,
                antialias=antialias,
            )
        
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Based on the original DiT implementation with sinusoidal embeddings.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Based on the original DiT implementation.
    """
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class TextEmbedder(nn.Module):
    """
    Embeds text features into vector representations for video conditioning.
    Replaces or augments the LabelEmbedder for text-to-video tasks.
    """
    def __init__(self, text_dim: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(text_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size)
        )
        self.dropout_prob = dropout_prob
        
    def forward(self, text_features, train=True):
        """
        Args:
            text_features: (batch, text_dim) - Text features from encoder
            train: Whether in training mode
        Returns:
            text_embeddings: (batch, hidden_size)
        """
        if train and self.dropout_prob > 0:
            # Apply dropout for classifier-free guidance during training
            drop_mask = torch.rand(text_features.shape[0], device=text_features.device) < self.dropout_prob
            text_features = text_features.clone()
            text_features[drop_mask] = 0  # Zero out for unconditional generation
            
        return self.projection(text_features)


#################################################################################
#                                 3D Patch Embedding                           #
#################################################################################

class PatchEmbed3D(nn.Module):
    """
    3D Video to Patch Embedding for DiT3D
    Handles video data (batch, channels, frames, height, width) -> (batch, num_patches, embed_dim)
    """
    def __init__(self, video_size=(16, 224, 224), patch_size=(2, 16, 16), in_channels=3, embed_dim=768, bias=True):
        super().__init__()
        self.video_size = video_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_frames_patches = video_size[0] // patch_size[0]
        self.num_spatial_patches = (video_size[1] // patch_size[1]) * (video_size[2] // patch_size[2])
        self.num_patches = self.num_frames_patches * self.num_spatial_patches
        
        # 3D convolution for patch embedding
        self.proj = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            bias=bias
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, frames, height, width)
        Returns:
            patches: (batch, num_patches, embed_dim)
        """
        B, C, F, H, W = x.shape
        
        # Ensure input size matches expected video size
        assert F == self.video_size[0] and H == self.video_size[1] and W == self.video_size[2], \
            f"Input size {(F, H, W)} doesn't match model video size {self.video_size}"
        
        # Apply 3D convolution to create patches
        x = self.proj(x)  # (batch, embed_dim, F//pF, H//pH, W//pW)
        
        # Flatten spatial and temporal dimensions
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Based on the original DiT implementation with video-specific enhancements.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Args:
            x: (batch, num_patches, hidden_size) - Input tokens
            c: (batch, hidden_size) - Conditioning vector (time + text/class)
        Returns:
            x: (batch, num_patches, hidden_size) - Output tokens
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), 
                                                   modulate(self.norm1(x), shift_msa, scale_msa), 
                                                   modulate(self.norm1(x), shift_msa, scale_msa))[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT3D.
    """
    def __init__(self, hidden_size: int, patch_size: Tuple[int, int, int], out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.linear = nn.Linear(hidden_size, patch_volume * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Args:
            x: (batch, num_patches, hidden_size)
            c: (batch, hidden_size) - Conditioning vector
        Returns:
            x: (batch, num_patches, patch_volume * out_channels)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_3d_sincos_pos_embed(embed_dim, grid_size_3d, cls_token=False, extra_tokens=0):
    """
    Create 3D sine-cosine positional embeddings for video data.
    
    Args:
        embed_dim: embedding dimension
        grid_size_3d: (frames, height, width) of the 3D grid
        cls_token: whether to include class token
        extra_tokens: number of extra tokens
    
    Returns:
        pos_embed: (frames*height*width, embed_dim) or (1+frames*height*width, embed_dim) with cls_token
    """
    frames, height, width = grid_size_3d
    
    # Split embedding dimension across the three dimensions
    # We'll use more dimensions for spatial than temporal
    embed_dim_t = embed_dim // 4  # temporal gets 1/4
    embed_dim_h = embed_dim_w = (embed_dim - embed_dim_t) // 2  # spatial gets rest split
    
    # Create grids
    grid_f = np.arange(frames, dtype=np.float32)
    grid_h = np.arange(height, dtype=np.float32)
    grid_w = np.arange(width, dtype=np.float32)
    
    # Get 1D embeddings for each dimension
    emb_f = get_1d_sincos_pos_embed_from_grid(embed_dim_t, grid_f)  # (F, embed_dim_t)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_h, grid_h)  # (H, embed_dim_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_w, grid_w)  # (W, embed_dim_w)
    
    # Create meshgrid and flatten
    grid = np.meshgrid(grid_f, grid_h, grid_w, indexing='ij')  # (3, F, H, W)
    grid = np.stack(grid, axis=0)  # (3, F, H, W)
    
    # Get positional embeddings for each position
    pos_embed = np.zeros((frames * height * width, embed_dim))
    
    idx = 0
    for f in range(frames):
        for h in range(height):
            for w in range(width):
                pos_embed[idx] = np.concatenate([emb_f[f], emb_h[h], emb_w[w]])
                idx += 1
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT3D Model                                 #
#################################################################################

class DiT3D(nn.Module):
    """
    Diffusion Transformer for 3D Video Generation (DiT3D) with Pretrained Backbone
    
    Based on the DiT architecture from Facebook Research, adapted for video diffusion.
    Uses pretrained DiT components wrapped with video-specific layers.
    
    Key features:
    - Pretrained DiT backbone with video adaptations
    - Frame-wise processing with temporal modeling
    - Video-specific positional embeddings
    - Text conditioning support
    - Scalable transformer architecture
    """
    
    def __init__(
        self,
        video_size: Tuple[int, int, int] = (16, 224, 224),  # (frames, height, width)
        patch_size: Tuple[int, int, int] = (2, 16, 16),      # (pF, pH, pW) - only spatial used for DiT
        in_channels: int = 3,
        out_channels: int = 3,
        freeze_dit_backbone: bool = True,  # Option to freeze pretrained backbone
        text_dim: Optional[int] = None,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        num_temporal_heads: int = 8,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = out_channels * 2 if learn_sigma else out_channels
        self.video_size = video_size
        self.frames, self.height, self.width = video_size
        self.spatial_patch_size = patch_size[1]  # Use spatial patch size for DiT
        
        # Pretrained DiT-S/2 backbone for spatial feature extraction (fixed to smallest model)
        self.dit_backbone = DiTBackboneExtractor(
            freeze_backbone=freeze_dit_backbone,
            input_size=max(self.height, self.width),  # Use larger dimension
            patch_size=self.spatial_patch_size
        )
        
        self.feature_dim = self.dit_backbone.embed_dim
        
        # Text conditioning (replaces or augments class conditioning)
        self.use_text_conditioning = text_dim is not None
        if self.use_text_conditioning:
            self.text_embedder = TextEmbedder(text_dim, self.feature_dim, class_dropout_prob)
        else:
            self.class_embedder = LabelEmbedder(num_classes, self.feature_dim, class_dropout_prob)
        
        # Video-specific temporal processing
        self.temporal_processor = VideoTemporalProcessor(
            feature_dim=self.feature_dim,
            num_frames=self.frames,
            num_heads=num_temporal_heads
        )
        
        # FiLM layers for conditioning integration
        self.conditioning_layers = nn.ModuleList([
            FiLMLayer(self.feature_dim, self.feature_dim) for _ in range(3)
        ])
        
        # Feature modulation for final processing
        self.feature_modulation = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Video upsampler to reconstruct spatial resolution
        self.video_upsampler = VideoFeatureUpsampler(
            feature_dim=self.feature_dim,
            out_channels=self.out_channels,
            target_size=(self.height, self.width)
        )
        
        # Final temporal processing with 3D convolutions
        self.final_temporal_conv = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(_get_group_count(self.out_channels * 2), self.out_channels * 2),
            nn.GELU(),
            nn.Conv3d(self.out_channels * 2, self.out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize video-specific components (DiT backbone uses pretrained weights)"""
        # Initialize temporal processor
        for layer in self.temporal_processor.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Initialize temporal positional embeddings
        nn.init.normal_(self.temporal_processor.temporal_pos_embed, std=0.02)
        
        # Initialize conditioning layers
        for film_layer in self.conditioning_layers:
            for layer in film_layer.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Initialize text embedder if used
        if self.use_text_conditioning:
            for layer in self.text_embedder.projection.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
        
        # Initialize feature modulation
        for layer in self.feature_modulation.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Initialize upsampler
        for layer in self.video_upsampler.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Initialize final temporal convolution
        for layer in self.final_temporal_conv.modules():
            if isinstance(layer, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: Optional[torch.Tensor] = None):
        """
        Forward pass with pretrained DiT backbone and video-specific processing
        
        Args:
            x: (batch, in_channels, frames, height, width) - Input video
            t: (batch,) - Diffusion timesteps
            conditioning: (batch, text_dim) for text or (batch,) for class labels
        
        Returns:
            output: (batch, out_channels, frames, height, width) - Predicted noise/video
        """
        batch_size, channels, frames, height, width = x.shape
        device = x.device
        
        # === CONDITIONING PREPARATION ===
        if conditioning is not None:
            if self.use_text_conditioning:
                cond_emb = self.text_embedder(conditioning, self.training)  # (batch, feature_dim)
            else:
                cond_emb = self.class_embedder(conditioning, self.training)  # (batch, feature_dim)
        else:
            cond_emb = torch.zeros(batch_size, self.feature_dim, device=device)
        
        # === SPATIAL FEATURE EXTRACTION WITH PRETRAINED DiT ===
        # Reshape for frame-wise processing
        x_frames = x.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        # (batch*frames, channels, height, width)
        
        # Expand timesteps for each frame
        t_frames = t.unsqueeze(1).expand(-1, frames).reshape(-1)  # (batch*frames,)
        
        # Extract features using pretrained DiT backbone
        global_features, patch_features = self.dit_backbone(x_frames, t_frames, return_features=True)
        # global_features: (batch*frames, feature_dim)
        
        # === TEMPORAL MODELING ===
        # Reshape global features for temporal processing
        temporal_features = global_features.view(batch_size, frames, self.feature_dim)
        # (batch, frames, feature_dim)
        
        # Apply temporal processing
        temporal_features = self.temporal_processor(temporal_features)
        # (batch, frames, feature_dim)
        
        # Apply conditioning through FiLM layers
        for film_layer in self.conditioning_layers:
            temporal_features = film_layer(temporal_features, cond_emb)
        
        # Apply feature modulation
        modulated_features = self.feature_modulation(temporal_features)
        # (batch, frames, feature_dim)
        
        # === SPATIAL RECONSTRUCTION ===
        # Reshape for spatial upsampling
        features_flat = modulated_features.view(batch_size * frames, self.feature_dim)
        
        # Upsample to spatial resolution
        spatial_output = self.video_upsampler(features_flat)
        # (batch*frames, out_channels, height, width)
        
        # Reshape back to video format
        video_output = spatial_output.view(batch_size, frames, self.out_channels, height, width)
        video_output = video_output.permute(0, 2, 1, 3, 4)
        # (batch, out_channels, frames, height, width)
        
        # === FINAL TEMPORAL PROCESSING ===
        # Apply 3D convolution for temporal consistency
        final_output = self.final_temporal_conv(video_output)
        
        # Residual connection with input (if same channels)
        if self.in_channels == self.out_channels // (2 if self.learn_sigma else 1):
            if self.learn_sigma:
                # Split output into noise prediction and variance prediction
                noise_pred, var_pred = final_output.chunk(2, dim=1)
                noise_pred = noise_pred + x
                final_output = torch.cat([noise_pred, var_pred], dim=1)
            else:
                final_output = final_output + x
        
        return final_output


#################################################################################
#                                FiLM Layer                                    #
#################################################################################

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation Layer"""
    def __init__(self, embed_dim: int, feature_dim: int):
        super().__init__()
        self.projection = nn.Linear(embed_dim, feature_dim * 2)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning
        Args:
            features: (batch, ..., feature_dim) - Input features
            conditioning: (batch, embed_dim) - Conditioning vector
        Returns:
            modulated_features: (batch, ..., feature_dim)
        """
        # Project conditioning to get scale and shift
        gamma, beta = self.projection(conditioning).chunk(2, dim=-1)
        
        # Reshape for broadcasting
        while len(gamma.shape) < len(features.shape):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            
        return features * (1 + gamma) + beta


#################################################################################
#                                   DiT3D Configs                               #
#################################################################################

def DiT3D_XL_2(**kwargs):
    """DiT3D-XL-2 with pretrained DiT-XL-2-256 backbone from HuggingFace"""
    return DiT3D(**kwargs)

# Model registry with HuggingFace pretrained model
DiT3D_models = {
    'DiT3D-XL-2': DiT3D_XL_2,
}


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_dit3d():
    """Comprehensive test of the DiT3D model with pretrained backbone"""
    print("🚀 Testing DiT3D Model with Pretrained Backbone...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    video_size = (16, 128, 128)  # (frames, height, width)
    in_channels = 3
    text_dim = 768  # TextEncoder outputs 768-dimensional embeddings
    
    print(f"📝 Test Configuration:")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Video size: {video_size}")
    print(f"   • In channels: {in_channels}")
    print(f"   • Text dim: {text_dim}")
    print()
    
    # Test different configurations
    configs = [
        {"name": "DiT3D-S/2 with Text (frozen)", "use_text": True, "freeze": True},
        {"name": "DiT3D-S/2 with Text (trainable)", "use_text": True, "freeze": False},
        {"name": "DiT3D-S/2 with Class (frozen)", "use_text": False, "freeze": True},
    ]
    
    for config in configs:
        print(f"🏗️  Testing {config['name']} Configuration...")
        
        try:
            # Create model
            model_kwargs = {
                'video_size': video_size,
                'in_channels': in_channels,
                'out_channels': in_channels,
                'freeze_dit_backbone': config['freeze'],
                'num_classes': 10,
                'learn_sigma': True,
            }
            
            if config['use_text']:
                model_kwargs['text_dim'] = text_dim
                
            model = DiT3D(**model_kwargs)
            
            # Count parameters
            total_params = count_parameters(model)
            
            print(f"   ✅ Model created successfully")
            print(f"   📊 Total parameters: {total_params:,}")
            
            # Create test inputs
            x = torch.randn(batch_size, in_channels, *video_size)
            t = torch.randint(0, 1000, (batch_size,))
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                if config['use_text']:
                    conditioning = torch.randn(batch_size, text_dim)
                    output = model(x, t, conditioning)
                    print(f"   ✅ Forward pass with text: {x.shape} → {output.shape}")
                else:
                    conditioning = torch.randint(0, 10, (batch_size,))
                    output = model(x, t, conditioning)
                    print(f"   ✅ Forward pass with class: {x.shape} → {output.shape}")
                
                # Test without conditioning
                output_no_cond = model(x, t)
                print(f"   ✅ Forward pass without conditioning: {x.shape} → {output_no_cond.shape}")
                
                # Verify shapes
                expected_shape = (batch_size, in_channels * 2, *video_size)  # learn_sigma=True
                assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
                
                # Check for NaN or Inf values
                assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
                
                print(f"   ✅ Output validation passed")
            
            print(f"   🎉 {config['name']} configuration test successful!")
            print()
            
        except Exception as e:
            print(f"   ❌ {config['name']} configuration failed: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    # Test model registry
    print("📚 Testing Model Registry...")
    try:
        for model_name, model_fn in DiT3D_models.items():
            print(f"   Testing {model_name}...")
            test_model = model_fn(
                video_size=(8, 64, 64),  # Smaller for registry test
                text_dim=768,
                num_classes=10
            )
            params = count_parameters(test_model)
            print(f"   ✅ {model_name}: {params:,} parameters")
            
    except Exception as e:
        print(f"   ⚠️  Registry test failed: {e}")
    
    print("=" * 60)
    print("🎊 DiT3D tests completed!")


def analyze_dit3d_architecture():
    """Analyze and display the DiT3D model architecture with pretrained backbone"""
    print("🔍 DiT3D Architecture Analysis (With Pretrained Backbone)")
    print("=" * 60)
    
    model = DiT3D_S_2(video_size=(16, 128, 128), text_dim=768)
    
    print("📋 Component Breakdown:")
    print("   1. Pretrained DiT Backbone Extractor")
    print("      • Uses pretrained DiT spatial transformer")
    print("      • Frame-wise feature extraction")
    print("      • Configurable backbone sizes (S/B/L/XL)")
    print("      • Optional backbone freezing")
    
    print("   2. Video Temporal Processor")
    print("      • Cross-frame temporal attention")
    print("      • Temporal positional embeddings")
    print("      • Residual connections and normalization")
    
    print("   3. FiLM Conditioning Layers")
    print("      • Feature-wise linear modulation")
    print("      • Text/class conditioning integration")
    print("      • Multiple conditioning layers")
    
    print("   4. Video Feature Upsampler")
    print("      • Progressive spatial upsampling")
    print("      • ConvNet-based reconstruction")
    print("      • Adaptive target size handling")
    
    print("   5. Final Temporal Convolution")
    print("      • 3D convolutions for temporal consistency")
    print("      • Residual connections when possible")
    print("      • Group normalization")
    
    print("\n🎯 Key Improvements over Standard DiT3D:")
    print("   ✅ Leverages pretrained DiT spatial representations")
    print("   ✅ Frame-wise processing with temporal modeling")
    print("   ✅ Modular design allows backbone freezing")
    print("   ✅ Better initialization from pretrained weights")
    print("   ✅ Scalable across DiT model sizes")
    print("   ✅ Memory-efficient video processing")
    
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics (DiT3D-S/2):")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print(f"   • Frozen parameters: {total_params - trainable_params:,}")
    print(f"   • Video size: {model.video_size}")
    print(f"   • DiT backbone: {model.dit_backbone.model_size}")
    print(f"   • Feature dimension: {model.feature_dim}")
    
    print(f"\n💾 Memory Estimates:")
    print(f"   • Model VRAM (fp32): ~{total_params * 4 / 1e9:.2f} GB")
    print(f"   • Model VRAM (fp16): ~{total_params * 2 / 1e9:.2f} GB")
    print(f"   • Trainable VRAM (fp32): ~{trainable_params * 4 / 1e9:.2f} GB")


if __name__ == "__main__":
    analyze_dit3d_architecture()
    print()
    test_dit3d()