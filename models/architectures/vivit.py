"""
Video Vision Transformer (ViViT) model for video diffusion
Based on HuggingFace's ViViT implementation, adapted for video diffusion tasks.

This module implements the ViViT architecture from Google Research for video understanding,
adapted for video generation and diffusion tasks. Key adaptations include:
- Direct HuggingFace transformers integration for pretrained models
- Video-specific temporal modeling with tubelet embeddings
- Text conditioning for controllable generation
- Diffusion-compatible forward pass design
- Configurable pretrained backbone freezing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Union
from transformers import VivitModel, VivitConfig

# Helper to choose a valid group count for GroupNorm
def _get_group_count(channels: int) -> int:
    """Return the largest group count <=8 that divides the number of channels"""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps"""
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        
        # Create frequency embeddings
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )
        
        # Expand dimensions for broadcasting
        time_expanded = time.float().unsqueeze(-1)  # (batch, 1)
        freqs_expanded = freqs.unsqueeze(0)  # (1, half_dim)
        
        # Compute embeddings
        args = time_expanded * freqs_expanded  # (batch, half_dim)
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
            
        return embeddings

class TextEmbedder(nn.Module):
    """Text embedding layer with dropout for classifier-free guidance"""
    def __init__(self, text_dim: int, embed_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.dropout_prob = dropout_prob
        
    def forward(self, text_emb: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply text embedding with optional dropout for classifier-free guidance
        Args:
            text_emb: (batch, text_dim) - Text embeddings
            training: Whether in training mode (for dropout)
        Returns:
            embedded: (batch, embed_dim) - Projected text embeddings
        """
        if training and self.dropout_prob > 0:
            # Apply dropout for classifier-free guidance
            mask = torch.rand(text_emb.shape[0], device=text_emb.device) >= self.dropout_prob
            text_emb = text_emb * mask.float().unsqueeze(-1)
        
        return self.text_proj(text_emb)

class ViViTBackboneExtractor(nn.Module):
    """ViViT backbone feature extractor using HuggingFace pretrained models"""
    def __init__(self, 
                 model_name: str = "google/vivit-b-16x2-kinetics400",
                 freeze_backbone: bool = True,
                 video_size: Tuple[int, int, int] = (32, 224, 224),  # (frames, height, width)
                 num_channels: int = 3):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.video_size = video_size
        self.num_frames, self.height, self.width = video_size
        self.num_channels = num_channels
        
        # Load pretrained ViViT model from HuggingFace
        try:
            if model_name != "default":
                print(f"🔍 Loading ViViT model: {model_name}")
                self.vivit = VivitModel.from_pretrained(model_name)
                print(f"✅ Loaded pretrained ViViT model: {model_name}")
            else:
                raise ValueError("Using default config")
        except Exception as e:
            print(f"⚠️  Failed to load pretrained model {model_name}: {e}")
            print("   Creating model with optimized configuration...")
            # Use a smaller, faster configuration optimized for our use case
            config = VivitConfig(
                image_size=224,
                num_frames=32,
                tubelet_size=[2, 16, 16],
                num_channels=num_channels,
                hidden_size=512,  # Smaller hidden size for efficiency
                num_hidden_layers=8,  # Fewer layers for speed
                num_attention_heads=8,  # Fewer heads
                intermediate_size=2048,  # Smaller intermediate size
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            self.vivit = VivitModel(config)
            print(f"✅ Created ViViT model with optimized configuration ({sum(p.numel() for p in self.vivit.parameters()):,} parameters)")
        
        # Extract configuration
        self.config = self.vivit.config
        self.embed_dim = self.config.hidden_size
        self.tubelet_size = self.config.tubelet_size
        
        # Calculate number of patches
        patch_frames = self.num_frames // self.tubelet_size[0]
        patch_height = self.height // self.tubelet_size[1]
        patch_width = self.width // self.tubelet_size[2]
        self.num_patches = patch_frames * patch_height * patch_width
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.vivit.parameters():
                param.requires_grad = False
            print(f"🔒 ViViT backbone frozen ({sum(p.numel() for p in self.vivit.parameters()):,} parameters)")
        else:
            print(f"🔓 ViViT backbone trainable ({sum(p.numel() for p in self.vivit.parameters() if p.requires_grad):,} parameters)")
        
        # Modify input projection if needed for different number of channels
        if num_channels != 3:
            print(f"🔧 Adapting input channels from 3 to {num_channels}")
            # Replace the patch embedding layer
            original_patch_embed = self.vivit.embeddings.patch_embeddings
            self.vivit.embeddings.patch_embeddings = nn.Conv3d(
                num_channels,
                original_patch_embed.out_channels,
                kernel_size=original_patch_embed.kernel_size,
                stride=original_patch_embed.stride,
                padding=original_patch_embed.padding
            )
            # Initialize new layer
            nn.init.xavier_uniform_(self.vivit.embeddings.patch_embeddings.weight)
            if self.vivit.embeddings.patch_embeddings.bias is not None:
                nn.init.zeros_(self.vivit.embeddings.patch_embeddings.bias)
    
    def forward(self, x: torch.Tensor, return_features: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from ViViT backbone
        Args:
            x: (batch, frames, channels, height, width) - Input video from text2sign framework
            return_features: Whether to return intermediate features
        Returns:
            global_features: (batch, embed_dim) - Global CLS token features
            patch_features: (batch, num_patches, embed_dim) - Patch token features
        """
        batch_size = x.shape[0]
        
        # Framework provides (B, C, F, H, W) but HuggingFace ViViT expects (B, F, C, H, W)
        # Convert from (B, C, F, H, W) to (B, F, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # B, C, F, H, W -> B, F, C, H, W
        
        # Simple spatial resize if needed
        if x.shape[-2:] != (self.height, self.width):
            B, num_frames, C, H, W = x.shape
            # Reshape to (B*F, C, H, W) for 2D interpolation
            x_reshaped = x.reshape(B * num_frames, C, H, W)
            x_resized = F.interpolate(x_reshaped, size=(self.height, self.width), mode='bilinear', align_corners=False)
            x = x_resized.reshape(B, num_frames, C, self.height, self.width)
        
        # Check frame count - ensure we have the right number
        if x.shape[1] != self.num_frames:  # Frame dimension is index 1
            if x.shape[1] > self.num_frames:
                # Sample frames uniformly
                indices = torch.linspace(0, x.shape[1]-1, self.num_frames).long()
                x = x[:, indices]
            else:
                # Repeat frames if we have fewer
                repeat_factor = self.num_frames // x.shape[1]
                remainder = self.num_frames % x.shape[1]
                x_repeated = x.repeat(1, repeat_factor, 1, 1, 1)
                if remainder > 0:
                    x_extra = x[:, :remainder]
                    x = torch.cat([x_repeated, x_extra], dim=1)
                else:
                    x = x_repeated
        
        # Forward through ViViT (expects B, F, C, H, W)
        outputs = self.vivit(pixel_values=x, output_hidden_states=return_features)
        
        # Extract features
        last_hidden_state = outputs.last_hidden_state  # (batch, num_patches + 1, embed_dim)
        
        # Separate CLS token and patch features
        global_features = last_hidden_state[:, 0]  # (batch, embed_dim) - CLS token
        patch_features = last_hidden_state[:, 1:] if return_features else None  # (batch, num_patches, embed_dim)
        
        return global_features, patch_features

class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer for video sequence modeling"""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Temporal self-attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and feed-forward
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temporal attention
        Args:
            x: (batch, sequence_length, embed_dim)
            mask: Optional attention mask
        Returns:
            x: (batch, sequence_length, embed_dim)
        """
        # Temporal self-attention with residual connection
        attn_out, _ = self.temporal_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class FeatureUpsampler(nn.Module):
    """Upsample features from ViViT patches to full video resolution"""
    def __init__(self, 
                 embed_dim: int, 
                 out_channels: int, 
                 video_size: Tuple[int, int, int],
                 tubelet_size: Tuple[int, int, int]):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.video_size = video_size
        self.tubelet_size = tubelet_size
        
        frames, height, width = video_size
        t_patch, h_patch, w_patch = tubelet_size
        
        # Calculate patch grid dimensions
        self.patch_frames = frames // t_patch
        self.patch_height = height // h_patch
        self.patch_width = width // w_patch
        
        # Feature projection to spatial features
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels * t_patch * h_patch * w_patch)
        )
        
        # 3D convolution layers for upsampling if needed
        self.conv_layers = nn.ModuleList()
        
        # Add conv layers only if we need spatial upsampling beyond patches
        current_size = (self.patch_frames * t_patch, self.patch_height * h_patch, self.patch_width * w_patch)
        if current_size != video_size:
            # Add 3D convolution for fine-tuning spatial resolution
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(out_channels, out_channels * 2, kernel_size=3, padding=1),
                    nn.GroupNorm(_get_group_count(out_channels * 2), out_channels * 2),
                    nn.GELU(),
                    nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(_get_group_count(out_channels), out_channels)
                )
            )
    
    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Upsample patch features to full video resolution
        Args:
            patch_features: (batch, num_patches, embed_dim)
        Returns:
            video: (batch, out_channels, frames, height, width)
        """
        batch_size = patch_features.shape[0]
        
        # Project features to tubelet space
        projected = self.feature_proj(patch_features)  # (batch, num_patches, out_channels * t_patch * h_patch * w_patch)
        
        # Reshape to tubelet grid
        tubelet_volume = self.tubelet_size[0] * self.tubelet_size[1] * self.tubelet_size[2]
        projected = projected.view(
            batch_size, 
            self.patch_frames, self.patch_height, self.patch_width,
            self.out_channels, self.tubelet_size[0], self.tubelet_size[1], self.tubelet_size[2]
        )
        
        # Rearrange to video format
        video = projected.permute(0, 4, 1, 5, 2, 6, 3, 7)  # (batch, out_channels, patch_f, t_patch, patch_h, h_patch, patch_w, w_patch)
        video = video.reshape(
            batch_size, 
            self.out_channels,
            self.patch_frames * self.tubelet_size[0],
            self.patch_height * self.tubelet_size[1],
            self.patch_width * self.tubelet_size[2]
        )
        
        # Apply additional conv layers if needed
        for conv_layer in self.conv_layers:
            video = conv_layer(video)
        
        # Final resize to exact target size
        if video.shape[2:] != self.video_size:
            video = F.interpolate(video, size=self.video_size, mode='trilinear', align_corners=False)
        
        return video

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation Layer for conditioning"""
    def __init__(self, embed_dim: int, feature_dim: int):
        super().__init__()
        self.projection = nn.Linear(embed_dim, feature_dim * 2)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning
        Args:
            features: (..., feature_dim) - Input features
            conditioning: (batch, embed_dim) - Conditioning vector
        Returns:
            modulated_features: (..., feature_dim)
        """
        # Project conditioning to get scale and shift
        gamma, beta = self.projection(conditioning).chunk(2, dim=-1)
        
        # Reshape for broadcasting
        while len(gamma.shape) < len(features.shape):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            
        return features * (1 + gamma) + beta

class ViViT(nn.Module):
    """
    Video Vision Transformer for noise prediction in video diffusion
    
    Based on Google's ViViT architecture, adapted for noise prediction in video diffusion tasks.
    This model predicts the noise ε that was added to clean videos during the forward diffusion process.
    
    Key features:
    - HuggingFace ViViT backbone integration
    - Tubelet-based video tokenization  
    - Text conditioning support
    - Noise prediction for DDPM training (ε-parameterization)
    - Configurable backbone freezing
    
    The model follows the standard DDPM objective: L = E[||ε - ε_θ(x_t, t)||²]
    where ε_θ is this ViViT model predicting noise from noisy input x_t at timestep t.
    """
    
    def __init__(self,
                 video_size: Tuple[int, int, int] = (32, 224, 224),  # (frames, height, width)
                 in_channels: int = 3,
                 out_channels: int = 3,
                 time_dim: int = 768,
                 text_dim: Optional[int] = None,
                 model_name: str = "google/vivit-b-16x2-kinetics400",
                 freeze_backbone: bool = True,
                 num_temporal_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 class_dropout_prob: float = 0.1):
        super().__init__()
        
        # Store configuration
        self.video_size = video_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        # ViViT backbone for spatiotemporal feature extraction from noisy videos
        self.backbone = ViViTBackboneExtractor(
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            video_size=video_size,
            num_channels=in_channels
        )
        
        self.embed_dim = self.backbone.embed_dim
        self.num_patches = self.backbone.num_patches
        self.tubelet_size = self.backbone.tubelet_size
        
        # Time embedding for diffusion timestep conditioning
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        
        # Text conditioning (optional) for controlled generation
        self.text_embedder = None
        if text_dim is not None:
            self.text_embedder = TextEmbedder(text_dim, time_dim, class_dropout_prob)
        
        # Project time/text embedding to feature space
        self.cond_proj = nn.Sequential(
            nn.Linear(time_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Temporal attention layers for video modeling
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionLayer(self.embed_dim, num_heads, dropout)
            for _ in range(num_temporal_layers)
        ])
        
        # FiLM layers for conditioning
        self.film_layers = nn.ModuleList([
            FiLMLayer(self.embed_dim, self.embed_dim) 
            for _ in range(num_temporal_layers)
        ])
        
        # Feature upsampler to reconstruct noise at video resolution
        self.upsampler = FeatureUpsampler(
            embed_dim=self.embed_dim,
            out_channels=out_channels,  # Should match input channels for noise prediction
            video_size=video_size,
            tubelet_size=self.tubelet_size
        )
        
        # Final output normalization for predicted noise
        self.output_norm = nn.GroupNorm(_get_group_count(out_channels), out_channels)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in [self.time_embedding, self.cond_proj, self.temporal_layers, 
                      self.film_layers, self.upsampler]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, (nn.Conv3d, nn.Conv2d)):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, 
                x: torch.Tensor, 
                time: torch.Tensor, 
                text_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for noise prediction in video diffusion
        Args:
            x: (batch, channels, frames, height, width) - Noisy input video from diffusion process
            time: (batch,) - Diffusion timesteps
            text_emb: (batch, text_dim) - Optional text conditioning
        Returns:
            predicted_noise: (batch, channels, frames, height, width) - Predicted noise ε
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Store original input dimensions for output matching
        original_shape = x.shape  # (B, C, F, H, W) from framework
        _, _, original_frames, original_height, original_width = original_shape
        
        # === CONDITIONING PREPARATION ===
        # Time embedding
        time_emb = self.time_embedding(time)  # (batch, time_dim)
        
        # Add text conditioning if provided
        if text_emb is not None and self.text_embedder is not None:
            text_emb_proj = self.text_embedder(text_emb, self.training)  # (batch, time_dim)
            time_emb = time_emb + text_emb_proj
        
        # Project to feature space
        cond_emb = self.cond_proj(time_emb)  # (batch, embed_dim)
        
        # === SPATIAL-TEMPORAL FEATURE EXTRACTION ===
        # Extract features from noisy input using ViViT backbone
        global_features, patch_features = self.backbone(x, return_features=True)
        # global_features: (batch, embed_dim)
        # patch_features: (batch, num_patches, embed_dim)
        
        # === TEMPORAL MODELING ===
        # Apply temporal attention with conditioning
        temporal_features = patch_features
        for temporal_layer, film_layer in zip(self.temporal_layers, self.film_layers):
            # Apply temporal attention
            temporal_features = temporal_layer(temporal_features)
            # Apply FiLM conditioning
            temporal_features = film_layer(temporal_features, cond_emb)
        
        # === NOISE PREDICTION ===
        # Upsample features to video resolution - this predicts the noise
        predicted_noise = self.upsampler(temporal_features)
        # (batch, out_channels, frames, height, width)
        
        # Apply final normalization
        predicted_noise = self.output_norm(predicted_noise)
        
        # Resize output to match original input spatial dimensions
        if predicted_noise.shape[2:] != (original_frames, original_height, original_width):
            predicted_noise = F.interpolate(
                predicted_noise, 
                size=(original_frames, original_height, original_width), 
                mode='trilinear', 
                align_corners=False
            )
        
        # Output format: (B, C, F, H, W) - predicted noise with same shape as input
        # No residual connection needed since we're predicting noise, not reconstructing data
        
        return predicted_noise

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Model registry for easy access
ViViT_models = {
    'ViViT-B-16x2': lambda **kwargs: ViViT(model_name="google/vivit-b-16x2-kinetics400", **kwargs),
    'ViViT-L-16x4': lambda **kwargs: ViViT(model_name="google/vivit-l-16x4-kinetics400", **kwargs),
    'ViViT-H-14x2': lambda **kwargs: ViViT(model_name="google/vivit-h-14x2-kinetics400", **kwargs),
}

def test_vivit():
    """Test the ViViT model implementation"""
    print("🚀 Testing ViViT Model...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    video_size = (16, 224, 224)  # (frames, height, width)
    in_channels = 3
    time_dim = 768
    text_dim = 768
    
    print(f"📝 Test Configuration:")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Video size: {video_size}")
    print(f"   • Input channels: {in_channels}")
    print(f"   • Time dim: {time_dim}")
    print(f"   • Text dim: {text_dim}")
    print()
    
    # Test different configurations
    configs = [
        {"name": "ViViT-B Frozen", "model_name": "google/vivit-b-16x2-kinetics400", "freeze": True},
        {"name": "ViViT-B Trainable", "model_name": "google/vivit-b-16x2-kinetics400", "freeze": False},
    ]
    
    for config in configs:
        print(f"🏗️  Testing {config['name']}...")
        
        try:
            # Create model
            model = ViViT(
                video_size=video_size,
                in_channels=in_channels,
                out_channels=in_channels,
                time_dim=time_dim,
                text_dim=text_dim,
                model_name=config['model_name'],
                freeze_backbone=config['freeze'],
                num_temporal_layers=2,
                num_heads=8
            )
            
            # Count parameters
            total_params = count_parameters(model)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   ✅ Model created successfully")
            print(f"   📊 Total parameters: {total_params:,}")
            print(f"   🎯 Trainable parameters: {trainable_params:,}")
            
            # Create test inputs - Framework expects (B, F, C, H, W) format
            frames, height, width = video_size
            x = torch.randn(batch_size, frames, in_channels, height, width)  # (B, F, C, H, W)
            time = torch.randint(0, 1000, (batch_size,))
            text_emb = torch.randn(batch_size, text_dim)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                # Test with text conditioning
                output_with_text = model(x, time, text_emb)
                print(f"   ✅ Forward pass with text: {x.shape} → {output_with_text.shape}")
                
                # Test without text conditioning
                output_no_text = model(x, time)
                print(f"   ✅ Forward pass without text: {x.shape} → {output_no_text.shape}")
                
                # Verify shapes
                assert output_with_text.shape == x.shape, f"Shape mismatch: {output_with_text.shape} vs {x.shape}"
                assert output_no_text.shape == x.shape, f"Shape mismatch: {output_no_text.shape} vs {x.shape}"
                
                # Check for NaN or Inf values
                assert torch.isfinite(output_with_text).all(), "Output contains NaN or Inf values"
                assert torch.isfinite(output_no_text).all(), "Output contains NaN or Inf values"
                
                print(f"   ✅ Output validation passed")
            
            print(f"   🎉 {config['name']} test successful!")
            print()
            
        except Exception as e:
            print(f"   ❌ {config['name']} test failed: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    print("=" * 60)
    print("🎊 ViViT tests completed!")
    
    return model

def analyze_vivit_architecture():
    """Analyze ViViT model architecture"""
    print("🔍 ViViT Architecture Analysis")
    print("=" * 60)
    
    try:
        model = ViViT(freeze_backbone=True)
        
        print("📋 Component Breakdown:")
        print("   1. ViViT Backbone (HuggingFace)")
        print("      • Pretrained tubelet-based video transformer")
        print("      • Spatial-temporal patch embeddings")
        print("      • Multi-head self-attention layers")
        
        print("   2. Temporal Attention Layers")
        print("      • Additional temporal modeling")
        print("      • Multi-head attention for frame sequences")
        print("      • Residual connections and layer normalization")
        
        print("   3. Feature Upsampler")
        print("      • Reconstruct video from patch features")
        print("      • Tubelet-to-video conversion")
        print("      • 3D convolutions for spatial refinement")
        
        print("   4. Conditioning Integration")
        print("      • Time embedding for diffusion timesteps")
        print("      • Text conditioning with FiLM layers")
        print("      • Feature-wise linear modulation")
        
        print("\n🎯 Key Features:")
        print("   ✅ HuggingFace pretrained backbone integration")
        print("   ✅ Tubelet-based video tokenization")
        print("   ✅ Text conditioning support")
        print("   ✅ Configurable backbone freezing")
        print("   ✅ Residual connections for training stability")
        print("   ✅ Compatible with video diffusion")
        
        total_params = count_parameters(model)
        print(f"\n📊 Model Statistics:")
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Backbone: ViViT-B-16x2 (HuggingFace)")
        print(f"   • Default video size: (32, 224, 224)")
        print(f"   • Tubelet size: {model.tubelet_size}")
        print(f"   • Embedding dimension: {model.embed_dim}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    analyze_vivit_architecture()
    print()
    test_vivit()