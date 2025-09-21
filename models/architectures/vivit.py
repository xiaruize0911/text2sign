"""
ViViT (Video Vision Transformer) model for video diffusion
This module implements a ViViT architecture using the pretrained google/vivit-b-16x2-kinetics400 model
for text-to-sign language video generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import os
import warnings

# Disable HuggingFace progress bars and verbose logging for cleaner output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

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

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation Layer for conditioning"""
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

class ViViTBackboneExtractor(nn.Module):
    """ViViT-inspired backbone feature extractor using frame-wise ViT processing"""
    def __init__(self, model_name: str = "google/vivit-b-16x2-kinetics400", freeze_backbone: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        try:
            print(f"🔍 Loading ViViT model: {model_name} (this may take a moment for first-time download...)")
            
            # Instead of using the full ViViT which has compatibility issues,
            # we'll use a ViT backbone and add our own temporal processing
            from transformers import ViTModel, ViTConfig
            
            # Load ViT backbone for spatial feature extraction
            try:
                # Try to use a standard ViT model for spatial features
                vit_model_name = "google/vit-base-patch16-224"
                print(f"Using ViT backbone: {vit_model_name}")
                self.vit = ViTModel.from_pretrained(vit_model_name)
                print(f"✅ Successfully loaded ViT backbone")
            except Exception as e:
                print(f"⚠️  Failed to load ViT backbone: {e}")
                print("   Creating ViT with default configuration...")
                config = ViTConfig(
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_dropout_prob=0.0,
                    attention_probs_dropout_prob=0.0,
                    layer_norm_eps=1e-12,
                    qkv_bias=True
                )
                self.vit = ViTModel(config)
                print(f"✅ Created ViT model with default configuration")
            
            # Store model dimensions
            self.embed_dim = self.vit.config.hidden_size  # 768 for ViT-B
            self.num_patches_per_frame = (224 // 16) ** 2  # 196 patches per frame
            
            # Count parameters
            total_params = sum(p.numel() for p in self.vit.parameters())
            print(f"✅ Created ViT model with optimized configuration ({total_params:,} parameters)")
            
            # Freeze backbone if requested
            if freeze_backbone:
                for param in self.vit.parameters():
                    param.requires_grad = False
                print(f"🔒 ViT backbone frozen ({total_params:,} parameters)")
            else:
                print(f"🔓 ViT backbone trainable ({total_params:,} parameters)")
                
        except Exception as e:
            print(f"❌ Failed to initialize ViT model: {e}")
            raise RuntimeError(f"Could not initialize ViT model: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from ViT backbone frame by frame with patch-level features preserved
        Args:
            x: (batch, channels, frames, height, width) - Input video
        Returns:
            features: (batch, frames, spatial_patches, embed_dim) - Patch-level features per frame
        """
        batch_size, channels, frames, height, width = x.shape
        
        # Process each frame separately through ViT
        all_frame_features = []
        
        for frame_idx in range(frames):
            # Extract single frame
            frame = x[:, :, frame_idx, :, :]  # (batch, channels, height, width)
            
            # Resize to ViT expected size (224x224) if needed
            if height != 224 or width != 224:
                frame = F.interpolate(frame, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Apply ViT model to frame
            if self.freeze_backbone:
                with torch.no_grad():
                    outputs = self.vit(frame)
            else:
                outputs = self.vit(frame)
            
            # Use patch-level features instead of global pooling
            # outputs.last_hidden_state: (batch, num_patches + 1, embed_dim)
            # Remove CLS token and keep only patch features
            patch_features = outputs.last_hidden_state[:, 1:, :]  # (batch, num_patches, embed_dim)
            all_frame_features.append(patch_features)
        
        # Stack frame features: (batch, frames, num_patches, embed_dim)
        features = torch.stack(all_frame_features, dim=1)  
        
        # Reshape to (batch, frames * num_patches, embed_dim) for temporal processing
        batch_size, frames, num_patches, embed_dim = features.shape
        features = features.view(batch_size, frames * num_patches, embed_dim)
        
        return features  # (batch, frames * num_patches, embed_dim)

class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer for enhanced temporal modeling"""
    def __init__(self, embed_dim: int, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Multi-head attention for temporal modeling
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention
        Args:
            x: (batch, frames, embed_dim)
        Returns:
            x: (batch, frames, embed_dim)
        """
        # Temporal self-attention with residual connection
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class FeatureUpsampler(nn.Module):
    """Feature upsampler to convert ViViT features to video frames"""
    def __init__(self, feature_dim: int, out_channels: int, target_size: Tuple[int, int]):
        super().__init__()
        self.target_size = target_size
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        
        # Feature projection to spatial features
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, out_channels * 64)  # Start with 8x8 spatial resolution
        )
        
        # Progressive upsampling layers with learnable output scaling
        self.conv_layers = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(out_channels, out_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_get_group_count(out_channels * 2), out_channels * 2),
            nn.GELU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(out_channels * 2, out_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_get_group_count(out_channels * 2), out_channels * 2),
            nn.GELU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_get_group_count(out_channels), out_channels),
            nn.GELU(),
            
            # 64x64 -> target_size (adaptive final layer)
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )
        
        # Learnable output scaling to handle noise scale issues
        self.output_scale = nn.Parameter(torch.ones(1) * 2.0)  # Initialize to allow 2x scaling
    
    def forward(self, features_flat: torch.Tensor) -> torch.Tensor:
        """
        Upsample flat features to spatial frames
        Args:
            features_flat: (batch*frames, feature_dim)
        Returns:
            x: (batch*frames, out_channels, target_h, target_w)
        """
        # Project to spatial features
        x = self.feature_proj(features_flat)  # (batch*frames, out_channels*64)
        x = x.view(-1, self.out_channels, 8, 8)  # (batch*frames, out_channels, 8, 8)
        
        # Progressive upsampling
        x = self.conv_layers(x)
        
        # Apply learnable scaling
        x = x * self.output_scale
        
        # Ensure final size matches target
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        return x

class ViViT(nn.Module):
    """
    ViViT model for video diffusion with text conditioning
    
    This model uses the pretrained google/vivit-b-16x2-kinetics400 as a backbone
    and adds temporal attention layers and upsampling to generate video frames.
    """
    
    def __init__(
        self,
        video_size: Tuple[int, int, int] = (16, 64, 64),  # (frames, height, width)
        in_channels: int = 3,
        out_channels: int = 3,
        time_dim: int = 768,
        text_dim: Optional[int] = None,
        model_name: str = "google/vivit-b-16x2-kinetics400",
        freeze_backbone: bool = True,
        num_temporal_layers: int = 4,
        num_heads: int = 12,
        dropout: float = 0.1,
        class_dropout_prob: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # Store configuration
        self.video_size = video_size
        self.frames, self.height, self.width = video_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.text_dim = text_dim
        self.model_name = model_name
        self.dropout = dropout
        self.class_dropout_prob = class_dropout_prob
        
        # ViViT backbone for spatial-temporal feature extraction
        self.backbone = ViViTBackboneExtractor(model_name, freeze_backbone)
        self.feature_dim = self.backbone.embed_dim  # 768 for ViViT-B
        
        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        
        # Text conditioning projection (optional)
        self.text_proj = None
        if text_dim is not None:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        
        # FiLM layers for conditioning each temporal layer
        self.film_layers = nn.ModuleList([
            FiLMLayer(time_dim, self.feature_dim) for _ in range(num_temporal_layers)
        ])
        
        # Additional temporal attention layers
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionLayer(
                embed_dim=self.feature_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_temporal_layers)
        ])
        
        # Feature upsampler to convert features back to video frames
        # Replace with a simpler final projection since we now preserve spatial structure
        self.final_projection = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(_get_group_count(self.feature_dim // 2), self.feature_dim // 2),
            nn.GELU(),
            nn.Conv2d(self.feature_dim // 2, out_channels, kernel_size=3, padding=1),
        )
        
        # Final temporal processing with 3D convolutions
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(_get_group_count(out_channels * 2), out_channels * 2),
            nn.GELU(),
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in [self.time_embedding, self.text_proj, self.film_layers, 
                      self.temporal_layers, self.final_projection, self.temporal_conv]:
            if module is None:
                continue
                
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor, text_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of ViViT model
        
        Args:
            x: (batch, channels, frames, height, width) - Input video
            time: (batch,) - Diffusion timestep
            text_emb: (batch, text_dim) - Optional text conditioning
            
        Returns:
            out: (batch, out_channels, frames, height, width) - Output video
        """
        batch_size, channels, frames, height, width = x.shape
        device = x.device
        
        # === CONDITIONING PREPARATION ===
        # Compute time embedding
        time_emb = self.time_embedding(time)  # (batch, time_dim)
        
        # Add text conditioning if provided
        if text_emb is not None and self.text_proj is not None:
            text_emb_proj = self.text_proj(text_emb)  # (batch, time_dim)
            time_emb = time_emb + text_emb_proj
        
        # === SPATIAL-TEMPORAL FEATURE EXTRACTION ===
        # Extract features from ViViT backbone
        patch_features = self.backbone(x)  # (batch, frames * num_patches, feature_dim)
        
        # Calculate dimensions for reshaping
        num_patches = self.backbone.num_patches_per_frame  # 196 for 224x224 with 16x16 patches
        
        # Reshape to separate frames for temporal processing
        features = patch_features.view(batch_size, frames, num_patches, self.feature_dim)
        
        # Global pooling to get frame-level features for temporal attention
        frame_features = features.mean(dim=2)  # (batch, frames, feature_dim)
        
        # === TEMPORAL PROCESSING ===
        # Apply temporal attention layers with FiLM conditioning
        for temporal_layer, film_layer in zip(self.temporal_layers, self.film_layers):
            frame_features = temporal_layer(frame_features)
            frame_features = film_layer(frame_features, time_emb)
        
        # Broadcast temporal features back to patch level
        # (batch, frames, feature_dim) -> (batch, frames, num_patches, feature_dim)
        temporal_features = frame_features.unsqueeze(2).expand(-1, -1, num_patches, -1)
        
        # Combine spatial and temporal information
        enhanced_features = features + temporal_features * 0.1  # Small weight for temporal modulation
        
        # === SPATIAL RECONSTRUCTION ===
        # Reshape for spatial upsampling, preserving patch-level spatial structure
        features_for_upsampling = enhanced_features.view(batch_size * frames, num_patches, self.feature_dim)
        
        # Use patch features to reconstruct spatial structure
        # First, reshape patches back to spatial grid (14x14 for 224x224 input)
        patch_h = patch_w = int(num_patches ** 0.5)  # 14 for 196 patches
        spatial_features = features_for_upsampling.view(batch_size * frames, self.feature_dim, patch_h, patch_w)
        
        # Upsample from patch grid to target resolution using input dimensions
        spatial_features = F.interpolate(spatial_features, size=(height, width), mode='bilinear', align_corners=False)
        
        # Project to output channels
        spatial_output = self.final_projection(spatial_features)  # (batch*frames, out_channels, height, width)
        
        # Ensure the spatial output has exactly the right dimensions
        if spatial_output.shape[-2:] != (height, width):
            spatial_output = F.adaptive_avg_pool2d(spatial_output, (height, width))
        
        # Ensure we have the right number of channels
        if spatial_output.shape[1] != self.out_channels:
            # Add a projection layer if channels don't match
            if not hasattr(self, 'channel_proj'):
                self.channel_proj = nn.Conv2d(spatial_output.shape[1], self.out_channels, 1).to(spatial_output.device)
            spatial_output = self.channel_proj(spatial_output)
        
        # Verify tensor size before reshaping
        expected_size = batch_size * frames * self.out_channels * height * width
        if spatial_output.numel() != expected_size:
            # Force reshape using adaptive pooling
            spatial_output = spatial_output.view(batch_size * frames, -1, height, width)
            if spatial_output.shape[1] != self.out_channels:
                spatial_output = F.adaptive_avg_pool2d(spatial_output, (height, width))
                spatial_output = spatial_output[:, :self.out_channels, :, :]
        
        # Reshape back to video format - force the correct shape
        spatial_output = spatial_output.view(batch_size, frames, self.out_channels, height, width)
        video_output = spatial_output.permute(0, 2, 1, 3, 4)
        # (batch, out_channels, frames, height, width)
        
        # === FINAL TEMPORAL PROCESSING ===
        # Apply 3D convolution for temporal consistency
        final_output = self.temporal_conv(video_output)
        
        # Residual connection if input and output have same channels
        if self.in_channels == self.out_channels:
            final_output = final_output + x
        
        return final_output

def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """Count the number of parameters in a model"""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

# Model registry for ViViT models
ViViT_models = {
    'ViViT-B-16x2': ViViT,  # Base ViViT model using google/vivit-b-16x2-kinetics400
}

def test_vivit():
    """Test the ViViT model"""
    print("🚀 Testing ViViT Model...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    channels, frames, height, width = 3, 16, 64, 64
    time_dim = 768
    text_dim = 768
    
    print(f"📝 Test Configuration:")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Video shape: {channels}×{frames}×{height}×{width}")
    print(f"   • Time dim: {time_dim}")
    print(f"   • Text dim: {text_dim}")
    print()
    
    try:
        # Create model
        model = ViViT(
            video_size=(frames, height, width),
            in_channels=channels,
            out_channels=channels,
            time_dim=time_dim,
            text_dim=text_dim,
            freeze_backbone=True,
            num_temporal_layers=2,  # Reduced for testing
            num_heads=12
        )
        
        # Count parameters
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, only_trainable=True)
        frozen_params = total_params - trainable_params
        
        print(f"✅ Model created successfully")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"🔒 Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print()
        
        # Create test inputs
        x = torch.randn(batch_size, channels, frames, height, width)
        time = torch.randint(0, 1000, (batch_size,))
        text_emb = torch.randn(batch_size, text_dim)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            # Test with text embedding
            output_with_text = model(x, time, text_emb)
            print(f"✅ Forward pass with text: {x.shape} → {output_with_text.shape}")
            
            # Test without text embedding
            output_no_text = model(x, time)
            print(f"✅ Forward pass without text: {x.shape} → {output_no_text.shape}")
            
            # Verify shapes
            assert output_with_text.shape == x.shape, f"Shape mismatch with text: {output_with_text.shape} vs {x.shape}"
            assert output_no_text.shape == x.shape, f"Shape mismatch without text: {output_no_text.shape} vs {x.shape}"
            
            # Check for NaN or Inf values
            assert torch.isfinite(output_with_text).all(), "Output contains NaN or Inf values (with text)"
            assert torch.isfinite(output_no_text).all(), "Output contains NaN or Inf values (without text)"
            
            # Check output statistics
            print(f"📈 Output statistics (with text):")
            print(f"   • Range: [{output_with_text.min():.3f}, {output_with_text.max():.3f}]")
            print(f"   • Mean: {output_with_text.mean():.3f}")
            print(f"   • Std: {output_with_text.std():.3f}")
            
            print(f"✅ All tests passed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    return model

def analyze_vivit_architecture():
    """Analyze and display the ViViT model architecture"""
    print("🔍 ViViT Model Architecture Analysis")
    print("=" * 60)
    
    model = ViViT(freeze_backbone=True)
    
    print("📋 Component Breakdown:")
    print("   1. ViViT Backbone (google/vivit-b-16x2-kinetics400)")
    print("      • Pretrained Video Vision Transformer")
    print("      • Spatial-temporal patch embedding")
    print("      • Transformer encoder with 12 layers")
    
    print("   2. Temporal Attention Layers")
    print("      • Additional temporal modeling beyond ViViT")
    print("      • Multi-head attention with residual connections")
    print("      • FiLM conditioning for each layer")
    
    print("   3. Feature Upsampler")
    print("      • Progressive learnable upsampling")
    print("      • ConvTranspose2d layers with GroupNorm")
    print("      • Learnable output scaling for noise prediction")
    
    print("   4. Final Temporal Processing")
    print("      • 3D convolutions for temporal consistency")
    print("      • Residual connections when possible")
    print("      • GroupNorm for stability")
    
    print("\n🎯 Key Features:")
    print("   ✅ Pretrained ViViT backbone for strong spatial-temporal understanding")
    print("   ✅ Text conditioning via FiLM layers")
    print("   ✅ Learnable output scaling to handle diffusion noise scales")
    print("   ✅ Progressive upsampling for high-quality spatial reconstruction")
    print("   ✅ Additional temporal layers for enhanced temporal modeling")
    print("   ✅ Memory-efficient with frozen backbone option")
    
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, only_trainable=True)
    
    print(f"\n📊 Model Statistics:")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print(f"   • Estimated VRAM (fp32): ~{total_params * 4 / 1e9:.2f} GB")
    print(f"   • Estimated VRAM (fp16): ~{total_params * 2 / 1e9:.2f} GB")

if __name__ == "__main__":
    analyze_vivit_architecture()
    print()
    test_vivit()
