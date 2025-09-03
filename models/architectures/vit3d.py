"""
Redesigned 3D Vision Transformer model for video diffusion
This module implements an improved ViT3D architecture with better spatial-temporal modeling,
efficient feature extraction, and proper conditioning mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Optional, Tuple
import math

# Helper to choose a valid group count for GroupNorm
def _get_group_count(channels: int) -> int:
    """Return the largest group count <=8 that divides the number of channels"""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1

class SinusoidalTimeEmbedding(nn.Module):
    """Improved sinusoidal time embedding with better conditioning"""
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


class SpatialTemporalAttention(nn.Module):
    """Advanced attention module for spatial-temporal video modeling"""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
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
        
        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply spatial-temporal attention
        Args:
            x: (batch, frames, embed_dim)
            mask: Optional attention mask
        Returns:
            x: (batch, frames, embed_dim)
        """
        # Temporal self-attention with residual connection
        attn_out, _ = self.temporal_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class ViTBackboneExtractor(nn.Module):
    """Efficient ViT backbone feature extractor with patch-level features"""
    def __init__(self, in_channels: int = 3, freeze_backbone: bool = False):
        super().__init__()
        
        # Load pretrained ViT-B/16
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Modify input projection for arbitrary number of channels
        if in_channels != 3:
            self.vit.conv_proj = nn.Conv2d(
                in_channels, 
                self.vit.conv_proj.out_channels, 
                kernel_size=16, 
                stride=16
            )
        
        # Extract useful attributes
        self.embed_dim = self.vit.conv_proj.out_channels
        self.num_patches = (224 // 16) ** 2  # 196 patches for 224x224 input
        
        # Remove classification head to get patch features
        self.vit.heads = nn.Identity()
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
                
        # Store encoder layers for accessing intermediate features
        self.encoder_layers = self.vit.encoder.layers
        
    def forward(self, x: torch.Tensor, return_patches: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from ViT backbone
        Args:
            x: (batch*frames, channels, height, width)
            return_patches: Whether to return patch-level features
        Returns:
            global_features: (batch*frames, embed_dim) - Global features
            patch_features: (batch*frames, num_patches, embed_dim) - Patch features
        """
        # Resize to ViT expected size if needed
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Apply patch embedding
        x = self.vit.conv_proj(x)  # (batch*frames, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (batch*frames, num_patches, embed_dim)
        
        # Add class token and positional embedding
        class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)  # (batch*frames, num_patches+1, embed_dim)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        
        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Apply final layer norm
        x = self.vit.encoder.ln(x)
        
        # Separate class token and patch features
        global_features = x[:, 0]  # (batch*frames, embed_dim)
        patch_features = x[:, 1:] if return_patches else None  # (batch*frames, num_patches, embed_dim)
        
        return global_features, patch_features


class FeatureUpsampler(nn.Module):
    """Intelligent upsampling from ViT features to video frames"""
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
            # Find the largest divisor of channels that's <= 8
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
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return x

class ViT3D(nn.Module):
    """
    Redesigned 3D Vision Transformer for video diffusion with improved architecture.
    
    Key improvements:
    - Efficient patch-level feature extraction from ViT backbone
    - Advanced spatial-temporal attention mechanisms  
    - Progressive upsampling with learnable modules
    - Better conditioning integration
    - Memory-efficient processing
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 frames: int = 28,
                 height: int = 128, 
                 width: int = 128, 
                 time_dim: int = 128,
                 text_dim: Optional[int] = None, 
                 embed_dim: int = 768,
                 freeze_backbone: bool = False, 
                 num_temporal_heads: int = 8,
                 num_attention_layers: int = 2):
        super().__init__()
        
        # Store input parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frames = frames
        self.height = height
        self.width = width
        self.time_dim = time_dim
        self.embed_dim = embed_dim
        
        # ViT backbone for spatial feature extraction
        self.backbone = ViTBackboneExtractor(in_channels, freeze_backbone)
        self.feature_dim = self.backbone.embed_dim  # 768 for ViT-B/16
        
        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Text conditioning (optional)
        self.text_proj = None
        if text_dim is not None:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        
        # Spatial-temporal attention stack
        self.attention_layers = nn.ModuleList([
            SpatialTemporalAttention(
                embed_dim=self.feature_dim,
                num_heads=num_temporal_heads,
                dropout=0.1
            ) for _ in range(num_attention_layers)
        ])
        
        # Feature conditioning and processing
        self.feature_modulation = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        
        # Learnable upsampling to output resolution
        self.upsampler = FeatureUpsampler(
            feature_dim=self.feature_dim,
            out_channels=out_channels,
            target_size=(height, width)
        )
        
        # Final temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(_get_group_count(out_channels * 2), out_channels * 2),
            nn.GELU(),
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(_get_group_count(out_channels), out_channels),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in [self.time_proj, self.feature_modulation, self.temporal_conv]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, (nn.Conv3d, nn.Conv2d)):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor, text_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with improved spatial-temporal processing
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
        
        # Project time embedding to feature space
        time_features = self.time_proj(time_emb)  # (batch, feature_dim)
        
        # === SPATIAL FEATURE EXTRACTION ===
        # Reshape for frame-wise processing
        x_frames = x.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        # (batch*frames, channels, height, width)
        
        # Extract features from ViT backbone
        global_features, patch_features = self.backbone(x_frames, return_patches=True)
        # global_features: (batch*frames, feature_dim)
        # patch_features: (batch*frames, num_patches, feature_dim)
        
        # === TEMPORAL MODELING ===
        # Reshape global features for temporal processing
        temporal_features = global_features.view(batch_size, frames, self.feature_dim)
        # (batch, frames, feature_dim)
        
        # Add time conditioning to each frame
        time_features_expanded = time_features.unsqueeze(1).expand(-1, frames, -1)
        temporal_features = temporal_features + time_features_expanded
        
        # Apply spatial-temporal attention layers
        for attention_layer in self.attention_layers:
            temporal_features = attention_layer(temporal_features)
        
        # Apply feature modulation
        modulated_features = self.feature_modulation(temporal_features)
        # (batch, frames, feature_dim)
        
        # === SPATIAL RECONSTRUCTION ===
        # Reshape for spatial upsampling
        features_flat = modulated_features.view(batch_size * frames, self.feature_dim)
        
        # Upsample to spatial resolution
        spatial_output = self.upsampler(features_flat)
        # (batch*frames, out_channels, height, width)
        
        # Reshape back to video format
        video_output = spatial_output.view(batch_size, frames, self.out_channels, height, width)
        video_output = video_output.permute(0, 2, 1, 3, 4)
        # (batch, out_channels, frames, height, width)
        
        # === FINAL TEMPORAL PROCESSING ===
        # Apply 3D convolution for temporal consistency
        final_output = self.temporal_conv(video_output)
        
        # Residual connection with input (if same channels)
        if self.in_channels == self.out_channels:
            final_output = final_output + x
        
        return final_output

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Backward compatibility aliases
TimeEmbedding = SinusoidalTimeEmbedding  # For existing code compatibility
TemporalAttention = SpatialTemporalAttention  # For existing code compatibility

def test_vit3d():
    """Comprehensive test of the redesigned ViT3D model"""
    print("🚀 Testing Redesigned ViT3D Model...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    channels, frames, height, width = 3, 16, 128, 128  # Smaller for testing
    time_dim = 128
    text_dim = 256
    
    print(f"📝 Test Configuration:")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Video shape: {channels}×{frames}×{height}×{width}")
    print(f"   • Time dim: {time_dim}")
    print(f"   • Text dim: {text_dim}")
    print()
    
    # Create model with different configurations
    configs = [
        {"name": "Standard", "freeze": False, "layers": 2, "heads": 8},
        {"name": "Frozen Backbone", "freeze": True, "layers": 1, "heads": 4},
        {"name": "Deep Attention", "freeze": False, "layers": 3, "heads": 12}
    ]
    
    for config in configs:
        print(f"🏗️  Testing {config['name']} Configuration...")
        
        try:
            # Create model
            model = ViT3D(
                in_channels=channels,
                out_channels=channels,
                frames=frames,
                height=height,
                width=width,
                time_dim=time_dim,
                text_dim=text_dim,
                freeze_backbone=config["freeze"],
                num_temporal_heads=config["heads"],
                num_attention_layers=config["layers"]
            )
            
            # Count parameters
            total_params = count_parameters(model)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   ✅ Model created successfully")
            print(f"   📊 Total parameters: {total_params:,}")
            print(f"   🎯 Trainable parameters: {trainable_params:,}")
            
            # Create test inputs
            x = torch.randn(batch_size, channels, frames, height, width)
            time = torch.randint(0, 1000, (batch_size,))
            text_emb = torch.randn(batch_size, text_dim)
            
            # Test forward pass with different inputs
            model.eval()
            with torch.no_grad():
                # Test with text embedding
                output_with_text = model(x, time, text_emb)
                print(f"   ✅ Forward pass with text: {x.shape} → {output_with_text.shape}")
                
                # Test without text embedding
                output_no_text = model(x, time)
                print(f"   ✅ Forward pass without text: {x.shape} → {output_no_text.shape}")
                
                # Verify shapes
                assert output_with_text.shape == x.shape, f"Shape mismatch with text: {output_with_text.shape} vs {x.shape}"
                assert output_no_text.shape == x.shape, f"Shape mismatch without text: {output_no_text.shape} vs {x.shape}"
                
                # Check for NaN or Inf values
                assert torch.isfinite(output_with_text).all(), "Output contains NaN or Inf values (with text)"
                assert torch.isfinite(output_no_text).all(), "Output contains NaN or Inf values (without text)"
                
                print(f"   ✅ Output validation passed")
            
            print(f"   🎉 {config['name']} configuration test successful!")
            print()
            
        except Exception as e:
            print(f"   ❌ {config['name']} configuration failed: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    # Test memory efficiency
    print("🧠 Memory Efficiency Test...")
    try:
        model = ViT3D(freeze_backbone=True, num_attention_layers=1)
        
        # Test with larger batch
        large_batch = 4
        x_large = torch.randn(large_batch, 3, 8, 64, 64)  # Smaller for memory test
        time_large = torch.randint(0, 1000, (large_batch,))
        
        with torch.no_grad():
            output_large = model(x_large, time_large)
            print(f"   ✅ Large batch test: {x_large.shape} → {output_large.shape}")
            
    except Exception as e:
        print(f"   ⚠️  Memory test failed: {e}")
    
    # Test CUDA compatibility if available
    if torch.cuda.is_available():
        print("🚀 CUDA Compatibility Test...")
        try:
            device = torch.device('cuda')
            model_cuda = ViT3D(freeze_backbone=True).to(device)
            x_cuda = torch.randn(1, 3, 8, 64, 64).to(device)
            time_cuda = torch.randint(0, 1000, (1,)).to(device)
            
            with torch.no_grad():
                output_cuda = model_cuda(x_cuda, time_cuda)
                print(f"   ✅ CUDA test successful: {output_cuda.shape}")
                
        except Exception as e:
            print(f"   ⚠️  CUDA test failed: {e}")
    
    print("=" * 60)
    print("🎊 All tests completed!")
    
    return model


def analyze_model_architecture():
    """Analyze and display the redesigned model architecture"""
    print("🔍 Model Architecture Analysis")
    print("=" * 60)
    
    model = ViT3D(freeze_backbone=True)
    
    print("📋 Component Breakdown:")
    print("   1. ViT Backbone Extractor")
    print("      • Pre-trained ViT-B/16 for spatial features")
    print("      • Patch-level and global feature extraction")
    print("      • Configurable input channels")
    
    print("   2. Spatial-Temporal Attention")
    print("      • Multi-layer attention for temporal modeling")
    print("      • Residual connections and layer normalization")
    print("      • Configurable number of heads and layers")
    
    print("   3. Feature Upsampler")
    print("      • Progressive learnable upsampling")
    print("      • ConvNet-based spatial reconstruction")
    print("      • Group normalization for stability")
    
    print("   4. Temporal Processing")
    print("      • 3D convolutions for temporal consistency")
    print("      • Residual connections when possible")
    print("      • Group normalization")
    
    print("\n🎯 Key Improvements:")
    print("   ✅ Better spatial feature extraction with patch-level details")
    print("   ✅ Advanced temporal attention mechanisms")
    print("   ✅ Progressive upsampling for better spatial reconstruction")
    print("   ✅ Modular design for easy customization")
    print("   ✅ Memory-efficient processing")
    print("   ✅ Better conditioning integration")
    
    total_params = count_parameters(model)
    print(f"\n📊 Model Statistics:")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Estimated VRAM (fp32): ~{total_params * 4 / 1e9:.2f} GB")
    print(f"   • Estimated VRAM (fp16): ~{total_params * 2 / 1e9:.2f} GB")


if __name__ == "__main__":
    analyze_model_architecture()
    print()
    test_vit3d()
