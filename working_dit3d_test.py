#!/usr/bin/env python3
"""
Working DiT3D implementation for testing
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/teamspace/studios/this_studio/text2sign')

def test_working_dit3d():
    print("🔍 Testing Working DiT3D Implementation")
    
    try:
        # Import our existing text encoder
        from models.text_encoder import TextEncoder
        
        # Simple working DiT3D
        class WorkingDiT3D(nn.Module):
            def __init__(self, video_size=(8, 64, 64), patch_size=(1, 8, 8), 
                         hidden_size=384, num_heads=6, depth=12, text_dim=768):  # Fixed text_dim
                super().__init__()
                
                self.video_size = video_size
                self.patch_size = patch_size
                self.hidden_size = hidden_size
                
                # Patch embedding for 3D videos
                self.patch_embed = nn.Conv3d(
                    3, hidden_size, 
                    kernel_size=patch_size, 
                    stride=patch_size
                )
                
                # Positional embedding
                num_patches = (video_size[0] // patch_size[0]) * \
                             (video_size[1] // patch_size[1]) * \
                             (video_size[2] // patch_size[2])
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
                
                # Time embedding
                self.time_embed = nn.Sequential(
                    nn.Linear(128, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
                
                # Text conditioning
                self.text_proj = nn.Linear(text_dim, hidden_size)
                
                # Transformer blocks
                self.blocks = nn.ModuleList([
                    self._make_block(hidden_size, num_heads) for _ in range(depth)
                ])
                
                # Final layer
                self.norm_final = nn.LayerNorm(hidden_size)
                patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
                self.final_proj = nn.Linear(hidden_size, patch_volume * 3)
                
                # Initialize
                nn.init.normal_(self.pos_embed, std=0.02)
                
            def _make_block(self, hidden_size, num_heads):
                return nn.ModuleDict({
                    'norm1': nn.LayerNorm(hidden_size),
                    'attn': nn.MultiheadAttention(hidden_size, num_heads, batch_first=True),
                    'norm2': nn.LayerNorm(hidden_size),
                    'mlp': nn.Sequential(
                        nn.Linear(hidden_size, hidden_size * 4),
                        nn.GELU(),
                        nn.Linear(hidden_size * 4, hidden_size)
                    ),
                    'adaLN': nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(hidden_size, hidden_size * 2)  # scale and shift
                    )
                })
            
            def timestep_embedding(self, timesteps, dim, max_period=10000):
                """Create sinusoidal timestep embeddings."""
                half = dim // 2
                freqs = torch.exp(
                    -math.log(max_period) * 
                    torch.arange(start=0, end=half, dtype=torch.float32) / half
                ).to(device=timesteps.device)
                args = timesteps[:, None].float() * freqs[None]
                embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                if dim % 2:
                    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
                return embedding
                
            def forward(self, x, t, text_emb):
                """
                Args:
                    x: (B, 3, T, H, W) - Input video
                    t: (B,) - Timesteps
                    text_emb: (B, text_dim) - Text embeddings
                """
                B, C, T, H, W = x.shape
                
                # Patch embedding
                x = self.patch_embed(x)  # (B, hidden_size, T', H', W')
                x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
                
                # Add positional embedding
                x = x + self.pos_embed
                
                # Time embedding
                t_emb = self.timestep_embedding(t, 128)
                t_emb = self.time_embed(t_emb)  # (B, hidden_size)
                
                # Text conditioning
                text_emb = self.text_proj(text_emb)  # (B, hidden_size)
                
                # Combine conditioning
                conditioning = t_emb + text_emb  # (B, hidden_size)
                
                # Transformer blocks
                for block in self.blocks:
                    # AdaLN conditioning
                    scale_shift = block['adaLN'](conditioning)
                    scale, shift = scale_shift.chunk(2, dim=1)
                    
                    # Self-attention with conditioning
                    x_norm = block['norm1'](x)
                    x_norm = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
                    x = x + block['attn'](x_norm, x_norm, x_norm)[0]
                    
                    # MLP
                    x = x + block['mlp'](block['norm2'](x))
                
                # Final layer
                x = self.norm_final(x)
                x = self.final_proj(x)  # (B, num_patches, patch_volume * 3)
                
                # Reshape back to video
                patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
                T_out = T // self.patch_size[0]
                H_out = H // self.patch_size[1] 
                W_out = W // self.patch_size[2]
                
                x = x.reshape(B, T_out, H_out, W_out, patch_volume, 3)
                x = x.permute(0, 5, 1, 4, 2, 3)  # Rearrange dimensions
                x = x.reshape(B, 3, T, H, W)
                
                return x
        
        print("✅ Creating working DiT3D model...")
        model = WorkingDiT3D()
        
        print("✅ Creating text encoder...")
        text_encoder = TextEncoder()
        
        print("✅ Testing forward pass...")
        # Test inputs
        x = torch.randn(2, 3, 8, 64, 64)
        t = torch.randint(0, 1000, (2,))
        
        # Get text embeddings
        texts = ["hello", "world"]  # Text strings
        text_emb = text_encoder(texts)
        
        print(f"   Input video: {x.shape}")
        print(f"   Timesteps: {t.shape}")
        print(f"   Text embeddings: {text_emb.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(x, t, text_emb)
        
        print(f"✅ Output shape: {output.shape}")
        print("🎯 Working DiT3D test PASSED!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import math
    test_working_dit3d()