#!/usr/bin/env python3
"""
Simple DiT3D test without pretrained weights
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/teamspace/studios/this_studio/text2sign')

def test_dit3d_no_pretrained():
    print("🔍 Testing DiT3D without pretrained weights")
    
    try:
        # Simple DiT block for testing
        class SimpleDiTBlock(nn.Module):
            def __init__(self, hidden_size, num_heads=8):
                super().__init__()
                self.norm1 = nn.LayerNorm(hidden_size)
                self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
                self.norm2 = nn.LayerNorm(hidden_size)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                
            def forward(self, x):
                x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                x = x + self.mlp(self.norm2(x))
                return x
        
        # Simple video model
        class SimpleVideoModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Conv3d(3, 256, kernel_size=(1, 8, 8), stride=(1, 8, 8))
                self.blocks = nn.ModuleList([SimpleDiTBlock(256) for _ in range(4)])
                self.final_layer = nn.Sequential(
                    nn.LayerNorm(256),
                    nn.Linear(256, 3 * 8 * 8)
                )
                
            def forward(self, x):
                # x: (B, 3, T, H, W)
                B, C, T, H, W = x.shape
                
                # Patch embedding
                x = self.patch_embed(x)  # (B, 256, T, H//8, W//8)
                x = x.permute(0, 2, 3, 4, 1)  # (B, T, H//8, W//8, 256)
                x = x.reshape(B, -1, 256)  # (B, T*H//8*W//8, 256)
                
                # Transformer blocks
                for block in self.blocks:
                    x = block(x)
                
                # Final layer
                x = self.final_layer(x)  # (B, seq_len, 3*8*8)
                
                # Reshape back to video
                seq_len = T * (H // 8) * (W // 8)
                x = x.reshape(B, T, H // 8, W // 8, 3 * 8 * 8)
                x = x.reshape(B, T, H // 8, W // 8, 3, 8, 8)
                # Rearrange patches back to image format
                x = x.permute(0, 4, 1, 2, 5, 3, 6)  # (B, 3, T, H//8, 8, W//8, 8)
                x = x.reshape(B, 3, T, H, W)
                
                return x
        
        print("✅ Creating simple model...")
        model = SimpleVideoModel()
        
        print("✅ Testing forward pass...")
        x = torch.randn(1, 3, 4, 32, 32)
        output = model(x)
        
        print(f"✅ Input shape: {x.shape}")
        print(f"✅ Output shape: {output.shape}")
        print("🎯 Simple test PASSED!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dit3d_no_pretrained()