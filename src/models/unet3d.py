import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock3D(nn.Module):
    """3D Attention block for spatial-temporal attention."""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        h = self.norm(x)
        
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = rearrange(q, 'b (heads dim) t height width -> b heads (t height width) dim', heads=self.num_heads)
        k = rearrange(k, 'b (heads dim) t height width -> b heads (t height width) dim', heads=self.num_heads)
        v = rearrange(v, 'b (heads dim) t height width -> b heads (t height width) dim', heads=self.num_heads)
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b heads (t height width) d -> b (heads d) t height width', 
                       heads=self.num_heads, t=T, height=H, width=W)
        
        out = self.proj_out(out)
        return x + out


class ResnetBlock3D(nn.Module):
    """3D ResNet block with time embedding conditioning."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
        h = h + time_emb
        
        h = self.block2(h)
        return h + self.shortcut(x)


class Downsample3D(nn.Module):
    """3D downsampling layer."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling layer."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D UNet for video diffusion with text conditioning.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        model_channels: Base number of channels
        num_res_blocks: Number of ResNet blocks per level
        attention_resolutions: Resolutions to apply attention at
        channel_mult: Channel multipliers for each level
        num_heads: Number of attention heads
        dropout: Dropout rate
        text_embed_dim: Dimension of text embeddings
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=32,
        num_res_blocks=1,
        attention_resolutions=[32, 64],
        channel_mult=[1, 2, 3],
        num_heads=2,
        dropout=0.1,
        text_embed_dim=256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Text conditioning
        self.text_proj = nn.Linear(text_embed_dim, time_embed_dim)
        
        # Input conv
        self.input_blocks = nn.ModuleList([
            nn.Conv3d(in_channels, model_channels, 3, padding=1)
        ])
        
        # Downsampling blocks
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResnetBlock3D(ch, mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock3D(ch, num_heads))
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample3D(ch))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResnetBlock3D(ch, ch, time_embed_dim, dropout),
            AttentionBlock3D(ch, num_heads),
            ResnetBlock3D(ch, ch, time_embed_dim, dropout)
        )
        
        # Upsampling blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResnetBlock3D(ch + ich, mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock3D(ch, num_heads))
                
                if level and i == num_res_blocks:
                    layers.append(Upsample3D(ch))
                    ds //= 2
                
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Output conv
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps, text_embeds=None):
        """
        Forward pass of UNet3D.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            timesteps: Timestep tensor of shape (B,)
            text_embeds: Text embeddings of shape (B, text_embed_dim)
        
        Returns:
            Output tensor of shape (B, C, T, H, W)
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Text conditioning
        if text_embeds is not None:
            text_emb = self.text_proj(text_embeds)
            t_emb = t_emb + text_emb
        
        # Store skip connections
        hs = []
        
        # Input
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResnetBlock3D):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResnetBlock3D):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Output
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResnetBlock3D):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
        
        return self.out(h)


def test_unet3d():
    """Test the UNet3D model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = UNet3D(
        in_channels=3,
        out_channels=3,
        model_channels=64,  # Reduced for testing
        num_res_blocks=1,
        attention_resolutions=[8, 16],
        channel_mult=[1, 2, 4],
        num_heads=4,
        text_embed_dim=512
    ).to(device)
    
    # Test input
    batch_size = 2
    frames = 16
    height = 64
    width = 64
    
    x = torch.randn(batch_size, 3, frames, height, width).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    text_embeds = torch.randn(batch_size, 512).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, timesteps, text_embeds)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    test_unet3d()
