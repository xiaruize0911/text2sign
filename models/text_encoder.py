"""
Text encoder for conditioning the diffusion model
Uses a simple transformer architecture
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        x2, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        # Feed forward
        x2 = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        
        return x


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder for conditioning
    Similar to CLIP text encoder but simplified
    """
    def __init__(
        self,
        vocab_size: int = 49408,
        max_length: int = 77,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_length)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embed_dim,
                num_heads=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
    
    def forward(
        self,
        tokens: torch.Tensor,  # (B, seq_len)
        return_pooled: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass
        Args:
            tokens: Token IDs (B, seq_len)
            return_pooled: Whether to return pooled output (first token)
        Returns:
            Text embeddings (B, seq_len, embed_dim) or (B, embed_dim) if pooled
        """
        # Token embedding
        x = self.token_embedding(tokens)  # (B, seq_len, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask for padding (token_id == 2)
        padding_mask = (tokens == 2)  # pad_token_id = 2
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=padding_mask)
        
        # Final norm
        x = self.final_norm(x)
        
        if return_pooled:
            # Return first token embedding (like [CLS])
            return x[:, 0]
        
        return x


class FrozenCLIPTextEncoder(nn.Module):
    """
    Wrapper for using pretrained CLIP text encoder (if available)
    Falls back to custom TextEncoder if CLIP is not available
    """
    def __init__(
        self,
        embed_dim: int = 512,
        max_length: int = 77,
        trainable_layers: int = 0,
        projection_mode: str = "identity",
        projection_hidden_mult: int = 2,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.trainable_layers = max(0, trainable_layers)
        self.projection_mode = projection_mode
        self.projection_hidden_mult = max(1, projection_hidden_mult)
        
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # Freeze the model
            for param in self.model.parameters():
                param.requires_grad = False

            if self.trainable_layers > 0:
                encoder_layers = list(self.model.text_model.encoder.layers)
                for layer in encoder_layers[-self.trainable_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                for param in self.model.text_model.final_layer_norm.parameters():
                    param.requires_grad = True
            
            # Project/adapt CLIP hidden states to target conditioning space
            clip_dim = self.model.config.hidden_size
            if projection_mode == "mlp":
                hidden_dim = max(embed_dim, clip_dim) * self.projection_hidden_mult
                self.proj = nn.Sequential(
                    nn.Linear(clip_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embed_dim),
                )
            elif projection_mode == "linear" or clip_dim != embed_dim:
                self.proj = nn.Linear(clip_dim, embed_dim)
            else:
                self.proj = nn.Identity()
            
            self.use_clip = True
            if self.trainable_layers > 0:
                print(
                    f"Using CLIP text encoder with last {self.trainable_layers} layers trainable "
                    f"and projection_mode={projection_mode}"
                )
            else:
                print(f"Using frozen pretrained CLIP text encoder (projection_mode={projection_mode})")
            
        except Exception as e:
            print(f"CLIP not available ({e}), using custom text encoder")
            self.model = TextEncoder(
                embed_dim=embed_dim,
                max_length=max_length,
            )
            self.proj = nn.Identity()
            self.use_clip = False
    
    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        text: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        Args:
            tokens: Pre-tokenized token IDs (B, seq_len)
            text: List of text strings
        Returns:
            Text embeddings (B, seq_len, embed_dim)
        """
        if self.use_clip:
            if tokens is None and text is not None:
                # Fallback to tokenizing here if not pre-tokenized
                inputs = self.tokenizer(
                    text,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                tokens = inputs["input_ids"].to(next(self.model.parameters()).device)
            elif tokens is not None:
                # Use pre-tokenized tokens
                tokens = tokens.to(next(self.model.parameters()).device)
            else:
                raise ValueError("Either tokens or text must be provided")
            
            if self.trainable_layers == 0:
                with torch.no_grad():
                    outputs = self.model(tokens)
                    hidden_states = outputs.last_hidden_state
            else:
                outputs = self.model(tokens)
                hidden_states = outputs.last_hidden_state
            
            return self.proj(hidden_states)
        else:
            return self.proj(self.model(tokens))


def create_text_encoder(config, use_clip: bool = True):
    """Create text encoder from config (default: pretrained CLIP)"""
    if use_clip:
        return FrozenCLIPTextEncoder(
            embed_dim=config.text_embed_dim,
            max_length=config.max_text_length,
            trainable_layers=getattr(config, "clip_trainable_layers", 0),
            projection_mode=getattr(config, "text_projection_mode", "identity"),
            projection_hidden_mult=getattr(config, "text_projection_hidden_mult", 2),
        )
    else:
        return TextEncoder(
            vocab_size=config.vocab_size,
            max_length=config.max_text_length,
            embed_dim=config.text_embed_dim,
        )


if __name__ == "__main__":
    # Test the encoder
    encoder = TextEncoder(
        vocab_size=49408,
        max_length=77,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
    )
    
    # Test input
    tokens = torch.randint(0, 49408, (2, 77))
    
    # Forward pass
    output = encoder(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
