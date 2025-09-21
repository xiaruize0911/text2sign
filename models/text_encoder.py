"""
Text encoder for text-to-sign language generation
This module provides text encoding capabilities for conditioning the diffusion model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    """
    Text encoder using pre-trained transformer models
    
    Args:
        model_name (str): Name of the pre-trained model to use
        embed_dim (int): Output embedding dimension
        max_length (int): Maximum sequence length
        freeze_backbone (bool): Whether to freeze the backbone model
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embed_dim: int = 768,
        max_length: int = 77,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Initialize tokenizer and model
        try:
            print(f"Loading text encoder: {model_name} (cached if available)...")
            # Use local cache and optimized loading
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=False,
                use_fast=True,  # Use fast tokenizer
                trust_remote_code=False
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=False,
                trust_remote_code=False
            )
            
            # Add special tokens if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"Warning: Could not load {model_name}, falling back to simple embedding: {e}")
            self.tokenizer = None
            self.model = None
            # Fallback to simple embedding
            self.vocab_size = 10000
            self.simple_embedding = nn.Embedding(self.vocab_size, embed_dim)
            
        # Projection layer to match desired embedding dimension
        if self.model is not None:
            model_dim = self.model.config.hidden_size
            if model_dim != embed_dim:
                self.projection = nn.Linear(model_dim, embed_dim)
            else:
                self.projection = nn.Identity()
        else:
            self.projection = nn.Identity()
            
        # Freeze backbone if requested
        if freeze_backbone and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, text: List[str]) -> torch.Tensor:
        """
        Encode text into embeddings
        
        Args:
            text (List[str]): List of text strings
            
        Returns:
            torch.Tensor: Text embeddings (batch_size, embed_dim)
        """
        if self.model is not None:
            # Use pre-trained model
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(next(self.parameters()).device)
            
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.model(**inputs)
                
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
            
        else:
            # Fallback to simple embedding
            # Simple tokenization by character
            batch_embeddings = []
            for t in text:
                # Simple character-level tokenization
                tokens = [ord(c) % self.vocab_size for c in t[:self.max_length]]
                if len(tokens) == 0:
                    tokens = [0]
                    
                # Pad or truncate
                if len(tokens) < self.max_length:
                    tokens.extend([0] * (self.max_length - len(tokens)))
                else:
                    tokens = tokens[:self.max_length]
                    
                # Convert to tensor and embed
                token_tensor = torch.tensor(tokens, device=next(self.parameters()).device)
                token_embeddings = self.simple_embedding(token_tensor)  # (max_length, embed_dim)
                sentence_embedding = token_embeddings.mean(dim=0)  # (embed_dim,)
                batch_embeddings.append(sentence_embedding)
                
            embeddings = torch.stack(batch_embeddings)  # (batch_size, embed_dim)
            
        # Project to desired dimension
        embeddings = self.projection(embeddings)
        
        return embeddings

class SimpleTextEncoder(nn.Module):
    """
    Simple text encoder using character-level embeddings
    
    Args:
        embed_dim (int): Embedding dimension
        max_length (int): Maximum text length
        vocab_size (int): Vocabulary size
    """
    
    def __init__(self, embed_dim: int = 768, max_length: int = 100, vocab_size: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, text: List[str]) -> torch.Tensor:
        """
        Encode text into embeddings
        
        Args:
            text (List[str]): List of text strings
            
        Returns:
            torch.Tensor: Text embeddings (batch_size, embed_dim)
        """
        batch_size = len(text)
        device = next(self.parameters()).device
        
        # Convert text to character indices
        char_indices = []
        for t in text:
            indices = [ord(c) % self.vocab_size for c in t[:self.max_length]]
            if len(indices) < self.max_length:
                indices.extend([0] * (self.max_length - len(indices)))
            char_indices.append(indices)
            
        # Convert to tensor
        char_tensor = torch.tensor(char_indices, device=device)  # (batch_size, max_length)
        
        # Get embeddings
        embeddings = self.embedding(char_tensor)  # (batch_size, max_length, embed_dim)
        
        # Add positional embeddings
        positions = torch.arange(self.max_length, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # (1, max_length, embed_dim)
        embeddings = embeddings + pos_emb
        
        # Apply transformer
        embeddings = self.transformer(embeddings)
        
        # Pool to get sentence embedding
        sentence_embedding = embeddings.mean(dim=1)  # (batch_size, embed_dim)
        
        # Final projection
        output = self.projection(sentence_embedding)
        
        return output

def create_text_encoder(config) -> TextEncoder:
    """
    Create a text encoder based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        TextEncoder: Configured text encoder
    """
    if hasattr(config, 'TEXT_ENCODER_MODEL'):
        model_name = config.TEXT_ENCODER_MODEL
    else:
        model_name = "distilbert-base-uncased"
        
    embed_dim = getattr(config, 'TEXT_EMBED_DIM', 768)
    max_length = getattr(config, 'TEXT_MAX_LENGTH', 77)
    freeze_backbone = getattr(config, 'TEXT_FREEZE_BACKBONE', True)
    
    try:
        encoder = TextEncoder(
            model_name=model_name,
            embed_dim=embed_dim,
            max_length=max_length,
            freeze_backbone=freeze_backbone
        )
        print(f"✅ Created text encoder with {model_name}")
    except Exception as e:
        print(f"⚠️ Failed to create text encoder with {model_name}: {e}")
        print("Falling back to simple text encoder")
        encoder = SimpleTextEncoder(
            embed_dim=embed_dim,
            max_length=max_length
        )
        
    return encoder

def test_text_encoder():
    """Test the text encoder"""
    print("Testing text encoder...")
    
    # Test with simple encoder
    encoder = SimpleTextEncoder(embed_dim=768, max_length=50)
    
    test_texts = [
        "Hello world",
        "This is a test sentence",
        "Sign language generation"
    ]
    
    embeddings = encoder(test_texts)
    print(f"Input texts: {test_texts}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    
    print("Text encoder test completed!")

if __name__ == "__main__":
    test_text_encoder()
