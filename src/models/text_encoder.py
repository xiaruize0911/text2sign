import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from typing import List, Union, Optional


class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder for conditioning the diffusion model.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        # Freeze the text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, text: List[str]) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: List of text strings
            
        Returns:
            Text embeddings of shape (batch_size, embed_dim)
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        # Move to same device as model
        inputs = {k: v.to(next(self.text_encoder.parameters()).device) for k, v in inputs.items()}
        
        # Get text embeddings
        outputs = self.text_encoder(**inputs)
        
        # Return pooled output (CLS token embedding)
        return outputs.pooler_output


class T5TextEncoder(nn.Module):
    """
    T5 text encoder for conditioning the diffusion model.
    """
    
    def __init__(self, model_name: str = "t5-small"):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.text_encoder = T5EncoderModel.from_pretrained(model_name)
        
        # Freeze the text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, text: List[str]) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: List of text strings
            
        Returns:
            Text embeddings of shape (batch_size, embed_dim)
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to same device as model
        inputs = {k: v.to(next(self.text_encoder.parameters()).device) for k, v in inputs.items()}
        
        # Get text embeddings
        outputs = self.text_encoder(**inputs)
        
        # Return mean pooled embeddings
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        
        # Mean pooling with attention mask
        embeddings = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        return embeddings


class SimpleTextEncoder(nn.Module):
    """
    Simple text encoder using word embeddings and LSTM.
    Useful for smaller datasets or when you want to train the text encoder.
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Simple tokenizer (word-based)
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = vocab_size
        self.current_vocab_size = 2
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word not in self.word_to_idx and self.current_vocab_size < self.vocab_size:
                    self.word_to_idx[word] = self.current_vocab_size
                    self.idx_to_word[self.current_vocab_size] = word
                    self.current_vocab_size += 1
    
    def tokenize(self, texts: List[str], max_length: int = 77) -> torch.Tensor:
        """Tokenize texts to indices."""
        tokenized = []
        for text in texts:
            words = text.lower().split()
            indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
            
            # Pad or truncate
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([0] * (max_length - len(indices)))  # 0 is <PAD>
            
            tokenized.append(indices)
        
        return torch.tensor(tokenized, dtype=torch.long)
    
    def forward(self, text: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: List of text strings or tensor of token indices
            
        Returns:
            Text embeddings of shape (batch_size, embed_dim)
        """
        if isinstance(text, list):
            # Tokenize if input is list of strings
            tokens = self.tokenize(text)
        else:
            tokens = text
        
        # Move to same device as model
        tokens = tokens.to(next(self.embedding.parameters()).device)
        
        # Embed tokens
        embeddings = self.embedding(tokens)  # (batch_size, seq_len, embed_dim)
        embeddings = self.dropout(embeddings)
        
        # Pass through LSTM
        lstm_out, (hidden, _) = self.lstm(embeddings)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Create attention mask (non-zero tokens)
        attention_mask = (tokens != 0).float().unsqueeze(-1)
        
        # Mean pooling with attention mask
        pooled = (lstm_out * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        # Project to final dimension
        output = self.projection(pooled)
        
        return output


def test_text_encoders():
    """Test the text encoders."""
    texts = [
        "Hello, how are you?",
        "I am learning sign language.",
        "This is a test sentence for text encoding."
    ]
    
    print("Testing CLIP Text Encoder...")
    try:
        clip_encoder = CLIPTextEncoder()
        clip_embeddings = clip_encoder(texts)
        print(f"CLIP embeddings shape: {clip_embeddings.shape}")
    except Exception as e:
        print(f"CLIP encoder failed: {e}")
    
    print("\nTesting T5 Text Encoder...")
    try:
        t5_encoder = T5TextEncoder()
        t5_embeddings = t5_encoder(texts)
        print(f"T5 embeddings shape: {t5_embeddings.shape}")
    except Exception as e:
        print(f"T5 encoder failed: {e}")
    
    print("\nTesting Simple Text Encoder...")
    simple_encoder = SimpleTextEncoder(vocab_size=1000, embed_dim=512)
    
    # Build vocabulary
    simple_encoder.build_vocab(texts)
    print(f"Vocabulary size: {simple_encoder.current_vocab_size}")
    
    # Test encoding
    simple_embeddings = simple_encoder(texts)
    print(f"Simple embeddings shape: {simple_embeddings.shape}")
    
    # Test with token tensor
    tokens = simple_encoder.tokenize(texts)
    print(f"Tokens shape: {tokens.shape}")
    
    embeddings_from_tokens = simple_encoder(tokens)
    print(f"Embeddings from tokens shape: {embeddings_from_tokens.shape}")
    
    # Verify they're the same
    print(f"Embeddings match: {torch.allclose(simple_embeddings, embeddings_from_tokens)}")


if __name__ == "__main__":
    test_text_encoders()
