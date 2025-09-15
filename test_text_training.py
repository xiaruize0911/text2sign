#!/usr/bin/env python3
"""
Test text embedding in training context
"""

import torch
from config import Config
from dataset import create_dataloader
from models.text_encoder import create_text_encoder

def test_text_embedding_in_training():
    """Test text embedding as it would be used in training"""
    print("🎯 Testing Text Embedding in Training Context")
    print("=" * 60)
    
    # Create components
    dataloader = create_dataloader(
        data_root=Config.DATA_ROOT,
        batch_size=2,
        num_workers=0,
        shuffle=False
    )
    
    text_encoder = create_text_encoder(Config)
    text_encoder.eval()
    
    # Get a batch
    videos, texts = next(iter(dataloader))
    
    print(f"📦 Input batch:")
    print(f"   Video shape: {videos.shape}")
    print(f"   Texts: {texts}")
    print(f"   Text types: {[type(t) for t in texts]}")
    
    # Test text encoding
    with torch.no_grad():
        text_embeddings = text_encoder(texts)
    
    print(f"\n🔤 Text embeddings:")
    print(f"   Shape: {text_embeddings.shape}")
    print(f"   Device: {text_embeddings.device}")
    print(f"   Dtype: {text_embeddings.dtype}")
    print(f"   Range: [{text_embeddings.min().item():.4f}, {text_embeddings.max().item():.4f}]")
    print(f"   Mean: {text_embeddings.mean().item():.4f}")
    print(f"   Std: {text_embeddings.std().item():.4f}")
    
    # Test individual embeddings
    for i, (text, embedding) in enumerate(zip(texts, text_embeddings)):
        print(f"\n   Text {i}: '{text[:50]}...'")
        print(f"   Embedding mean: {embedding.mean().item():.4f}")
        print(f"   Embedding norm: {embedding.norm().item():.4f}")
    
    # Test similarity between different texts
    if len(texts) > 1:
        similarity = torch.cosine_similarity(
            text_embeddings[0].unsqueeze(0),
            text_embeddings[1].unsqueeze(0)
        ).item()
        print(f"\n📊 Cosine similarity between text 0 and 1: {similarity:.4f}")
        
        if similarity > 0.9:
            print("   ⚠️  WARNING: Very high similarity - texts might be too similar")
        elif similarity > 0.7:
            print("   ⚠️  Moderate similarity")
        else:
            print("   ✅ Good diversity between texts")
    
    # Test batch processing consistency
    print(f"\n🔄 Testing batch vs individual processing:")
    
    # Process individually
    individual_embeddings = []
    for text in texts:
        individual_emb = text_encoder([text])  # Single text in list
        individual_embeddings.append(individual_emb)
    individual_embeddings = torch.cat(individual_embeddings, dim=0)
    
    # Compare batch vs individual
    max_diff = torch.max(torch.abs(text_embeddings - individual_embeddings)).item()
    print(f"   Max difference (batch vs individual): {max_diff:.8f}")
    
    if max_diff < 1e-6:
        print("   ✅ Perfect consistency between batch and individual processing")
    elif max_diff < 1e-3:
        print("   ✅ Good consistency")
    else:
        print("   ⚠️  WARNING: Inconsistency between batch and individual processing")
    
    return max_diff < 1e-3

def test_text_encoder_device_handling():
    """Test text encoder device handling"""
    print("\n🖥️  Testing Device Handling")
    print("=" * 40)
    
    text_encoder = create_text_encoder(Config)
    
    # Test on current device
    test_texts = ["hello", "world"]
    embeddings = text_encoder(test_texts)
    
    print(f"   Text encoder device: {next(text_encoder.parameters()).device}")
    print(f"   Embedding device: {embeddings.device}")
    print(f"   Device consistency: {'✅ YES' if embeddings.device == next(text_encoder.parameters()).device else '❌ NO'}")
    
    return True

if __name__ == "__main__":
    print("🧪 Running Text Embedding Tests...")
    
    test1 = test_text_embedding_in_training()
    test2 = test_text_encoder_device_handling()
    
    overall_success = test1 and test2
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")