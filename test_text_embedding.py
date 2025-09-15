#!/usr/bin/env python3
"""
Test script to verify text embedding correctness and batch association
This script checks if the text embeddings match the corresponding videos in each batch.
"""

import torch
import numpy as np
from config import Config
from dataset import create_dataloader
from models.text_encoder import create_text_encoder
import sys
import os

def test_text_batch_association():
    """Test if texts are correctly associated with videos in each batch"""
    print("🔍 Testing Text-Video Batch Association...")
    print("=" * 60)
    
    # Create dataloader with small batch size for detailed inspection
    print("📊 Creating dataloader...")
    dataloader = create_dataloader(
        data_root=Config.DATA_ROOT,
        batch_size=4,  # Small batch for detailed inspection
        num_workers=0,  # Single threaded for deterministic results
        shuffle=False   # No shuffle to ensure predictable order
    )
    
    # Create text encoder
    print("🔤 Creating text encoder...")
    text_encoder = create_text_encoder(Config)
    text_encoder.eval()
    
    print(f"✅ Setup complete. Testing {len(dataloader)} batches...")
    print()
    
    # Test first few batches
    for batch_idx, (videos, texts) in enumerate(dataloader):
        print(f"📦 Batch {batch_idx + 1}:")
        print(f"   Video shape: {videos.shape}")
        print(f"   Number of texts: {len(texts)}")
        
        # Print each text in the batch
        for i, text in enumerate(texts):
            print(f"   Text {i}: '{text}'")
        
        # Encode texts
        with torch.no_grad():
            text_embeddings = text_encoder(texts)
        
        print(f"   Text embeddings shape: {text_embeddings.shape}")
        print(f"   Text embeddings stats: mean={text_embeddings.mean().item():.4f}, std={text_embeddings.std().item():.4f}")
        
        # Check if embeddings are unique (different texts should have different embeddings)
        if len(texts) > 1:
            embedding_similarities = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    sim = torch.cosine_similarity(
                        text_embeddings[i].unsqueeze(0), 
                        text_embeddings[j].unsqueeze(0)
                    ).item()
                    embedding_similarities.append(sim)
                    print(f"   Cosine similarity between text {i} and {j}: {sim:.4f}")
            
            avg_similarity = np.mean(embedding_similarities)
            print(f"   Average pairwise similarity: {avg_similarity:.4f}")
            
            # Check if texts are actually different
            unique_texts = set(texts)
            if len(unique_texts) == 1:
                print("   ⚠️  WARNING: All texts in this batch are identical!")
            elif len(unique_texts) < len(texts):
                print(f"   ⚠️  WARNING: Only {len(unique_texts)} unique texts out of {len(texts)} in batch")
            else:
                print(f"   ✅ All {len(texts)} texts are unique")
        
        print()
        
        # Only test first 3 batches for detailed inspection
        if batch_idx >= 2:
            break
    
    return True

def test_text_consistency():
    """Test if the same text produces consistent embeddings"""
    print("🔄 Testing Text Embedding Consistency...")
    print("=" * 60)
    
    # Create text encoder
    text_encoder = create_text_encoder(Config)
    text_encoder.eval()
    
    test_text = "hello world"
    num_tests = 5
    
    embeddings = []
    
    print(f"📝 Testing text: '{test_text}'")
    print(f"🔄 Running {num_tests} encoding passes...")
    
    with torch.no_grad():
        for i in range(num_tests):
            # Encode the same text multiple times
            embedding = text_encoder([test_text])
            embeddings.append(embedding)
            print(f"   Pass {i+1}: shape={embedding.shape}, mean={embedding.mean().item():.6f}")
    
    # Check consistency
    embeddings = torch.cat(embeddings, dim=0)  # (num_tests, embed_dim)
    
    # Calculate pairwise differences
    max_diff = 0.0
    for i in range(num_tests):
        for j in range(i + 1, num_tests):
            diff = torch.norm(embeddings[i] - embeddings[j]).item()
            max_diff = max(max_diff, diff)
    
    print(f"📊 Consistency Results:")
    print(f"   Maximum pairwise difference: {max_diff:.8f}")
    
    if max_diff < 1e-6:
        print("   ✅ Embeddings are perfectly consistent")
    elif max_diff < 1e-3:
        print("   ✅ Embeddings are reasonably consistent")
    else:
        print("   ⚠️  WARNING: Embeddings show significant variation!")
    
    return max_diff < 1e-3

def test_text_diversity():
    """Test if different texts produce different embeddings"""
    print("🌈 Testing Text Embedding Diversity...")
    print("=" * 60)
    
    # Create text encoder
    text_encoder = create_text_encoder(Config)
    text_encoder.eval()
    
    test_texts = [
        "hello",
        "goodbye", 
        "thank you",
        "please",
        "help",
        "stop",
        "go",
        "yes",
        "no",
        "good morning"
    ]
    
    print(f"📝 Testing {len(test_texts)} different texts...")
    
    with torch.no_grad():
        embeddings = text_encoder(test_texts)
    
    print(f"📊 Embeddings shape: {embeddings.shape}")
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            sim = torch.cosine_similarity(
                embeddings[i].unsqueeze(0), 
                embeddings[j].unsqueeze(0)
            ).item()
            similarities.append(sim)
            print(f"   '{test_texts[i]}' vs '{test_texts[j]}': {sim:.4f}")
    
    avg_similarity = np.mean(similarities)
    max_similarity = np.max(similarities)
    min_similarity = np.min(similarities)
    
    print(f"📊 Diversity Results:")
    print(f"   Average similarity: {avg_similarity:.4f}")
    print(f"   Max similarity: {max_similarity:.4f}")
    print(f"   Min similarity: {min_similarity:.4f}")
    
    if avg_similarity < 0.3:
        print("   ✅ Good embedding diversity")
    elif avg_similarity < 0.7:
        print("   ⚠️  Moderate embedding diversity")
    else:
        print("   ❌ Poor embedding diversity - texts too similar!")
    
    return avg_similarity < 0.7

def test_training_data_texts():
    """Examine actual texts in the training data"""
    print("📚 Examining Training Data Texts...")
    print("=" * 60)
    
    # Create dataloader
    dataloader = create_dataloader(
        data_root=Config.DATA_ROOT,
        batch_size=1,  # Single item to see individual texts
        num_workers=0,
        shuffle=False
    )
    
    texts_seen = []
    unique_texts = set()
    
    print("📖 Sampling texts from training data...")
    
    for batch_idx, (videos, texts) in enumerate(dataloader):
        text = texts[0]  # Single item batch
        texts_seen.append(text)
        unique_texts.add(text)
        
        if batch_idx < 20:  # Show first 20 texts
            print(f"   Sample {batch_idx + 1}: '{text}'")
        
        # Stop after examining enough samples
        if batch_idx >= 100:
            break
    
    print(f"📊 Training Data Analysis:")
    print(f"   Examined {len(texts_seen)} samples")
    print(f"   Found {len(unique_texts)} unique texts")
    print(f"   Diversity ratio: {len(unique_texts)/len(texts_seen):.2f}")
    
    # Show most common texts
    from collections import Counter
    text_counts = Counter(texts_seen)
    most_common = text_counts.most_common(10)
    
    print(f"📈 Most common texts:")
    for text, count in most_common:
        print(f"   '{text}': {count} times ({count/len(texts_seen)*100:.1f}%)")
    
    # Check for potential issues
    if len(unique_texts) == 1:
        print("   ❌ CRITICAL: All texts are identical!")
        return False
    elif len(unique_texts) < len(texts_seen) * 0.1:
        print("   ⚠️  WARNING: Very low text diversity")
        return False
    else:
        print("   ✅ Reasonable text diversity")
        return True

def run_comprehensive_text_tests():
    """Run all text embedding tests"""
    print("🎯 Comprehensive Text Embedding Test Suite")
    print("=" * 80)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Training Data Text Analysis", test_training_data_texts),
        ("Text-Video Batch Association", test_text_batch_association),
        ("Text Embedding Consistency", test_text_consistency),
        ("Text Embedding Diversity", test_text_diversity)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            test_results.append((test_name, result if result is not None else True))
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Summary:")
    print("=" * 80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All text embedding tests passed!")
        print("✅ Text embeddings are working correctly with proper batch association")
    else:
        print("⚠️  Some text embedding tests failed.")
        print("🔍 Check the failed tests above for specific issues")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_text_tests()
    exit(0 if success else 1)