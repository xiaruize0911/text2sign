#!/usr/bin/env python3
"""
Quick test to check text-batch association
"""

import torch
from config import Config
from dataset import create_dataloader

def quick_text_batch_test():
    """Quick test for text-batch association"""
    print("🔍 Quick Text-Batch Association Test")
    print("=" * 50)
    
    # Create small dataloader
    dataloader = create_dataloader(
        data_root=Config.DATA_ROOT,
        batch_size=3,
        num_workers=0,
        shuffle=False
    )
    
    # Test first batch
    videos, texts = next(iter(dataloader))
    
    print(f"✅ Batch loaded successfully:")
    print(f"   Video shape: {videos.shape}")
    print(f"   Number of texts: {len(texts)}")
    print(f"   Texts:")
    
    for i, text in enumerate(texts):
        print(f"     {i}: '{text[:100]}...' (length: {len(text)})")
    
    # Check if texts are different
    unique_texts = set(texts)
    print(f"\n📊 Analysis:")
    print(f"   Unique texts in batch: {len(unique_texts)}")
    print(f"   All texts unique: {'✅ YES' if len(unique_texts) == len(texts) else '❌ NO'}")
    
    if len(unique_texts) != len(texts):
        print("   ⚠️  WARNING: Some texts are duplicated in this batch!")
        for i, text in enumerate(texts):
            duplicates = [j for j, other_text in enumerate(texts) if j != i and other_text == text]
            if duplicates:
                print(f"      Text {i} is duplicated at positions: {duplicates}")
    
    return len(unique_texts) == len(texts)

if __name__ == "__main__":
    success = quick_text_batch_test()
    print(f"\n🎯 Result: {'✅ PASSED' if success else '❌ FAILED'}")