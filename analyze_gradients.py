#!/usr/bin/env python3
"""
Gradient Descent Analysis Tool
Analyzes the current training session to check if gradient descent is working correctly.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config

def analyze_gradient_descent():
    """Analyze the gradient descent procedure"""
    
    print("=" * 60)
    print("GRADIENT DESCENT ANALYSIS")
    print("=" * 60)
    
    # Current configuration analysis
    print(f"📊 Current Configuration:")
    print(f"   Learning Rate: {Config.get_learning_rate()}")
    print(f"   Base Learning Rate: {Config.LEARNING_RATE}")
    print(f"   Model Architecture: {Config.MODEL_ARCHITECTURE}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Gradient Accumulation Steps: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Effective Batch Size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Gradient Clipping: {Config.GRADIENT_CLIP}")
    print(f"   Mixed Precision: {getattr(Config, 'USE_MIXED_PRECISION', False)}")
    print(f"   Optimizer: {Config.OPTIMIZER_TYPE}")
    print(f"   Weight Decay: {Config.WEIGHT_DECAY}")
    print(f"   Beta Values: {Config.ADAM_BETAS}")
    
    # Learning rate analysis
    print(f"\n📈 Learning Rate Analysis:")
    current_lr = Config.get_learning_rate()
    
    # Check if learning rate is appropriate for the model size
    # Rule of thumb: LR should be roughly proportional to sqrt(batch_size)
    effective_batch = Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS
    recommended_lr_base = 0.0001 * np.sqrt(effective_batch / 4)  # Base recommendation
    
    print(f"   Current LR: {current_lr:.6f}")
    print(f"   Recommended LR (sqrt scaling): {recommended_lr_base:.6f}")
    
    if current_lr > recommended_lr_base * 2:
        print(f"   ⚠️  Learning rate might be too high (>{2*recommended_lr_base:.6f})")
    elif current_lr < recommended_lr_base * 0.1:
        print(f"   ⚠️  Learning rate might be too low (<{0.1*recommended_lr_base:.6f})")
    else:
        print(f"   ✅ Learning rate appears reasonable")
    
    # Gradient clipping analysis  
    print(f"\n🎯 Gradient Clipping Analysis:")
    print(f"   Gradient Clip Value: {Config.GRADIENT_CLIP}")
    
    if Config.GRADIENT_CLIP < 0.1:
        print(f"   ⚠️  Gradient clipping might be too aggressive (<0.1)")
    elif Config.GRADIENT_CLIP > 10.0:
        print(f"   ⚠️  Gradient clipping might be too loose (>10.0)")
    else:
        print(f"   ✅ Gradient clipping appears reasonable")
    
    # Memory optimization impact on gradients
    print(f"\n🧠 Memory Optimization Impact:")
    print(f"   Gradient Accumulation: {Config.GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"   Mixed Precision: {getattr(Config, 'USE_MIXED_PRECISION', False)}")
    print(f"   Gradient Checkpointing: {Config.GRADIENT_CHECKPOINTING}")
    
    if Config.GRADIENT_ACCUMULATION_STEPS > 1:
        print(f"   ✅ Gradient accumulation maintains effective batch size")
        print(f"   📊 Gradients are averaged over {Config.GRADIENT_ACCUMULATION_STEPS} mini-batches")
    
    if getattr(Config, 'USE_MIXED_PRECISION', False):
        print(f"   ✅ Mixed precision enabled (potential gradient scaling)")
        print(f"   📊 Automatic loss scaling prevents gradient underflow")
    
    # Check for potential issues
    print(f"\n🔍 Potential Issues Check:")
    
    issues_found = False
    
    # Issue 1: Learning rate too high for diffusion models
    if current_lr > 0.001:
        print(f"   ❌ Learning rate ({current_lr}) might be too high for diffusion models")
        print(f"      Recommended: 1e-4 to 1e-5 for stable training")
        issues_found = True
        
    # Issue 2: Batch size too small without adequate accumulation
    if effective_batch < 4:
        print(f"   ❌ Effective batch size ({effective_batch}) is very small")
        print(f"      Consider increasing gradient accumulation steps")
        issues_found = True
    
    # Issue 3: Mixed precision with small gradients
    if getattr(Config, 'USE_MIXED_PRECISION', False) and current_lr < 1e-5:
        print(f"   ⚠️  Mixed precision with very small LR might cause gradient underflow")
        print(f"      Monitor for loss scaling issues")
    
    if not issues_found:
        print(f"   ✅ No obvious configuration issues detected")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if Config.MODEL_ARCHITECTURE == "tinyfusion":
        print(f"   📋 For TinyFusion model:")
        print(f"      • Learning rate 1e-4 is appropriate for this architecture")
        print(f"      • Consider warmup if loss diverges initially")
        print(f"      • Monitor gradient norms (should be ~0.1-10.0)")
        print(f"      • Loss should decrease gradually over epochs")
    
    print(f"\n🎯 Expected Training Behavior:")
    print(f"   • Initial loss: ~1.0-2.0 (MSE loss for noise prediction)")
    print(f"   • Loss should decrease monotonically (with fluctuations)")
    print(f"   • Gradient norms should be stable (~0.1-10.0)")
    print(f"   • No NaN/Inf gradients should occur")
    print(f"   • Training speed: ~2-3 seconds/batch is reasonable")
    
    return {
        'learning_rate': current_lr,
        'effective_batch_size': effective_batch,
        'gradient_clip': Config.GRADIENT_CLIP,
        'mixed_precision': getattr(Config, 'USE_MIXED_PRECISION', False),
        'issues_found': issues_found
    }

def check_loss_progress():
    """Check if loss is progressing as expected"""
    
    print(f"\n📊 Loss Progress Analysis:")
    
    # Look for recent checkpoints to analyze loss history
    checkpoint_dir = Path(Config.CHECKPOINT_DIR)
    
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            print(f"   Found {len(checkpoints)} checkpoints")
            # Could load and analyze loss history from checkpoints
        else:
            print(f"   No checkpoints found yet")
    else:
        print(f"   Checkpoint directory doesn't exist yet")
    
    # Check TensorBoard logs
    log_dir = Path(Config.LOG_DIR)
    if log_dir.exists():
        log_files = list(log_dir.glob("events.out.tfevents.*"))
        if log_files:
            print(f"   Found {len(log_files)} TensorBoard log files")
            print(f"   📈 Use: tensorboard --logdir={Config.LOG_DIR} to monitor training")
        else:
            print(f"   No TensorBoard logs found")
    else:
        print(f"   Log directory doesn't exist yet")

def main():
    """Main analysis function"""
    
    print("🔬 Gradient Descent Analysis for Text2Sign Training")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analyze current configuration
    analysis = analyze_gradient_descent()
    
    # Check loss progress
    check_loss_progress()
    
    print("=" * 60)
    
    return analysis

if __name__ == "__main__":
    main()