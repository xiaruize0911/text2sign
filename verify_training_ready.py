#!/usr/bin/env python3
"""
Final verification script - Run this before starting training
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config

def check_config():
    """Verify config settings"""
    print("=" * 70)
    print("1. Configuration Check")
    print("=" * 70)
    
    issues = []
    
    # Check critical settings
    if Config.TINYFUSION_FREEZE_BACKBONE:
        issues.append("❌ TINYFUSION_FREEZE_BACKBONE is True - model can't learn!")
    else:
        print("✅ Backbone is unfrozen (trainable)")
    
    if Config.MODEL_ARCHITECTURE != "tinyfusion":
        issues.append(f"❌ MODEL_ARCHITECTURE is {Config.MODEL_ARCHITECTURE}, should be 'tinyfusion'")
    else:
        print("✅ Model architecture is 'tinyfusion'")
    
    if not os.path.exists(Config.TINYFUSION_CHECKPOINT):
        issues.append(f"❌ Checkpoint not found: {Config.TINYFUSION_CHECKPOINT}")
    else:
        print(f"✅ Checkpoint exists: {Config.TINYFUSION_CHECKPOINT}")
    
    if Config.LEARNING_RATE > 1e-3:
        print(f"⚠️  Warning: High learning rate ({Config.LEARNING_RATE})")
    else:
        print(f"✅ Learning rate: {Config.LEARNING_RATE}")
    
    return len(issues) == 0, issues

def check_model():
    """Verify model can be created and produces output"""
    print("\n" + "=" * 70)
    print("2. Model Check")
    print("=" * 70)
    
    try:
        from models.architectures.tinyfusion import TinyFusionVideoWrapper
        
        model = TinyFusionVideoWrapper(
            video_size=Config.TINYFUSION_VIDEO_SIZE,
            in_channels=3,
            out_channels=3,
            text_dim=Config.TEXT_EMBED_DIM,
            variant=Config.TINYFUSION_VARIANT,
            checkpoint_path=Config.TINYFUSION_CHECKPOINT,
            freeze_backbone=Config.TINYFUSION_FREEZE_BACKBONE,
            enable_temporal_post=Config.TINYFUSION_ENABLE_TEMPORAL_POST,
            temporal_kernel=Config.TINYFUSION_TEMPORAL_KERNEL,
            frame_chunk_size=Config.TINYFUSION_FRAME_CHUNK_SIZE,
        )
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Model created successfully")
        print(f"   Total params: {total:,}")
        print(f"   Trainable params: {trainable:,} ({trainable/total*100:.1f}%)")
        
        if trainable == 0:
            return False, ["❌ No trainable parameters!"]
        
        # Test forward pass
        model.eval()
        x = torch.randn(1, 3, Config.NUM_FRAMES, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        t = torch.tensor([500])
        
        with torch.no_grad():
            out = model(x, t)
        
        zero_pct = (out == 0).sum().item() / out.numel() * 100
        
        if zero_pct > 99:
            return False, [f"❌ Output is {zero_pct:.1f}% zeros"]
        
        if out.std() < 0.001:
            return False, [f"❌ Output has very low variance ({out.std().item():.6f})"]
        
        print(f"✅ Model forward pass works")
        print(f"   Output range: [{out.min():.3f}, {out.max():.3f}]")
        print(f"   Output std: {out.std():.6f}")
        
        return True, []
        
    except Exception as e:
        return False, [f"❌ Model creation failed: {e}"]

def check_training():
    """Verify training setup"""
    print("\n" + "=" * 70)
    print("3. Training Setup Check")
    print("=" * 70)
    
    try:
        from diffusion import create_diffusion_model
        
        model = create_diffusion_model(Config)
        model.train()
        
        # Quick gradient check
        x = torch.randn(1, *Config.INPUT_SHAPE)
        text = ["test"]
        
        loss, _, _ = model(x, text)
        loss.backward()
        
        has_grads = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grads = True
                break
        
        if not has_grads:
            return False, ["❌ No gradients computed!"]
        
        print("✅ Diffusion model created")
        print("✅ Forward pass works")
        print("✅ Backward pass works (gradients computed)")
        
        return True, []
        
    except Exception as e:
        return False, [f"❌ Training setup failed: {e}"]

def main():
    print("\n" + "=" * 70)
    print("TinyFusion Training Verification")
    print("=" * 70)
    print("\nThis script checks if everything is ready for training.\n")
    
    all_passed = True
    all_issues = []
    
    # Run checks
    config_ok, config_issues = check_config()
    all_passed = all_passed and config_ok
    all_issues.extend(config_issues)
    
    model_ok, model_issues = check_model()
    all_passed = all_passed and model_ok
    all_issues.extend(model_issues)
    
    training_ok, training_issues = check_training()
    all_passed = all_passed and training_ok
    all_issues.extend(training_issues)
    
    # Print summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if all_issues:
        print("\n❌ Issues Found:")
        for issue in all_issues:
            print(f"   {issue}")
        print("\nPlease fix these issues before training.")
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED!")
        print("\nYour model is ready for training. You can now run:")
        print("   python main.py train")
        print("\nTo monitor training:")
        print("   python start_tensorboard.sh")
        print("   # Then open http://localhost:6006")
    else:
        print("\n❌ VERIFICATION FAILED")
        print("\nPlease fix the issues above before training.")
    
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
