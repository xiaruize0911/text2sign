#!/usr/bin/env python3
"""
Pre-Training Checklist - Run this before starting final training
Ensures all configurations are optimal and ready for production training.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(text):
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)

def check_workspace_clean():
    """Check that workspace is clean of test/debug files"""
    print_header("Workspace Cleanliness Check")
    
    test_patterns = [
        'test_*.py',
        'debug_*.py',
        'trace_*.py',
        'analyze_*.py',
        'check_*.py'
    ]
    
    import glob
    found_files = []
    for pattern in test_patterns:
        found_files.extend(glob.glob(pattern))
    
    if found_files:
        print("⚠️  Found test/debug files:")
        for f in found_files:
            print(f"   - {f}")
        return False
    else:
        print("✅ Workspace is clean - no test/debug files found")
        return True

def check_config():
    """Verify production-ready configuration"""
    print_header("Configuration Check")
    
    from config import Config
    
    issues = []
    warnings = []
    
    # Critical settings
    if Config.MODEL_ARCHITECTURE != "tinyfusion":
        issues.append(f"MODEL_ARCHITECTURE should be 'tinyfusion', is '{Config.MODEL_ARCHITECTURE}'")
    else:
        print("✅ Model architecture: tinyfusion")
    
    if Config.TINYFUSION_FREEZE_BACKBONE:
        issues.append("TINYFUSION_FREEZE_BACKBONE must be False for training")
    else:
        print("✅ Backbone is trainable (not frozen)")
    
    if not os.path.exists(Config.TINYFUSION_CHECKPOINT):
        issues.append(f"Checkpoint not found: {Config.TINYFUSION_CHECKPOINT}")
    else:
        print(f"✅ Checkpoint exists: {Config.TINYFUSION_CHECKPOINT}")
    
    # Training settings
    print(f"\n📊 Training Settings:")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    print(f"   Learning rate: {Config.LEARNING_RATE}")
    print(f"   Epochs: {Config.NUM_EPOCHS}")
    print(f"   Timesteps: {Config.TIMESTEPS}")
    print(f"   Noise scheduler: {Config.NOISE_SCHEDULER}")
    
    # Video settings
    print(f"\n🎬 Video Settings:")
    print(f"   Frames: {Config.NUM_FRAMES}")
    print(f"   Resolution: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"   Input shape: {Config.INPUT_SHAPE}")
    
    # Memory optimization
    print(f"\n💾 Memory Optimization:")
    print(f"   Frame chunk size: {Config.TINYFUSION_FRAME_CHUNK_SIZE}")
    print(f"   Gradient accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Mixed precision: {Config.USE_MIXED_PRECISION}")
    print(f"   Gradient checkpointing: {Config.GRADIENT_CHECKPOINTING}")
    
    # Logging
    print(f"\n📝 Logging:")
    print(f"   Experiment: {Config.EXPERIMENT_NAME}")
    print(f"   Log dir: {Config.LOG_DIR}")
    print(f"   Checkpoint dir: {Config.CHECKPOINT_DIR}")
    print(f"   Sample every: {Config.SAMPLE_EVERY_EPOCHS} epochs")
    print(f"   Save every: {Config.SAVE_EVERY_EPOCHS} epochs")
    
    # Warnings
    if Config.BATCH_SIZE > 2:
        warnings.append(f"Large batch size ({Config.BATCH_SIZE}) may cause OOM")
    
    if Config.LEARNING_RATE > 1e-3:
        warnings.append(f"High learning rate ({Config.LEARNING_RATE})")
    
    if Config.NUM_FRAMES > 28:
        warnings.append(f"Many frames ({Config.NUM_FRAMES}) increases memory usage")
    
    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"   - {w}")
    
    return len(issues) == 0, issues

def check_directories():
    """Ensure required directories exist"""
    print_header("Directory Check")
    
    from config import Config
    
    required_dirs = [
        'training_data',
        'pretrained',
        'external/TinyFusion'
    ]
    
    created_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing: {dir_path}")
            return False
        print(f"✅ {dir_path}")
    
    # Create output directories if needed
    output_dirs = [
        Config.LOG_DIR,
        Config.CHECKPOINT_DIR,
        Config.SAMPLES_DIR
    ]
    
    for dir_path in output_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            created_dirs.append(dir_path)
    
    if created_dirs:
        print(f"\n📁 Created output directories:")
        for d in created_dirs:
            print(f"   - {d}")
    
    return True

def check_data():
    """Verify training data exists"""
    print_header("Training Data Check")
    
    import glob
    
    gif_files = glob.glob('training_data/*.gif')
    txt_files = glob.glob('training_data/*.txt')
    
    print(f"Found {len(gif_files)} GIF files")
    print(f"Found {len(txt_files)} text files")
    
    if len(gif_files) == 0:
        print("❌ No training data found!")
        return False
    
    if len(txt_files) == 0:
        print("⚠️  No text descriptions found")
    
    print(f"✅ Training data available ({len(gif_files)} samples)")
    return True

def final_model_check():
    """Quick model creation test"""
    print_header("Model Creation Test")
    
    try:
        from diffusion import create_diffusion_model
        from config import Config
        
        print("Creating model...")
        model = create_diffusion_model(Config)
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully")
        print(f"   Total params: {total:,}")
        print(f"   Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
        
        if trainable == 0:
            print("❌ ERROR: No trainable parameters!")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def main():
    print("\n" + "=" * 70)
    print("PRE-TRAINING FINAL CHECKLIST".center(70))
    print("=" * 70)
    print("\nThis script performs final verification before production training.")
    
    all_passed = True
    
    # Run all checks
    checks = [
        ("Workspace Clean", check_workspace_clean),
        ("Configuration", check_config),
        ("Directories", check_directories),
        ("Training Data", check_data),
        ("Model Creation", final_model_check)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            if callable(check_func):
                result = check_func()
            else:
                result = check_func
            results.append((check_name, result))
            all_passed = all_passed and result
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}")
            results.append((check_name, False))
            all_passed = False
    
    # Summary
    print_header("FINAL CHECKLIST SUMMARY")
    
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED - READY FOR TRAINING! 🎉".center(70))
        print("=" * 70)
        print("\n📚 Quick Start Commands:")
        print("   1. Start training:  python main.py train")
        print("   2. Monitor:         python start_tensorboard.sh")
        print("   3. Resume:          python main.py train --resume")
        print("\n💡 Training Tips:")
        print("   - Monitor loss in TensorBoard (should decrease within 50 steps)")
        print("   - Check generated samples every 5 epochs")
        print("   - Training recommended for 1000+ epochs")
        print("   - Checkpoints saved every 10 epochs")
        print("\n🔗 TensorBoard will be at: http://localhost:6006")
        print("=" * 70 + "\n")
        return 0
    else:
        print("⚠️  CHECKLIST FAILED - FIX ISSUES BEFORE TRAINING".center(70))
        print("=" * 70)
        print("\nPlease fix the issues above before starting training.")
        print("=" * 70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
