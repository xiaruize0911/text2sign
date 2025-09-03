#!/usr/bin/env python3
"""
Test script to validate epoch-based logging frequencies
"""

from config import Config

def test_logging_frequencies():
    """Test the new epoch-based logging frequencies"""
    
    print("🧪 Testing Epoch-based Logging Frequencies")
    print("=" * 50)
    
    # Test configuration validation
    try:
        Config.validate_config()
        print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Print epoch-based frequencies
    print(f"\n📊 Epoch-based Logging Configuration:")
    print(f"  Sample every: {Config.SAMPLE_EVERY_EPOCHS} epochs")
    print(f"  Log every: {Config.LOG_EVERY_EPOCHS} epochs") 
    print(f"  Save checkpoint every: {Config.SAVE_EVERY_EPOCHS} epochs")
    print(f"  Log parameters every: {Config.PARAM_LOG_EVERY_EPOCHS} epochs")
    print(f"  Log summary every: {Config.SUMMARY_LOG_EVERY_EPOCHS} epochs")
    
    # Print step-based diagnostic frequencies  
    print(f"\n🔍 Step-based Diagnostic Configuration:")
    print(f"  Noise display every: {Config.NOISE_DISPLAY_EVERY_STEPS} steps")
    print(f"  Diagnostic log every: {Config.DIAGNOSTIC_LOG_EVERY_STEPS} steps")
    print(f"  TensorBoard flush every: {Config.TENSORBOARD_FLUSH_EVERY_STEPS} steps")
    
    # Simulate training calculations
    batch_size = Config.BATCH_SIZE
    num_epochs = 5  # Test with few epochs
    
    # Estimate steps per epoch (this would normally come from dataloader)
    # For testing, assume we have 100 samples
    estimated_samples = 100
    steps_per_epoch = estimated_samples // batch_size
    
    print(f"\n🎯 Example Training Calculation:")
    print(f"  Estimated samples: {estimated_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Test epochs: {num_epochs}")
    
    print(f"\n📈 Logging Schedule for {num_epochs} epochs:")
    
    for epoch in range(num_epochs):
        messages = []
        
        # Check epoch-based events
        if epoch % Config.SAMPLE_EVERY_EPOCHS == 0:
            messages.append("📱 Generate samples")
        if epoch % Config.LOG_EVERY_EPOCHS == 0:
            messages.append("📝 Log metrics")
        if epoch % Config.SAVE_EVERY_EPOCHS == 0:
            messages.append("💾 Save checkpoint")
        if epoch % Config.PARAM_LOG_EVERY_EPOCHS == 0:
            messages.append("📊 Log parameters")
        if epoch % Config.SUMMARY_LOG_EVERY_EPOCHS == 0:
            messages.append("📋 Log summary")
        
        if messages:
            print(f"  Epoch {epoch}: {', '.join(messages)}")
        else:
            print(f"  Epoch {epoch}: Regular training")
    
    print(f"\n✅ Epoch-based logging frequency test completed successfully!")
    return True

if __name__ == "__main__":
    test_logging_frequencies()
