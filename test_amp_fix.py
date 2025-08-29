#!/usr/bin/env python3
"""
Test script to verify AMP scaler fix
"""

import torch
import sys
import os

# Add the project root to Python path
sys.path.append('/teamspace/studios/this_studio/text2sign')

from config import Config
from methods.trainer import setup_training

def test_amp_fix():
    """Test that the AMP scaler works correctly"""
    print("Testing AMP scaler fix...")
    
    # Create config
    config = Config()
    
    # Print device and AMP status
    print(f"Device: {config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"AMP requested: {config.USE_AMP}")
    
    try:
        # Create trainer (this will initialize AMP settings)
        trainer = setup_training(config)
        
        print("✅ Trainer setup successful")
        print(f"✅ AMP enabled: {trainer.use_amp}")
        print(f"✅ Scaler initialized: {trainer.scaler is not None}")
        
        # Test a few training steps
        print("Testing a few training steps...")
        
        # Manually run a few steps to test the scaler
        trainer.model.train()
        num_test_steps = 3
        
        for step in range(num_test_steps):
            print(f"  Step {step + 1}/{num_test_steps}")
            
            # Get one batch from dataloader
            try:
                batch = next(iter(trainer.dataloader))
                videos, texts = batch
                videos = videos.to(trainer.device)
                
                # Zero gradients
                trainer.optimizer.zero_grad()
                
                # Forward pass
                if trainer.use_amp and trainer.scaler is not None:
                    with torch.autocast(device_type='cuda', dtype=trainer.amp_dtype):
                        loss, predicted_noise, noise = trainer.model(videos)
                    
                    # Backward pass
                    trainer.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    trainer.scaler.unscale_(trainer.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), config.GRADIENT_CLIP)
                    
                    # Optimizer step
                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                else:
                    # Regular training without AMP
                    loss, predicted_noise, noise = trainer.model(videos)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), config.GRADIENT_CLIP)
                    trainer.optimizer.step()
                
                print(f"    Loss: {loss.item():.4f}, Grad norm: {grad_norm:.4f}")
                
            except Exception as e:
                print(f"    ❌ Error in step {step + 1}: {e}")
                return False
        
        print("✅ All test steps completed successfully")
        print("✅ AMP scaler fix verified")
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_amp_fix()
    if success:
        print("\n🎉 AMP fix test PASSED - training should work now!")
    else:
        print("\n💥 AMP fix test FAILED - there may still be issues")
