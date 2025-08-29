#!/usr/bin/env python3
"""
Quick test to verify training works without AMP scaler errors
"""

import sys
import os
sys.path.append('/teamspace/studios/this_studio/text2sign')

import torch
from config import Config

def main():
    print("=== AMP Scaler Fix Verification ===")
    
    config = Config()
    print(f"Device: {config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"AMP enabled: {config.USE_AMP}")
    
    # Test if we can import and create the trainer without errors
    try:
        from methods.trainer import Trainer
        from dataset import create_dataloader
        from diffusion import create_diffusion_model
        import torch.optim as optim
        
        print("\n=== Creating Components ===")
        
        # Create a minimal dataloader (just 1 batch for testing)
        print("Creating dataloader...")
        dataloader = create_dataloader(
            data_root=config.DATA_ROOT,
            batch_size=1,  # Small batch for testing
            num_workers=0,  # No multiprocessing for testing
            shuffle=False
        )
        
        # Create model
        print("Creating model...")
        model = create_diffusion_model(config)
        model.to(config.DEVICE)
        
        # Create optimizer
        print("Creating optimizer...")
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        
        # Create trainer
        print("Creating trainer...")
        trainer = Trainer(
            config=config,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=None
        )
        
        print(f"✅ Trainer created successfully")
        print(f"✅ AMP enabled: {trainer.use_amp}")
        print(f"✅ Scaler available: {trainer.scaler is not None}")
        
        # Test one training step
        print("\n=== Testing One Training Step ===")
        trainer.model.train()
        
        # Get one batch
        batch = next(iter(dataloader))
        videos, texts = batch
        videos = videos.to(trainer.device)
        
        # Test forward pass
        trainer.optimizer.zero_grad()
        
        if trainer.use_amp and trainer.scaler is not None:
            print("Testing AMP training step...")
            with torch.autocast(device_type='cuda', dtype=trainer.amp_dtype):
                loss, predicted_noise, noise = trainer.model(videos)
            
            trainer.scaler.scale(loss).backward()
            trainer.scaler.unscale_(trainer.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), config.GRADIENT_CLIP)
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        else:
            print("Testing regular training step (no AMP)...")
            loss, predicted_noise, noise = trainer.model(videos)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), config.GRADIENT_CLIP)
            trainer.optimizer.step()
        
        print(f"✅ Training step completed successfully")
        print(f"✅ Loss: {loss.item():.4f}")
        print(f"✅ Gradient norm: {grad_norm:.4f}")
        
        print("\n🎉 AMP scaler fix verified - training should work now!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
