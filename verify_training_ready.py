import os
import sys
import torch
import logging
import numpy as np
from config import Config
from diffusion import create_diffusion_model
from dataset import create_dataloader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_config():
    logger.info("🔍 Checking configuration...")
    required_attrs = [
        'MODEL_ARCHITECTURE', 'BATCH_SIZE', 'LEARNING_RATE', 
        'NUM_EPOCHS', 'INPUT_SHAPE', 'DEVICE'
    ]
    for attr in required_attrs:
        if not hasattr(Config, attr):
            logger.error(f"❌ Missing config attribute: {attr}")
            return False
    
    logger.info(f"  Architecture: {Config.MODEL_ARCHITECTURE}")
    logger.info(f"  Device: {Config.DEVICE}")
    logger.info("✅ Configuration check passed")
    return True

def check_data():
    logger.info("🔍 Checking data...")
    if not os.path.exists(Config.DATA_ROOT):
        logger.error(f"❌ Data root not found: {Config.DATA_ROOT}")
        return False
    
    files = os.listdir(Config.DATA_ROOT)
    gifs = [f for f in files if f.endswith('.gif') or f.endswith('.mp4')] # Assuming mp4 also possible based on dataset.py usually
    txts = [f for f in files if f.endswith('.txt')]
    
    if not txts:
         # Fallback: check if there are text files in the list provided in workspace info
         # The workspace info showed many .txt files in training_data
         pass

    logger.info(f"  Found {len(files)} files in {Config.DATA_ROOT}")
    
    # Try creating dataloader
    try:
        dataloader = create_dataloader(
            data_root=Config.DATA_ROOT,
            batch_size=2, # Small batch for testing
            num_workers=0,
            shuffle=True,
            num_frames=Config.NUM_FRAMES
        )
        batch = next(iter(dataloader))
        videos, texts = batch
        logger.info(f"  DataLoader works. Batch shape: {videos.shape}")
        logger.info("✅ Data check passed")
        return True
    except Exception as e:
        logger.error(f"❌ Data check failed: {e}")
        return False

def check_model():
    logger.info("🔍 Checking model initialization...")
    try:
        model = create_diffusion_model(Config)
        model.to(Config.DEVICE)
        logger.info(f"  Model created: {type(model).__name__}")
        
        # Check for trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            logger.error("❌ No trainable parameters! Check FREEZE_BACKBONE setting.")
            return False
        
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info("✅ Model initialization passed")
        return model
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_training_step(model):
    logger.info("🔍 Checking training step (Forward + Backward)...")
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dummy batch
        batch_size = 2
        frames = Config.NUM_FRAMES
        height = Config.IMAGE_SIZE
        width = Config.IMAGE_SIZE
        channels = 4 # RGBA
        
        dummy_video = torch.randn(batch_size, channels, frames, height, width).to(Config.DEVICE)
        # Normalize to [-1, 1]
        dummy_video = torch.clamp(dummy_video, -1, 1)
        
        dummy_text = ["test prompt"] * batch_size
        
        # Run a few steps to see if loss changes
        logger.info("  Running 5 training steps...")
        for i in range(5):
            optimizer.zero_grad()
            loss, predicted_noise, noise, t = model(dummy_video, dummy_text)
            
            # Analyze predicted noise
            pred_mean = predicted_noise.mean().item()
            pred_std = predicted_noise.std().item()
            noise_mean = noise.mean().item()
            noise_std = noise.std().item()
            
            logger.info(f"  Step {i+1}: Loss={loss.item():.6f}")
            logger.info(f"    Pred Noise: mean={pred_mean:.6f}, std={pred_std:.6f}")
            logger.info(f"    Real Noise: mean={noise_mean:.6f}, std={noise_std:.6f}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("❌ Loss is NaN or Inf")
                return False
                
            loss.backward()
            
            # Check gradients
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
            
            logger.info(f"    Total Grad Norm: {total_grad_norm:.6f}")
            
            if total_grad_norm == 0:
                logger.warning("⚠️  Total gradient norm is ZERO! Model is not learning.")
            
            optimizer.step()
            
        logger.info("✅ Backward pass and optimizer step successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("========================================")
    print("   Text2Sign Training Readiness Check   ")
    print("========================================")
    
    if not check_config():
        sys.exit(1)
        
    if not check_data():
        sys.exit(1)
        
    model = check_model()
    if model is None:
        sys.exit(1)
        
    if not check_training_step(model):
        sys.exit(1)
        
    print("\n" + "="*40)
    print("✅ ALL CHECKS PASSED!")
    print("="*40)
    print("\nYour model is ready for training. You can now run:")
    print("   python main.py train")

if __name__ == "__main__":
    main()
