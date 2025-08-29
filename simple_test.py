#!/usr/bin/env python3
"""
Simple test to verify the sampling fix is working
"""

import sys
import os
sys.path.append('/teamspace/studios/this_studio/text2sign')

try:
    import torch
    print(f"✅ PyTorch loaded successfully: {torch.__version__}")
    
    from config import Config
    print("✅ Config loaded successfully")
    
    from diffusion.text2sign import Text2SignDiffusion
    print("✅ Text2SignDiffusion loaded successfully")
    
    from models.model_factory import create_model
    print("✅ Model factory loaded successfully")
    
    config = Config()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    
    # Test if we can create the model
    model = create_model(config)
    print("✅ Model created successfully")
    
    # Test if we can create diffusion
    diffusion = Text2SignDiffusion(model, config, device)
    print("✅ Diffusion model created successfully")
    
    # Test if we can load checkpoint
    checkpoint_path = "checkpoints/latest_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        diffusion.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint loaded successfully from step {checkpoint.get('step', 'unknown')}")
    else:
        print("❌ Latest checkpoint not found")
        sys.exit(1)
    
    # Quick test of one step of sampling
    with torch.no_grad():
        x = torch.randn(1, 3, config.num_frames, config.image_height, config.image_width, device=device)
        t = torch.tensor([10], device=device)
        
        # Test model prediction
        noise_pred = diffusion.model(x, t, text="test")
        print(f"✅ Model prediction shape: {noise_pred.shape}")
        print(f"✅ Model prediction mean: {noise_pred.mean().item():.6f}")
        print(f"✅ Model prediction std: {noise_pred.std().item():.6f}")
        
        # Test one sampling step
        x_prev = diffusion.p_sample_step(x, t)
        print(f"✅ Sampling step completed")
        print(f"✅ Output shape: {x_prev.shape}")
        print(f"✅ Output mean: {x_prev.mean().item():.6f}")
        print(f"✅ Output std: {x_prev.std().item():.6f}")
        
        # Test if output is different from input (should be less noisy)
        diff = torch.abs(x - x_prev).mean().item()
        print(f"✅ Difference from input: {diff:.6f}")
        
        if diff > 1e-6:
            print("✅ SUCCESS: Sampling step produces different output (denoising is working)")
        else:
            print("❌ ISSUE: Sampling step produces identical output")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
