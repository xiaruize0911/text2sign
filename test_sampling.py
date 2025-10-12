"""
Quick test to validate the sampling procedure works correctly.
"""

import torch
from config import Config
from diffusion.text2sign import create_diffusion_model

def test_sampling():
    """Test that sampling produces valid output."""
    print("=" * 60)
    print("Testing Sampling Procedure")
    print("=" * 60)
    
    # Load config
    config = Config()
    config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {config.DEVICE}")
    
    # Create model
    print("\n📦 Loading diffusion model...")
    model = create_diffusion_model(config).to(config.DEVICE)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/text2sign_vivit7/checkpoint_epoch_latest.pt"
    print(f"📥 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully")
    
    # Test sampling
    print("\n" + "=" * 60)
    print("Testing Sampling Methods")
    print("=" * 60)
    
    test_text = "hello"
    
    # Test 1: Fast deterministic DDIM sampling (50 steps)
    print("\n1️⃣ Fast DDIM Sampling (50 steps, deterministic)")
    with torch.no_grad():
        sample = model.sample(
            text=test_text,
            batch_size=1,
            num_frames=28,
            height=128,
            width=128,
            deterministic=True,
            num_inference_steps=50,
        )
    print(f"✅ Output shape: {sample.shape}")
    print(f"   Range: [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"   Mean: {sample.mean():.3f}, Std: {sample.std():.3f}")
    assert sample.shape == (1, 3, 28, 128, 128), f"Wrong shape: {sample.shape}"
    assert sample.min() >= -1.0 and sample.max() <= 1.0, "Values outside [-1, 1] range"
    
    # Test 2: Full timestep sampling
    print("\n2️⃣ Full DDIM Sampling (1000 steps, deterministic)")
    with torch.no_grad():
        sample = model.sample(
            text=test_text,
            batch_size=1,
            num_frames=28,
            height=128,
            width=128,
            deterministic=True,
            num_inference_steps=1000,
        )
    print(f"✅ Output shape: {sample.shape}")
    print(f"   Range: [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"   Mean: {sample.mean():.3f}, Std: {sample.std():.3f}")
    
    # Test 3: Stochastic sampling
    print("\n3️⃣ Stochastic DDIM Sampling (50 steps, η=0.5)")
    with torch.no_grad():
        sample = model.sample(
            text=test_text,
            batch_size=1,
            num_frames=28,
            height=128,
            width=128,
            deterministic=False,
            num_inference_steps=50,
            eta=0.5,
        )
    print(f"✅ Output shape: {sample.shape}")
    print(f"   Range: [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"   Mean: {sample.mean():.3f}, Std: {sample.std():.3f}")
    
    print("\n" + "=" * 60)
    print("✅ All sampling tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_sampling()
