"""
Diffusion module for video generation
"""

from .text2sign import DiffusionModel, create_diffusion_model

def test_diffusion():
    """Test diffusion model functionality"""
    print("🧪 Testing diffusion model...")
    try:
        from config import Config
        import torch
        
        # Create model
        model = create_diffusion_model(Config)
        model = model.to(Config.DEVICE)
        print(f"✅ Model created: {type(model.model).__name__}")
        
        # Test forward pass
        batch_size = 1
        test_input = torch.randn(batch_size, *Config.INPUT_SHAPE, device=Config.DEVICE)
        
        dummy_text = ["hello world"] * batch_size

        with torch.no_grad():
            loss, pred_noise, actual_noise = model(test_input, text=dummy_text)
            print(f"✅ Forward pass successful - Loss: {loss.item():.6f}")
            
        print("✅ Diffusion model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Diffusion model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

__all__ = ['DiffusionModel', 'create_diffusion_model', 'test_diffusion']
