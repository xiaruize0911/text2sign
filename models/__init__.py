# Models package for Text2Sign
from .architectures.unet3d import UNet3D, test_unet3d
from .architectures.vit3d import ViT3D, test_vit3d
from .architectures.tinyfusion import TinyFusionVideoWrapper

def count_parameters(model) -> int:
    """
    Count trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

__all__ = [
    'UNet3D', 'ViT3D', 'TinyFusionVideoWrapper', 'count_parameters', 'test_unet3d', 'test_vit3d'
]
