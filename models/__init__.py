# Models package for Text2Sign
from .architectures.unet3d import UNet3D, count_parameters as count_unet_parameters, test_unet3d
from .architectures.vit3d import ViT3D, count_parameters as count_vit_parameters, test_vit3d

# For backward compatibility, use UNet3D count_parameters as default
count_parameters = count_unet_parameters

__all__ = [
    'UNet3D', 'ViT3D', 'count_parameters', 'count_unet_parameters', 
    'count_vit_parameters', 'test_unet3d', 'test_vit3d'
]
