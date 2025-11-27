
import torch
import torch.nn as nn

def test_dirac_init():
    channels = 4
    kernel_size = 3
    conv = nn.Conv3d(
        channels,
        channels,
        kernel_size=(kernel_size, 1, 1),
        padding='same',
        bias=False,
    )
    nn.init.dirac_(conv.weight)
    
    print(f"Weight shape: {conv.weight.shape}")
    print("Weight values:")
    print(conv.weight)
    
    # Test forward pass
    x = torch.randn(1, channels, 16, 64, 64)
    y = conv(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Check if it preserves identity roughly
    diff = (x - y).abs().mean()
    print(f"Mean difference from identity: {diff.item()}")

if __name__ == "__main__":
    test_dirac_init()
