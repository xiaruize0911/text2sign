# Bug Fix: TinyFusion Temporal Shape Mismatch

## Problem
Training failed with the following error:
```
RuntimeError: The size of tensor a (29) must match the size of tensor b (28) at non-singleton dimension 2
```

This occurred in `diffusion/text2sign.py` line 373:
```python
loss = F.mse_loss(predicted_noise, noise)
```

The `predicted_noise` had 29 frames while `noise` had 28 frames.

## Root Cause
The issue was in the `TemporalPostProcessor` class in `models/architectures/tinyfusion.py`. 

The temporal convolution used manual padding calculation:
```python
padding = kernel_size // 2
self.conv = nn.Conv3d(
    channels,
    channels,
    kernel_size=(kernel_size, 1, 1),
    padding=(padding, 0, 0),
    bias=False,
)
```

With `TINYFUSION_TEMPORAL_KERNEL = 2` (even number):
- `kernel_size = 2`
- `padding = 2 // 2 = 1`

This caused Conv3d to add an extra frame to the output:
- Input: (batch, 3, 28, 128, 128)
- Output: (batch, 3, **29**, 128, 128) ❌

## Solution
Changed the padding mode from manual calculation to `'same'` mode:

```python
self.conv = nn.Conv3d(
    channels,
    channels,
    kernel_size=(kernel_size, 1, 1),
    padding='same',  # Automatically handles padding to maintain shape
    bias=False,
)
```

The `'same'` padding mode automatically ensures the output has the same temporal dimension as the input, regardless of whether the kernel size is even or odd.

## Result
Now the temporal dimension is preserved correctly:
- Input: (batch, 3, 28, 128, 128)
- Output: (batch, 3, **28**, 128, 128) ✅

## Files Modified
- `models/architectures/tinyfusion.py`: Updated `TemporalPostProcessor.__init__()` method (2 occurrences)

## Testing
Verified that `padding='same'` works correctly:
```python
conv = nn.Conv3d(3, 3, kernel_size=(2, 1, 1), padding='same', bias=False)
x = torch.randn(1, 3, 28, 128, 128)
out = conv(x)
# Input shape: torch.Size([1, 3, 28, 128, 128])
# Output shape: torch.Size([1, 3, 28, 128, 128]) ✓
```

## Note
PyTorch may emit a warning about even kernel lengths with 'same' padding:
```
UserWarning: Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created
```

This is informational only and does not affect functionality. The padding behavior is correct.
