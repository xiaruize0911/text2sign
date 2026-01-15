# Temporal Consistency Fix - Positional Encoding for Video Frames

## Problem Identified
Generated videos lacked **temporal consistency** - frames appeared independent and didn't form coherent motion sequences. This is a critical issue for sign language videos where smooth temporal transitions are essential.

## Root Cause

### The Confusion: Two Different "Time" Concepts

The model was using **only** the diffusion timestep embedding (`time_emb`), which represents:
- **Diffusion noise level** (how noisy the current sample is in the denoising process)
- **NOT** the position of frames in the video sequence

This meant the model had **no way to distinguish** which frame is frame 0, frame 5, or frame 15 in the sequence!

### Original Code (WRONG)
```python
# Only diffusion timestep embedding - noise level only!
self.time_embed = nn.Sequential(
    nn.Linear(model_channels, time_embed_dim),
    nn.SiLU(),
    nn.Linear(time_embed_dim, time_embed_dim),
)
```

Without temporal positional encoding, the model treats all frames as **exchangeable**, like a bag of independent images rather than a coherent video sequence.

## Solution: Add Temporal Positional Embeddings

### What Was Added

1. **Learnable Temporal Positional Embeddings**
   - Added to UNet3D constructor:
   ```python
   # Temporal positional embedding (frame position in sequence)
   # This is critical for temporal consistency!
   self.temporal_pos_embed = nn.Parameter(
       torch.randn(1, in_channels, num_frames, 1, 1) * 0.02
   )
   ```
   - Shape: `(1, C=3, T=16, 1, 1)` - one unique position vector per frame
   - Learnable during training to adapt to the data

2. **Applied in Forward Pass**
   - Before processing, add position encoding to input:
   ```python
   # Add temporal positional encoding for frame position awareness
   B, C, T, H, W = x.shape
   if T <= self.num_frames:
       x = x + self.temporal_pos_embed[:, :, :T, :, :]
   else:
       # Interpolate if video has more frames than expected
       pos_embed = F.interpolate(...)
       x = x + pos_embed
   ```

3. **Clear Naming Convention**
   - `time_emb` / `t_emb` → **Diffusion timestep** (noise level)
   - `temporal_pos_embed` → **Frame position** in video sequence

### Now the Model Has Two Types of Time Information:

| Type | Purpose | Shape | Example |
|------|---------|-------|---------|
| **Diffusion Timestep** (`t_emb`) | Denoising progress | `(B, time_embed_dim)` | "This sample is at 30% noise" |
| **Frame Position** (`temporal_pos_embed`) | Temporal order | `(1, C, T, 1, 1)` | "This is frame 5 of 16" |

## Changes Made

### 1. [models/unet3d.py](models/unet3d.py)

#### Constructor Changes (Lines 704-751)
- Added `num_frames: int = 16` parameter
- Added `self.num_frames = num_frames` attribute
- Added `self.temporal_pos_embed` learnable parameter
- Renamed `time_embed` comments to clarify it's for diffusion timestep

#### Forward Pass Changes (Lines 853-888)
- Added temporal positional encoding before processing
- Supports variable frame counts via interpolation
- Clear comments distinguishing diffusion time from frame position

#### Factory Function Update (Line 907-921)
- Updated `create_unet()` to pass `num_frames` from config

### 2. Test Files Created

#### [test_temporal_fix.py](test_temporal_fix.py)
- Comprehensive test of temporal positional embeddings
- Verifies model can use frame position information
- Measures temporal consistency via frame differences

## Testing Results

```
✅ Model forward pass successful!
✅ Generation successful!
   Temporal pos embed shape: torch.Size([1, 3, 16, 1, 1])
   Video shape: torch.Size([1, 3, 16, 64, 64])
   Temporal frame difference: 0.366880
```

## Why This Fixes Temporal Inconsistency

### Before (No Positional Encoding):
```
Frame 0: [no position info] → Model treats as random frame
Frame 1: [no position info] → Model treats as random frame  
Frame 2: [no position info] → Model treats as random frame
...
Result: Frames are independent, no temporal coherence
```

### After (With Positional Encoding):
```
Frame 0: [pos vector 0] → Model knows "this is the start"
Frame 1: [pos vector 1] → Model knows "this comes after frame 0"
Frame 2: [pos vector 2] → Model knows "this continues the sequence"
...
Result: Frames form coherent temporal progression
```

## Expected Improvements

1. **Smoother Frame Transitions**
   - Adjacent frames will have more similar content
   - Motion will appear more continuous

2. **Better Temporal Structure**
   - Beginning, middle, and end of videos will be distinguishable
   - Sign language gestures will have proper temporal flow

3. **Improved Long-Range Coherence**
   - Later frames can reference earlier frames
   - Complete sign language phrases will be more coherent

## Important Notes

### ⚠️ Training Required
- This adds new learnable parameters (`temporal_pos_embed`)
- **Cannot load old checkpoints directly** - will have missing keys
- Options:
  1. **Start fresh training** (recommended for best results)
  2. Load old checkpoint with `strict=False` and train temporal embeddings
  3. Fine-tune from existing checkpoint

### Model Parameter Count
- Added: `3 × 16 × 1 × 1 = 48` parameters per input channel
- Total new params: **48 parameters** (negligible)
- These are critical despite being few in number!

## How to Resume Training

### Option 1: Fresh Training (Recommended)
```bash
cd /teamspace/studios/this_studio/text_to_sign
python main.py train
```

### Option 2: Fine-tune from Existing Checkpoint
Modify trainer.py to load with `strict=False`:
```python
checkpoint = torch.load(checkpoint_path)
self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
# temporal_pos_embed will be randomly initialized
```

## Verification

Run the test to verify temporal positional embeddings work:
```bash
cd /teamspace/studios/this_studio/text_to_sign
python test_temporal_fix.py
```

Expected output:
```
✅ Temporal positional embeddings added to UNet3D
✅ Each frame now has unique position encoding  
✅ Model can distinguish between frames in sequence
✅ Should improve temporal consistency in generated videos
```

## Technical Details

### Why Learnable vs Fixed Positional Encoding?

**Learnable** (our choice):
- ✅ Adapts to data distribution
- ✅ Can learn task-specific temporal patterns
- ✅ Works well for video diffusion models

**Fixed** (sinusoidal):
- Used in original Transformer for NLP
- Less flexible for video domain

### Broadcasting Behavior
```python
x.shape:                (B=2, C=3, T=16, H=64, W=64)
temporal_pos_embed:     (1,   C=3, T=16, 1,    1   )
                         ↑            ↑    ↑    ↑
                    broadcasts across these dimensions
Result:                 (B=2, C=3, T=16, H=64, W=64)
```

Each spatial location (H, W) in each batch item (B) gets the same temporal position encoding for each frame (T).

## References

- Video Diffusion Models (Ho et al., 2022)
- Attention is All You Need (Vaswani et al., 2017) - for positional encoding concept
- DiT: Scalable Diffusion Transformers (Peebles & Xie, 2023)

## Date
January 15, 2026

## Summary

✅ **Root cause**: Missing temporal positional information  
✅ **Solution**: Added learnable frame position embeddings  
✅ **Impact**: Videos should now have temporal consistency  
✅ **Trade-off**: Need to retrain or fine-tune model  
✅ **Benefit**: Proper sign language temporal dynamics
