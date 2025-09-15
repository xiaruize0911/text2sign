# DiT3D Tensor Size Mismatch Fix

## Problem Solved ✅

**Error**: `RuntimeError: The size of tensor a (28) must match the size of tensor b (16) at non-singleton dimension 1`

**Root Cause**: The dataset was hardcoded to use 28 frames, but the config and model were set to use 16 frames, causing a mismatch in the temporal positional embedding.

## Changes Made

### 1. Updated Dataset Class (`dataset.py`)
- **Added `num_frames` parameter** to `SignLanguageDataset.__init__()`
- **Made frame count configurable** instead of hardcoded 28
- **Updated frame padding/truncation logic** to use `self.num_frames`

```python
# Before (hardcoded)
if num_frames < 28:
    padding = 28 - num_frames
elif num_frames > 28:
    frames = frames[:28]

# After (configurable)
if num_frames < self.num_frames:
    padding = self.num_frames - num_frames  
elif num_frames > self.num_frames:
    frames = frames[:self.num_frames]
```

### 2. Updated DataLoader Creation (`dataset.py`)
- **Added `num_frames` parameter** to `create_dataloader()` function
- **Made image size configurable** using `Config.IMAGE_SIZE` instead of hardcoded 128
- **Pass num_frames to dataset** when creating the dataset instance

```python
def create_dataloader(data_root: str, batch_size: int, num_workers: int = 2, 
                     shuffle: bool = True, num_frames: int = 16) -> DataLoader:
```

### 3. Updated Trainer Setup (`methods/trainer.py`)
- **Added `num_frames=config.NUM_FRAMES`** parameter when calling `create_dataloader()`
- **Ensures consistent frame count** throughout the pipeline

```python
dataloader = create_dataloader(
    data_root=config.DATA_ROOT,
    batch_size=config.BATCH_SIZE, 
    num_workers=config.NUM_WORKERS,
    shuffle=True,
    num_frames=config.NUM_FRAMES  # ← Added this line
)
```

### 4. Added Tiny DiT Models to Registry (`models/architectures/dit3d.py`)
 
- **Ensures model creation works** for memory-optimized variants

## Configuration Alignment

All components now use consistent values from `config.py`:

```python
NUM_FRAMES = 16          # Dataset uses this for frame count
IMAGE_SIZE = 64          # Dataset uses this for cropping 
INPUT_SHAPE = (3, 16, 64, 64)  # Model expects this shape
DIT_VIDEO_SIZE = (16, 64, 64)   # DiT3D uses this for temporal embedding
```

## Validation Results ✅

- ✅ **Dataset**: Produces `torch.Size([1, 3, 16, 64, 64])` tensors
- ✅ **Model**: Expects `(batch, 3, 16, 64, 64)` input shape  
- ✅ **Temporal embedding**: Correctly sized for 16 frames
- ✅ **Training**: Can start without tensor size mismatches

## Memory Impact

The fix also **reduces memory usage** significantly:
- **From**: 28 frames × 128×128 = 458,752 pixels per video
- **To**: 16 frames × 64×64 = 65,536 pixels per video  
- **Reduction**: ~7x fewer pixels per video = much lower memory usage

This change supports the overall goal of fitting the model in 16GB memory constraints while maintaining functionality.

## Next Steps

The tensor size mismatch is now resolved. Training should be able to start successfully with the memory-optimized DiT3D-S/2 configuration.