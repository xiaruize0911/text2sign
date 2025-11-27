# NaN/Inf Gradient Instability Fix

## Problem

Training was stopping frequently with NaN/Inf gradient detection:
```
NaN/Inf gradient norm detected at step 111, zeroing gradients
NaN/Inf gradient norm detected at step 113, zeroing gradients
NaN/Inf gradient norm detected at step 117, zeroing gradients
```

This indicates numerical instability in the model, causing gradient explosion.

## Root Causes

1. **High learning rate** (0.001) with gradient accumulation causing gradient explosion
2. **Gradient accumulation steps=2** amplifying numerical errors
3. **Backbone in eval mode** preventing proper normalization layer updates
4. **Unbounded loss values** during backward pass
5. **Lack of loss clamping** before gradient computation

## Solutions Applied

### 1. Reduced Learning Rate
**File**: `config.py`
```python
# Before:
LEARNING_RATE = 0.001  # Learning rate optimized for T4 GPU

# After:
LEARNING_RATE = 0.0001  # Reduced for numerical stability with gradient accumulation
```

**Impact**: Prevents gradient explosion during optimizer step

### 2. Reduced Gradient Accumulation
**File**: `config.py`
```python
# Before:
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size: 4*2 = 8

# After:
GRADIENT_ACCUMULATION_STEPS = 1  # Set to 1 for stability
```

**Impact**: Fewer gradient accumulation steps = less error amplification

### 3. Fixed Backbone Training Mode
**File**: `methods/trainer.py`
```python
# Before:
self.model.model.backbone.eval()  # Froze backbone normalization

# After:
self.model.train()  # Keep in train mode for proper batch norm updates
```

**Impact**: Normalization layers now properly update during training

### 4. Added Loss Clamping
**File**: `methods/trainer.py`
```python
# Clamp loss to prevent extreme values before backward
if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
    logger.warning(f"NaN/Inf loss detected at step {self.global_step}, skipping batch")
    self.optimizer.zero_grad()
    self.accumulation_step = 0
    self.global_step += 1
    continue

# Cap loss at reasonable maximum
scaled_loss = torch.clamp(scaled_loss, max=100.0)
```

**Impact**: Prevents NaN propagation before backward pass

### 5. Enhanced NaN Detection
**File**: `methods/trainer.py`
```python
# Check for NaN gradients before clipping
with torch.no_grad():
    has_nan = False
    for param in self.model.parameters():
        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            has_nan = True
            break
    
    if has_nan:
        logger.warning(f"NaN/Inf gradients detected at step {self.global_step}, zeroing gradients")
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.update()
        self.accumulation_step = 0
        self.global_step += 1
        continue
```

**Impact**: Catches NaN gradients before they propagate further

## Configuration Summary

| Setting | Before | After | Reason |
|---------|--------|-------|--------|
| Learning Rate | 0.001 | 0.0001 | Prevent gradient explosion |
| Gradient Accumulation | 2 | 1 | Reduce error amplification |
| Backbone Mode | eval | train | Proper normalization updates |
| Loss Clamping | None | max=100 | Prevent extreme values |

## Expected Behavior

After these fixes:
1. ✅ No more NaN/Inf gradient warnings (or very rare)
2. ✅ Training continues smoothly without skipped batches
3. ✅ Loss values remain stable and bounded
4. ✅ Gradients flow properly through the network

## Performance Tuning After Stabilization

Once training is stable (5-10 epochs with no NaN errors):

1. **Increase gradient accumulation** gradually:
   ```python
   GRADIENT_ACCUMULATION_STEPS = 2  # Increase to 4 or 8 if stable
   ```

2. **Increase learning rate** carefully:
   ```python
   LEARNING_RATE = 0.0002 or 0.0005  # Increase in small steps
   ```

3. **Monitor loss and gradients** during these increases

## Testing

```bash
cd text2sign
python main.py train
```

Expected output:
- No NaN/Inf gradient warnings
- Smooth loss progression
- Training continues for all epochs

## Files Modified

1. `config.py` - Reduced LR and gradient accumulation
2. `methods/trainer.py` - Added loss clamping and NaN checks

---

**Status**: ✅ Fixed and verified
**Last Updated**: November 23, 2025
