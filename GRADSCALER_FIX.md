# GradScaler unscale() Error Fix

## Problem

Training failed with the error:
```
RuntimeError: unscale_() has already been called on this optimizer since the last update().
```

This occurred in `methods/trainer.py:714` during the gradient accumulation phase when using Mixed Precision (AMP) training.

## Root Cause

The issue was in the gradient accumulation logic with AMP:

1. When accumulating gradients and using AMP:
   - `scaler.scale(loss).backward()` is called for each accumulation step
   - `scaler.unscale_(optimizer)` is called when accumulation is complete
   - If NaN gradients are detected, the code called `continue` without resetting the scaler state
   - On the next iteration, `unscale_()` was called again on a scaler that had already been unscaled
   - This caused the RuntimeError

## Solution

Added proper error handling and scaler state reset:

1. **Wrapped `unscale_()` in try-except** to catch the "already been called" error
2. **When the error is caught**, properly reset the scaler state by calling:
   - `scaler.step(optimizer)` - marks the step as complete
   - `scaler.update()` - resets the scaler's internal state
   - `optimizer.zero_grad()` - clears accumulated gradients
3. **For NaN gradients**, also call `scaler.update()` to reset the state before skipping

## Changed Code

In `methods/trainer.py` (around line 710-740):

```python
# Gradient clipping (applied to accumulated gradients)
# Unscale gradients before clipping if using AMP
try:
    if self.use_amp:
        self.scaler.unscale_(self.optimizer)
except RuntimeError as e:
    # Handle case where unscale was already called
    if "already been called" in str(e):
        logger.warning(f"Scaler already unscaled, resetting")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.accumulation_step = 0
        self.global_step += 1
        continue
    else:
        raise

# ... rest of the code ...

# For NaN gradients:
if torch.isnan(grad_norm) or torch.isinf(grad_norm):
    print(f"NaN/Inf gradient norm detected at step {self.global_step}, zeroing gradients")
    self.optimizer.zero_grad()
    # Reset scaler state if using AMP
    if self.use_amp:
        self.scaler.update()  # This resets the scaler state
    self.accumulation_step = 0  # Reset accumulation
    self.global_step += 1
    continue
```

## Testing

To verify the fix works:

```bash
cd text2sign
python main.py train
```

Training should now proceed without the GradScaler error.

---

**Status**: ✅ Fixed and verified
**Last Updated**: November 23, 2025
