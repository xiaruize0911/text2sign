# Sample Generation Logging Fix

## Problem
The logging generation during training was failing silently. While `main.py` generation worked correctly, the periodic sample generation during training (every `sample_every` steps) was not working.

## Root Cause
In the `generate_samples` method in [trainer.py](trainer.py), there was a **type mismatch** when passing the timestep to the scheduler's `step` method:

### The Bug (Line 354-370)
```python
# Denoising loop
for t in tqdm(self.scheduler.timesteps, desc="Generating", leave=False):
    # ...
    timestep = torch.tensor([t] * latent_model_input.shape[0], device=self.device)
    noise_pred = self.model(latent_model_input, timestep, text_embeddings)
    # ...
    # DDIM step - BUG: passing tensor 't' instead of int
    latents, _ = self.scheduler.step(noise_pred, t, latents, eta=eta)
```

**Issue**: The variable `t` from `self.scheduler.timesteps` is a `torch.Tensor`, but `scheduler.step()` expects an `int` for the `timestep` parameter.

### The scheduler.step method expects:
```python
def step(
    self,
    model_output: torch.Tensor,
    timestep: int,  # <-- Expects int, not tensor!
    sample: torch.Tensor,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

## Solution

### Fixed Code (Lines 354-372)
```python
# Denoising loop
for t in tqdm(self.scheduler.timesteps, desc="Generating", leave=False):
    latent_model_input = latents
    
    if guidance_scale > 1.0:
        latent_model_input = torch.cat([latents] * 2)
    
    # Convert timestep to int for scheduler.step, keep tensor for model
    t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
    timestep = torch.tensor([t_int] * latent_model_input.shape[0], device=self.device)
    
    noise_pred = self.model(latent_model_input, timestep, text_embeddings)
    
    # Apply classifier-free guidance
    if guidance_scale > 1.0:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # DDIM step - use int timestep for scheduler
    latents, _ = self.scheduler.step(noise_pred, t_int, latents, eta=eta)
```

**Key change**: Convert the tensor timestep `t` to an integer `t_int` before passing to `scheduler.step()`.

## Additional Improvements

### Enhanced Error Handling and Logging (Lines 561-599)
Added comprehensive logging to track sample generation success/failure:

```python
# Generate samples
if self.global_step % self.train_config.sample_every == 0:
    try:
        print(f"\n🎨 Generating samples at step {self.global_step}...")
        videos = self.generate_samples(...)
        
        # Save samples...
        
        print(f"✅ Samples saved successfully!")
        
        # Log sample generation success
        self.metrics_logger.log_step(
            step=self.global_step,
            metrics={"sample_generated": 1.0},
            phase="generation"
        )
    except Exception as e:
        print(f"❌ Error generating samples at step {self.global_step}: {e}")
        import traceback
        traceback.print_exc()
        
        # Log sample generation failure
        self.metrics_logger.log_step(
            step=self.global_step,
            metrics={"sample_generated": 0.0, "sample_error": str(e)[:100]},
            phase="generation"
        )
```

**Benefits**:
1. Visible progress messages during generation
2. Full stack traces on errors (not just silent failures)
3. Metric logging for tracking generation success rate
4. Phase-specific logging for better analysis

## Testing

Created `test_generation_fix.py` to verify the fix works:
```bash
cd /teamspace/studios/this_studio/text_to_sign
python test_generation_fix.py
```

**Test Results**:
```
✅ SUCCESS! Generated videos with shape: torch.Size([2, 3, 16, 64, 64])
   Expected: (2, 3, 16, 64, 64)
   Got:      (2, 3, 16, 64, 64)
   Value range: [0.000, 1.000]

🎉 All checks passed! Generation is working correctly.
```

## Impact

- ✅ Sample generation during training now works correctly
- ✅ Better visibility into generation process with progress messages
- ✅ Errors are logged and tracked for debugging
- ✅ Main.py generation continues to work as before
- ✅ No impact on training loss convergence

## Files Modified

1. `trainer.py`:
   - Fixed timestep type conversion in `generate_samples()` method
   - Enhanced error handling and logging for sample generation

2. `test_generation_fix.py` (new):
   - Test script to verify the fix

## Date
January 15, 2026
