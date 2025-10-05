# ViViT Model Fixes - Summary

## Issues Identified

### 1. **Poor Low-Timestep Performance**
- **Problem**: Model had MSE of 1.92 at t=0 (clean images) vs 0.001 at t=49 (noisy images)
- **Impact**: Generated samples were extremely noisy because the model couldn't denoise the final steps properly
- **Root Cause**: Model was not learning to predict noise accurately at low timesteps where details matter most

### 2. **Residual Connection Interference**
- **Problem**: The model had `final_output = temporal_conv(video_output) + x` which added noisy input to predicted noise
- **Impact**: This interfered with proper noise prediction since diffusion models should predict pure noise, not add residuals
- **Root Cause**: Residual connections work for image generation but not for noise prediction tasks

### 3. **Weak Temporal Modulation**
- **Problem**: Temporal features were only weighted by 0.1 (10%)
- **Impact**: Temporal information was barely affecting the spatial features
- **Root Cause**: Conservative initial weight choice

### 4. **Missing Output Scaling**
- **Problem**: No learnable output scaling parameter
- **Impact**: Model couldn't adapt its output magnitude to match different noise scales at different timesteps
- **Root Cause**: Architecture didn't account for timestep-dependent noise magnitudes

### 5. **No Loss Weighting for Timesteps**
- **Problem**: All timesteps had equal loss weight during training
- **Impact**: Model focused on easy high-timestep predictions, neglecting difficult low-timestep predictions
- **Root Cause**: Standard DDPM training treats all timesteps equally

## Fixes Applied

### 1. **Removed Residual Connection** ✅
```python
# OLD (WRONG):
if self.in_channels == self.out_channels:
    final_output = final_output + x

# NEW (CORRECT):
# Removed residual connection - model predicts pure noise
return final_output * self.output_scale
```

### 2. **Added Learnable Output Scaling** ✅
```python
# Added to __init__:
self.output_scale = nn.Parameter(torch.ones(1) * 0.5)

# Applied in forward:
final_output = final_output * self.output_scale
```
- Initialized to 0.5 (moderate scaling)
- Allows model to learn appropriate output magnitude
- Critical for matching noise prediction scales

### 3. **Increased Temporal Modulation Weight** ✅
```python
# OLD: 0.1 (too weak)
enhanced_features = features + temporal_features * 0.1

# NEW: 0.5 (stronger)
enhanced_features = features + temporal_features * 0.5
```

### 4. **Improved Final Projection Layer** ✅
```python
# Added deeper projection with regularization:
self.final_projection = nn.Sequential(
    nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=3, padding=1),
    nn.GroupNorm(_get_group_count(self.feature_dim // 2), self.feature_dim // 2),
    nn.GELU(),
    nn.Dropout2d(dropout * 0.5),  # Light dropout
    nn.Conv2d(self.feature_dim // 2, self.feature_dim // 4, kernel_size=3, padding=1),
    nn.GroupNorm(_get_group_count(self.feature_dim // 4), self.feature_dim // 4),
    nn.GELU(),
    nn.Conv2d(self.feature_dim // 4, out_channels, kernel_size=1),
)
```

### 5. **Added Timestep-Aware Loss Weighting** ✅
```python
# Compute SNR (Signal-to-Noise Ratio)
alpha_bar_t = self.alphas_cumprod[t]
snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)

# Higher weight for low SNR (low timesteps = harder to predict)
loss_weight = 1.0 / torch.clamp(snr, min=0.1, max=10.0)
weighted_loss = base_loss * loss_weight
loss = weighted_loss.mean()
```

### 6. **Added Configuration Options** ✅
```python
# In config.py:
USE_TIMESTEP_WEIGHTING = True
TIMESTEP_WEIGHT_MIN_SNR = 0.1
TIMESTEP_WEIGHT_MAX_SNR = 10.0
```

## Expected Improvements

### Training
- **Loss should decrease more evenly across timesteps**: Low-timestep loss should improve
- **Better convergence**: Weighted loss focuses training on difficult examples
- **More stable gradients**: Output scaling prevents gradient explosion/vanishing

### Sampling
- **Much less noisy outputs**: Better low-timestep prediction = cleaner final images
- **Better temporal consistency**: Stronger temporal modulation
- **More accurate noise prediction**: Proper output scaling

## Testing the Fixes

### Option 1: Continue Training (Recommended)
The model architecture has changed, so you should:
1. Start fresh training with the new architecture
2. The old checkpoints won't load properly due to new parameters

```bash
# Start new training
python main.py train
```

### Option 2: Compare Old vs New
Generate samples with both architectures to see the difference:

```bash
# Generate with old model (if you have old checkpoint)
python main.py sample --checkpoint checkpoints/text2sign_vivit6/latest_checkpoint.pt --text "hello" --output_dir samples_old

# After retraining with new architecture
python main.py sample --checkpoint checkpoints/text2sign_vivit6/latest_checkpoint.pt --text "hello" --output_dir samples_new
```

## What to Monitor

### During Training
1. **Loss at different timestep ranges**: Should be more balanced
2. **Output scale parameter**: Should stabilize around 0.5-2.0
3. **Generated samples**: Should become less noisy over epochs

### Red Flags
- ⚠️ **Output scale > 10**: Model outputs exploding
- ⚠️ **Output scale < 0.1**: Model outputs suppressed
- ⚠️ **Loss oscillating**: May need to adjust learning rate
- ⚠️ **Samples still very noisy after 20+ epochs**: Architecture issue remains

## Architecture Changes Summary

| Component | Old | New | Impact |
|-----------|-----|-----|--------|
| Residual connection | ✅ Enabled | ❌ Removed | Fixes noise prediction |
| Output scaling | ❌ None | ✅ Learnable (0.5 init) | Adapts to timesteps |
| Temporal weight | 0.1 | 0.5 | Stronger temporal features |
| Loss weighting | Equal all timesteps | SNR-based | Focuses on hard examples |
| Final projection | 2 layers | 3 layers + dropout | Better spatial detail |

## Next Steps

1. **Start new training run** with fixed architecture
2. **Monitor TensorBoard** for loss curves and sample quality
3. **Compare samples** at epoch 5, 10, 20 to old samples
4. **Adjust hyperparameters** if needed:
   - If outputs too small: Increase initial output_scale to 1.0
   - If loss not decreasing: Reduce learning rate
   - If samples still noisy: Increase NUM_EPOCHS or check data quality

## Files Modified

1. `/teamspace/studios/this_studio/text2sign/models/architectures/vivit.py`
   - Removed residual connection
   - Added output_scale parameter
   - Improved final_projection
   - Increased temporal modulation weight

2. `/teamspace/studios/this_studio/text2sign/diffusion/text2sign.py`
   - Added timestep-aware loss weighting
   - Added configuration parameters for loss weighting

3. `/teamspace/studios/this_studio/text2sign/config.py`
   - Added USE_TIMESTEP_WEIGHTING flag
   - Added TIMESTEP_WEIGHT_MIN_SNR and TIMESTEP_WEIGHT_MAX_SNR

## Troubleshooting

### If samples are still noisy:
1. Check if model is actually training (loss decreasing?)
2. Verify output_scale is reasonable (0.1 - 10.0 range)
3. Ensure data is properly normalized to [-1, 1]
4. Try increasing TIMESTEP_WEIGHT_MAX_SNR to 20.0

### If training is unstable:
1. Reduce learning rate to 5e-5
2. Enable gradient clipping (already at 1.0)
3. Reduce temporal modulation weight back to 0.3
4. Disable timestep weighting temporarily

### If output scale explodes:
1. Add constraint: `output_scale.data.clamp_(0.1, 10.0)` after optimizer step
2. Reduce learning rate
3. Initialize to 0.3 instead of 0.5
