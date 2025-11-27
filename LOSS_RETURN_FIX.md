# Loss Return Fix Summary

## Issue
The user reported that the loss returns to 1.0 after reaching 0.8. This indicates that the model is able to learn initially but then becomes unstable or fails to converge further, bouncing back to a higher loss state.

## Investigation
1.  **Learning Rate**: The `LEARNING_RATE` in `config.py` was set to `0.0001` (`1e-4`).
2.  **Previous Fixes**: `LOSS_JUMP_FIX.md` explicitly recommended reducing the learning rate to `1e-5` for fine-tuning, but this setting was not present in the current `config.py`.
3.  **Scheduler**: No learning rate scheduler was active (`USE_SCHEDULER = False`), so the learning rate remained constant at `1e-4`.
4.  **Model**: The model is `TinyFusion` (fine-tuning a pre-trained `TinyDiT` checkpoint). Fine-tuning typically requires a lower learning rate to preserve pre-trained knowledge and avoid destroying the weights.
5.  **Temporal Post-Processing**: The `TINYFUSION_TEMPORAL_KERNEL` was set to 2. Tests revealed that `nn.init.dirac_` with an even kernel size (2) does NOT produce an identity mapping (mean difference > 1.0), whereas an odd kernel size (3) does (mean difference 0.0). This means the model was starting with a distorted output, contributing to instability.

## Fix Applied
1.  **Reduced Learning Rate**: Modified `text2sign/config.py` to set `LEARNING_RATE = 0.00001` (`1e-5`).
2.  **Fixed Temporal Kernel**: Modified `text2sign/config.py` to set `TINYFUSION_TEMPORAL_KERNEL = 3`.

## Explanation
- **Learning Rate**: A learning rate of `1e-4` is likely too high for this fine-tuning task. The model descends into a loss basin (around 0.8) but the step size is too large to stay there or descend further, causing it to oscillate or jump back out to a higher loss (1.0). Reducing the learning rate to `1e-5` allows for smaller, more precise updates.
- **Temporal Kernel**: Using an odd kernel size (3) ensures that the `TemporalPostProcessor` starts as a true identity function. This prevents the model from having to "unlearn" an initial shift or distortion, providing a stable starting point for the temporal smoothing layer.

## Next Steps
- Run training again: `python main.py train`
- Monitor the loss. It should decrease more steadily and stay below 0.8 once it reaches that level.
