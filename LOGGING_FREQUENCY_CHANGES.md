## TensorBoard Logging Frequency Reduction Summary

### Changes Made

#### 1. **Config Updates** (`config.py`)
- **LOG_EVERY**: Changed from `10` to `1020` (once per epoch)
- **SAMPLE_EVERY**: Changed from `1000` to `1020` (once per epoch) 
- **SAVE_EVERY**: Changed from `1000` to `2040` (every two epochs)

#### 2. **Training Loop Updates** (`methods/trainer.py`)
- **Epoch-end logging**: Added comprehensive logging at the end of each epoch
- **Epoch-end sampling**: Generate samples at the end of each epoch regardless of step-based timing
- **Checkpoint frequency**: Save epoch checkpoints only every 2 epochs (epochs 0, 2, 4, etc.)
- **TensorBoard flushing**: Reduced from every 3 steps to every 100 steps
- **Enhanced logging**: Added frequency information to startup logs

### Calculation Basis
- **Training dataset**: ~4,082 samples
- **Batch size**: 4
- **Steps per epoch**: ~1,020 (4,082 ÷ 4)
- **Steps per 2 epochs**: ~2,040

### Expected Behavior
1. **Logging**: TensorBoard will receive updates approximately once per epoch
2. **Sampling**: Video samples will be generated once per epoch  
3. **Checkpointing**: 
   - Step-based checkpoints: Every ~2,040 steps (every 2 epochs)
   - Epoch-based checkpoints: Every 2 epochs (0, 2, 4, 6, ...)
   - Latest checkpoint: Still saved every epoch
4. **Storage efficiency**: Significantly reduced disk I/O and storage usage
5. **TensorBoard performance**: Faster loading with fewer data points

### Benefits
- **Reduced storage**: Fewer checkpoint files and log entries
- **Faster TensorBoard**: Less frequent updates improve UI responsiveness
- **Better overview**: Epoch-level metrics provide cleaner training curves
- **Maintained monitoring**: Still get comprehensive progress tracking
- **Preserved functionality**: All logging features remain, just less frequent

### File Locations
- **Logs**: `logs/` directory (TensorBoard events)
- **Checkpoints**: `checkpoints/` directory (model states)
- **Samples**: `generated_samples/` directory (GIF videos)
