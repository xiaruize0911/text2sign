# Resume Training Functionality for Text2Sign Diffusion Model

## Overview

Added comprehensive resume training functionality to the Text2Sign diffusion model, allowing seamless continuation of training from previously saved checkpoints.

## Features

### 1. Resume Command (`--resume` flag)
- **Usage**: `python main.py train --resume`
- **Functionality**: Automatically loads the latest checkpoint and continues training
- **Fallback**: If no checkpoint found, starts fresh training with a warning

### 2. Checkpoint Management Commands

#### List Available Checkpoints
```bash
python main.py checkpoints
```
- **Displays**: All available checkpoints with epoch, step, and file size information
- **Filtering**: Excludes temporary files (starting with `._`)
- **Information**: Shows epoch number, global step, and file size for each checkpoint

#### Enhanced Sample Generation
```bash
python main.py sample --checkpoint checkpoints/latest_checkpoint.pt --num_samples 4
```
- **Improved**: Now includes checkpoint info (epoch/step) in logs
- **Output**: Saves generated samples as GIF files with progress tracking

### 3. Checkpoint Information Display

#### Example Output:
```
INFO:__main__:Available checkpoints in checkpoints:
INFO:__main__:  • checkpoint_epoch_7.pt (4.2 MB) - Epoch: 7, Step: 16328
INFO:__main__:  • checkpoint_step_17800.pt (4.2 MB) - Epoch: 8, Step: 17800
INFO:__main__:  • latest_checkpoint.pt (4.2 MB) - Epoch: 7, Step: 16328
INFO:__main__:  • interrupted_checkpoint.pt (4.2 MB) - Epoch: 8, Step: 17899
```

## Technical Implementation

### 1. PyTorch 2.6 Compatibility
- **Fixed**: `weights_only=False` parameter added to `torch.load()` calls
- **Reason**: PyTorch 2.6 changed default behavior for security
- **Impact**: Enables loading of complete checkpoint data (not just model weights)

### 2. Smart Checkpoint Loading
- **Latest Checkpoint**: Automatically finds and loads `latest_checkpoint.pt`
- **Error Handling**: Graceful fallback if checkpoint is missing or corrupted
- **Logging**: Clear information about what checkpoint was loaded

### 3. Training State Restoration
When resuming, the following state is fully restored:
- **Model weights**: Complete neural network parameters
- **Optimizer state**: Adam/AdamW momentum and learning rate schedules
- **Training progress**: Current epoch and global step counters
- **Scheduler state**: Learning rate scheduling progress

## Usage Examples

### Basic Resume Training
```bash
# Start fresh training
python main.py train

# Resume from latest checkpoint
python main.py train --resume
```

### Check Available Checkpoints
```bash
# List all checkpoints
python main.py checkpoints

# Generate samples from specific checkpoint
python main.py sample --checkpoint checkpoints/checkpoint_epoch_5.pt
```

### Training Workflow
```bash
# 1. Start training
python main.py train

# 2. Stop training (Ctrl+C) - creates interrupted_checkpoint.pt

# 3. Check available checkpoints
python main.py checkpoints

# 4. Resume training from where you left off
python main.py train --resume
```

## Checkpoint Types

### Automatic Checkpoints
- **`latest_checkpoint.pt`**: Always updated with the most recent state
- **`checkpoint_epoch_X.pt`**: Saved at the end of each epoch
- **`checkpoint_step_X.pt`**: Saved every N training steps (configurable)

### Emergency Checkpoints
- **`interrupted_checkpoint.pt`**: Created when training is stopped with Ctrl+C
- **`error_checkpoint.pt`**: Created when training fails due to an error

## Configuration

Resume behavior can be customized through `config.py`:
- **`SAVE_EVERY`**: How often to save step checkpoints (default: 100 steps)
- **`SAMPLE_EVERY`**: How often to generate samples during training
- **`CHECKPOINT_DIR`**: Directory to store checkpoints (default: "checkpoints")

## Benefits

### Development Workflow
- **Experimentation**: Easy to stop and resume training during hyperparameter tuning
- **Resource Management**: Efficient use of compute resources (can pause/resume as needed)
- **Debugging**: Can inspect model state at different training stages

### Production Training
- **Reliability**: Automatic recovery from system crashes or interruptions
- **Long Training**: Essential for multi-day training sessions
- **Monitoring**: Easy to track training progress across sessions

### MacBook M4 Optimization
- **Memory Management**: Can pause training to free up memory for other tasks
- **Thermal Management**: Can pause during high-load periods to prevent overheating
- **Battery Conservation**: Pause training when running on battery power

## Error Handling

### Common Issues and Solutions
1. **Checkpoint Loading Errors**: Fixed with `weights_only=False` parameter
2. **Missing Checkpoints**: Graceful fallback to fresh training
3. **Corrupted Checkpoints**: Clear error messages and recovery options
4. **Version Compatibility**: Robust loading that handles minor version differences

## Future Enhancements

### Potential Improvements
- **Selective Loading**: Option to load only model weights (not optimizer state)
- **Checkpoint Cleanup**: Automatic removal of old checkpoints to save disk space
- **Distributed Training**: Support for multi-GPU checkpoint synchronization
- **Cloud Integration**: Automatic backup to cloud storage services

This resume functionality makes the Text2Sign diffusion model much more practical for real-world training scenarios, especially on MacBook M4 hardware where training sessions may need to be interrupted and resumed frequently.
