# TensorBoard Monitoring Guide for Text2Sign Project

## Overview

Your text2sign diffusion model now includes comprehensive TensorBoard monitoring that tracks:

- 📊 **Loss curves** (training and validation)
- 🧠 **Network architecture** (UNet3D graph visualization)
- 🎬 **Video samples** (training, validation, and generated clips)
- 📝 **Text prompts** (associated with each video sample)
- 📈 **Training metrics** (learning rate, gradient norms)
- 🎯 **Model performance** (epoch-wise comparisons)

## Quick Start

### 1. Start Training with TensorBoard
```bash
cd /Volumes/Extreme\ Pro/Polygence/text2sign

# Train with TensorBoard enabled (default)
python train.py \
    --data_dir ./training_data \
    --batch_size 4 \
    --num_epochs 100 \
    --use_tensorboard
```

### 2. Launch TensorBoard
```bash
# Option 1: Use the helper script (recommended)
python launch_tensorboard.py

# Option 2: Manual launch
tensorboard --logdir ./checkpoints/tensorboard_logs --port 6006
```

### 3. Open Browser
Navigate to: `http://localhost:6006`

## TensorBoard Features

### 📊 SCALARS Tab
- **Loss/Train**: Real-time training loss
- **Loss/Validation**: Validation loss per epoch
- **Loss/Epoch**: Combined train/val loss comparison
- **Learning_Rate**: Learning rate schedule
- **Gradients/Total_Norm**: Gradient magnitude monitoring

### 🧠 GRAPHS Tab
- **UNet3D Architecture**: Complete network visualization
- **Model Structure**: Layer connections and data flow
- **Parameter Shapes**: Tensor dimensions at each layer

### 🖼️ IMAGES Tab
- **training/sample_X**: Training video frames
- **validation/sample_X**: Validation video frames
- **generated/sample_X**: Generated sign language clips

### 📝 TEXT Tab
- **training/text_X**: Training text prompts
- **validation/text_X**: Validation text prompts
- **generated/prompt_X**: Generation prompts

### 🎬 VIDEOS Tab (if supported)
- **Training Videos**: Original sign language clips
- **Generated Videos**: AI-generated sign language

## Advanced Usage

### Custom TensorBoard Launch
```bash
# Custom port
python launch_tensorboard.py --port 8080

# Different log directory
python launch_tensorboard.py --logdir ./custom_logs

# Don't auto-open browser
python launch_tensorboard.py --no-browser
```

### Training Arguments
```bash
# Disable TensorBoard
python train.py --data_dir ./training_data --no-use_tensorboard

# Enable both TensorBoard and Weights & Biases
python train.py --data_dir ./training_data --use_tensorboard --use_wandb
```

## Monitoring Best Practices

### 1. **Loss Monitoring**
- Watch for steady decrease in training loss
- Validation loss should follow training loss
- Large gap indicates overfitting

### 2. **Learning Rate**
- Should decrease gradually with scheduler
- Too high: loss oscillates or increases
- Too low: slow convergence

### 3. **Gradient Norms**
- Should be stable, not exploding
- Values > 10 may indicate gradient explosion
- Values < 0.1 may indicate vanishing gradients

### 4. **Sample Quality**
- Generated samples should improve over epochs
- Check for mode collapse (repetitive outputs)
- Validate text-video alignment

## Troubleshooting

### TensorBoard Won't Start
```bash
# Install TensorBoard
pip install tensorboard

# Clear cache
rm -rf ./checkpoints/tensorboard_logs/.tensorboard_cache

# Use different port
python launch_tensorboard.py --port 6007
```

### No Graphs Showing
- Ensure training has started (graph logged after first batch)
- Check log directory exists
- Verify TensorBoard is reading correct directory

### Video/Image Issues
- Samples logged every 100 training steps
- Check tensor shapes are correct (N, C, T, H, W)
- Ensure values are in [0, 1] range

## File Structure
```
checkpoints/
├── tensorboard_logs/           # TensorBoard log files
│   ├── events.out.tfevents.*  # Scalar/text logs
│   ├── events.out.images.*    # Image logs
│   └── events.out.videos.*    # Video logs (if any)
├── checkpoint_epoch_*.pt      # Model checkpoints
├── best_model.pt             # Best validation model
└── final_model.pt            # Final trained model
```

## Tips for Better Monitoring

1. **Regular Checkpoints**: Save models frequently to track progress
2. **Sample Generation**: Generate samples every few epochs to see improvement
3. **Multiple Metrics**: Monitor both loss and sample quality
4. **Comparative Analysis**: Compare different training runs
5. **Resource Monitoring**: Watch GPU/CPU usage alongside training metrics

## Integration with Training

The TensorBoard integration automatically:
- ✅ Logs model architecture on first run
- ✅ Records training/validation loss every step/epoch
- ✅ Saves video samples every 100 steps
- ✅ Tracks learning rate and gradients
- ✅ Stores generated samples during training
- ✅ Provides text-video pair associations

Start training and open TensorBoard to see your diffusion model's progress in real-time! 🚀
