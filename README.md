````markdown
# Text2Sign Diffusion Model

A PyTorch implementation of a TinyFusion-based diffusion model for generating sign language videos from text descriptions.

## Features

- **TinyFusion Architecture**: Pre-trained DiT (Diffusion Transformer) backbone fine-tuned for sign language video generation
- **Text-Conditioned Generation**: Uses DistilBERT for text encoding and conditioning
- **Memory Efficient**: Frame chunking and gradient checkpointing for large-scale training
- **Cross-Platform**: Supports CUDA, MPS (Apple Silicon), and CPU devices
- **Comprehensive Logging**: TensorBoard integration with detailed metrics, sample generation, and gradient analysis
- **Modular Design**: Clean, well-documented code structure for easy modification and extension

## Project Structure

```
text2sign/
├── config.py              # Configuration and hyperparameters
├── dataset.py             # Data loading and preprocessing
├── main.py                # Command-line interface
├── requirements.txt       # Python dependencies
├── verify_training_ready.py  # Pre-training verification script
├── diffusion/             # Diffusion model implementation
│   ├── __init__.py
│   └── text2sign.py       # Core diffusion logic
├── models/                # Model architectures
│   ├── __init__.py
│   ├── text_encoder.py    # DistilBERT text encoder
│   └── architectures/
│       ├── tinyfusion.py  # TinyFusion video wrapper (ACTIVE)
│       ├── vivit.py       # ViViT architecture
│       ├── vit3d.py       # ViT3D architecture
│       ├── unet3d.py      # UNet3D architecture
│       └── dit3d.py       # DiT3D architecture
├── methods/               # Training utilities
│   ├── __init__.py
│   └── trainer.py         # Main training loop
├── schedulers/            # Noise schedulers
│   ├── __init__.py
│   └── noise_schedulers.py
├── utils/                 # Utility functions
├── external/              # External dependencies
│   └── TinyFusion/        # TinyFusion backbone implementation
├── pretrained/            # Pre-trained checkpoints
│   └── TinyDiT-D14-MaskedKD-500K.pt
├── training_data/         # Training data (GIF files and text descriptions)
├── checkpoints/           # Model checkpoints (created during training)
├── logs/                  # TensorBoard logs (created during training)
└── generated_samples/     # Generated sample videos
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ with appropriate device support (CUDA/MPS/CPU)
- 8GB+ GPU memory recommended (or 16GB+ for Apple Silicon)

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Verify the installation and model setup:
```bash
python verify_training_ready.py
```

This will check:
- Configuration is correct
- Model can be loaded
- Pre-trained weights are accessible
- Forward and backward passes work
- Gradients flow properly

## Quick Start

### 1. Verify Everything is Ready

```bash
python verify_training_ready.py
```

Expected output:
```
✅ ALL CHECKS PASSED!

Your model is ready for training. You can now run:
   python main.py train
```

### 2. Start Training

```bash
# Start fresh training
python main.py train

# Resume from checkpoint
python main.py train --resume
```

### 3. Monitor Training

In a separate terminal:
```bash
python start_tensorboard.sh
# Then open http://localhost:6006
```

### 4. Generate Samples

```bash
python main.py sample \
  --checkpoint checkpoints/tinyfusion_test_3/latest_checkpoint.pt \
  --text "hello" \
  --num_samples 4
```

## Usage

### Command Line Interface

The project provides a comprehensive CLI through `main.py`:

```bash
# Show all available commands
python main.py --help

# Show current configuration
python main.py config

# List available checkpoints
python main.py checkpoints

# Train the model
python main.py train

# Resume training from checkpoint
python main.py train --resume

# Generate samples (eta: 0=deterministic, 1=stochastic)
python main.py sample \
  --checkpoint checkpoints/tinyfusion_test_3/latest_checkpoint.pt \
  --text "hello" \
  --num_samples 4 \
  --eta 0.0

# Fix config to match a checkpoint architecture
python main.py fix-config --checkpoint path/to/checkpoint.pt
```

### Training Configuration

All hyperparameters are in `config.py`. Key settings:

```python
# Model Architecture
MODEL_ARCHITECTURE = "tinyfusion"
TINYFUSION_VARIANT = "DiT-D14/2"
TINYFUSION_FREEZE_BACKBONE = False  # MUST be False for training

# Training Settings
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000
GRADIENT_CLIP = 1.0

# Video Dimensions
NUM_FRAMES = 16
IMAGE_SIZE = 64
INPUT_SHAPE = (3, 16, 64, 64)  # (channels, frames, height, width)

# Diffusion Process
TIMESTEPS = 50
INFERENCE_TIMESTEPS = 50
NOISE_SCHEDULER = "cosine"
```

### Training

To start training:

```bash
python main.py train
```

Training features:
- **Automatic checkpoint saving**: Every 10 epochs + latest checkpoint
- **TensorBoard logging**: Comprehensive metrics including:
  - Loss curves and learning rates
  - Generated sample videos
  - Gradient statistics and histograms
  - Parameter distributions
  - Memory usage tracking
  - Model architecture graph
- **Sample generation**: Every 5 epochs for quality monitoring
- **Resume support**: `python main.py train --resume`
- **Gradient clipping**: Stable training with gradient norm clipping
- **Memory optimization**: Frame chunking, gradient checkpointing

Monitor training with TensorBoard:
```bash
python start_tensorboard.sh
# Or manually:
tensorboard --logdir logs/tinyfusion_test_3
```

### Expected Training Behavior

**First 100 Steps:**
- Loss should start around 0.5-1.0
- Should begin decreasing within 20-50 steps
- Gradient norm stable at 1.0-10.0 (after clipping)

**After 1000 Steps:**
- Loss should decrease to ~0.1-0.3
- Samples should show structure instead of pure noise
- Quality improves gradually

**Signs of Success:**
- ✅ Loss consistently decreasing
- ✅ Generated samples improving quality
- ✅ No NaN/Inf warnings in logs
- ✅ Gradient norms stable

**Signs of Problems:**
- ❌ Loss stuck at initial value → Try reducing learning rate
- ❌ Out of memory → Reduce batch size or frame chunk size
- ❌ NaN loss → Check data quality, reduce learning rate

## Model Architecture

### TinyFusion Video Wrapper

The model uses **TinyFusion** (DiT-D14/2), a Diffusion Transformer pre-trained on ImageNet, adapted for video generation:

- **Backbone**: DiT-D14/2 with 340M+ parameters
- **Frame Processing**: Frame-by-frame with temporal post-processing
- **Text Conditioning**: DistilBERT text encoder (768-dim embeddings)
- **Video Output**: 16 frames @ 64x64 RGB

### Architecture Components

1. **Text Encoder** (`models/text_encoder.py`)
   - Pre-trained DistilBERT
   - Frozen during training
   - Produces 768-dimensional embeddings

2. **TinyFusion Backbone** (`models/architectures/tinyfusion.py`)
   - Pre-trained DiT-D14/2 transformer
   - Trainable (not frozen)
   - Frame-by-frame processing with memory optimization
   - Temporal post-processing with 3D convolution

3. **Diffusion Process** (`diffusion/text2sign.py`)
   - Cosine noise schedule
   - 50 training timesteps
   - Text-conditioned noise prediction
   - DDIM sampling support

### Key Features

- **Memory Efficient**: Frame chunking (4 frames per chunk)
- **Gradient Checkpointing**: Reduces memory usage during training
- **Smart Initialization**: Properly adapted pre-trained weights
- **NaN/Inf Handling**: Robust numerical stability checks

## Monitoring and Visualization

### TensorBoard Logs

The training process logs to 15 organized categories:

1. **Training**: Core metrics (loss, learning rate, gradient norm)
2. **Loss Components**: Detailed loss breakdown
3. **Epoch Summary**: Aggregated epoch-level metrics
4. **Learning Rate**: LR scheduling history
5. **Performance**: Training throughput and timing
6. **Diffusion**: Noise prediction MSE, SNR analysis
7. **Noise Analysis**: Detailed noise statistics
8. **Model Architecture**: Parameter counts and structure
9. **Parameter Stats**: Layer-wise parameter analysis
10. **Parameter Histograms**: Distribution visualization
11. **Gradient Stats**: Gradient norms and flow
12. **Generated Samples**: Video samples as GIFs
13. **Noise Visualization**: Prediction vs ground truth
14. **System**: GPU/MPS memory usage
15. **Configuration**: Training settings

### Sample Generation

During training:
- Samples generated every 5 epochs
- Saved as GIFs in `generated_samples/tinyfusion_test_3/`
- Also logged to TensorBoard for easy viewing
- Text prompts from validation set

### Real-Time Monitoring

TensorBoard provides:
- Live loss curves
- Sample quality progression
- Memory usage tracking
- Gradient flow visualization
- Parameter evolution over time

## Performance Optimization

### Memory Management

- **Frame Chunking**: Process 4 frames at a time (configurable)
- **Gradient Checkpointing**: Reduces memory by recomputing activations
- **Mixed Precision**: Automatic mixed precision training (optional)
- **CUDA Cache Management**: Periodic cache clearing

### Current Settings (Optimized for 8-16GB GPU)

```python
BATCH_SIZE = 1
NUM_FRAMES = 16
IMAGE_SIZE = 64
TINYFUSION_FRAME_CHUNK_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
```

### Scaling Up

For more GPU memory (24GB+):
```python
BATCH_SIZE = 2-4
NUM_FRAMES = 28
IMAGE_SIZE = 128
TINYFUSION_FRAME_CHUNK_SIZE = 8
```

### Scaling Down

For limited memory (4-8GB):
```python
BATCH_SIZE = 1
NUM_FRAMES = 8
IMAGE_SIZE = 32
TINYFUSION_FRAME_CHUNK_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
```

## Troubleshooting

### Common Issues

1. **"No trainable parameters" Error**
   - Check: `TINYFUSION_FREEZE_BACKBONE = False` in config.py
   - The model must be trainable for training to work

2. **Out of Memory**
   - Reduce `BATCH_SIZE` to 1
   - Reduce `TINYFUSION_FRAME_CHUNK_SIZE` to 2
   - Reduce `NUM_FRAMES` or `IMAGE_SIZE`
   - Enable gradient checkpointing

3. **Loss Not Decreasing**
   - Check that model has trainable parameters
   - Verify data is loading correctly
   - Try reducing learning rate to 1e-5
   - Check for NaN/Inf warnings

4. **Pure Black/Zero Output**
   - **FIXED**: This was caused by frozen backbone with random output layers
   - Ensure `TINYFUSION_FREEZE_BACKBONE = False`
   - Run `python verify_training_ready.py` to check

5. **Checkpoint Loading Errors**
   - Architecture mismatch between config and checkpoint
   - Use: `python main.py fix-config --checkpoint path/to/checkpoint.pt`

### Performance Tips

1. **Monitor GPU/MPS Usage**: 
   - Linux/CUDA: `nvidia-smi -l 1`
   - macOS: Activity Monitor → GPU tab

2. **Speed Up Training**:
   - Reduce `INFERENCE_TIMESTEPS` for faster sampling
   - Reduce sample generation frequency
   - Disable some diagnostic logging

3. **Improve Quality**:
   - Train for more epochs (1000+)
   - Increase resolution (IMAGE_SIZE=128)
   - Add more frames (NUM_FRAMES=28)
   - Use cosine noise schedule (already default)

## Data Format

The model expects:
- **GIF Files**: Sign language videos in `training_data/` directory
- **Text Files**: Corresponding `.txt` files with the same name
- **Format**: Each GIF should contain sign language motion
- **Preprocessing**: Automatic center cropping and frame extraction

Example:
```
training_data/
├── hello.gif       # Sign language video for "hello"
├── hello.txt       # Text: "hello"
├── goodbye.gif     # Sign language video for "goodbye"
└── goodbye.txt     # Text: "goodbye"
```

## Development

### Project Status

**Current State**: ✅ Ready for Training
- Model architecture: TinyFusion (DiT-D14/2)
- Text encoding: DistilBERT
- All components tested and verified
- Memory optimization implemented
- Comprehensive logging enabled

### Adding Features

1. **New Architectures**: 
   - Add to `models/architectures/`
   - Register in `diffusion/text2sign.py` create function
   - Update `config.py` with new settings

2. **Loss Functions**: 
   - Modify `diffusion/text2sign.py` forward method
   - Add logging in `methods/trainer.py`

3. **Data Processing**: 
   - Extend `dataset.py` TextToSignDataset class
   - Add augmentation in __getitem__ method

4. **Hyperparameters**: 
   - Add to `config.py` Config class
   - Update documentation

### Available Architectures

The codebase includes multiple architectures (change in config.py):

- **tinyfusion** (ACTIVE): TinyFusion DiT-D14/2 with temporal processing
- **vivit**: Video Vision Transformer with temporal attention layers
- **vit3d**: 3D Vision Transformer for video
- **dit3d**: 3D Diffusion Transformer
- **unet3d**: 3D UNet with residual blocks

## Recent Fixes (October 2025)

### TinyFusion Zero Output Fix

**Problem**: Model was producing pure black (all zeros) output, causing training loss to not decrease.

**Root Causes**:
1. Frozen backbone with randomly initialized output layers
2. Output layers skipped during checkpoint loading
3. Aggressive NaN-to-zero conversion

**Solutions Applied**:
1. Unfroze backbone (`TINYFUSION_FREEZE_BACKBONE = False`)
2. Proper output layer initialization from checkpoint statistics
3. Smart NaN/Inf handling that preserves valid data
4. Better checkpoint loading with shape adaptation

**Verification**: Run `python verify_training_ready.py` to ensure everything works.

See `TRAINING_FIX_SUMMARY.md` for detailed technical explanation.

## License

This project is part of the Polygence research program.

## Acknowledgments

- [TinyFusion](https://github.com/VainF/TinyFusion) for the efficient DiT backbone
- [Diffusers](https://github.com/huggingface/diffusers) library by Hugging Face
- PyTorch team for the deep learning framework
- Diffusion model research community

---

**Ready to train?** Run `python verify_training_ready.py` then `python main.py train`
````
