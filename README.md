# Text2Sign Diffusion Model

A PyTorch implementation of a 3D UNet-based diffusion model for generating sign language videos from text descriptions.

## Features

- **3D UNet Architecture**: Specifically designed for video generation with temporal consistency
- **Diffusion Model**: DDPM-style training and sampling for high-quality video generation
- **MacBook M4 Optimized**: Smaller model size and optimized settings for Apple Silicon
- **Cross-Platform**: Supports both CUDA and MPS (Apple Silicon) devices
- **Comprehensive Logging**: TensorBoard integration for monitoring training and visualizing results
- **Modular Design**: Clean, well-documented code structure for easy modification

## Project Structure

```
text2sign/
├── config.py          # Configuration and hyperparameters
├── dataset.py         # Data loading and preprocessing
├── model.py           # 3D UNet architecture
├── diffusion.py       # Diffusion model implementation
├── train.py           # Training utilities and main training loop
├── utils.py           # Utility functions
├── main.py            # Command-line interface
├── requirements.txt   # Python dependencies
├── training_data/     # Training data (GIF files and text descriptions)
├── logs/              # TensorBoard logs (created during training)
└── checkpoints/       # Model checkpoints (created during training)
```

## Setup

### Prerequisites

- Python 3.8+
- Conda environment named `text2sign` (already created)
- PyTorch 2.0+ with appropriate device support (CUDA/MPS)

### Installation

1. Activate the conda environment:
```bash
conda activate text2sign
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or use the built-in installer:
```bash
python main.py install
```

## Usage

### Command Line Interface

The project provides a comprehensive CLI through `main.py`:

```bash
# Show all available commands
python main.py --help

# Show current configuration
python main.py config

# Test all components
python main.py test

# Train the model
python main.py train

# Generate samples (eta controls stochasticity: 0=deterministic, 1=ancestral)
python main.py sample --checkpoint checkpoints/latest_checkpoint.pt --num_samples 8 --eta 0.0

# Visualize model architecture
python main.py visualize

# Launch Tune-A-Video style fine-tuning (see section below)
python -m tuneavideo_text2sign.training.train --data_root training_data
```

### Training

To start training:

```bash
python main.py train
```

Training features:
- Automatic checkpoint saving
- TensorBoard logging
- Sample generation during training
- Resume from checkpoint support
- Gradient clipping and learning rate scheduling

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs
```

## Tune-A-Video Text2Sign

We now provide a self-contained fork of the [Tune-A-Video](https://github.com/showlab/Tune-A-Video) pipeline adapted to the Text2Sign dataset. The code lives under `tuneavideo_text2sign/` and reuses the existing GIF/text pairs from `training_data/`.

### Highlights

- **Independent module** – mirrors Tune-A-Video's structure (`configs/`, `data/`, `models/`, `pipelines/`, `training/`, `utils/`).
- **Shared dataset** – wraps `training_data/` through `SignTuneAVideoDataset`, reusing the same center-crop transform as the main project.
- **Diffusers-based training loop** – loads Stable Diffusion 2.x components, inflates them to 3D, and fine-tunes with classifier-free guidance.
- **Validation previews** – optional GIF grids saved to `logs/tuneavideo_text2sign/samples` during training.

### Quickstart

```bash
# Install dependencies (diffusers, accelerate, einops were added to requirements.txt)
pip install -r requirements.txt

# Run fine-tuning with defaults targeting stabilityai/stable-diffusion-2-1-base
python -m tuneavideo_text2sign.training.train \
	--data_root training_data \
	--output_dir checkpoints/tuneavideo_text2sign \
	--logging_dir logs/tuneavideo_text2sign

# Override any hyper-parameter, e.g. a different base model and batch size
python -m tuneavideo_text2sign.training.train \
	--pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
	--train_batch_size 2 \
	--num_frames 28
```

Key arguments mirror the upstream repository. Run `python -m tuneavideo_text2sign.training.train --help` for the full list.

### Configuration

All hyperparameters are centralized in `config.py`. Key settings:

- **Model Size**: Optimized for MacBook M4 with smaller dimensions
- **Batch Size**: Set to 2 for memory efficiency
- **Device**: Automatically detects and uses MPS/CUDA/CPU
- **Logging**: Configurable sample and checkpoint intervals

### Data Format

The model expects:
- **Input**: GIF files (28 frames, 180x320 pixels, RGB)
- **Text**: Corresponding .txt files with descriptions
- **Processing**: Automatic center cropping to 128x128 pixels

## Model Architecture

### 3D UNet Features

- **Encoder-Decoder Structure**: Multi-scale feature processing
- **Residual Blocks**: 3D convolutions with time conditioning
- **Attention Mechanisms**: Self-attention for improved quality
- **Skip Connections**: Preserves fine details across scales

### Diffusion Process

- **Forward Process**: Gradual noise addition over 1000 timesteps
- **Reverse Process**: Learned denoising for sample generation
- **Time Embedding**: Sinusoidal position encoding for timesteps
- **Noise Schedule**: Linear beta schedule for stable training

## Monitoring and Visualization

### TensorBoard Logs

The training process logs:
- Loss curves and learning rates
- Generated sample videos (every 100 steps)
- Model architecture graph
- Training configuration
- Device and system information

### Sample Generation

During training, the model generates sample videos to monitor progress:
- Samples saved as frame sequences in TensorBoard
- Checkpoint-based sample generation available
- Configurable number of samples and output formats

## Performance Optimization

### MacBook M4 Specific

- **Model Size**: Reduced from typical diffusion models
- **Batch Size**: Small batches for memory efficiency
- **MPS Backend**: Optimized for Apple Silicon
- **Memory Management**: Efficient tensor operations

### Cross-Platform Support

- **Device Detection**: Automatic CUDA/MPS/CPU selection
- **Memory Monitoring**: Adaptive batch sizing
- **Gradient Accumulation**: Effective batch size scaling

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size in `config.py`
2. **Slow Training**: Check device utilization and reduce model dimensions
3. **Import Errors**: Ensure all dependencies are installed in the correct environment

### Performance Tips

1. **Monitor GPU/MPS Usage**: Use Activity Monitor (macOS) or nvidia-smi (CUDA)
2. **Adjust Learning Rate**: Based on loss curves in TensorBoard
3. **Sample Quality**: Increase model dimensions if memory allows

## Development

### Adding Features

1. **New Architectures**: Modify `model.py`
2. **Loss Functions**: Update `diffusion.py`
3. **Data Processing**: Extend `dataset.py`
4. **Hyperparameters**: Add to `config.py`

### Testing

```bash
# Test individual components
python dataset.py
python model.py
python diffusion.py

# Full system test
python main.py test
```

## License

This project is part of the Polygence research program.

## Acknowledgments

- PyTorch team for the deep learning framework
- Diffusion model research community
- Apple for MPS backend support
