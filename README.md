# Text-to-Sign Language Diffusion Model

A PyTorch implementation of a diffusion model for generating sign language videos from English text descriptions.

## Features

- **3D UNet Architecture**: Custom 3D UNet with spatial-temporal attention for video generation
- **Multiple Text Encoders**: Support for CLIP, T5, and simple LSTM-based text encoders
- **Flexible Schedulers**: DDPM and DDIM schedulers for training and inference
- **Complete Pipeline**: End-to-end training and inference pipeline
- **Video Processing**: Automatic GIF loading and processing for training data

## Project Structure

```
text2sign/
├── src/
│   ├── models/
│   │   ├── unet3d.py          # 3D UNet implementation
│   │   ├── scheduler.py       # DDPM/DDIM schedulers
│   │   ├── text_encoder.py    # Text encoding models
│   │   └── pipeline.py        # Complete diffusion pipeline
│   ├── data/
│   │   └── dataset.py         # Dataset and data loading utilities
│   └── training/
├── training_data/             # Your GIF and text files
├── train.py                   # Training script
├── inference.py              # Inference script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. **Clone or download the project files to your workspace**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare your data:**
   - Ensure your training data is in the `training_data/` directory
   - Each GIF should have a corresponding TXT file with the same name
   - Example: `video_001.gif` and `video_001.txt`

## Quick Start

### Training

Basic training with default settings:
```bash
python train.py --data_dir ./training_data --batch_size 4 --num_epochs 50
```

Advanced training options:
```bash
python train.py \
    --data_dir ./training_data \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --max_frames 16 \
    --frame_size 64 \
    --model_channels 128 \
    --text_encoder simple \
    --scheduler ddpm \
    --save_dir ./checkpoints \
    --use_wandb
```

### Inference

Generate videos from text prompts:
```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --prompts "Hello" "How are you?" "Thank you" \
    --output_dir ./outputs \
    --num_frames 16 \
    --steps 50
```

## Model Architecture

### 3D UNet
- **Input/Output**: 3-channel RGB videos
- **Architecture**: Encoder-decoder with skip connections
- **Attention**: Spatial-temporal attention at multiple resolutions
- **Conditioning**: Text embeddings injected via cross-attention and time embedding

### Text Encoders
1. **Simple Encoder**: LSTM-based with learnable vocabulary
2. **CLIP**: Pre-trained CLIP text encoder (frozen)
3. **T5**: Pre-trained T5 encoder (frozen)

### Schedulers
1. **DDPM**: Standard denoising diffusion process
2. **DDIM**: Deterministic sampling for faster inference

## Training Configuration

Key hyperparameters:

- **Learning Rate**: 1e-4 (with cosine annealing)
- **Batch Size**: 4-8 (depending on GPU memory)
- **Max Frames**: 16 frames per video
- **Frame Size**: 64x64 pixels
- **Timesteps**: 1000 (training), 50 (inference)
- **Model Channels**: 128 (base channels, scaled with channel_mult)

## Data Format

Your training data should follow this structure:

```
training_data/
├── video_001.gif    # Sign language video
├── video_001.txt    # "Hello, how are you?"
├── video_002.gif
├── video_002.txt    # "I am fine, thank you"
└── ...
```

**Requirements:**
- GIF files containing sign language demonstrations
- Corresponding text files with English descriptions
- Consistent naming (same filename, different extensions)

## Inference Options

The inference script supports various output formats:

- **GIF**: Animated GIF files (default)
- **Frames**: Individual PNG frames
- **Both**: Both GIF and frame outputs

Additional options:
- Custom resolution (height/width)
- Number of inference steps (speed vs quality trade-off)
- Classifier-free guidance scale
- Random seed for reproducibility

## Performance Tips

### Training
- Use GPU for faster training (`CUDA_VISIBLE_DEVICES=0`)
- Reduce batch size if you encounter memory issues
- Use mixed precision training for better memory efficiency
- Monitor training with Weights & Biases (`--use_wandb`)

### Inference
- Use DDIM scheduler for faster sampling
- Reduce inference steps for speed (at cost of quality)
- Lower resolution for faster generation
- Use CPU if GPU memory is limited

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size (`--batch_size 2`)
   - Reduce model size (`--model_channels 64`)
   - Reduce frame size (`--frame_size 32`)

2. **Slow Training**
   - Reduce number of workers if I/O bound
   - Use SSD storage for training data
   - Enable mixed precision training

3. **Poor Generation Quality**
   - Train for more epochs
   - Increase model capacity
   - Use better text encoder (CLIP/T5)
   - Increase inference steps

### Dependencies Issues

If you encounter import errors:
```bash
# For transformers-related errors
pip install transformers>=4.21.0

# For image processing
pip install Pillow>=9.0.0

# For optional dependencies
pip install wandb  # For experiment tracking
pip install matplotlib  # For visualization
```

## Model Checkpoints

The training script saves:
- `best_model.pt`: Best model based on validation loss
- `final_model.pt`: Final model after training
- `checkpoint_epoch_X_step_Y.pt`: Regular checkpoints

Checkpoint contents:
- Model state dict
- Optimizer state
- Scheduler state
- Training metadata

## Advanced Usage

### Custom Text Encoder

To use your own text encoder:

```python
from src.models.text_encoder import SimpleTextEncoder

# Create custom encoder
custom_encoder = SimpleTextEncoder(
    vocab_size=20000,
    embed_dim=512,
    hidden_dim=1024
)

# Use in pipeline
pipeline = create_pipeline(text_encoder=custom_encoder)
```

### Custom Training Loop

```python
from src.models.pipeline import create_pipeline
from src.data.dataset import create_dataloader

# Create components
pipeline = create_pipeline()
dataloader = create_dataloader("./training_data")

# Custom training
for batch in dataloader:
    videos = batch['videos']
    texts = batch['texts']
    
    loss = pipeline.train_step(videos, texts)
    # Your custom training logic here
```

## Contributing

This is a research implementation. Feel free to:
- Experiment with different architectures
- Add new text encoders
- Improve the training process
- Add evaluation metrics

## License

This project is for educational and research purposes.

## Acknowledgments

- Based on DDPM and DDIM papers
- Inspired by Stable Diffusion and video generation research
- Uses Hugging Face Transformers for text encoding
