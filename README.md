# Text-to-Sign Language DDIM Diffusion Model

A PyTorch implementation of a DDIM (Denoising Diffusion Implicit Models) diffusion model for generating sign language GIF animations from text descriptions.

## Features

- **3D UNet Architecture with DiT-style Transformers**: Video-aware model with spatial, temporal, and cross-attention mechanisms enhanced with Diffusion Transformer (DiT) blocks
- **Spatio-Temporal Transformer**: Advanced transformer architecture that processes spatial attention within each frame and temporal attention across frames
- **Adaptive Layer Normalization**: DiT-style conditioning for better timestep integration
- **Text Conditioning**: Transformer-based text encoder for text-to-video generation
- **DDIM Scheduler**: Supports both deterministic (DDIM) and stochastic (DDPM) sampling
- **TensorBoard Logging**: Real-time training monitoring
- **Progress Bars**: tqdm progress bars for training and generation
- **Mixed Precision Training**: FP16 training for faster training on modern GPUs
- **Classifier-Free Guidance**: Improved generation quality with guidance

## Architecture

### DiT-style Transformer Blocks

The model uses enhanced transformer blocks inspired by Diffusion Transformers (DiT):

1. **SpatioTemporalTransformer**: Combines spatial and temporal attention
   - Spatial attention processes each frame independently
   - Temporal attention captures motion across frames
   
2. **DiTBlock**: Full transformer block with adaptive layer norm
   - Self-attention with adaptive normalization
   - Cross-attention for text conditioning
   - Feed-forward network with gating

3. **Adaptive Layer Normalization**: Timestep embeddings modulate the layer norm parameters for better conditioning

## Project Structure

```
text_to_sign/
├── main.py              # CLI interface for training and generation
├── config.py            # Configuration dataclasses
├── dataset.py           # Dataset and data loading
├── trainer.py           # Training loop with TensorBoard
├── pipeline.py          # End-to-end generation pipeline
├── models/
│   ├── __init__.py
│   ├── unet3d.py        # 3D UNet with DiT-style transformers
│   └── text_encoder.py  # Text encoder
├── schedulers/
│   ├── __init__.py
│   └── ddim.py          # DDIM noise scheduler
├── checkpoints/         # Saved model checkpoints
├── logs/                # TensorBoard logs
└── generated/           # Generated GIFs
```

## Requirements

```bash
pip install torch torchvision tqdm tensorboard pillow numpy einops
```

## Quick Start

### 1. Test the Setup

```bash
cd text_to_sign
python main.py test --data-dir ../text2sign/training_data
```

### 2. Train the Model

```bash
python main.py train \
    --data-dir ../text2sign/training_data \
    --epochs 100 \
    --batch-size 4 \
    --image-size 64 \
    --num-frames 16 \
    --lr 1e-4
```

### 3. Monitor Training with TensorBoard

```bash
tensorboard --logdir text_to_sign/logs
```

### 4. Generate Sign Language GIFs

```bash
python main.py generate \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Hello" "Thank you" "I love you" \
    --steps 50 \
    --guidance-scale 7.5
```

## Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `text2sign/training_data` | Path to training data |
| `--batch-size` | 4 | Training batch size |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--image-size` | 64 | Image size (width=height) |
| `--num-frames` | 16 | Number of video frames |
| `--model-channels` | 128 | Base model channels |
| `--timesteps` | 1000 | Diffusion timesteps |
| `--beta-schedule` | linear | Beta schedule (linear/cosine) |
| `--save-every` | 5 | Save checkpoint every N epochs |
| `--no-amp` | False | Disable mixed precision |
| `--use-transformer` | True | Use DiT-style transformer blocks |
| `--transformer-depth` | 1 | Number of transformer layers per block |

## Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--prompt` | - | Text prompt(s) for generation |
| `--steps` | 50 | Number of denoising steps |
| `--guidance-scale` | 7.5 | CFG guidance scale |
| `--eta` | 0.0 | DDIM stochasticity (0=deterministic) |

## Model Architecture

### 3D UNet
- Input: Noisy video tensor `(B, C, T, H, W)`
- Encoder with residual blocks and downsampling
- Spatial, temporal, and cross-attention at multiple resolutions
- Decoder with skip connections and upsampling
- Output: Predicted noise `(B, C, T, H, W)`

### Text Encoder
- Token embedding with learned positional encoding
- 6-layer transformer encoder
- Outputs contextualized text embeddings `(B, seq_len, embed_dim)`

### DDIM Scheduler
- Linear or cosine beta schedules
- Supports epsilon and v-prediction
- Deterministic (eta=0) or stochastic (eta>0) sampling

## Data Format

The training data should be organized as pairs of GIF and text files:
```
training_data/
├── video1.gif
├── video1.txt
├── video2.gif
├── video2.txt
└── ...
```

Each `.txt` file contains the text description for the corresponding `.gif` file.

## Example Usage in Python

```python
from text_to_sign.pipeline import Text2SignPipeline

# Load trained model
pipeline = Text2SignPipeline.from_pretrained("checkpoints/best_model.pt")

# Generate sign language video
videos = pipeline(
    prompt=["Hello world"],
    num_inference_steps=50,
    guidance_scale=7.5,
)

# Save as GIF
pipeline.save_gif(videos[0], "hello_world.gif")
```

## TensorBoard Metrics

During training, the following metrics are logged:
- `train/loss`: Training loss per step
- `train/lr`: Learning rate
- `epoch/train_loss`: Average training loss per epoch
- `epoch/val_loss`: Validation loss per epoch
- `samples/generated`: Generated sample images

## License

MIT License
