# Text-to-Sign Language DDIM Diffusion Model

A PyTorch implementation of a text-to-sign diffusion model for generating short sign-language video clips from text descriptions, with an emphasis on practical single-GPU training and evaluation.

## Features

- **3D UNet Architecture with DiT-style Transformers**: Video-aware model with spatial, temporal, and cross-attention mechanisms enhanced with Diffusion Transformer (DiT) blocks
- **Spatio-Temporal Transformer**: Advanced transformer architecture that processes spatial attention within each frame and temporal attention across frames
- **Adaptive Layer Normalization**: DiT-style conditioning for better timestep integration
- **Text Conditioning**: Frozen CLIP text conditioning by default, plus custom and partially trainable variants
- **DDIM Scheduler**: Supports both deterministic (DDIM) and stochastic (DDPM) sampling
- **TensorBoard Logging**: Real-time training monitoring
- **Progress Bars**: tqdm progress bars for training and generation
- **Mixed Precision Training**: FP16 training for faster training on modern GPUs
- **Training Optimizations**: Optional `torch.compile`, TF32, channels-last 3D, and configurable precision modes
- **Classifier-Free Guidance**: Improved generation quality with guidance
- **Signer-Disjoint Evaluation**: Dataset split protocol with zero signer overlap between train and validation
- **Benchmarking + Back-Translation**: Runtime benchmarking and optional GloFE-based back-translation proxy evaluation

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

Use the current workspace path if you are running from this repository root:

```bash
python main.py test --data-dir /teamspace/studios/this_studio/text_to_sign/training_data
```

### 2. Train the Model

```bash
python main.py train \
    --data-dir /teamspace/studios/this_studio/text_to_sign/training_data \
    --model-size base \
    --epochs 100 \
    --batch-size 2 \
    --grad-accum-steps 8 \
    --lr 5e-5 \
    --precision auto \
    --split-mode signer_disjoint
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

### 5. Validate a Checkpoint

```bash
python main.py validate \
    --checkpoint text_to_sign/checkpoints/<run_name>/best_model.pt \
    --data-dir /teamspace/studios/this_studio/text_to_sign/training_data \
    --benchmark-repeats 5
```

If the GloFE dependencies are unavailable in your environment, add:

```bash
--skip-backtranslation
```
```

## Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `text_to_sign/training_data` | Path to training data |
| `--batch-size` | 2 | Training batch size |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 5e-5 | Learning rate |
| `--image-size` | 64 | Image size (width=height) |
| `--num-frames` | 24 | Number of video frames |
| `--model-channels` | 64 | Base model channels |
| `--model-size` | preset-driven | Named model/training size preset (`small`, `base`, `large`) |
| `--grad-accum-steps` | 8 | Gradient accumulation steps |
| `--timesteps` | 1000 | Diffusion timesteps |
| `--beta-schedule` | cosine | Beta schedule (linear/cosine) |
| `--split-mode` | `signer_disjoint` | Train/validation split protocol |
| `--text-conditioning-mode` | `normal` | Conditioning ablation mode (`normal`, `none`, `random`) |
| `--clip-trainable-layers` | 0 | Unfreeze last N CLIP layers |
| `--precision` | `auto` | Precision mode (`auto`, `fp16`, `bf16`, `fp32`) |
| `--no-compile` | False | Disable `torch.compile` |
| `--save-every` | 5 | Save checkpoint every N epochs |
| `--no-amp` | False | Disable mixed precision |

### Recommended full-training baseline

For a first serious run, start with the balanced preset and the automatic CUDA optimizations enabled:

```bash
python main.py train \
    --data-dir /teamspace/studios/this_studio/text_to_sign/training_data \
    --model-size base \
    --epochs 100 \
    --lr 5e-5 \
    --split-mode signer_disjoint \
    --precision auto \
    --compile-mode reduce-overhead
```

## Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--prompt` | - | Text prompt(s) for generation |
| `--steps` | 50 | Number of denoising steps |
| `--guidance-scale` | 7.5 | CFG guidance scale |
| `--eta` | 0.0 | DDIM stochasticity (0=deterministic) |

## Validation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-samples` | 50 | Number of validation samples to score |
| `--benchmark-repeats` | 5 | Number of repeated latency runs |
| `--skip-backtranslation` | False | Skip GloFE back-translation proxy |

## Model Architecture

### 3D UNet
- Input: Noisy video tensor `(B, C, T, H, W)`
- Encoder with residual blocks and downsampling
- Spatial, temporal, and cross-attention at multiple resolutions
- Decoder with skip connections and upsampling
- Output: Predicted noise `(B, C, T, H, W)`

### Text Encoder
- Frozen CLIP text encoder by default
- Optional custom transformer encoder / partial CLIP unfreezing for ablations
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

Signer-disjoint splitting is inferred from the filename prefix before the first underscore. If your dataset uses a different naming convention, update `SignLanguageDataset._extract_signer_id()` in `dataset.py`.

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

During validation, the following artifacts are produced:
- `validation_results.json`
- `comparison.png`
- Real/generated GIF pairs
- Runtime benchmark summary
- Optional back-translation examples and scores

## License

MIT License
