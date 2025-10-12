# 🚀 Quick Start - TinyFusion Training

## Pre-Training Checklist
```bash
python pre_training_checklist.py
```
Expected: ✅ ALL CHECKS PASSED

## Start Training
```bash
python main.py train
```

## Monitor Training
```bash
# Terminal 1: Training
python main.py train

# Terminal 2: TensorBoard
python start_tensorboard.sh
# Open: http://localhost:6006
```

## Resume Training
```bash
python main.py train --resume
```

## Generate Samples
```bash
python main.py sample \
  --checkpoint checkpoints/tinyfusion_test_3/latest_checkpoint.pt \
  --text "hello" \
  --num_samples 4
```

## Key Settings (config.py)

```python
MODEL_ARCHITECTURE = "tinyfusion"
TINYFUSION_FREEZE_BACKBONE = False  # MUST BE FALSE!

BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000
NUM_FRAMES = 16
IMAGE_SIZE = 64

TIMESTEPS = 50
NOISE_SCHEDULER = "cosine"
```

## Expected Training Progress

| Milestone | Loss | Behavior |
|-----------|------|----------|
| Step 1-20 | 0.5-1.0 | Initial high loss |
| Step 20-50 | Decreasing | Loss starts dropping |
| Step 100 | 0.3-0.5 | Stable training |
| Step 1000 | 0.1-0.3 | Clear structure in samples |
| Epoch 100+ | <0.1 | Good quality samples |

## Signs of Success ✅
- Loss consistently decreasing
- No NaN/Inf warnings
- Samples improving quality
- Gradient norms stable (1-10)

## Signs of Problems ❌
- Loss stuck or increasing → Reduce LR
- Out of memory → Reduce batch size
- NaN loss → Check data quality
- Black outputs → Check backbone is unfrozen

## Quick Fixes

### Out of Memory
```python
BATCH_SIZE = 1
TINYFUSION_FRAME_CHUNK_SIZE = 2
NUM_FRAMES = 8
IMAGE_SIZE = 32
```

### Loss Not Decreasing
```python
LEARNING_RATE = 0.00001  # Reduce by 10x
GRADIENT_CLIP = 0.5      # Tighter clipping
```

### Want Better Quality
```python
NUM_FRAMES = 28
IMAGE_SIZE = 128
NUM_EPOCHS = 2000
```

## TensorBoard Categories

Navigate to these tabs:
1. **01_Training** - Main loss curves
2. **12_Generated_Samples** - Video outputs
3. **03_Epoch_Summary** - Progress overview
4. **14_System** - Memory usage

## Checkpoints

Located in: `checkpoints/tinyfusion_test_3/`
- `latest_checkpoint.pt` - Most recent
- `checkpoint_epoch_X.pt` - Every 10 epochs

## Generated Samples

Located in: `generated_samples/tinyfusion_test_3/`
- GIF files generated every 5 epochs
- Named: `epoch_X_sample_Y.gif`

## Commands Reference

```bash
# Configuration
python main.py config

# List checkpoints
python main.py checkpoints

# Fix config mismatch
python main.py fix-config --checkpoint path/to/checkpoint.pt

# Verify setup
python verify_training_ready.py
python pre_training_checklist.py
```

## Training Time Estimates

| Hardware | Epoch Time | 100 Epochs | 1000 Epochs |
|----------|------------|------------|-------------|
| RTX 3090 | ~15 min | ~25 hours | ~10 days |
| RTX 4090 | ~10 min | ~17 hours | ~7 days |
| A100 | ~8 min | ~13 hours | ~5.5 days |
| Apple M4 | ~30 min | ~50 hours | ~21 days |

*With 4082 samples, batch_size=1*

## Recommended Training Duration

- **Minimum**: 100 epochs (~1 day)
- **Good**: 500 epochs (~3-5 days)
- **Best**: 1000+ epochs (~7-10 days)

## When to Stop Training

Stop when:
1. Loss plateaus for 100+ epochs
2. Sample quality stops improving
3. Validation loss starts increasing (overfitting)

Continue if:
1. Loss still decreasing
2. Samples improving
3. No signs of overfitting

---

**Need Help?** Check:
1. `README.md` - Full documentation
2. `TRAINING_FIX_SUMMARY.md` - Common issues
3. `CLEANUP_SUMMARY.md` - Workspace status
