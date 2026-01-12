# Ablation Study for Text2Sign Model

This folder contains the complete implementation of the ablation study for validating architectural choices in the Text2Sign diffusion model.

## Quick Start

### 1. Test Setup (Recommended First)
Verify all configurations and logging infrastructure work:

```bash
cd scripts
python test_ablation_setup.py
```

Expected output: All tests should pass ✓

### 2. Run Individual Ablations
Each ablation can be run independently:

```bash
# Baseline (Frozen text encoder, EMA enabled)
python run_ablation.py --config baseline --save-dir ../results --scale full

# Text encoder finetuned
python run_ablation.py --config text_finetuned --save-dir ../results --scale full

# EMA disabled
python run_ablation.py --config no_ema --save-dir ../results --scale full
```

### 3. Analyze Results
After experiments complete:

```bash
python analyze_results.py --results-dir ../results --output-format both
```

This generates:
- `comparison_table.csv` - Numerical comparison
- `comparison_table.md` - Markdown formatted table
- `ABLATION_RESULTS_REPORT.txt` - Detailed text report

## Directory Structure

```
ablations/
├── configs/                 # Configuration variants
│   ├── config_baseline.py              # Baseline: frozen text, EMA enabled
│   ├── config_text_finetuned.py        # Ablation: unfrozen text encoder
│   └── config_no_ema.py                # Ablation: EMA disabled
│
├── scripts/                 # Python scripts
│   ├── metrics_logger.py               # Metrics collection & logging
│   ├── run_ablation.py                 # Main experiment runner
│   ├── analyze_results.py              # Results analysis
│   └── test_ablation_setup.py          # Setup validation
│
├── results/                 # Results directory (created after runs)
│   ├── logs/                           # Training logs (CSV, JSON)
│   ├── tensorboard/                    # TensorBoard event files
│   ├── baseline_checkpoints/           # Model checkpoints
│   ├── text_finetuned_checkpoints/
│   ├── no_ema_checkpoints/
│   └── comparison_table.csv            # Results summary
│
└── README.md                # This file

```

## Configuration Overview

### Baseline Configuration
```
Freeze text encoder: True
Use EMA: True
Model channels: 96
Learning rate: 5e-5
Batch size: 2
```

### Ablation 1: Text Encoder Finetuned
**Change**: `freeze_text_encoder = False`

**Expected Impact**: +2-8% quality improvement, 20-30% slower training

**Monitors**: Overfitting risk on small sign language dataset

### Ablation 2: EMA Disabled
**Change**: `use_ema = False`

**Expected Impact**: -2-5% quality degradation

**Benefits**: Same training time, isolates EMA contribution

## Data Logging

### Training Metrics (Per Step)
Logged to `logs/{ablation}_training/training_metrics.csv`:
- Epoch, Step
- Loss
- Learning rate
- Training time (seconds)
- Peak GPU memory (GB)
- Average GPU memory (GB)
- Timestamp

### GPU Memory Tracking
Logged to `logs/{ablation}_training/gpu_memory.csv`:
- Epoch, Step
- GPU memory usage (GB)
- Timestamp

### Training History (Full)
Logged to `logs/{ablation}_training/training_metrics.json`:
- Complete training history in JSON format for easy loading into pandas/analysis tools

### Evaluation Metrics
Logged to `logs/{ablation}_evaluation/evaluation_metrics.csv|json`:
- Model name, config name
- FVD (Fréchet Video Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Temporal consistency
- Inference time (ms)
- Inference memory (GB)
- Parameter count (millions)
- Timestamp

### Experiment Metadata
Saved to `results/{ablation}_metadata.json`:
- Configuration settings
- Model parameters
- Training parameters
- Start/end time
- Total training time

### Summary Reports
- `{ablation}_summary.json` - Structured summary for programmatic access
- `ABLATION_RESULTS_REPORT.txt` - Human-readable text report
- `comparison_table.csv` - Side-by-side comparison
- `comparison_table.md` - Markdown formatted comparison

## TensorBoard Monitoring

Monitor training in real-time with TensorBoard:

```bash
# Start TensorBoard (from text_to_sign/ablations directory)
tensorboard --logdir results/tensorboard

# Open browser to http://localhost:6006
```

TensorBoard logs include:
- Training loss per step
- Learning rate over time
- Evaluation metrics (FVD, LPIPS, etc.)
- Inference performance metrics

## Small-Scale Testing

For quick validation before full training:

```bash
python run_ablation.py --config baseline --save-dir ../results --epochs 2 --scale small
```

This runs only 2 epochs with reduced batch size to:
- Verify configuration loads correctly
- Test data loading pipeline
- Validate metrics logging
- Ensure no import errors

Results are saved the same way as full experiments.

## Integration with Training Code

The `AblationRunner` class in `run_ablation.py` provides hooks for integrating with your actual training loop:

```python
from run_ablation import AblationRunner

# Initialize runner
runner = AblationRunner(
    config_name="baseline",
    save_dir="results",
    experiment_scale="full"
)

# In your training loop, log metrics:
for epoch in range(num_epochs):
    for step, (images, texts) in enumerate(dataloader):
        # ... training code ...
        
        # Log training step
        runner.log_training_step(
            epoch=epoch,
            step=step,
            loss=loss_value,
            learning_rate=optimizer.param_groups[0]['lr'],
            elapsed_time=time_elapsed
        )

# After training, log evaluation
runner.log_evaluation(
    fvd=fvd_score,
    lpips=lpips_score,
    temporal_consistency=temporal_consistency,
    inference_time_ms=inference_time,
    inference_memory_gb=inference_memory,
    num_parameters_millions=param_count
)

# Save results
runner.save_results()
```

## Code Comments and Documentation

All code includes detailed comments:
- **Function docstrings**: Purpose, arguments, returns
- **Class docstrings**: Overall design and responsibility
- **Inline comments**: Complex logic and key decisions
- **Config comments**: Explanation of each configuration parameter

## Metrics Explained

### Quality Metrics

**FVD (Fréchet Video Distance)**
- Measures perceptual quality of generated videos
- Lower is better
- Range: 0-100 (0 = identical to real videos)
- Computed using InceptionV3 features

**LPIPS (Learned Perceptual Image Patch Similarity)**
- Learned perceptual metric for frame quality
- Lower is better
- Range: 0-1 (0 = identical, 1 = very different)

**Temporal Consistency**
- Measures smoothness between frames
- Higher is better
- Range: 0-1 (1 = perfectly consistent)

### Efficiency Metrics

**Training Time**
- Total wall-clock training time in hours
- Includes data loading, forward pass, backward pass, optimization

**Peak Memory**
- Maximum GPU memory used during training
- Useful for understanding memory constraints

**Inference Time**
- Time to generate one video sample
- In milliseconds
- Single video, single GPU

**Parameters**
- Total trainable parameters in millions
- Indicator of model complexity

## Expected Results Summary

Based on ablation plan:

| Ablation | Quality Δ | Train Time Δ | Memory Δ | Key Finding |
|----------|-----------|--------------|----------|-------------|
| Baseline | 0% | 0% | 0% | Reference |
| Text Finetuned | +0.4% | +22% | +2% | Frozen sufficient |
| No EMA | -0.6% | 0% | 0% | EMA helps quality |

## Troubleshooting

### Config Loading Errors
```
FileNotFoundError: Config not found
```
Solution: Verify config files exist in `configs/` directory

### Import Errors
```
ModuleNotFoundError: No module named 'metrics_logger'
```
Solution: Ensure you're running from `scripts/` directory or add to PYTHONPATH

### GPU Memory Issues
```
RuntimeError: CUDA out of memory
```
Solution: Reduce batch size in config or use gradient checkpointing

### TensorBoard Not Found
```
ModuleNotFoundError: No module named 'torch.utils.tensorboard'
```
Solution: Install PyTorch with tensorboard support

## Future Extensions

- Implement full 3D attention for Ablation 3
- Add UNet3D baseline (Ablation 1.1)
- Implement statistical significance testing
- Add visualization of generated samples
- Implement user study interface for quality assessment
- Add per-layer learning rate scheduling for text encoder

## References

See `ABLATION_STUDY_PLAN.md` in parent directory for:
- Detailed ablation methodology
- Theoretical justification for each ablation
- Computational complexity analysis
- Additional ablation options for future work

## Contact

For questions about this ablation study, refer to the main planning document:
`/teamspace/studios/this_studio/ABLATION_STUDY_PLAN.md`
