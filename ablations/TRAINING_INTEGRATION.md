# Ablation Study Training Integration Guide

## Overview

The ablation study has been integrated with the Text2Sign text2sign training infrastructure. The `run_ablation.py` script now:

1. **Loads ablation configurations** from `configs/config_*.py`
2. **Applies settings** to the text2sign model and trainer
3. **Integrates metrics logging** throughout training
4. **Captures all results** in structured format (CSV, JSON, TensorBoard)

## Architecture

```
run_ablation.py (main entry point)
  ├─ Loads ablation config (baseline, text_finetuned, no_ema)
  ├─ Applies config to text2sign Config object
  ├─ Creates trainer with configuration
  ├─ Wraps trainer with TrainerWithMetrics for logging
  ├─ Runs training loop
  ├─ Captures metrics throughout
  └─ Saves results and reports
```

## Key Changes to Config

### Model Architecture
- `image_size`: Video frame resolution (64x64)
- `num_frames`: Frames per video (16)
- `freeze_text_encoder`: Controls text encoder training (ABLATION VARIABLE)

### Training
- `num_epochs`: 150 (full) or 2 (test)
- `batch_size`: 2 (with 8x gradient accumulation = effective batch 16)
- `learning_rate`: 5e-5
- `use_ema`: Enable EMA weights (ABLATION VARIABLE)

## Configuration Variants

### 1. Baseline (config_baseline.py)
```
freeze_text_encoder = True  ✓
use_ema = True              ✓
```
**Purpose**: Control group with frozen text encoder and EMA enabled
**Expected**: Best quality baseline

### 2. Text Finetuned (config_text_finetuned.py)
```
freeze_text_encoder = False  ← ABLATION
use_ema = True              ✓
```
**Purpose**: Test impact of finetuning text encoder
**Expected**: +2-8% quality improvement, 20-30% slower training

### 3. No EMA (config_no_ema.py)
```
freeze_text_encoder = True   ✓
use_ema = False             ← ABLATION
```
**Purpose**: Test impact of EMA for training stability
**Expected**: -2-5% quality, same training speed

## Running the Ablation Study

### Step 1: Verify Setup
```bash
cd text_to_sign/ablations/scripts
python test_ablation_setup.py
```
Expected: All 4 tests pass ✓

### Step 2: Quick Test (2 epochs)
```bash
python run_ablation.py --config baseline --save-dir ../results --epochs 2 --scale small
```

### Step 3: Full Training
```bash
# Run all three in parallel (requires 3 GPUs) or sequentially:

python run_ablation.py --config baseline --save-dir ../results --scale full
python run_ablation.py --config text_finetuned --save-dir ../results --scale full
python run_ablation.py --config no_ema --save-dir ../results --scale full
```

### Step 4: Monitor Progress
```bash
# In separate terminal
cd text_to_sign/ablations
tensorboard --logdir results/tensorboard --port 6006
```

### Step 5: Analyze Results
```bash
python analyze_results.py --results-dir ../results --output-format both
```

## Integration Points

### 1. Config Application (run_ablation.py)
```python
def _apply_config_to_text2sign(self, Config):
    """Apply ablation configuration to text2sign Config object."""
    # Model settings
    Config.IMAGE_SIZE = self.config['model'].image_size
    Config.TEXT_FREEZE_BACKBONE = self.config['model'].freeze_text_encoder
    
    # Training settings
    Config.USE_EMA = self.config['training'].use_ema
    # ... and many more settings
```

### 2. Metrics Logging (trainer_integration.py)
```python
class TrainerWithMetrics:
    """Wraps trainer to add metrics logging."""
    
    def train(self):
        """Run training with integrated metrics logging."""
        # Call base trainer
        self.base_trainer.train()
        
        # Log metrics automatically
        self.metrics_logger.log_training_step(...)
        self.tb_writer.add_scalar(...)
```

### 3. Results Saving (run_ablation.py)
```python
def save_results(self):
    """Save all results and logs."""
    # Metrics CSV and JSON
    self.logger.save_all()
    
    # TensorBoard logs
    self.tb_writer.close()
    
    # Metadata and summaries
    # ...
```

## Output Structure

```
results/
├── baseline_checkpoints/         # Model checkpoints
├── baseline_config.json          # Configuration used
├── baseline_metadata.json        # Experiment metadata
├── baseline_summary.json         # Training summary
├── logs/
│   ├── training_metrics.csv      # Step-by-step training losses
│   ├── training_metrics.json     # JSON format
│   ├── evaluation_metrics.csv    # FVD, LPIPS, etc.
│   ├── evaluation_metrics.json
│   └── gpu_memory.csv            # GPU memory tracking
├── tensorboard/
│   ├── baseline/
│   │   └── events.out.tfevents.* # TensorBoard event files
│   ├── text_finetuned/
│   └── no_ema/
└── comparison_table.*            # Generated after all runs
```

## Text2Sign Integration Details

The ablation runner integrates with text2sign by:

1. **Importing text2sign modules**
   ```python
   from config import Config
   from training_loop import setup_training, TrainerWithMetrics
   from dataset import get_dataloader
   ```

2. **Applying ablation config to text2sign Config**
   ```python
   Config.TEXT_FREEZE_BACKBONE = ablation_config.freeze_text_encoder
   Config.USE_EMA = ablation_config.use_ema
   # ... etc
   ```

3. **Creating trainer**
   ```python
   trainer = setup_training(Config)
   trainer = TrainerWithMetrics(trainer, metrics_logger, tb_writer)
   ```

4. **Running training**
   ```python
   trainer.train()
   ```

## Handling Import Errors

If text2sign modules can't be imported, the ablation runner will:

1. Print a warning
2. Fall back to demonstration mode
3. Generate dummy metrics for testing logging infrastructure
4. Allow you to test configuration loading and metrics logging

This enables testing the ablation setup even if the full text2sign code isn't available.

## GPU Memory Tracking

The metrics logger automatically tracks GPU memory:

```python
# Logged automatically during training
gpu_memory.csv
├── timestamp
├── epoch
├── step
├── gpu_memory_allocated_mb  # Current GPU memory
├── gpu_memory_reserved_mb   # Reserved GPU memory
├── gpu_memory_max_mb        # Peak GPU memory
└── gpu_memory_avg_mb        # Average GPU memory
```

## TensorBoard Visualization

View real-time metrics:

```bash
tensorboard --logdir results/tensorboard
```

Available metrics:
- `training/loss` - Training loss per step
- `training/learning_rate` - Learning rate schedule
- `evaluation/fvd` - Fréchet Video Distance
- `evaluation/lpips` - LPIPS score
- `evaluation/temporal_consistency` - Temporal stability
- `evaluation/inference_time_ms` - Inference speed
- `evaluation/inference_memory_gb` - Inference memory

## Comparison Analysis

After all ablations complete, generate comparison tables:

```bash
python analyze_results.py --results-dir ../results --output-format both
```

Generates:
- `comparison_table.csv` - Spreadsheet compatible
- `comparison_table.md` - Markdown formatted
- `ABLATION_RESULTS_REPORT.txt` - Human-readable report

Shows:
- Training loss progression
- Final metrics comparison
- Statistical analysis
- Key findings

## Troubleshooting

### Issue: "Could not import metrics_logger"
**Solution**: Ensure you're running from `text_to_sign/ablations/scripts/` directory

### Issue: "Could not import text2sign training modules"
**Solution**: Falls back to demo mode automatically. Full integration requires text2sign in path.

### Issue: Out of memory
**Solution**: Reduce batch size in config (currently 2 with 8x accumulation)

### Issue: GPU not detected
**Solution**: Check `torch.cuda.is_available()` and CUDA installation

## Advanced: Custom Metrics

To add custom metrics logging:

```python
# In your trainer training loop:
from ablations.scripts.metrics_logger import MetricsLogger

logger = MetricsLogger(...)
logger.log_training_step(
    epoch=epoch,
    step=step,
    loss=loss,
    learning_rate=lr,
    elapsed_time=elapsed
)
```

## References

- Main runner: `text_to_sign/ablations/scripts/run_ablation.py`
- Trainer integration: `text_to_sign/ablations/scripts/trainer_integration.py`
- Metrics logging: `text_to_sign/ablations/scripts/metrics_logger.py`
- Analysis tool: `text_to_sign/ablations/scripts/analyze_results.py`
- Test suite: `text_to_sign/ablations/scripts/test_ablation_setup.py`
