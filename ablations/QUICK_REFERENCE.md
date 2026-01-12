# Ablation Study - Quick Reference Card

**Status**: ✅ COMPLETE AND READY  
**Last Updated**: January 12, 2026

## 📋 TL;DR - Get Started in 30 Seconds

```bash
cd /teamspace/studios/this_studio/text_to_sign/ablations/scripts

# Test setup (recommended)
python test_ablation_setup.py

# Run quick test (2 epochs, ~5 min)
python run_ablation.py --config baseline --epochs 2 --scale small

# Or run full training (~2 hours per variant)
python run_ablation.py --config baseline --save-dir ../results --scale full
```

## 🎯 Three Ablations

| Name | freeze_text | use_ema | Expected | Runtime |
|------|------------|---------|----------|---------|
| **Baseline** | ✅ True | ✅ True | Control group | ~2h |
| **Text Finetuned** | ❌ False | ✅ True | +2-8% quality | ~2.5h |
| **No EMA** | ✅ True | ❌ False | -2-5% quality | ~2h |

## 📂 Key Locations

```
text_to_sign/ablations/
├── configs/                    ← Configuration files
│   ├── config_baseline.py
│   ├── config_text_finetuned.py
│   └── config_no_ema.py
├── scripts/                    ← Executable scripts
│   ├── run_ablation.py        ← Main runner
│   ├── trainer_integration.py ← Metrics wrapper
│   ├── metrics_logger.py      ← Logging
│   ├── analyze_results.py     ← Analysis
│   └── test_ablation_setup.py ← Tests
├── results/                    ← Generated outputs
└── README_INTEGRATION.md       ← Full docs
```

## 🚀 Common Commands

### Test Setup
```bash
cd text_to_sign/ablations/scripts
python test_ablation_setup.py
```
✓ Verify all configs load correctly

### Quick Verification (2 epochs)
```bash
python run_ablation.py --config baseline --epochs 2 --scale small
```
✓ Test pipeline, config, and logging

### Full Baseline Training
```bash
python run_ablation.py --config baseline --save-dir ../results --scale full
```
✓ Run 150 epochs with baseline config

### All Three Ablations (Parallel)
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python run_ablation.py --config baseline --save-dir ../results

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python run_ablation.py --config text_finetuned --save-dir ../results

# Terminal 3
CUDA_VISIBLE_DEVICES=2 python run_ablation.py --config no_ema --save-dir ../results
```
✓ Run all variants in parallel (requires 3 GPUs)

### Monitor with TensorBoard
```bash
cd text_to_sign/ablations
tensorboard --logdir results/tensorboard
```
✓ Open http://localhost:6006 in browser

### Analyze Results
```bash
python analyze_results.py --results-dir ../results --output-format both
```
✓ Generate comparison tables (CSV + Markdown)

## 📊 Output Files

After each run, you get:

```
results/{variant}/
├── {variant}_config.json              ← Config used
├── {variant}_metadata.json            ← Timestamps, hardware
├── {variant}_summary.json             ← Training summary
├── logs/
│   ├── training_metrics.csv           ← Loss per step
│   ├── evaluation_metrics.csv         ← Final FVD, LPIPS
│   └── gpu_memory.csv                 ← GPU usage over time
├── tensorboard/{variant}/
│   └── events.out.tfevents.*          ← TensorBoard data
└── {variant}_checkpoints/             ← Model weights
    ├── epoch_5.pt
    ├── epoch_10.pt
    └── ...
```

## 🔧 Configuration Parameters

### Model
```python
image_size: int = 64              # Video resolution
num_frames: int = 16              # Frames per video
freeze_text_encoder: bool = True  # ← ABLATION VAR 1
```

### Training
```python
batch_size: int = 2               # Batch size
num_epochs: int = 150             # Epochs (or override with --epochs)
learning_rate: float = 5e-5       # Learning rate
use_ema: bool = True              # ← ABLATION VAR 2
ema_decay: float = 0.9999         # EMA decay rate
```

### To Change
Edit the corresponding `config_*.py` file and re-run:
```bash
python run_ablation.py --config baseline ...
```

## 🐛 Troubleshooting

### "Config not found"
```bash
# Make sure you're in scripts directory
cd text_to_sign/ablations/scripts
```

### "Could not import metrics_logger"
```bash
# Again, must be in scripts directory
python run_ablation.py ...
```

### "No module named 'text2sign'"
- Falls back to demo mode automatically
- Logging infrastructure still works
- For full integration, ensure text2sign accessible

### Out of Memory
```python
# Reduce batch size in config file:
batch_size: int = 1              # Was 2
```

### Training Too Slow
```python
# Reduce gradient accumulation:
gradient_accumulation_steps: int = 4  # Was 8
# (Effective batch size becomes 4 instead of 16)
```

## 📈 Expected Results

After 150 epochs on typical hardware:

| Metric | Baseline | Text FT | No EMA |
|--------|----------|---------|--------|
| Final Loss | ~0.003 | ~0.002* | ~0.004* |
| Training Time | ~2h | ~2.5h | ~2h |
| Quality | Reference | Better* | Worse* |
| GPU Memory | ~20GB | ~20GB | ~20GB |

*Relative to baseline

## 📝 Important Files

| File | Purpose |
|------|---------|
| [README_INTEGRATION.md](README_INTEGRATION.md) | Full integration guide |
| [TRAINING_INTEGRATION.md](TRAINING_INTEGRATION.md) | Architecture details |
| [INTEGRATION_CHECKLIST.md](INTEGRATION_CHECKLIST.md) | Implementation status |
| [run_ablation.py](scripts/run_ablation.py) | Main runner |
| [trainer_integration.py](scripts/trainer_integration.py) | Metrics wrapper |

## 🎓 What Gets Tested

```
✓ baseline         - Frozen text encoder + EMA enabled
✓ text_finetuned   - Unfrozen text encoder + EMA enabled
✓ no_ema           - Frozen text encoder + EMA disabled
```

These answer:
1. **Does unfreezing text encoder help?** → Compare baseline vs text_finetuned
2. **Is EMA important?** → Compare baseline vs no_ema
3. **What's the best configuration?** → See comparison_table

## 🚦 Workflow

```
1. Test Setup
   python test_ablation_setup.py
                    ↓
2. Quick Verification (optional)
   python run_ablation.py --config baseline --epochs 2
                    ↓
3. Run Full Ablations
   For each of baseline, text_finetuned, no_ema:
   python run_ablation.py --config <name> --save-dir ../results
                    ↓
4. Monitor Progress (optional)
   tensorboard --logdir results/tensorboard
                    ↓
5. Analyze Results
   python analyze_results.py --results-dir ../results
                    ↓
6. Review Findings
   cat results/ABLATION_RESULTS_REPORT.txt
```

## 💾 Saving Space

To save space, delete demo results:
```bash
rm -rf results/baseline_2epoch_test/
```

Or archive old runs:
```bash
tar -czf results_backup.tar.gz results/
rm -rf results/
```

## 📞 Getting Help

1. **Setup issues** → Run `python test_ablation_setup.py`
2. **Config issues** → Check [TRAINING_INTEGRATION.md](TRAINING_INTEGRATION.md)
3. **Training issues** → Check GPU memory, reduce batch size
4. **Results issues** → Check `results/{variant}/logs/` for CSV files

## ✅ Before You Start

- [ ] Read this quick reference
- [ ] Run `test_ablation_setup.py`
- [ ] Review the 3 ablations above
- [ ] Pick one to try first (baseline recommended)
- [ ] Check you have 20GB+ GPU memory
- [ ] Estimate time (2-4 hours per variant)

## 🎬 Ready?

```bash
cd text_to_sign/ablations/scripts
python test_ablation_setup.py
```

Then see [README_INTEGRATION.md](README_INTEGRATION.md) for full details!
