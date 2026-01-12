# Research Paper Logging Guide

**Date**: January 12, 2026  
**Purpose**: Comprehensive metrics tracking for academic publication

---

## 📊 Overview

The training system now includes **research-grade logging** with:
- ✅ CSV exports for easy analysis in Excel/Python/R
- ✅ JSON metadata with full experiment configuration
- ✅ TensorBoard visualizations
- ✅ Automatic statistics (mean, std, min, max)
- ✅ Step-level and epoch-level tracking
- ✅ Publication-ready plots and LaTeX tables

---

## 🗂️ Output Structure

After training, you'll have:

```
text_to_sign/
├── logs/
│   └── text2sign_YYYYMMDD_HHMMSS/
│       ├── events.out.tfevents.*        # TensorBoard logs
│       ├── csv/
│       │   ├── text2sign_*_steps.csv    # Every training step
│       │   └── text2sign_*_epochs.csv   # Every epoch summary
│       └── json/
│           ├── text2sign_*_config.json  # Experiment configuration
│           └── text2sign_*_summary.json # Final training summary
├── experiments/
│   └── experiment_registry.json         # All experiments catalog
└── checkpoints/
    └── text2sign_YYYYMMDD_HHMMSS/
        ├── checkpoint_epoch_*.pt
        ├── best_model.pt
        └── final_model.pt
```

---

## 📈 Logged Metrics

### Step-Level (Every Training Step)
Saved to: `logs/*/csv/*_steps.csv`

| Column | Description |
|--------|-------------|
| `step` | Global training step number |
| `phase` | "train", "val", or "test" |
| `timestamp` | Seconds since training start |
| `loss` | Training loss value |
| `lr` | Current learning rate |
| `grad_norm_total` | Total gradient norm (L2) |
| `grad_norm_avg` | Average gradient norm per parameter |
| `grad_norm_max` | Maximum gradient norm |

### Epoch-Level (Every Epoch)
Saved to: `logs/*/csv/*_epochs.csv`

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number |
| `duration_seconds` | Epoch duration |
| `duration_minutes` | Epoch duration in minutes |
| `total_runtime_hours` | Cumulative training time |
| `train_loss_mean` | Mean training loss |
| `train_loss_std` | Standard deviation of training loss |
| `train_loss_min` | Minimum training loss |
| `train_loss_max` | Maximum training loss |
| `val_loss` | Validation loss |
| `learning_rate` | Current learning rate |
| `grad_norm_avg` | Average gradient norm |
| `grad_norm_max` | Maximum gradient norm |
| `num_batches` | Number of batches processed |
| `samples_processed` | Total samples in epoch |
| `ema_step_counter` | EMA update counter (if enabled) |

---

## 📊 Analysis Tools

### 1. Python Analysis Script

Generate publication-ready plots and statistics:

```bash
python analyze_results.py --log_dir logs/text2sign_YYYYMMDD_HHMMSS --output_dir paper_results
```

**Outputs:**
- `training_curves.png` - 4-panel training visualization
  - Training/validation loss with confidence intervals
  - Learning rate schedule
  - Gradient norms over time
  - Loss improvement percentage
- `loss_distribution.png` - Loss distribution analysis
  - Histogram of loss values
  - Box plots showing loss evolution
- `statistics_table.tex` - LaTeX table for paper
- `statistics_table.md` - Markdown table for README

### 2. Pandas Analysis (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
steps_df = pd.read_csv('logs/.../csv/*_steps.csv')
epochs_df = pd.read_csv('logs/.../csv/*_epochs.csv')

# Quick analysis
print(steps_df.describe())
print(f"Best validation loss: {epochs_df['val_loss'].min():.6f}")
print(f"Best epoch: {epochs_df['val_loss'].idxmin() + 1}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_df['epoch'], epochs_df['train_loss_mean'], label='Train')
plt.plot(epochs_df['epoch'], epochs_df['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curves.png', dpi=300)
```

### 3. R Analysis

```r
# Load data
library(readr)
library(ggplot2)

epochs <- read_csv('logs/.../csv/*_epochs.csv')

# Summary statistics
summary(epochs)

# Plot with confidence intervals
ggplot(epochs, aes(x=epoch)) +
  geom_ribbon(aes(ymin=train_loss_mean - train_loss_std,
                  ymax=train_loss_mean + train_loss_std),
              alpha=0.3, fill='blue') +
  geom_line(aes(y=train_loss_mean, color='Train'), size=1) +
  geom_line(aes(y=val_loss, color='Validation'), size=1) +
  labs(x='Epoch', y='Loss', title='Training Progress') +
  theme_minimal()

ggsave('loss_curves.pdf', width=8, height=6, dpi=300)
```

### 4. Excel/Spreadsheet

Simply open the CSV files in Excel, Google Sheets, or LibreOffice Calc:
- `*_steps.csv` for detailed step analysis
- `*_epochs.csv` for epoch summaries

---

## 📝 For Your Paper

### Training Setup Section

```latex
\subsection{Training Configuration}

The model was trained for \textit{N} epochs using the following configuration:

\begin{itemize}
    \item Optimizer: AdamW with learning rate $5 \times 10^{-5}$
    \item Learning rate schedule: Linear warmup (2000 steps) followed by cosine annealing
    \item Batch size: 2 (effective batch size: 16 with gradient accumulation)
    \item Diffusion scheduler: DDIM with cosine beta schedule
    \item EMA: Enabled with decay rate 0.9999
    \item Training duration: \textit{X} hours on \textit{GPU model}
\end{itemize}

All experiments were tracked with comprehensive logging, including per-step gradients,
per-epoch statistics (mean, standard deviation, min, max), and validation metrics.
```

### Results Section

Use the generated LaTeX table:

```latex
\subsection{Training Results}

Table~\ref{tab:training_stats} shows the training statistics. The model achieved
a final validation loss of \textit{X.XXXX} after \textit{N} epochs.

\input{paper_results/statistics_table.tex}
\label{tab:training_stats}
```

### Figures

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{paper_results/training_curves.png}
    \caption{Training progress showing (a) training and validation loss with 
             standard deviation bands, (b) learning rate schedule, (c) gradient 
             norms, and (d) cumulative loss improvement.}
    \label{fig:training_curves}
\end{figure}
```

---

## 🔍 Monitoring During Training

### 1. TensorBoard (Real-time)

```bash
# Start TensorBoard
tensorboard --logdir text_to_sign/logs

# Open browser to http://localhost:6006
```

**Available visualizations:**
- `train/loss` - Training loss per step
- `train/lr` - Learning rate per step
- `train/grad_norm_*` - Gradient statistics
- `epoch/train_loss` - Training loss per epoch
- `epoch/val_loss` - Validation loss per epoch
- `samples/generated` - Generated samples

### 2. Live CSV Monitoring

The CSVs are updated in real-time. You can monitor progress with:

```bash
# Watch latest epoch metrics
watch -n 5 'tail -n 1 logs/*/csv/*_epochs.csv | column -t -s,'

# Count total steps
wc -l logs/*/csv/*_steps.csv
```

### 3. Terminal Output

The training prints comprehensive summaries:

```
======================================================================
📈 Epoch 5 Summary:
======================================================================
  Duration: 12.34 minutes
  Total Runtime: 1.23 hours
  train_loss_mean: 0.123456
  val_loss: 0.098765
  grad_norm_avg: 1.234
======================================================================
```

---

## 📊 Comparison Between Runs

To compare multiple experiments:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load multiple experiments
exp1 = pd.read_csv('logs/exp1/csv/*_epochs.csv')
exp2 = pd.read_csv('logs/exp2/csv/*_epochs.csv')

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(exp1['epoch'], exp1['val_loss'], label='Baseline')
plt.plot(exp2['epoch'], exp2['val_loss'], label='With EMA')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Model Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('comparison.png', dpi=300)
plt.show()

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_ind(exp1['val_loss'], exp2['val_loss'])
print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
```

---

## 🎯 Key Metrics for Your Paper

### Report These Metrics:

1. **Best Validation Loss**: `min(epochs_df['val_loss'])`
2. **Final Training Loss**: `epochs_df['train_loss_mean'].iloc[-1]`
3. **Training Stability**: `mean(epochs_df['train_loss_std'])`
4. **Convergence Speed**: Epoch where loss plateaus
5. **Gradient Health**: `mean(epochs_df['grad_norm_avg'])`
6. **Total Training Time**: From `summary.json`
7. **Loss Improvement**: `(initial_loss - final_loss) / initial_loss * 100%`

### Example Results Paragraph:

```
The model was trained for 150 epochs, achieving a final validation loss of 
0.0234 (±0.0012 std). Training converged after approximately 80 epochs, with 
the best validation loss of 0.0198 observed at epoch 125. The total training 
time was 18.5 hours on a single NVIDIA A100 GPU. Gradient norms remained 
stable throughout training (mean: 1.23 ± 0.45), indicating healthy 
optimization. Compared to the baseline, our method achieved a 23% reduction 
in validation loss (p < 0.001, paired t-test).
```

---

## 🚀 Quick Start

### During Training
Training automatically logs everything. Just run:

```bash
./start_training.sh
```

Logs are saved to: `text_to_sign/logs/text2sign_YYYYMMDD_HHMMSS/`

### After Training
Generate publication materials:

```bash
# 1. Analyze results
python analyze_results.py \
    --log_dir logs/text2sign_YYYYMMDD_HHMMSS \
    --output_dir paper_results

# 2. Check outputs
ls paper_results/
# training_curves.png
# loss_distribution.png
# statistics_table.tex
# statistics_table.md

# 3. Use in your paper
# - Copy .png files to your paper's figures/
# - \input{} the .tex table
# - Reference the metrics
```

---

## 📋 Checklist for Paper

- [ ] Report all hyperparameters (from `*_config.json`)
- [ ] Include training curves figure (`training_curves.png`)
- [ ] Report final metrics (from `*_summary.json`)
- [ ] Compare with baseline (if applicable)
- [ ] Statistical significance tests (t-test, ANOVA)
- [ ] Report training time and computational resources
- [ ] Include loss statistics (mean ± std)
- [ ] Report gradient norms (for stability claim)
- [ ] Mention EMA usage and impact
- [ ] Cite TensorBoard for visualization

---

## 💡 Tips

1. **Keep Everything**: Don't delete log directories - they're small and valuable
2. **Name Experiments**: Use meaningful run names in your code
3. **Compare Fairly**: Use same random seeds when comparing methods
4. **Multiple Runs**: Train 3-5 times for statistical significance
5. **Document Changes**: Track config changes in experiment descriptions

---

## 🔧 Customization

To log additional metrics, modify `trainer.py`:

```python
# In train_step(), add to metrics dict:
metrics = {
    "loss": loss.item(),
    "lr": lr,
    "my_custom_metric": my_value,  # Add here
}

# Automatically logged to CSV and TensorBoard!
```

---

**The logging system is now production-ready for research publication!** 🎉
