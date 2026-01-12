#!/bin/bash
# Quick reference for research logging

cat << 'EOF'

╔═══════════════════════════════════════════════════════════════════╗
║                  RESEARCH LOGGING QUICK REFERENCE                 ║
╚═══════════════════════════════════════════════════════════════════╝

📊 TRAINING (Automatic Logging)
────────────────────────────────────────────────────────────────────
  ./start_training.sh
  → Logs to: logs/text2sign_YYYYMMDD_HHMMSS/

📈 MONITORING (Real-Time)
────────────────────────────────────────────────────────────────────
  tensorboard --logdir text_to_sign/logs
  → Open: http://localhost:6006

📉 ANALYSIS (After Training)
────────────────────────────────────────────────────────────────────
  python analyze_results.py \
      --log_dir logs/text2sign_20260112_123456 \
      --output_dir paper_results

📁 OUTPUT FILES
────────────────────────────────────────────────────────────────────
  CSV Files (Data Analysis):
    logs/*/csv/*_steps.csv     → Every training step
    logs/*/csv/*_epochs.csv    → Every epoch summary
  
  JSON Files (Metadata):
    logs/*/json/*_config.json   → Full configuration
    logs/*/json/*_summary.json  → Training summary
  
  Publication Materials:
    paper_results/training_curves.png     → 4-panel plot (300 DPI)
    paper_results/loss_distribution.png   → Loss analysis
    paper_results/statistics_table.tex    → LaTeX table
    paper_results/statistics_table.md     → Markdown table

📊 LOGGED METRICS
────────────────────────────────────────────────────────────────────
  Step-Level:
    • loss, lr, grad_norm_total, grad_norm_avg, grad_norm_max
  
  Epoch-Level:
    • train_loss (mean/std/min/max)
    • val_loss, learning_rate
    • grad_norm_avg/max
    • timing, samples_processed

📝 FOR YOUR PAPER
────────────────────────────────────────────────────────────────────
  1. Figures:   \includegraphics{paper_results/training_curves.png}
  2. Tables:    \input{paper_results/statistics_table.tex}
  3. Report:    Best loss, training time, gradient norms
  4. Data:      Use CSV files for additional analysis

🔬 PYTHON ANALYSIS
────────────────────────────────────────────────────────────────────
  import pandas as pd
  
  steps = pd.read_csv('logs/.../csv/*_steps.csv')
  epochs = pd.read_csv('logs/.../csv/*_epochs.csv')
  
  print(epochs['val_loss'].min())  # Best validation loss
  print(epochs['val_loss'].idxmin())  # Best epoch

📚 DOCUMENTATION
────────────────────────────────────────────────────────────────────
  RESEARCH_LOGGING_GUIDE.md     → Complete guide
  RESEARCH_LOGGING_SUMMARY.md   → Feature summary
  test_logging.py                → Test the system

✨ Everything logs automatically - just train normally!

EOF
