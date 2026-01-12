#!/usr/bin/env python3
"""
Analysis script for research paper
Generates publication-ready plots and statistics from training logs
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import argparse

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_experiment_data(log_dir: str):
    """Load all experiment data"""
    log_path = Path(log_dir)
    
    # Find CSV files
    csv_dir = log_path / "csv"
    steps_csv = list(csv_dir.glob("*_steps.csv"))
    epochs_csv = list(csv_dir.glob("*_epochs.csv"))
    
    if not steps_csv or not epochs_csv:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    
    # Load data
    steps_df = pd.read_csv(steps_csv[0])
    epochs_df = pd.read_csv(epochs_csv[0])
    
    # Load summary
    json_dir = log_path / "json"
    summary_json = list(json_dir.glob("*_summary.json"))
    summary = None
    if summary_json:
        with open(summary_json[0], 'r') as f:
            summary = json.load(f)
    
    return steps_df, epochs_df, summary


def plot_training_curves(epochs_df, output_dir):
    """Plot training and validation loss curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs_df['epoch'], epochs_df['train_loss_mean'], 'b-', label='Train Loss', linewidth=2)
    ax.fill_between(
        epochs_df['epoch'],
        epochs_df['train_loss_mean'] - epochs_df['train_loss_std'],
        epochs_df['train_loss_mean'] + epochs_df['train_loss_std'],
        alpha=0.3, color='blue', label='±1 std'
    )
    ax.plot(epochs_df['epoch'], epochs_df['val_loss'], 'r--', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[0, 1]
    ax.plot(epochs_df['epoch'], epochs_df['learning_rate'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Gradient norms
    ax = axes[1, 0]
    ax.plot(epochs_df['epoch'], epochs_df['grad_norm_avg'], 'purple', label='Avg', linewidth=2)
    ax.plot(epochs_df['epoch'], epochs_df['grad_norm_max'], 'orange', label='Max', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norms', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Loss improvement
    ax = axes[1, 1]
    initial_loss = epochs_df['train_loss_mean'].iloc[0]
    improvement = ((initial_loss - epochs_df['train_loss_mean']) / initial_loss) * 100
    ax.plot(epochs_df['epoch'], improvement, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Training Loss Improvement', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training curves to {output_path}")
    
    return fig


def generate_statistics_table(epochs_df, summary, output_dir):
    """Generate LaTeX table with training statistics"""
    
    # Calculate statistics
    stats = {
        'Total Epochs': int(epochs_df['epoch'].max()),
        'Best Train Loss': f"{epochs_df['train_loss_mean'].min():.6f}",
        'Best Val Loss': f"{epochs_df['val_loss'].min():.6f}",
        'Final Train Loss': f"{epochs_df['train_loss_mean'].iloc[-1]:.6f}",
        'Final Val Loss': f"{epochs_df['val_loss'].iloc[-1]:.6f}",
        'Avg Gradient Norm': f"{epochs_df['grad_norm_avg'].mean():.4f}",
        'Training Time (hours)': f"{summary['total_duration_hours']:.2f}" if summary else "N/A",
        'Total Steps': summary['total_steps'] if summary else epochs_df['num_batches'].sum(),
    }
    
    # Create LaTeX table
    latex_table = "\\begin{table}[h]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Training Statistics}\n"
    latex_table += "\\begin{tabular}{lr}\n"
    latex_table += "\\toprule\n"
    latex_table += "Metric & Value \\\\\n"
    latex_table += "\\midrule\n"
    
    for key, value in stats.items():
        latex_table += f"{key} & {value} \\\\\n"
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    
    # Save LaTeX table
    output_path = Path(output_dir) / 'statistics_table.tex'
    with open(output_path, 'w') as f:
        f.write(latex_table)
    print(f"✅ Saved LaTeX table to {output_path}")
    
    # Also save as markdown for README
    md_table = "| Metric | Value |\n"
    md_table += "|--------|-------|\n"
    for key, value in stats.items():
        md_table += f"| {key} | {value} |\n"
    
    md_path = Path(output_dir) / 'statistics_table.md'
    with open(md_path, 'w') as f:
        f.write(md_table)
    print(f"✅ Saved Markdown table to {md_path}")
    
    return stats, latex_table


def plot_loss_distribution(steps_df, output_dir):
    """Plot loss distribution across training"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(steps_df['loss'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Loss Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Loss Distribution', fontsize=14, fontweight='bold')
    ax.axvline(steps_df['loss'].mean(), color='r', linestyle='--', 
               linewidth=2, label=f'Mean: {steps_df["loss"].mean():.4f}')
    ax.axvline(steps_df['loss'].median(), color='g', linestyle='--', 
               linewidth=2, label=f'Median: {steps_df["loss"].median():.4f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Box plot over time (split into 10 segments)
    ax = axes[1]
    steps_df['segment'] = pd.cut(steps_df['step'], bins=10, labels=False)
    steps_df.boxplot(column='loss', by='segment', ax=ax)
    ax.set_xlabel('Training Progress (Decile)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Evolution', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove default title
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'loss_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved loss distribution to {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Analyze training logs for research paper')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to log directory')
    parser.add_argument('--output_dir', type=str, default='paper_results', help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("📊 Research Paper Analysis")
    print("="*70)
    print(f"Log directory: {args.log_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    print("Loading experiment data...")
    steps_df, epochs_df, summary = load_experiment_data(args.log_dir)
    print(f"✅ Loaded {len(steps_df)} steps and {len(epochs_df)} epochs")
    print()
    
    # Generate plots
    print("Generating plots...")
    plot_training_curves(epochs_df, output_dir)
    plot_loss_distribution(steps_df, output_dir)
    print()
    
    # Generate statistics
    print("Generating statistics tables...")
    stats, latex_table = generate_statistics_table(epochs_df, summary, output_dir)
    print()
    
    # Print summary
    print("="*70)
    print("📈 Training Summary")
    print("="*70)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*70)
    print()
    
    print("✅ Analysis complete! Results saved to:", output_dir)
    print()
    print("Files generated:")
    print("  - training_curves.png (publication-quality plots)")
    print("  - loss_distribution.png (loss analysis)")
    print("  - statistics_table.tex (LaTeX table)")
    print("  - statistics_table.md (Markdown table)")


if __name__ == "__main__":
    main()
