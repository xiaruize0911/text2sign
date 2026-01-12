#!/usr/bin/env python3
"""
Test research logging system
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.metrics_logger import MetricsLogger, ExperimentTracker
import numpy as np

print("="*70)
print("Testing Research Logging System")
print("="*70)
print()

# Test MetricsLogger
print("1. Testing MetricsLogger...")
config = {
    "model": {"channels": 96, "depth": 2},
    "training": {"lr": 5e-5, "epochs": 150}
}

logger = MetricsLogger(
    log_dir="test_logs",
    experiment_name="test_experiment",
    config=config
)

# Simulate training
print("\n2. Simulating training...")
for epoch in range(3):
    logger.start_epoch(epoch + 1)
    
    # Simulate steps
    for step in range(10):
        metrics = {
            "loss": np.random.uniform(0.5, 1.0),
            "lr": 5e-5 * (1 - step/100),
            "grad_norm": np.random.uniform(0.5, 2.0)
        }
        logger.log_step(step + epoch*10, metrics, phase="train")
    
    # End epoch
    epoch_metrics = {
        "train_loss_mean": np.random.uniform(0.5, 1.0),
        "val_loss": np.random.uniform(0.4, 0.9),
        "learning_rate": 5e-5,
        "grad_norm_avg": 1.0
    }
    logger.end_epoch(epoch + 1, epoch_metrics)

# Save summary
print("\n3. Saving summary...")
summary = logger.save_summary()

logger.close()

print("\n4. Testing ExperimentTracker...")
tracker = ExperimentTracker(base_dir="test_experiments")
tracker.register_experiment(
    experiment_name="test_exp_1",
    config=config,
    description="Test experiment for logging system"
)
tracker.list_experiments()

print("\n" + "="*70)
print("✅ All Tests Passed!")
print("="*70)
print("\nGenerated files:")
print("  - test_logs/csv/test_experiment_steps.csv")
print("  - test_logs/csv/test_experiment_epochs.csv")
print("  - test_logs/json/test_experiment_config.json")
print("  - test_logs/json/test_experiment_summary.json")
print("  - test_experiments/experiment_registry.json")
print("\nYou can inspect these files to see the logging format.")
print()
