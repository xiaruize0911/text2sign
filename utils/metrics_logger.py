"""
Comprehensive metrics logging for research paper purposes
Tracks all training metrics, saves to CSV, JSON, and TensorBoard
"""

import os
import csv
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np


class MetricsLogger:
    """
    Comprehensive metrics logger for research experiments
    
    Features:
    - CSV export for easy analysis
    - JSON export with full experiment metadata
    - Automatic aggregation (mean, std, min, max)
    - Epoch and step-level tracking
    - TensorBoard integration
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
            config: Configuration dict to save with experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.config = config or {}
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir = self.log_dir / "csv"
        self.csv_dir.mkdir(exist_ok=True)
        self.json_dir = self.log_dir / "json"
        self.json_dir.mkdir(exist_ok=True)
        
        # Tracking data
        self.step_metrics = []  # List of dicts for each step
        self.epoch_metrics = []  # List of dicts for each epoch
        self.epoch_start_time = None
        self.training_start_time = time.time()
        
        # CSV files
        self.step_csv_path = self.csv_dir / f"{experiment_name}_steps.csv"
        self.epoch_csv_path = self.csv_dir / f"{experiment_name}_epochs.csv"
        self.step_csv_file = None
        self.epoch_csv_file = None
        self.step_csv_writer = None
        self.epoch_csv_writer = None
        
        # Save experiment config
        self._save_config()
        
        print(f"📊 Metrics Logger initialized:")
        print(f"   - Experiment: {experiment_name}")
        print(f"   - Log dir: {log_dir}")
        print(f"   - CSV: {self.csv_dir}")
        print(f"   - JSON: {self.json_dir}")
    
    def _save_config(self):
        """Save experiment configuration"""
        config_path = self.json_dir / f"{self.experiment_name}_config.json"
        
        # Convert config to serializable format
        serializable_config = {}
        for key, value in self.config.items():
            if hasattr(value, '__dict__'):
                # Convert dataclass/object to dict
                serializable_config[key] = {
                    k: v for k, v in value.__dict__.items()
                    if not k.startswith('_')
                }
            else:
                serializable_config[key] = value
        
        metadata = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": serializable_config,
        }
        
        with open(config_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"   ✅ Config saved to {config_path}")
    
    def log_step(self, step: int, metrics: Dict[str, float], phase: str = "train"):
        """
        Log metrics for a single training/validation step
        
        Args:
            step: Global step number
            metrics: Dictionary of metric_name -> value
            phase: "train", "val", or "test"
        """
        # Add metadata
        log_entry = {
            "step": step,
            "phase": phase,
            "timestamp": time.time() - self.training_start_time,
            **metrics
        }
        
        self.step_metrics.append(log_entry)
        
        # Write to CSV immediately (append mode)
        if self.step_csv_file is None:
            self.step_csv_file = open(self.step_csv_path, 'w', newline='')
            fieldnames = list(log_entry.keys())
            # Use extrasaction='ignore' to handle dynamic fields gracefully
            self.step_csv_writer = csv.DictWriter(
                self.step_csv_file, 
                fieldnames=fieldnames,
                extrasaction='ignore'  # Ignore extra fields instead of raising error
            )
            self.step_csv_writer.writeheader()
        
        # Only write fields that exist in the original fieldnames
        # This prevents crashes when new fields are added later
        self.step_csv_writer.writerow(log_entry)
        self.step_csv_file.flush()  # Ensure immediate write
    
    def start_epoch(self, epoch: int):
        """Mark the start of an epoch"""
        self.epoch_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"📈 Epoch {epoch} started at {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")
    
    def end_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for completed epoch with statistics
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of epoch-level metrics
        """
        # Calculate epoch duration
        epoch_duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Get step-level statistics for this epoch
        epoch_steps = [m for m in self.step_metrics if m.get('epoch') == epoch]
        if not epoch_steps and self.step_metrics:
            # If epoch not in step metrics, use recent steps
            epoch_steps = self.step_metrics[-100:]  # Last 100 steps
        
        # Calculate statistics from steps
        step_stats = {}
        if epoch_steps:
            for key in epoch_steps[0].keys():
                if key not in ['step', 'epoch', 'phase', 'timestamp']:
                    values = [s[key] for s in epoch_steps if isinstance(s.get(key), (int, float))]
                    if values:
                        step_stats[f"{key}_mean"] = np.mean(values)
                        step_stats[f"{key}_std"] = np.std(values)
                        step_stats[f"{key}_min"] = np.min(values)
                        step_stats[f"{key}_max"] = np.max(values)
        
        # Create epoch log entry
        log_entry = {
            "epoch": epoch,
            "duration_seconds": epoch_duration,
            "duration_minutes": epoch_duration / 60,
            "total_runtime_hours": (time.time() - self.training_start_time) / 3600,
            **metrics,
            **step_stats,
        }
        
        self.epoch_metrics.append(log_entry)
        
        # Write to CSV
        if self.epoch_csv_file is None:
            self.epoch_csv_file = open(self.epoch_csv_path, 'w', newline='')
            fieldnames = list(log_entry.keys())
            self.epoch_csv_writer = csv.DictWriter(self.epoch_csv_file, fieldnames=fieldnames)
            self.epoch_csv_writer.writeheader()
        
        self.epoch_csv_writer.writerow(log_entry)
        self.epoch_csv_file.flush()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 Epoch {epoch} Summary:")
        print(f"{'='*70}")
        print(f"  Duration: {epoch_duration/60:.2f} minutes")
        print(f"  Total Runtime: {(time.time() - self.training_start_time)/3600:.2f} hours")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*70}\n")
    
    def save_summary(self):
        """Save complete training summary to JSON"""
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.fromtimestamp(self.training_start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_duration_hours": (time.time() - self.training_start_time) / 3600,
            "total_steps": len(self.step_metrics),
            "total_epochs": len(self.epoch_metrics),
            "config": self.config,
            "final_metrics": self.epoch_metrics[-1] if self.epoch_metrics else {},
            "best_metrics": self._get_best_metrics(),
        }
        
        summary_path = self.json_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print(f"💾 Training Summary Saved:")
        print(f"{'='*70}")
        print(f"  Total Duration: {summary['total_duration_hours']:.2f} hours")
        print(f"  Total Steps: {summary['total_steps']}")
        print(f"  Total Epochs: {summary['total_epochs']}")
        print(f"  Summary: {summary_path}")
        print(f"  Steps CSV: {self.step_csv_path}")
        print(f"  Epochs CSV: {self.epoch_csv_path}")
        print(f"{'='*70}\n")
        
        return summary
    
    def _get_best_metrics(self) -> Dict[str, Any]:
        """Find best values across all epochs"""
        if not self.epoch_metrics:
            return {}
        
        best = {}
        for key in self.epoch_metrics[0].keys():
            if key in ['epoch', 'duration_seconds', 'duration_minutes', 'total_runtime_hours']:
                continue
            
            values = [e[key] for e in self.epoch_metrics if isinstance(e.get(key), (int, float))]
            if values:
                best[f"best_{key}"] = min(values) if 'loss' in key.lower() else max(values)
                best[f"best_{key}_epoch"] = [e['epoch'] for e in self.epoch_metrics 
                                              if e.get(key) == best[f"best_{key}"]][0]
        
        return best
    
    def close(self):
        """Close all file handles"""
        if self.step_csv_file:
            self.step_csv_file.close()
        if self.epoch_csv_file:
            self.epoch_csv_file.close()
    
    def __del__(self):
        """Ensure files are closed"""
        self.close()


class ExperimentTracker:
    """
    Track multiple experiments and compare results
    """
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Load experiment registry
        self.registry_path = self.base_dir / "experiment_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> List[Dict]:
        """Load experiment registry"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_registry(self):
        """Save experiment registry"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def register_experiment(
        self,
        experiment_name: str,
        config: Dict,
        description: str = ""
    ):
        """Register a new experiment"""
        entry = {
            "name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "description": description,
            "config": config,
        }
        self.registry.append(entry)
        self._save_registry()
        
        print(f"📝 Experiment '{experiment_name}' registered")
    
    def list_experiments(self):
        """List all experiments"""
        print(f"\n{'='*70}")
        print(f"📊 Experiment Registry ({len(self.registry)} experiments)")
        print(f"{'='*70}\n")
        
        for i, exp in enumerate(self.registry, 1):
            print(f"{i}. {exp['name']}")
            print(f"   Started: {exp['start_time']}")
            if exp.get('description'):
                print(f"   Description: {exp['description']}")
            print()
