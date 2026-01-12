"""
Metrics Collection and Logging Module for Ablation Studies

This module provides comprehensive logging of:
- Training metrics (loss, time, memory)
- Evaluation metrics (FVD, LPIPS, temporal consistency)
- System resources (GPU memory, CPU usage)
- Inference performance (time, memory)
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import psutil
import torch
from dataclasses import dataclass, asdict, field
from collections import defaultdict


@dataclass
class TrainingMetrics:
    """Metrics collected during training"""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    training_time_seconds: float = 0.0
    peak_memory_gb: float = 0.0
    avg_memory_gb: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvaluationMetrics:
    """Metrics collected during evaluation"""
    model_name: str = ""
    config_name: str = ""
    fvd: Optional[float] = None  # Fréchet Video Distance (lower is better)
    lpips: Optional[float] = None  # Learned Perceptual Image Patch Similarity (lower is better)
    temporal_consistency: Optional[float] = None  # Temporal consistency (higher is better)
    inference_time_ms: float = 0.0
    inference_memory_gb: float = 0.0
    num_parameters_millions: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentMetrics:
    """Complete metrics for a single ablation experiment"""
    ablation_name: str = ""
    config_name: str = ""
    total_training_time_hours: float = 0.0
    total_training_epochs: int = 0
    peak_training_memory_gb: float = 0.0
    final_training_loss: float = 0.0
    evaluation_metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    training_history: List[TrainingMetrics] = field(default_factory=list)


class MetricsLogger:
    """
    Comprehensive metrics logger for ablation studies.
    
    Logs to multiple formats:
    - JSON: For structured data and loading into pandas
    - CSV: For easy viewing in spreadsheet apps
    - TensorBoard: For real-time monitoring
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save all logs
            experiment_name: Name of the experiment (e.g., 'baseline', 'text_finetuned')
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.training_log_dir = self.log_dir / f"{experiment_name}_training"
        self.evaluation_log_dir = self.log_dir / f"{experiment_name}_evaluation"
        self.training_log_dir.mkdir(exist_ok=True)
        self.evaluation_log_dir.mkdir(exist_ok=True)
        
        # Initialize CSV files
        self.training_csv = self.training_log_dir / "training_metrics.csv"
        self.evaluation_csv = self.evaluation_log_dir / "evaluation_metrics.csv"
        
        # Initialize JSON storage
        self.training_json = self.training_log_dir / "training_metrics.json"
        self.evaluation_json = self.evaluation_log_dir / "evaluation_metrics.json"
        
        # In-memory storage
        self.training_metrics: List[TrainingMetrics] = []
        self.evaluation_metrics: List[EvaluationMetrics] = []
        
        # GPU memory tracking
        self.gpu_memory_log = self.training_log_dir / "gpu_memory.csv"
        self.gpu_memory_history: List[Dict[str, Any]] = []
        
        print(f"[MetricsLogger] Initialized for experiment: {experiment_name}")
        print(f"[MetricsLogger] Log directory: {self.log_dir}")
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        elapsed_time: float
    ) -> None:
        """
        Log a single training step.
        
        Args:
            epoch: Current epoch
            step: Current step
            loss: Training loss
            learning_rate: Current learning rate
            elapsed_time: Time elapsed since training start (seconds)
        """
        # Get current memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory = psutil.virtual_memory().used / 1e9
        
        metrics = TrainingMetrics(
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            training_time_seconds=elapsed_time,
            peak_memory_gb=peak_memory,
            avg_memory_gb=peak_memory  # Simplified: using peak as average
        )
        
        self.training_metrics.append(metrics)
        
        # Also track GPU memory history
        if torch.cuda.is_available():
            self.gpu_memory_history.append({
                "epoch": epoch,
                "step": step,
                "gpu_memory_gb": peak_memory,
                "timestamp": datetime.now().isoformat()
            })
    
    def log_evaluation(
        self,
        model_name: str,
        config_name: str,
        fvd: Optional[float] = None,
        lpips: Optional[float] = None,
        temporal_consistency: Optional[float] = None,
        inference_time_ms: float = 0.0,
        inference_memory_gb: float = 0.0,
        num_parameters_millions: float = 0.0
    ) -> None:
        """
        Log evaluation metrics.
        
        Args:
            model_name: Name of the model variant
            config_name: Configuration name
            fvd: Fréchet Video Distance (lower is better)
            lpips: LPIPS score (lower is better)
            temporal_consistency: Temporal consistency metric (0-1, higher is better)
            inference_time_ms: Time to inference one sample (milliseconds)
            inference_memory_gb: Memory used during inference (GB)
            num_parameters_millions: Number of parameters in millions
        """
        metrics = EvaluationMetrics(
            model_name=model_name,
            config_name=config_name,
            fvd=fvd,
            lpips=lpips,
            temporal_consistency=temporal_consistency,
            inference_time_ms=inference_time_ms,
            inference_memory_gb=inference_memory_gb,
            num_parameters_millions=num_parameters_millions
        )
        
        self.evaluation_metrics.append(metrics)
        
        print(f"\n[Evaluation Results for {model_name}]")
        print(f"  FVD: {fvd if fvd else 'N/A'}")
        print(f"  LPIPS: {lpips if lpips else 'N/A'}")
        print(f"  Temporal Consistency: {temporal_consistency if temporal_consistency else 'N/A'}")
        print(f"  Inference Time: {inference_time_ms:.2f} ms")
        print(f"  Inference Memory: {inference_memory_gb:.2f} GB")
        print(f"  Parameters: {num_parameters_millions:.2f}M\n")
    
    def save_training_metrics(self) -> None:
        """Save training metrics to CSV and JSON files."""
        if not self.training_metrics:
            print("[MetricsLogger] No training metrics to save")
            return
        
        # Save to CSV
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.training_metrics[0].__dataclass_fields__.keys())
            writer.writeheader()
            for metric in self.training_metrics:
                writer.writerow(asdict(metric))
        
        # Save to JSON
        with open(self.training_json, 'w') as f:
            json.dump(
                [asdict(m) for m in self.training_metrics],
                f,
                indent=2
            )
        
        print(f"[MetricsLogger] Saved training metrics to:")
        print(f"  CSV: {self.training_csv}")
        print(f"  JSON: {self.training_json}")
        
        # Save GPU memory history
        if self.gpu_memory_history:
            with open(self.gpu_memory_log, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'step', 'gpu_memory_gb', 'timestamp'])
                writer.writeheader()
                writer.writerows(self.gpu_memory_history)
            
            print(f"  GPU Memory: {self.gpu_memory_log}")
    
    def save_evaluation_metrics(self) -> None:
        """Save evaluation metrics to CSV and JSON files."""
        if not self.evaluation_metrics:
            print("[MetricsLogger] No evaluation metrics to save")
            return
        
        # Save to CSV
        with open(self.evaluation_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.evaluation_metrics[0].__dataclass_fields__.keys())
            writer.writeheader()
            for metric in self.evaluation_metrics:
                writer.writerow(asdict(metric))
        
        # Save to JSON
        with open(self.evaluation_json, 'w') as f:
            json.dump(
                [asdict(m) for m in self.evaluation_metrics],
                f,
                indent=2
            )
        
        print(f"[MetricsLogger] Saved evaluation metrics to:")
        print(f"  CSV: {self.evaluation_csv}")
        print(f"  JSON: {self.evaluation_json}")
    
    def save_all(self) -> None:
        """Save all metrics (training and evaluation)."""
        self.save_training_metrics()
        self.save_evaluation_metrics()
        print(f"[MetricsLogger] All metrics saved to {self.log_dir}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary statistics of training."""
        if not self.training_metrics:
            return {}
        
        losses = [m.loss for m in self.training_metrics]
        memory = [m.peak_memory_gb for m in self.training_metrics]
        times = [m.training_time_seconds for m in self.training_metrics]
        
        return {
            "num_steps": len(self.training_metrics),
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "max_loss": max(losses),
            "avg_loss": sum(losses) / len(losses),
            "peak_memory_gb": max(memory),
            "avg_memory_gb": sum(memory) / len(memory),
            "total_time_hours": times[-1] / 3600 if times else 0.0
        }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of evaluation."""
        if not self.evaluation_metrics:
            return {}
        
        latest = self.evaluation_metrics[-1]
        
        return {
            "model_name": latest.model_name,
            "config_name": latest.config_name,
            "fvd": latest.fvd,
            "lpips": latest.lpips,
            "temporal_consistency": latest.temporal_consistency,
            "inference_time_ms": latest.inference_time_ms,
            "inference_memory_gb": latest.inference_memory_gb,
            "num_parameters_millions": latest.num_parameters_millions
        }
    
    def save_summary_report(self, experiment_summary: Dict[str, Any]) -> None:
        """
        Save a comprehensive summary report.
        
        Args:
            experiment_summary: Dictionary containing experiment summary
        """
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "training_summary": self.get_training_summary(),
            "evaluation_summary": self.get_evaluation_summary(),
            "experiment_summary": experiment_summary
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[MetricsLogger] Summary report saved to {summary_file}")
