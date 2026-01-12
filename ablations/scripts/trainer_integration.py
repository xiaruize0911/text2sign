"""
Trainer Integration Module for Ablation Study

Provides wrappers and utilities to integrate ablation metrics logging
with the actual Text2Sign training loop.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import time


class TrainerWithMetrics:
    """
    Wrapper around Text2Sign Trainer that integrates ablation metrics logging.
    
    This class wraps an existing trainer instance and intercepts training/evaluation
    to log metrics to the ablation study logging infrastructure.
    
    Falls back gracefully if metrics logging is not available.
    """
    
    def __init__(self, base_trainer, metrics_logger, tb_writer):
        """
        Initialize trainer with metrics logging.
        
        Args:
            base_trainer: The actual Text2Sign Trainer instance
            metrics_logger: MetricsLogger instance for ablation study (can be None)
            tb_writer: TensorBoard SummaryWriter instance (can be None)
        """
        self.base_trainer = base_trainer
        self.metrics_logger = metrics_logger
        self.tb_writer = tb_writer
        self._training_started = False
        
        # Delegate attributes to base trainer
        for attr in dir(base_trainer):
            if not attr.startswith('_') and not callable(getattr(base_trainer, attr)):
                try:
                    setattr(self, attr, getattr(base_trainer, attr))
                except (AttributeError, TypeError):
                    pass
    
    def __getattr__(self, name):
        """Delegate attribute access to base trainer."""
        return getattr(self.base_trainer, name)
    
    def train(self):
        """Run training with metrics logging."""
        print("[TrainerWithMetrics] Starting training with metrics logging...")
        self._training_started = True
        
        try:
            # Get reference to base trainer's training method
            if hasattr(self.base_trainer, 'train'):
                self.base_trainer.train()
            else:
                print("[TrainerWithMetrics] Warning: base_trainer.train() not found")
                return
            
            # Log final metrics if available
            self._log_final_metrics()
            
        except Exception as e:
            print(f"[TrainerWithMetrics] Error during training: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._training_started = False
    
    def _log_final_metrics(self):
        """Log final metrics after training completes."""
        if not self.metrics_logger or not self.tb_writer:
            return
        
        try:
            # Try to extract final metrics from trainer
            if hasattr(self.base_trainer, 'epoch_losses'):
                losses = self.base_trainer.epoch_losses
                if losses:
                    print(f"[TrainerWithMetrics] Final loss: {losses[-1]:.6f}")
                    print(f"[TrainerWithMetrics] Best loss: {min(losses):.6f}")
            
            # Log to metrics logger
            if hasattr(self.metrics_logger, 'log_evaluation'):
                self.metrics_logger.log_evaluation(
                    model_name=getattr(self.base_trainer, 'config_name', 'unknown'),
                    config_name=getattr(self.base_trainer, 'config_name', 'unknown'),
                )
        except Exception as e:
            print(f"[TrainerWithMetrics] Could not log final metrics: {e}")
    
    def log_evaluation(self, fvd: Optional[float] = None, lpips: Optional[float] = None, **kwargs):
        """Log evaluation metrics."""
        print("[TrainerWithMetrics] Logging evaluation metrics...")
        
        self.metrics_logger.log_evaluation(
            model_name=getattr(self.base_trainer, 'config_name', 'unknown'),
            config_name=getattr(self.base_trainer, 'config_name', 'unknown'),
            fvd=fvd,
            lpips=lpips,
            **kwargs
        )
        
        # Log to TensorBoard
        if fvd is not None:
            self.tb_writer.add_scalar("evaluation/fvd", fvd, 0)
        if lpips is not None:
            self.tb_writer.add_scalar("evaluation/lpips", lpips, 0)


def create_metrics_aware_trainer(
    base_trainer,
    metrics_logger,
    tb_writer
):
    """
    Factory function to create a metrics-aware trainer.
    
    Args:
        base_trainer: The actual Text2Sign Trainer instance
        metrics_logger: MetricsLogger instance
        tb_writer: TensorBoard SummaryWriter
        
    Returns:
        TrainerWithMetrics instance that wraps the base trainer
    """
    return TrainerWithMetrics(base_trainer, metrics_logger, tb_writer)


def integrate_metrics_logging(trainer, metrics_logger, tb_writer):
    """
    Integrate metrics logging into an existing trainer.
    
    This is an alternative to using TrainerWithMetrics that directly
    modifies the trainer instance to add logging.
    
    Args:
        trainer: Text2Sign Trainer instance
        metrics_logger: MetricsLogger instance
        tb_writer: TensorBoard SummaryWriter
    """
    
    # Store references
    trainer._metrics_logger = metrics_logger
    trainer._tb_writer = tb_writer
    trainer._training_step_count = 0
    
    # Wrap training loop if it exists
    if hasattr(trainer, 'train'):
        original_train = trainer.train
        
        def train_with_logging():
            """Training loop with integrated logging."""
            original_train()
            
            # Log final metrics
            if hasattr(trainer, 'epoch_losses') and trainer.epoch_losses:
                final_loss = trainer.epoch_losses[-1]
                best_loss = min(trainer.epoch_losses)
                best_epoch = trainer.epoch_losses.index(best_loss)
                
                metrics_logger.log_evaluation(
                    model_name=getattr(trainer, 'config_name', 'unknown'),
                    config_name=getattr(trainer, 'config_name', 'unknown'),
                    fvd=None,
                    lpips=None
                )
        
        trainer.train = train_with_logging
    
    return trainer


class MetricsCapture:
    """Context manager to capture metrics during training."""
    
    def __init__(self, metrics_logger, tb_writer):
        self.metrics_logger = metrics_logger
        self.tb_writer = tb_writer
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            if self.metrics_logger:
                self.metrics_logger.log_training_step(
                    epoch=0,
                    step=0,
                    loss=0.0,
                    learning_rate=0.0,
                    elapsed_time=elapsed
                )
    
    def log_loss(self, epoch: int, step: int, loss: float, lr: float):
        """Log training loss."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        self.metrics_logger.log_training_step(
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=lr,
            elapsed_time=elapsed
        )
        
        # TensorBoard logging
        global_step = epoch * 1000 + step
        self.tb_writer.add_scalar("training/loss", loss, global_step)
        self.tb_writer.add_scalar("training/learning_rate", lr, global_step)
