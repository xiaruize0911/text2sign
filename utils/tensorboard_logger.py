"""
Comprehensive TensorBoard Logging System for Text2Sign Diffusion Model
This module provides structured and organized logging for all training metrics,
visualizations, and model monitoring.
"""

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union
import matplotlib.pyplot as plt
import io
from PIL import Image
import warnings

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """
    Comprehensive TensorBoard logging system with organized metric categories
    """
    
    def __init__(self, log_dir: str, config=None):
        """
        Initialize TensorBoard logger with organized structure
        
        Args:
            log_dir: Directory for TensorBoard logs
            config: Configuration object for logging settings
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.config = config
        self.step_counters = {
            'train': 0,
            'epoch': 0,
            'sample': 0,
            'debug': 0
        }
        
        # Metric buffers for aggregation
        self.metric_buffers = {
            'train_loss': [],
            'train_metrics': {},
            'debug_metrics': {}
        }
        
        logger.info(f"🔍 TensorBoard logger initialized: {log_dir}")
    
    def flush(self):
        """Flush TensorBoard writer"""
        self.writer.flush()
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()
    
    # =============================================================================
    # TRAINING METRICS LOGGING
    # =============================================================================
    
    def log_training_step(self, metrics: Dict[str, float], step: int = None):
        """
        Log training step metrics
        
        Args:
            metrics: Dictionary of training metrics
            step: Global step (auto-incremented if None)
        """
        if step is None:
            step = self.step_counters['train']
            self.step_counters['train'] += 1
        
        # Core training metrics
        if 'loss' in metrics:
            self.writer.add_scalar('01_Training/Loss', metrics['loss'], step)
            self.metric_buffers['train_loss'].append(metrics['loss'])
        
        if 'learning_rate' in metrics:
            self.writer.add_scalar('01_Training/Learning_Rate', metrics['learning_rate'], step)
            # Also log to dedicated LR category for step-level tracking
            self.writer.add_scalar('04_Learning_Rate/Step_LR', metrics['learning_rate'], step)
        
        if 'grad_norm' in metrics:
            self.writer.add_scalar('01_Training/Gradient_Norm', metrics['grad_norm'], step)
        
        if 'batch_size' in metrics:
            self.writer.add_scalar('01_Training/Batch_Size', metrics['batch_size'], step)
            # Also log to performance category
            self.writer.add_scalar('05_Performance/Effective_Batch_Size', metrics['batch_size'], step)
        
        # Advanced training metrics
        if 'loss_components' in metrics:
            for component, value in metrics['loss_components'].items():
                self.writer.add_scalar(f'02_Loss_Components/{component}', value, step)
    
    def log_epoch_summary(self, metrics: Dict[str, float], epoch: int):
        """
        Log comprehensive epoch summary
        
        Args:
            metrics: Dictionary of epoch metrics
            epoch: Current epoch number
        """
        self.step_counters['epoch'] = epoch
        
        # Primary epoch metrics
        if 'loss' in metrics:
            self.writer.add_scalar('03_Epoch_Summary/Average_Loss', metrics['loss'], epoch)
        
        if 'time' in metrics:
            self.writer.add_scalar('03_Epoch_Summary/Epoch_Time_Seconds', metrics['time'], epoch)
        
        if 'grad_norm' in metrics:
            self.writer.add_scalar('03_Epoch_Summary/Average_Grad_Norm', metrics['grad_norm'], epoch)
        
        if 'optimizer_steps' in metrics:
            self.writer.add_scalar('03_Epoch_Summary/Optimizer_Steps', metrics['optimizer_steps'], epoch)
        
        # Learning rate tracking
        if 'learning_rate' in metrics:
            self.writer.add_scalar('04_Learning_Rate/Current_LR', metrics['learning_rate'], epoch)
        
        # Performance metrics
        if 'samples_per_second' in metrics:
            self.writer.add_scalar('05_Performance/Samples_Per_Second', metrics['samples_per_second'], epoch)
        
        if 'steps_per_second' in metrics:
            self.writer.add_scalar('05_Performance/Steps_Per_Second', metrics['steps_per_second'], epoch)
    
    def log_step_performance(self, metrics: Dict[str, float], step: int):
        """
        Log step-level performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
            step: Current global step
        """
        if 'step_time' in metrics:
            self.writer.add_scalar('05_Performance/Step_Time_Seconds', metrics['step_time'], step)
        
        if 'samples_per_second' in metrics:
            self.writer.add_scalar('05_Performance/Step_Samples_Per_Second', metrics['samples_per_second'], step)
        
        if 'memory_usage' in metrics:
            self.writer.add_scalar('05_Performance/Memory_Usage_MB', metrics['memory_usage'], step)

    # =============================================================================
    # DIFFUSION-SPECIFIC METRICS
    # =============================================================================
    
    def log_diffusion_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log diffusion model specific metrics
        
        Args:
            metrics: Dictionary of diffusion metrics
            step: Current global step
        """
        # Noise prediction quality
        if 'noise_mse' in metrics:
            self.writer.add_scalar('06_Diffusion/Noise_Prediction_MSE', metrics['noise_mse'], step)
        
        if 'noise_mae' in metrics:
            self.writer.add_scalar('06_Diffusion/Noise_Prediction_MAE', metrics['noise_mae'], step)
        
        # Timestep analysis
        if 'timestep_distribution' in metrics:
            self.writer.add_histogram('06_Diffusion/Timestep_Distribution', 
                                     torch.tensor(metrics['timestep_distribution']), step)
        
        # Signal-to-noise ratio tracking
        if 'snr' in metrics:
            self.writer.add_scalar('06_Diffusion/Signal_to_Noise_Ratio', metrics['snr'], step)
        
        # Noise schedule analysis
        if 'beta_values' in metrics:
            self.writer.add_scalar('06_Diffusion/Current_Beta', metrics['beta_values'], step)
        
        if 'alpha_values' in metrics:
            self.writer.add_scalar('06_Diffusion/Current_Alpha', metrics['alpha_values'], step)
    
    def log_noise_statistics(self, predicted_noise: torch.Tensor, actual_noise: torch.Tensor, 
                           video_batch: torch.Tensor, timesteps: torch.Tensor, step: int):
        """
        Log detailed noise prediction statistics
        
        Args:
            predicted_noise: Model predicted noise
            actual_noise: Ground truth noise
            video_batch: Input video batch
            timesteps: Diffusion timesteps
            step: Current global step
        """
        with torch.no_grad():
            # Basic statistics
            pred_stats = {
                'mean': predicted_noise.mean().item(),
                'std': predicted_noise.std().item(),
                'min': predicted_noise.min().item(),
                'max': predicted_noise.max().item(),
                'abs_mean': predicted_noise.abs().mean().item()
            }
            
            actual_stats = {
                'mean': actual_noise.mean().item(),
                'std': actual_noise.std().item(),
                'min': actual_noise.min().item(),
                'max': actual_noise.max().item(),
                'abs_mean': actual_noise.abs().mean().item()
            }
            
            video_stats = {
                'mean': video_batch.mean().item(),
                'std': video_batch.std().item(),
                'min': video_batch.min().item(),
                'max': video_batch.max().item()
            }
            
            # Log predicted noise statistics
            for stat_name, value in pred_stats.items():
                self.writer.add_scalar(f'07_Noise_Analysis/Predicted_{stat_name.title()}', value, step)
            
            # Log actual noise statistics
            for stat_name, value in actual_stats.items():
                self.writer.add_scalar(f'07_Noise_Analysis/Actual_{stat_name.title()}', value, step)
            
            # Log video statistics
            for stat_name, value in video_stats.items():
                self.writer.add_scalar(f'07_Noise_Analysis/Video_{stat_name.title()}', value, step)
            
            # Correlation and error metrics
            noise_correlation = F.cosine_similarity(
                predicted_noise.flatten(), actual_noise.flatten(), dim=0
            ).item()
            self.writer.add_scalar('07_Noise_Analysis/Noise_Correlation', noise_correlation, step)
            
            # Per-timestep analysis
            timestep_mean = timesteps.float().mean().item()
            timestep_std = timesteps.float().std().item()
            self.writer.add_scalar('07_Noise_Analysis/Timestep_Mean', timestep_mean, step)
            self.writer.add_scalar('07_Noise_Analysis/Timestep_Std', timestep_std, step)
    
    # =============================================================================
    # MODEL ARCHITECTURE METRICS
    # =============================================================================
    
    def log_model_architecture(self, model: torch.nn.Module, epoch: int):
        """
        Log model architecture information
        
        Args:
            model: PyTorch model
            epoch: Current epoch
        """
        # Parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.writer.add_scalar('08_Model_Architecture/Total_Parameters', total_params, epoch)
        self.writer.add_scalar('08_Model_Architecture/Trainable_Parameters', trainable_params, epoch)
        self.writer.add_scalar('08_Model_Architecture/Parameter_Efficiency', 
                              trainable_params / total_params, epoch)
        
        # Layer-wise parameter analysis
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Clean parameter name for TensorBoard
                clean_name = name.replace('.', '/')
                param_norm = param.norm().item()
                param_mean = param.mean().item()
                param_std = param.std().item()
                
                self.writer.add_scalar(f'09_Parameter_Stats/{clean_name}/Norm', param_norm, epoch)
                self.writer.add_scalar(f'09_Parameter_Stats/{clean_name}/Mean', param_mean, epoch)
                self.writer.add_scalar(f'09_Parameter_Stats/{clean_name}/Std', param_std, epoch)
                
                # Add parameter histograms periodically
                if epoch % 10 == 0:  # Every 10 epochs
                    self.writer.add_histogram(f'10_Parameter_Histograms/{clean_name}', param, epoch)
    
    def log_gradient_statistics(self, model: torch.nn.Module, step: int):
        """
        Log gradient statistics
        
        Args:
            model: PyTorch model
            step: Current global step
        """
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                clean_name = name.replace('.', '/')
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                self.writer.add_scalar(f'11_Gradient_Stats/{clean_name}/Norm', grad_norm, step)
                self.writer.add_scalar(f'11_Gradient_Stats/{clean_name}/Mean', grad_mean, step)
                self.writer.add_scalar(f'11_Gradient_Stats/{clean_name}/Std', grad_std, step)
                
                total_norm += grad_norm ** 2
                param_count += 1
        
        # Overall gradient statistics
        if param_count > 0:
            total_norm = (total_norm ** 0.5)
            self.writer.add_scalar('11_Gradient_Stats/Total_Gradient_Norm', total_norm, step)
            self.writer.add_scalar('11_Gradient_Stats/Average_Gradient_Norm', 
                                  total_norm / param_count, step)
    
    # =============================================================================
    # LEARNING RATE SCHEDULING
    # =============================================================================
    
    def log_learning_rate_schedule(self, scheduler, optimizer, config, epoch: int):
        """
        Log learning rate scheduling information
        
        Args:
            scheduler: Learning rate scheduler
            optimizer: PyTorch optimizer
            config: Configuration object
            epoch: Current epoch
        """
        current_lr = optimizer.param_groups[0]['lr']
        self.writer.add_scalar('04_Learning_Rate/Current_LR', current_lr, epoch)
        
        # Scheduler-specific metrics
        if hasattr(scheduler, 'last_epoch'):
            self.writer.add_scalar('04_Learning_Rate/Scheduler_Epoch', scheduler.last_epoch, epoch)
        
        # Plateau scheduler metrics
        if hasattr(scheduler, 'num_bad_epochs'):
            self.writer.add_scalar('04_Learning_Rate/Plateau_Patience', scheduler.num_bad_epochs, epoch)
        
        if hasattr(scheduler, 'best'):
            self.writer.add_scalar('04_Learning_Rate/Best_Metric', scheduler.best, epoch)
        
        # Predict future learning rates for visualization
        if hasattr(config, 'NUM_EPOCHS'):
            future_epochs = list(range(epoch, min(epoch + 50, config.NUM_EPOCHS)))
            future_lrs = []
            
            # This is a simplified prediction - actual implementation may vary
            for future_epoch in future_epochs:
                if hasattr(scheduler, 'get_lr'):
                    try:
                        # Temporarily set epoch for prediction
                        original_epoch = scheduler.last_epoch
                        scheduler.last_epoch = future_epoch
                        future_lr = scheduler.get_lr()[0]
                        scheduler.last_epoch = original_epoch
                        future_lrs.append(future_lr)
                    except:
                        break
                else:
                    break
            
            if future_lrs:
                for i, lr in enumerate(future_lrs):
                    self.writer.add_scalar('04_Learning_Rate/Predicted_Schedule', lr, epoch + i)
    
    # =============================================================================
    # SAMPLE GENERATION AND VISUALIZATION
    # =============================================================================
    
    def log_generated_samples(self, samples: torch.Tensor, step: int, tag: str = "Generated_Samples"):
        """
        Log generated video samples to TensorBoard
        
        Args:
            samples: Generated video samples (B, C, T, H, W)
            step: Current step
            tag: Tag for the samples
        """
        try:
            # Ensure samples are in correct format and range
            if samples.dim() == 5:  # (B, C, T, H, W)
                # Normalize to [0, 1] range
                samples_norm = (samples - samples.min()) / (samples.max() - samples.min() + 1e-8)
                
                # Log as video (take first sample in batch)
                # Note: TensorBoard expects (N, T, C, H, W) format, so we need to permute
                samples_for_tb = samples_norm[:4].permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                self.writer.add_video(
                    f'12_Generated_Samples/{tag}',
                    samples_for_tb,
                    global_step=step,
                    fps=8
                )
                
                # Log sample statistics
                self.writer.add_scalar(f'12_Generated_Samples/Sample_Mean', samples.mean().item(), step)
                self.writer.add_scalar(f'12_Generated_Samples/Sample_Std', samples.std().item(), step)
                self.writer.add_scalar(f'12_Generated_Samples/Sample_Min', samples.min().item(), step)
                self.writer.add_scalar(f'12_Generated_Samples/Sample_Max', samples.max().item(), step)
                
                self.step_counters['sample'] += 1
                logger.info(f"📹 Generated samples logged to TensorBoard (step {step})")
                
            else:
                logger.warning(f"Invalid sample dimensions: {samples.shape}")
                
        except Exception as e:
            logger.error(f"Failed to log generated samples: {e}")
    
    def log_noise_visualization(self, predicted_noise: torch.Tensor, actual_noise: torch.Tensor, 
                               original_video: torch.Tensor, step: int):
        """
        Log noise visualization videos
        
        Args:
            predicted_noise: Model predicted noise
            actual_noise: Ground truth noise  
            original_video: Original video
            step: Current step
        """
        try:
            # Normalize all tensors for visualization
            def normalize_for_viz(tensor):
                return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
            
            pred_viz = normalize_for_viz(predicted_noise[:1])  # Take first sample
            actual_viz = normalize_for_viz(actual_noise[:1])
            orig_viz = normalize_for_viz(original_video[:1])
            
            # Convert to TensorBoard format (N, T, C, H, W)
            pred_viz_tb = pred_viz.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
            actual_viz_tb = actual_viz.permute(0, 2, 1, 3, 4)
            orig_viz_tb = orig_viz.permute(0, 2, 1, 3, 4)
            
            # Log separate videos
            self.writer.add_video('13_Noise_Visualization/Predicted_Noise', pred_viz_tb, step, fps=8)
            self.writer.add_video('13_Noise_Visualization/Actual_Noise', actual_viz_tb, step, fps=8)
            self.writer.add_video('13_Noise_Visualization/Original_Video', orig_viz_tb, step, fps=8)
            
            logger.debug(f"Noise visualization logged to TensorBoard (step {step})")
            
        except Exception as e:
            logger.error(f"Failed to log noise visualization: {e}")
    
    # =============================================================================
    # SYSTEM METRICS
    # =============================================================================
    
    def log_system_metrics(self, epoch: int):
        """
        Log system performance metrics
        
        Args:
            epoch: Current epoch
        """
        # GPU memory metrics
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            self.writer.add_scalar('14_System/GPU_Memory_Allocated_GB', memory_allocated, epoch)
            self.writer.add_scalar('14_System/GPU_Memory_Reserved_GB', memory_reserved, epoch)
            self.writer.add_scalar('14_System/GPU_Memory_Utilization', 
                                  memory_allocated / (memory_reserved + 1e-8), epoch)
        
        # Apple Silicon memory metrics
        elif torch.backends.mps.is_available():
            try:
                memory_allocated = torch.mps.current_allocated_memory() / 1024**3  # GB
                self.writer.add_scalar('14_System/MPS_Memory_Allocated_GB', memory_allocated, epoch)
            except:
                pass  # MPS memory tracking might not be available
    
    # =============================================================================
    # CONFIGURATION LOGGING
    # =============================================================================
    
    def log_configuration(self, config, epoch: int = 0):
        """
        Log configuration parameters
        
        Args:
            config: Configuration object
            epoch: Current epoch (default 0 for initial logging)
        """
        # Core training configuration
        config_metrics = {
            'batch_size': getattr(config, 'BATCH_SIZE', 0),
            'learning_rate': getattr(config, 'LEARNING_RATE', 0),
            'num_epochs': getattr(config, 'NUM_EPOCHS', 0),
            'gradient_accumulation_steps': getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1),
            'timesteps': getattr(config, 'TIMESTEPS', 1000),
        }
        
        for key, value in config_metrics.items():
            self.writer.add_scalar(f'15_Configuration/{key.title()}', value, epoch)
        
        # Model architecture configuration
        if hasattr(config, 'MODEL_ARCHITECTURE'):
            # Convert string to numeric for logging
            arch_map = {'unet3d': 0, 'vit3d': 1, 'dit3d': 2}
            arch_value = arch_map.get(config.MODEL_ARCHITECTURE, -1)
            self.writer.add_scalar('15_Configuration/Model_Architecture', arch_value, epoch)
    
    # =============================================================================
    # COMPREHENSIVE DASHBOARD CREATION
    # =============================================================================
    
    def create_training_dashboard(self, metrics: Dict[str, Any], epoch: int, step: int):
        """
        Create a comprehensive training dashboard with all key metrics
        
        Args:
            metrics: Dictionary containing all training metrics
            epoch: Current epoch
            step: Current global step
        """
        # Log all categories of metrics
        if 'training' in metrics:
            self.log_training_step(metrics['training'], step)
        
        if 'epoch_summary' in metrics:
            self.log_epoch_summary(metrics['epoch_summary'], epoch)
        
        if 'diffusion' in metrics:
            self.log_diffusion_metrics(metrics['diffusion'], step)
        
        if 'model' in metrics and 'model_object' in metrics:
            self.log_model_architecture(metrics['model_object'], epoch)
        
        if 'system' in metrics:
            self.log_system_metrics(epoch)
        
        # Flush the writer to ensure all metrics are written
        self.flush()
    
    def get_logging_summary(self) -> Dict[str, Any]:
        """
        Get summary of logging activity
        
        Returns:
            Dictionary with logging statistics
        """
        return {
            'step_counters': self.step_counters.copy(),
            'metric_buffer_sizes': {
                'train_loss': len(self.metric_buffers['train_loss']),
                'train_metrics': len(self.metric_buffers['train_metrics']),
                'debug_metrics': len(self.metric_buffers['debug_metrics'])
            },
            'log_dir': str(self.writer.log_dir) if hasattr(self.writer, 'log_dir') else 'unknown'
        }


def create_tensorboard_logger(config) -> TensorBoardLogger:
    """
    Factory function to create TensorBoard logger
    
    Args:
        config: Configuration object
        
    Returns:
        Configured TensorBoardLogger instance
    """
    log_dir = getattr(config, 'LOG_DIR', 'logs/default')
    return TensorBoardLogger(log_dir, config)
