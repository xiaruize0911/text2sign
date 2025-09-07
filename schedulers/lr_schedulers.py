"""
Dynamic Learning Rate Schedulers for Text2Sign Training
This module provides various learning rate scheduling strategies for optimal training dynamics.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    CosineAnnealingWarmRestarts, 
    ReduceLROnPlateau, 
    ExponentialLR,
    LambdaLR
)
import math
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LinearWarmupCosineAnnealingLR(LambdaLR):
    """
    Linear warmup followed by cosine annealing learning rate scheduler
    """
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 warmup_start_lr: float = 1e-6, eta_min: float = 1e-7):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        # Get the initial learning rate from optimizer
        self.base_lr = optimizer.param_groups[0]['lr']
        
        super().__init__(optimizer, self.lr_lambda)
    
    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup phase
            return (self.warmup_start_lr + 
                   (self.base_lr - self.warmup_start_lr) * epoch / self.warmup_epochs) / self.base_lr
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return (self.eta_min + (self.base_lr - self.eta_min) * cosine_factor) / self.base_lr


class PolynomialLR(LambdaLR):
    """
    Polynomial learning rate decay scheduler
    """
    def __init__(self, optimizer, max_epochs: int, power: float = 0.9, eta_min: float = 1e-7):
        self.max_epochs = max_epochs
        self.power = power
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        
        super().__init__(optimizer, self.lr_lambda)
    
    def lr_lambda(self, epoch):
        factor = (1 - epoch / self.max_epochs) ** self.power
        return max(self.eta_min / self.base_lr, factor)


class AdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler that monitors training progress
    and adjusts learning rate based on loss trends
    """
    def __init__(self, optimizer, patience: int = 10, factor: float = 0.5, 
                 min_lr: float = 1e-7, threshold: float = 1e-4):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        
        self.best_loss = float('inf')
        self.wait = 0
        self.loss_history = []
        
    def step(self, loss: float):
        """Step the scheduler with current loss"""
        self.loss_history.append(loss)
        
        # Check if we have improvement
        if loss < self.best_loss - self.threshold:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            
        # Reduce learning rate if no improvement for patience epochs
        if self.wait >= self.patience:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * self.factor, self.min_lr)
            
            if new_lr < current_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                logger.info(f"Reducing learning rate to {new_lr:.2e}")
                self.wait = 0
        
        return self.optimizer.param_groups[0]['lr']


def create_lr_scheduler(optimizer: optim.Optimizer, config) -> Optional[Any]:
    """
    Factory function to create learning rate scheduler based on config
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration object containing scheduler settings
        
    Returns:
        Learning rate scheduler or None if disabled
    """
    if not getattr(config, 'USE_SCHEDULER', False):
        logger.info("Learning rate scheduler disabled")
        return None
    
    scheduler_type = getattr(config, 'SCHEDULER_TYPE', 'cosine_annealing')
    logger.info(f"Creating {scheduler_type} learning rate scheduler")
    
    try:
        if scheduler_type == "cosine_annealing":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=getattr(config, 'COSINE_T_MAX', 50),
                eta_min=getattr(config, 'COSINE_ETA_MIN', 1e-7)
            )
            
        elif scheduler_type == "cosine_annealing_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=getattr(config, 'COSINE_RESTARTS_T_0', 25),
                T_mult=getattr(config, 'COSINE_RESTARTS_T_MULT', 2),
                eta_min=getattr(config, 'COSINE_RESTARTS_ETA_MIN', 1e-7)
            )
            
        elif scheduler_type == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=getattr(config, 'PLATEAU_FACTOR', 0.5),
                patience=getattr(config, 'PLATEAU_PATIENCE', 15),
                threshold=getattr(config, 'PLATEAU_THRESHOLD', 1e-4),
                min_lr=getattr(config, 'PLATEAU_MIN_LR', 1e-7),
                verbose=True
            )
            
        elif scheduler_type == "exponential":
            scheduler = ExponentialLR(
                optimizer,
                gamma=getattr(config, 'EXPONENTIAL_GAMMA', 0.95)
            )
            
        elif scheduler_type == "linear_warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=getattr(config, 'WARMUP_EPOCHS', 10),
                max_epochs=getattr(config, 'NUM_EPOCHS', 500),
                warmup_start_lr=getattr(config, 'WARMUP_START_LR', 1e-6),
                eta_min=getattr(config, 'COSINE_ETA_MIN', 1e-7)
            )
            
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                max_epochs=getattr(config, 'NUM_EPOCHS', 500),
                power=getattr(config, 'POLYNOMIAL_POWER', 0.9),
                eta_min=getattr(config, 'COSINE_ETA_MIN', 1e-7)
            )
            
        elif scheduler_type == "adaptive":
            scheduler = AdaptiveLRScheduler(
                optimizer,
                patience=getattr(config, 'PLATEAU_PATIENCE', 10),
                factor=getattr(config, 'PLATEAU_FACTOR', 0.5),
                min_lr=getattr(config, 'PLATEAU_MIN_LR', 1e-7),
                threshold=getattr(config, 'PLATEAU_THRESHOLD', 1e-4)
            )
            
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}")
            return None
            
        logger.info(f"✅ Created {scheduler_type} scheduler successfully")
        return scheduler
        
    except Exception as e:
        logger.error(f"Failed to create scheduler {scheduler_type}: {e}")
        return None


def get_lr_schedule_info(scheduler, config) -> Dict[str, Any]:
    """
    Get information about the learning rate schedule
    
    Args:
        scheduler: Learning rate scheduler
        config: Configuration object
        
    Returns:
        Dictionary with schedule information
    """
    info = {
        'type': getattr(config, 'SCHEDULER_TYPE', 'none'),
        'enabled': getattr(config, 'USE_SCHEDULER', False)
    }
    
    if scheduler is None:
        return info
    
    if hasattr(scheduler, 'T_max'):
        info['period'] = scheduler.T_max
    if hasattr(scheduler, 'eta_min'):
        info['min_lr'] = scheduler.eta_min
    if hasattr(scheduler, 'factor'):
        info['reduction_factor'] = scheduler.factor
    if hasattr(scheduler, 'patience'):
        info['patience'] = scheduler.patience
        
    return info


def log_lr_schedule(writer, scheduler, optimizer, config, epoch: int):
    """
    Log learning rate schedule information to TensorBoard
    
    Args:
        writer: TensorBoard writer
        scheduler: Learning rate scheduler
        optimizer: PyTorch optimizer
        config: Configuration object
        epoch: Current epoch
    """
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('learning_rate/current', current_lr, epoch)
    
    # Log scheduler-specific metrics
    if hasattr(scheduler, 'last_epoch'):
        writer.add_scalar('learning_rate/scheduler_epoch', scheduler.last_epoch, epoch)
    
    if isinstance(scheduler, ReduceLROnPlateau):
        writer.add_scalar('learning_rate/plateau_patience', scheduler.num_bad_epochs, epoch)
        writer.add_scalar('learning_rate/best_metric', scheduler.best, epoch)
    
    # Log schedule parameters
    schedule_info = get_lr_schedule_info(scheduler, config)
    for key, value in schedule_info.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f'learning_rate/schedule_{key}', value, epoch)


def simulate_lr_schedule(config, num_epochs: Optional[int] = None) -> Dict[str, list]:
    """
    Simulate learning rate schedule for visualization
    
    Args:
        config: Configuration object
        num_epochs: Number of epochs to simulate (defaults to config.NUM_EPOCHS)
        
    Returns:
        Dictionary with epochs and corresponding learning rates
    """
    if num_epochs is None:
        num_epochs = getattr(config, 'NUM_EPOCHS', 500)
    
    # Create dummy optimizer
    dummy_param = torch.tensor([1.0], requires_grad=True)
    optimizer = optim.Adam([dummy_param], lr=config.get_learning_rate())
    
    # Create scheduler
    scheduler = create_lr_scheduler(optimizer, config)
    
    epochs = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        epochs.append(epoch)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Step the scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                # For plateau scheduler, simulate some loss values
                fake_loss = 1.0 + 0.1 * math.sin(epoch * 0.1) + 0.01 * epoch
                scheduler.step(fake_loss)
            elif isinstance(scheduler, AdaptiveLRScheduler):
                fake_loss = 1.0 + 0.1 * math.sin(epoch * 0.1) + 0.01 * epoch
                scheduler.step(fake_loss)
            else:
                scheduler.step()
    
    return {
        'epochs': epochs,
        'learning_rates': learning_rates,
        'scheduler_type': getattr(config, 'SCHEDULER_TYPE', 'none')
    }


if __name__ == "__main__":
    # Test the schedulers
    from config import Config
    
    print("Testing Learning Rate Schedulers...")
    print("=" * 50)
    
    # Test different scheduler types
    scheduler_types = [
        "cosine_annealing",
        "cosine_annealing_with_restarts", 
        "linear_warmup_cosine",
        "polynomial",
        "exponential"
    ]
    
    for sched_type in scheduler_types:
        print(f"\n📊 Testing {sched_type}...")
        
        # Temporarily set scheduler type
        original_type = getattr(Config, 'SCHEDULER_TYPE', 'none')
        Config.SCHEDULER_TYPE = sched_type
        Config.USE_SCHEDULER = True
        
        try:
            # Simulate schedule
            schedule_data = simulate_lr_schedule(Config, num_epochs=100)
            
            print(f"   ✅ Scheduler created successfully")
            print(f"   📈 LR range: {min(schedule_data['learning_rates']):.2e} - {max(schedule_data['learning_rates']):.2e}")
            print(f"   🎯 Final LR: {schedule_data['learning_rates'][-1]:.2e}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Restore original type
        Config.SCHEDULER_TYPE = original_type
    
    print("\n🎉 Scheduler testing completed!")
