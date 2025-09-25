"""
Training utilities and functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import os
import time
import pickle
from typing import Optional, Dict, Any
import numpy as np
import imageio
from tqdm import tqdm
from config import Config
from dataset import create_dataloader
from diffusion import create_diffusion_model
from schedulers.lr_schedulers import create_lr_scheduler, log_lr_schedule
from utils.tensorboard_logger import TensorBoardLogger, create_tensorboard_logger
import logging
import utils.gif as gif_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for the diffusion model
    
    Args:
        config: Configuration object containing all hyperparameters
        model: Diffusion model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler (optional)
    """
    
    def __init__(
        self,
        config,
        model: nn.Module,
        dataloader,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None
    ):
        # Set deterministic behavior for training
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        if hasattr(config, 'DETERMINISTIC') and config.DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = config.DEVICE
        
        # Initialize AMP scaler if using AMP and CUDA is available
        self.use_amp = getattr(config, 'USE_AMP', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        self.amp_dtype = torch.float16 if self.use_amp else torch.float32
        
        # Create directories
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.SAMPLES_DIR, exist_ok=True)
        os.makedirs('./noise_display', exist_ok=True)  # Directory for noise display GIFs
        
        # Initialize comprehensive tensorboard logger
        self.tb_logger = create_tensorboard_logger(config)
        self.writer = self.tb_logger.writer  # Keep backward compatibility
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.accumulation_step = 0  # Track gradient accumulation steps
        
        # Calculate steps per epoch for conversion between epoch and step-based logging
        self.steps_per_epoch = len(dataloader)
        
        # Convert epoch-based frequencies to step-based for compatibility
        self.sample_every_steps = config.SAMPLE_EVERY_EPOCHS * self.steps_per_epoch
        self.log_every_steps = config.LOG_EVERY_EPOCHS * self.steps_per_epoch  
        self.save_every_steps = config.SAVE_EVERY_EPOCHS * self.steps_per_epoch
        
        # Track learning rate history for adaptive scheduling
        self.lr_history = []
        self.loss_history = []
        
        # Log model structure
        self.log_model_structure()
    
    def get_effective_batch_size(self) -> int:
        """Get the effective batch size considering gradient accumulation"""
        return self.config.BATCH_SIZE * self.config.GRADIENT_ACCUMULATION_STEPS
        
    def log_model_structure(self):
        """
        Log the model structure to tensorboard
        """
        # Skip model graph logging if it causes issues
        log_graph = getattr(self.config, 'LOG_MODEL_GRAPH', False)
        if not log_graph:
            logger.info("Model graph logging disabled")
            return
            
        try:
            # Create a dummy input for the model graph
            dummy_input = torch.randn(1, *self.config.INPUT_SHAPE).to(self.device)
            dummy_time = torch.randint(0, self.config.TIMESTEPS, (1,)).to(self.device)
            
            # Log the model graph with strict=False to avoid tracing issues
            self.writer.add_graph(self.model.model, (dummy_input, dummy_time), strict=False)
            self.writer.flush()  # Flush model graph immediately
            logger.info("Model structure logged to tensorboard")
        except Exception as e:
            logger.warning(f"Could not log model structure: {e}")
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint
        
        Args:
            filename (str): Filename for the checkpoint
        """
        # Convert config to a serializable dictionary
        config_dict = {}
        if hasattr(self.config, '__dict__'):
            for key, value in self.config.__dict__.items():
                # Only save basic types and skip problematic attributes
                if not key.startswith('_') and not callable(value):
                    if isinstance(value, (int, float, str, bool, list, tuple, torch.device)):
                        config_dict[key] = str(value) if isinstance(value, torch.device) else value
                    else:
                        logger.debug(f"Skipping config item: {key} (type: {type(value)})")
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'accumulation_step': self.accumulation_step,  # Save accumulation state
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': config_dict
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.config.CHECKPOINT_DIR, filename)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename: str, override_lr: float = Config.LEARNING_RATE):
        """
        Load model checkpoint
        
        Args:
            filename (str): Filename of the checkpoint to load
        """
        filepath = os.path.join(self.config.CHECKPOINT_DIR, filename)
        if not isinstance(filename, str) or not filename.strip():
            logger.error(f"Invalid checkpoint filename: {filename}")
            return
        if not os.path.exists(filepath):
            logger.warning(f"Checkpoint not found: {filepath}")
            return
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.accumulation_step = checkpoint.get('accumulation_step', 0)  # Load accumulation state with default
            # Optionally override the resumed optimizer learning rate
            if override_lr is not None:
                for i, pg in enumerate(self.optimizer.param_groups):
                    old_lr = pg.get('lr', None)
                    pg['lr'] = float(override_lr)
                    logger.info(f"Overrode param_group[{i}] lr: {old_lr} -> {pg['lr']}")
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    logger.warning(f"Could not load scheduler state: {e}")
            logger.info(f"Checkpoint loaded: {filepath}")
        except Exception as e:
            logger.error(f"Failed to securely load checkpoint: {e}")
            return
    
    def generate_samples(self, num_samples: Optional[int] = None) -> torch.Tensor:
        """
        Generate samples for logging
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            torch.Tensor: Generated samples
        """
        if num_samples is None:
            num_samples = self.config.NUM_SAMPLES
        
        self.model.eval()
        with torch.no_grad():
            shape = (num_samples, *self.config.INPUT_SHAPE)
            # Generate samples with progress tracking
            logger.debug(f"Generating {num_samples} video samples...")
            
            # Provide a text prompt for conditioned models
            text_prompt = "A sample sign language translation"  # Replace with your desired prompt
            
            # Use AMP for inference if available and enabled
            if self.use_amp and self.scaler is not None:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    samples = self.model.p_sample(shape, text=text_prompt)
            else:
                samples = self.model.p_sample(shape, text=text_prompt)
                
            # Note: For ε-parameterization, do NOT clamp samples during generation
            # This allows the model to learn the proper data distribution naturally
        return samples
    
    def save_samples_as_gifs(self, samples: torch.Tensor, step: int):
        """
        Save generated samples as GIF files
        
        Args:
            samples (torch.Tensor): Generated samples with shape (batch_size, channels, frames, height, width)
            step (int): Current training step for filename
        """
        batch_size, channels, frames, height, width = samples.shape
        
        # Convert samples to numpy and process for GIF saving
        samples_np = samples.detach().cpu().numpy()
        
        for sample_idx in range(batch_size):
            sample = samples_np[sample_idx]  # (channels, frames, height, width)
            # Convert from CHW to HWC format and scale from [-1,1] to [0, 255]
            video_frames = []
            for frame_idx in range(frames):
                frame = sample[:, frame_idx]  # (channels, height, width)
                frame = np.transpose(frame, (1, 2, 0))  # Convert to (height, width, channels)
                # Convert from [-1, 1] to [0, 255]
                frame = np.clip((frame + 1) * 127.5, 0, 255).astype(np.uint8)
                video_frames.append(frame)
            # Save as GIF
            gif_filename = f"sample_step_{step}_idx_{sample_idx}.gif"
            gif_path = os.path.join(self.config.SAMPLES_DIR, gif_filename)
            imageio.mimsave(gif_path, video_frames, duration=0.1)
    
    def log_training_initialization(self):
        """
        Log initial training configuration and model setup
        """
        # Log configuration at the start of training
        self.tb_logger.log_configuration(self.config, epoch=0)
        
        # Log initial model architecture
        self.tb_logger.log_model_architecture(self.model, epoch=0)
        
        # Log initial system metrics
        self.tb_logger.log_system_metrics(epoch=0)
        
        logger.debug("Training initialization logged to TensorBoard")
    
    def create_training_summary(self, epoch: int):
        """
        Create a summary of training progress for TensorBoard
        
        Args:
            epoch (int): Current epoch number
        """
        # Create text summary of current training state
        summary_text = f"""
        Epoch: {epoch}
        Global Step: {self.global_step}
        Current Learning Rate: {(self.optimizer.param_groups[0]['lr']):.6f}
        Device: {self.device}
        Batch Size: {self.config.BATCH_SIZE}
        Effective Batch Size: {self.get_effective_batch_size()}
        Architecture: {self.config.MODEL_ARCHITECTURE}
        Scheduler: {getattr(self.config, 'SCHEDULER_TYPE', 'None')}
        """
        self.writer.add_text('training_summary', summary_text, epoch)
        # Log training curves as custom plots (if we have enough data points)
        if epoch >= 5:
            # This would create comparison plots, but TensorBoard scalars already provide this
            pass
        self.writer.flush()
    
    def log_lr_schedule(self):
        """
        Log the complete learning rate schedule to TensorBoard
        """
        if self.scheduler is None:
            return
            
        # Generate learning rate values for the entire training
        lr_values = []
        steps = []
        
        # Save current state
        current_lr = self.optimizer.param_groups[0]['lr']
        current_step = self.global_step
        
        # Simulate the learning rate schedule
        for step in range(self.config.NUM_EPOCHS * len(self.dataloader)):
            lr_at_step = self.scheduler.get_last_lr()[0]
            lr_values.append(lr_at_step)
            steps.append(step)
            
            # Step the scheduler
            self.scheduler.step()
        
        # Restore original state
        self.optimizer.param_groups[0]['lr'] = current_lr
        self.global_step = current_step
        
        # Log the learning rate schedule
        for step, lr in zip(steps, lr_values):
            self.writer.add_scalar('lr_schedule/full_schedule', lr, step)
        
        self.writer.flush()
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            dict: Training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_grad_norm = 0.0
        num_optimizer_steps = 0  # Track actual optimizer steps for gradient norm averaging
        
        # Create progress bar for training batches
        current_lr = self.optimizer.param_groups[0]['lr']
        progress_bar = tqdm(
            self.dataloader, 
            desc=f"Epoch {self.epoch}/{self.config.NUM_EPOCHS} | LR: {current_lr:.2e}", 
            leave=False,
            unit="batch"
        )
        
        for batch_idx, (videos, texts) in enumerate(progress_bar):
            step_start_time = time.time()  # Track step timing
            
            # Move video data to device
            videos = videos.to(self.device)
            # Sanity check: input videos should be normalized to [-1,1]
            vmin, vmax = videos.min().item(), videos.max().item()
            assert -1.0001 <= vmin <= 1.0001 and -1.0001 <= vmax <= 1.0001, \
                f"Input videos not normalized to [-1,1]: min={vmin}, max={vmax}"
            # Texts is a list, do not move to device
            
            # Zero gradients only at the start of accumulation cycle
            if self.accumulation_step == 0:
                self.optimizer.zero_grad()
            
            loss, predicted_noise, noise = self.model(videos, texts)
            
            # Scale loss by accumulation steps for correct gradients
            scaled_loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
            
            # Reduced diagnostic logging - only log critical issues
            if self.global_step % (self.config.DIAGNOSTIC_LOG_EVERY_STEPS * 5) == 0:
                with torch.no_grad():
                    # Check for critical issues only
                    pred_noise_abs_max = predicted_noise.abs().max().item()
                    actual_noise_abs_max = noise.abs().max().item()
                    
                    # Only log if there's a potential issue
                    if pred_noise_abs_max > 10.0 or actual_noise_abs_max > 10.0:
                        print(f"⚠️ [Step {self.global_step}] High noise values detected - pred_max: {pred_noise_abs_max:.2f}, actual_max: {actual_noise_abs_max:.2f}")
                    
                    # Essential TensorBoard logging only
                    self.writer.add_scalar('debug/pred_noise_max', pred_noise_abs_max, self.global_step)
                    self.writer.add_scalar('debug/actual_noise_max', actual_noise_abs_max, self.global_step)
            # Backward pass with scaled loss
            scaled_loss.backward()
            
            # Increment accumulation step
            self.accumulation_step += 1
            
            # Only perform optimizer step when accumulation is complete
            if self.accumulation_step == self.config.GRADIENT_ACCUMULATION_STEPS:
                # Gradient clipping (applied to accumulated gradients)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                
                # Check for NaN gradients
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"NaN/Inf gradient norm detected at step {self.global_step}, zeroing gradients")
                    self.optimizer.zero_grad()
                    self.accumulation_step = 0  # Reset accumulation
                    continue
                
                # Optimizer step (applies accumulated gradients)
                self.optimizer.step()
                
                # Update learning rate
                if self.scheduler is not None:
                    # Handle different scheduler types
                    from torch.optim.lr_scheduler import ReduceLROnPlateau
                    from schedulers.lr_schedulers import AdaptiveLRScheduler
                    
                    if isinstance(self.scheduler, (ReduceLROnPlateau, AdaptiveLRScheduler)):
                        # These schedulers need loss values, we'll step them at epoch end
                        pass
                    else:
                        # Step-based schedulers
                        self.scheduler.step()
                
                # Reset accumulation counter
                self.accumulation_step = 0
                num_optimizer_steps += 1  # Count actual optimizer steps
                
                # Log gradient norm (only when we actually step)
                epoch_grad_norm += grad_norm.item()
            else:
                # For incomplete accumulation, set grad_norm to 0 for logging
                grad_norm = torch.tensor(0.0)
            
            # Accumulate metrics (always accumulate loss, but only accumulate grad_norm when stepping)
            epoch_loss += loss.item()  # Use original unscaled loss for logging
            num_batches += 1
            
            # Update progress bar with accumulation info
            effective_batch_size = self.config.BATCH_SIZE * self.config.GRADIENT_ACCUMULATION_STEPS
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss/num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                'acc': f"{self.accumulation_step}/{self.config.GRADIENT_ACCUMULATION_STEPS}",
                'eff_bs': effective_batch_size
            })
            
            # Basic scalar logging every few steps for immediate feedback
            if self.global_step % 5 == 0:  # Log basic metrics every 5 steps
                running_avg_loss = epoch_loss / num_batches if num_batches > 0 else loss.item()
                
                # Calculate basic loss components for logging
                with torch.no_grad():
                    basic_noise_mse = F.mse_loss(predicted_noise, noise, reduction='mean')
                
                self.tb_logger.log_training_step({
                    'loss': running_avg_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    'loss_components': {
                        'total_loss': running_avg_loss,
                        'noise_mse': basic_noise_mse.item()
                    }
                }, self.global_step)
            
            # Comprehensive logging using new TensorBoard logger (more frequent)
            if self.global_step % self.config.DIAGNOSTIC_LOG_EVERY_STEPS == 0:
                # Prepare comprehensive metrics
                running_avg_loss = epoch_loss / num_batches if num_batches > 0 else loss.item()
                
                # Calculate additional diffusion metrics
                with torch.no_grad():
                    noise_mse = F.mse_loss(predicted_noise, noise, reduction='none').mean()
                    noise_mae = F.l1_loss(predicted_noise, noise, reduction='none').mean()
                    
                    # Signal-to-noise ratio
                    signal_power = videos.pow(2).mean()
                    noise_power = noise.pow(2).mean()
                    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
                
                # Training metrics
                training_metrics = {
                    'loss': running_avg_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'grad_norm': grad_norm.item(),
                    'batch_size': effective_batch_size,
                    'loss_components': {
                        'total_loss': running_avg_loss,
                        'noise_mse': noise_mse.item(),
                        'noise_mae': noise_mae.item()
                    }
                }
                
                # Diffusion-specific metrics
                # Create representative timesteps for logging (since we don't have access to actual timesteps)
                dummy_timesteps = torch.randint(0, self.config.TIMESTEPS, (videos.size(0),), device=videos.device)
                
                diffusion_metrics = {
                    'noise_mse': noise_mse.item(),
                    'noise_mae': noise_mae.item(),
                    'snr': snr.item(),
                    'timestep_distribution': dummy_timesteps.cpu().numpy()
                }
                
                # Log using new structured system
                self.tb_logger.log_training_step(training_metrics, self.global_step)
                self.tb_logger.log_diffusion_metrics(diffusion_metrics, self.global_step)
                self.tb_logger.log_noise_statistics(predicted_noise, noise, videos, dummy_timesteps, self.global_step)
                self.tb_logger.log_gradient_statistics(self.model, self.global_step)
                
                # Add step-level performance metrics
                step_time = time.time() - step_start_time
                step_performance = {
                    'step_time': step_time,
                    'samples_per_second': effective_batch_size / step_time if step_time > 0 else 0.0,
                }
                
                # Add memory usage if available
                if self.device.type == 'cuda':
                    memory_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    step_performance['memory_usage'] = memory_usage
                elif self.device.type == 'mps':
                    try:
                        memory_usage = torch.mps.current_allocated_memory() / 1024 / 1024  # MB
                        step_performance['memory_usage'] = memory_usage
                    except:
                        pass
                
                self.tb_logger.log_step_performance(step_performance, self.global_step)
                
                # Update progress bar with enhanced info
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_description(
                    f"Epoch {self.epoch} [Step {self.global_step}] | LR: {current_lr:.2e} - "
                    f"Loss: {loss.item():.4f}, MSE: {noise_mse.item():.4f}, SNR: {snr.item():.1f}dB"
                )
    

            # Generate and log samples (epoch-based frequency)
            if self.global_step % self.sample_every_steps == 0 and self.global_step > 0:
                progress_bar.set_description(f"Epoch {self.epoch} | LR: {current_lr:.2e} - Generating samples...")
                samples = self.generate_samples()
                self.save_samples_as_gifs(samples, self.global_step)
                
                # Log generated samples using new structured logging
                self.tb_logger.log_generated_samples(samples, self.global_step, "Training_Samples")
                progress_bar.set_description(f"Epoch {self.epoch} | LR: {current_lr:.2e}")
            
            # Save noise display at configured intervals (step-based)
            if self.global_step % self.config.NOISE_DISPLAY_EVERY_STEPS == 0:
                try:
                    # Normalize noise to [-1, 1] range for visualization
                    # Clamp to reasonable range (±3 std) then normalize
                    pred_noise_viz = torch.clamp(predicted_noise[0], -3, 3) / 3.0
                    real_noise_viz = torch.clamp(noise[0], -3, 3) / 3.0
                    x_0_viz = torch.clamp(videos[0], -1, 1) / 1.0  # Original video for reference
                    
                    # Save as GIFs (keep existing functionality)
                    gif_utils.save_video_as_gif(pred_noise_viz, f'./noise_display/pred_noise_{self.global_step}.gif')
                    gif_utils.save_video_as_gif(real_noise_viz, f'./noise_display/real_noise_{self.global_step}.gif')
                    gif_utils.save_video_as_gif(x_0_viz, f'./noise_display/original_video_{self.global_step}.gif')
                    
                    # Log using new structured noise visualization
                    self.tb_logger.log_noise_visualization(
                        predicted_noise, noise, videos, self.global_step
                    )
                except Exception as e:
                    logger.debug(f"Failed to save noise display GIFs: {e}")
                    # Traceback only in debug mode to avoid console clutter
            
            # Save checkpoint (epoch-based frequency)
            if self.global_step % self.save_every_steps == 0 and self.global_step > 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_description(f"Epoch {self.epoch} | LR: {current_lr:.2e} - Saving checkpoint...")
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
                progress_bar.set_description(f"Epoch {self.epoch} | LR: {current_lr:.2e}")
            
            # Flush TensorBoard at configured intervals (step-based)
            if self.global_step % self.config.TENSORBOARD_FLUSH_EVERY_STEPS == 0:
                self.writer.flush()
            
            self.global_step += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = epoch_grad_norm / num_optimizer_steps if num_optimizer_steps > 0 else 0.0
        
        return {'loss': avg_loss, 'grad_norm': avg_grad_norm, 'optimizer_steps': num_optimizer_steps}
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting training...")
        logger.info(f"📊 Config: {self.config.NUM_EPOCHS} epochs, batch size {self.config.BATCH_SIZE} "
                   f"(effective: {self.config.BATCH_SIZE * self.config.GRADIENT_ACCUMULATION_STEPS}), "
                   f"device: {self.device}")
        
        # Initialize comprehensive logging
        self.log_training_initialization()
        config_text = "\n".join([f"{k}: {v}" for k, v in self.config.__dict__.items() if not k.startswith('_')])
        self.writer.add_text('config', config_text, 0)
        self.writer.flush()  # Flush configuration immediately
        
        # Log learning rate schedule
        self.log_lr_schedule()
        
        try:
            # Create progress bar for epochs
            epoch_progress = tqdm(
                range(self.epoch, self.config.NUM_EPOCHS),
                desc="Training Progress",
                unit="epoch",
                initial=self.epoch,
                total=self.config.NUM_EPOCHS,
                ncols=100  # Set width for better display
            )
            
            for epoch in epoch_progress:
                self.epoch = epoch
                start_time = time.time()
                
                # Update epoch progress description
                epoch_progress.set_description(f"Training - Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
                
                # Train for one epoch
                metrics = self.train_epoch()
                
                epoch_time = time.time() - start_time
                
                # Calculate performance metrics
                samples_processed = len(self.dataloader) * self.config.BATCH_SIZE
                samples_per_second = samples_processed / epoch_time if epoch_time > 0 else 0
                steps_per_second = len(self.dataloader) / epoch_time if epoch_time > 0 else 0
                
                # Update epoch progress with metrics
                current_lr = self.optimizer.param_groups[0]['lr']
                epoch_progress.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'time': f"{epoch_time:.1f}s"
                })
                
                # Prepare comprehensive epoch metrics
                epoch_summary = {
                    'loss': metrics['loss'],
                    'time': epoch_time,
                    'grad_norm': metrics['grad_norm'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'optimizer_steps': metrics['optimizer_steps'],
                    'samples_per_second': samples_per_second,
                    'steps_per_second': steps_per_second
                }
                
                # Log comprehensive epoch summary using new structured logging
                self.tb_logger.log_epoch_summary(epoch_summary, epoch)
                self.tb_logger.log_model_architecture(self.model, epoch)
                self.tb_logger.log_system_metrics(epoch)
                
                # Update learning rate scheduler at epoch end
                if self.scheduler is not None:
                    from torch.optim.lr_scheduler import ReduceLROnPlateau
                    from schedulers.lr_schedulers import AdaptiveLRScheduler
                    
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(metrics['loss'])
                    elif isinstance(self.scheduler, AdaptiveLRScheduler):
                        current_lr = self.scheduler.step(metrics['loss'])
                    else:
                        # For step-based schedulers, step once per epoch
                        if not hasattr(self, '_scheduler_stepped_this_epoch'):
                            self.scheduler.step()
                            self._scheduler_stepped_this_epoch = True
                    
                    # Log learning rate schedule using new structured logging
                    self.tb_logger.log_learning_rate_schedule(self.scheduler, self.optimizer, self.config, epoch)
                
                # Track learning rate and loss history
                self.lr_history.append(self.optimizer.param_groups[0]['lr'])
                self.loss_history.append(metrics['loss'])
                
                # Reset scheduler step flag for next epoch
                self._scheduler_stepped_this_epoch = False
                
                # Generate and log samples at the end of each epoch
                samples = self.generate_samples()
                self.save_samples_as_gifs(samples, self.global_step)
                
                # Log epoch-end samples using structured logging
                self.tb_logger.log_generated_samples(samples, epoch, f"Epoch_{epoch}_Samples")
                
                # Log training throughput
                samples_processed = len(self.dataloader) * self.config.BATCH_SIZE
                throughput = samples_processed / epoch_time  # samples per second
                self.writer.add_scalar('epoch/throughput_samples_per_sec', throughput, epoch)
                self.writer.add_scalar('epoch/samples_processed', samples_processed, epoch)
                
                # Log model parameter histograms at configured intervals
                if epoch % self.config.PARAM_LOG_EVERY_EPOCHS == 0:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(f'parameters/{name}', param.data, epoch)
                            self.writer.add_histogram(f'gradients/{name}', param.grad.data, epoch)
                
                # Log training statistics
                self.writer.add_scalar('stats/epoch', epoch, epoch)
                self.writer.add_scalar('stats/global_step', self.global_step, epoch)
                
                self.writer.flush()  # Flush epoch metrics immediately
                
                # Log comprehensive training summary periodically
                if epoch % self.config.SUMMARY_LOG_EVERY_EPOCHS == 0:
                    self.create_training_summary(epoch)
                
                # Flush TensorBoard logs periodically
                if self.global_step % self.config.TENSORBOARD_FLUSH_EVERY_STEPS == 0:
                    self.tb_logger.flush()
                
                # Update epoch progress bar with completion info
                epoch_progress.set_postfix({
                    'avg_loss': f"{metrics['loss']:.4f}",
                    'time': f"{epoch_time:.1f}s",
                    'step': self.global_step
                })
                
                # Save epoch checkpoint at configured intervals
                if epoch % self.config.SAVE_EVERY_EPOCHS == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
                    logger.debug(f"Saved checkpoint for epoch {epoch}")
                
                # Always save latest checkpoint
                self.save_checkpoint('latest_checkpoint.pt')
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint('interrupted_checkpoint.pt')
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            self.save_checkpoint('error_checkpoint.pt')
            raise
        
        finally:
            # Close comprehensive logging system
            self.tb_logger.close()
            logger.info("Training completed")
            
            # Print final logging summary in debug mode only
            if logger.isEnabledFor(logging.DEBUG):
                logging_summary = self.tb_logger.get_logging_summary()
                logger.debug(f"Logging Summary: {logging_summary}")

def setup_training(config) -> Trainer:
    """
    Setup training components
    
    Args:
        config: Configuration object
        
    Returns:
        Trainer: Configured trainer
    """
    # Print configuration
    config.print_config()
    
    # Create dataloader with progress indication
    print("📊 Setting up data loader...")
    dataloader = create_dataloader(
        data_root=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        num_frames=config.NUM_FRAMES
    )
    
    # Create model with progress indication
    model = create_diffusion_model(config)
    model.to(config.DEVICE)
    
    # Count parameters
    from models import count_parameters
    num_params = count_parameters(model)
    print(f"✅ Setup complete: {len(dataloader)} batches, {num_params:,} parameters on {config.DEVICE}")
    
    # Create optimizer
    print("⚙️  Setting up optimizer...")
    lr = config.get_learning_rate()  # Get architecture-specific learning rate
    if config.OPTIMIZER_TYPE == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.ADAM_BETAS
        )
        print(f"✅ AdamW optimizer ready - LR: {lr}, Weight Decay: {config.WEIGHT_DECAY}")
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=config.ADAM_BETAS
        )
        print(f"✅ Adam optimizer ready - LR: {lr}")
    
    # Create learning rate scheduler
    print("📈 Setting up learning rate scheduler...")
    scheduler = create_lr_scheduler(optimizer, config)
    if scheduler is not None:
        scheduler_type = getattr(config, 'SCHEDULER_TYPE', 'none')
        print(f"✅ {scheduler_type} scheduler created")
    else:
        print("✅ No scheduler (constant learning rate)")
    
    # Create trainer
    print("🚀 Creating trainer...")
    trainer = Trainer(
        config=config,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    print("✅ Training setup complete!")
    
    return trainer

if __name__ == "__main__":
    # Setup and start training
    trainer = setup_training(Config)
    trainer.train()
