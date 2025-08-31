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
        torch.manual_seed(42)
        np.random.seed(42)
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
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Log model structure
        self.log_model_structure()
        
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
        
        if not os.path.exists(filepath):
            logger.warning(f"Checkpoint not found: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        # Optionally override the resumed optimizer learning rate
        # By default we keep the learning rate saved in the checkpoint. Passing
        # a numeric value to `override_lr` will set that LR across all param groups
        # after the optimizer state is restored.
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
            logger.info(f"🎬 Generating {num_samples} video samples...")
            
            # Use AMP for inference if available and enabled
            if self.use_amp and self.scaler is not None:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    samples = self.model.p_sample(shape)
            else:
                samples = self.model.p_sample(shape)
                
            # Note: For ε-parameterization, do NOT clamp samples during generation
            # This allows the model to learn the proper data distribution naturally
            logger.info(f"✅ Generated {num_samples} samples successfully")
        self.model.train()
        
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
            # Assert sample properties
            assert sample.shape[0] == 3, f"Expected 3 channels, got {sample.shape[0]}"
            assert sample.dtype == np.float32, f"Expected float32, got {sample.dtype}"
            smin, smax = sample.min(), sample.max()
            assert -1.0001 <= smin <= 1.0001 and -1.0001 <= smax <= 1.0001, \
                f"Sample not in [-1,1]: min={smin}, max={smax}"
            
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
        # Log model parameter statistics
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                # Log parameter statistics
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                self.writer.add_scalar(f'param_stats/{name}_mean', param_mean, self.epoch)
                self.writer.add_scalar(f'param_stats/{name}_std', param_std, self.epoch)
            total_params += param.numel()

        # Log training configuration stats
        self.writer.add_scalar('config/total_parameters', total_params, self.epoch)
        self.writer.add_scalar('config/trainable_parameters', trainable_params, self.epoch)
        self.writer.add_scalar('config/batch_size', self.config.BATCH_SIZE, self.epoch)
        self.writer.add_scalar('config/learning_rate', self.config.LEARNING_RATE, self.epoch)

        # Log memory usage if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            self.writer.add_scalar('memory/allocated_gb', memory_allocated, self.epoch)
            self.writer.add_scalar('memory/reserved_gb', memory_reserved, self.epoch)

        self.writer.flush()
    
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
        
        # Create progress bar for training batches
        progress_bar = tqdm(
            self.dataloader, 
            desc=f"Epoch {self.epoch}/{self.config.NUM_EPOCHS}", 
            leave=False,
            unit="batch"
        )
        
        for batch_idx, (videos, texts) in enumerate(progress_bar):
            # Move video data to device
            videos = videos.to(self.device)
            # Sanity check: input videos should be normalized to [-1,1]
            vmin, vmax = videos.min().item(), videos.max().item()
            assert -1.0001 <= vmin <= 1.0001 and -1.0001 <= vmax <= 1.0001, \
                f"Input videos not normalized to [-1,1]: min={vmin}, max={vmax}"
            # Texts is a list, do not move to device
            # Zero gradients
            self.optimizer.zero_grad()
            
            loss, predicted_noise, noise = self.model(videos,texts)
            
            # Diagnostic logging for noise statistics (should be mean≈0, std≈1)
            if self.global_step % 100 == 0:  # Log every 100 steps
                with torch.no_grad():
                    # Check input data range
                    video_min, video_max = videos.min().item(), videos.max().item()
                    video_mean, video_std = videos.mean().item(), videos.std().item()
                    
                    # Check noise statistics
                    noise_mean, noise_std = noise.mean().item(), noise.std().item()
                    pred_noise_mean, pred_noise_std = predicted_noise.mean().item(), predicted_noise.std().item()
                    
                    # CRITICAL: Check if model is predicting reasonable noise scale
                    pred_noise_abs_max = predicted_noise.abs().max().item()
                    actual_noise_abs_max = noise.abs().max().item()
                    
                    # Log to console and tensorboard
                    print(f"[Step {self.global_step}] Diagnostics:")
                    print(f"  Video: range=[{video_min:.3f}, {video_max:.3f}], mean={video_mean:.3f}, std={video_std:.3f}")
                    print(f"  Real noise: mean={noise_mean:.3f}, std={noise_std:.3f}, abs_max={actual_noise_abs_max:.3f}")
                    print(f"  Pred noise: mean={pred_noise_mean:.3f}, std={pred_noise_std:.3f}, abs_max={pred_noise_abs_max:.3f}")
                    print(f"  Loss: {loss.item():.6f}")
                    
                    # TensorBoard logging
                    self.writer.add_scalar('debug/video_range_min', video_min, self.global_step)
                    self.writer.add_scalar('debug/video_range_max', video_max, self.global_step)
                    self.writer.add_scalar('debug/video_mean', video_mean, self.global_step)
                    self.writer.add_scalar('debug/video_std', video_std, self.global_step)
                    self.writer.add_scalar('debug/noise_mean', noise_mean, self.global_step)
                    self.writer.add_scalar('debug/noise_std', noise_std, self.global_step)
                    self.writer.add_scalar('debug/pred_noise_mean', pred_noise_mean, self.global_step)
                    self.writer.add_scalar('debug/pred_noise_std', pred_noise_std, self.global_step)
               
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
            
            # Check for NaN gradients
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"NaN/Inf gradient norm detected at step {self.global_step}, zeroing gradients")
                self.optimizer.zero_grad()
                continue
            
            # Optimizer step
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_grad_norm += grad_norm.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss/num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Logging (now less frequent - once per epoch via config)
            if self.global_step % self.config.LOG_EVERY == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('train/grad_norm', grad_norm.item(), self.global_step)
                
                # Critical ε-parameterization diagnostics
                with torch.no_grad():
                    # Noise prediction quality metrics
                    noise_mse = F.mse_loss(predicted_noise, noise, reduction='none').mean()
                    self.writer.add_scalar('debug/noise_prediction_mse', noise_mse.item(), self.global_step)
                    
                    # Model output statistics (should be unbiased for ε-parameterization)
                    self.writer.add_scalar('debug/predicted_noise_abs_mean', predicted_noise.abs().mean().item(), self.global_step)
                    self.writer.add_scalar('debug/actual_noise_abs_mean', noise.abs().mean().item(), self.global_step)

                # Update progress bar with detailed step info
                progress_bar.set_description(
                    f"Epoch {self.epoch} [Step {self.global_step}] - "
                    f"Loss: {loss.item():.4f}, Noise MSE: {noise_mse.item():.4f}"
                )
    

            # Generate and log samples (now less frequent - once per epoch via config)
            if self.global_step % self.config.SAMPLE_EVERY == 0 and self.global_step > 0:
                progress_bar.set_description(f"Epoch {self.epoch} - Generating samples...")
                samples = self.generate_samples()
                self.save_samples_as_gifs(samples, self.global_step)
                progress_bar.set_description(f"Epoch {self.epoch}")
            
            # Save noise display every 100 steps
            if self.global_step % 100 == 0:
                try:
                    # Debug: Check tensor shapes and values
                    print(f"Predicted noise shape: {predicted_noise[0].shape}, range: [{predicted_noise[0].min():.3f}, {predicted_noise[0].max():.3f}]")
                    print(f"Real noise shape: {noise[0].shape}, range: [{noise[0].min():.3f}, {noise[0].max():.3f}]")
                    
                    # Normalize noise to [-1, 1] range for visualization
                    # Clamp to reasonable range (±3 std) then normalize
                    pred_noise_viz = torch.clamp(predicted_noise[0], -3, 3) / 3.0
                    real_noise_viz = torch.clamp(noise[0], -3, 3) / 3.0
                    x_0_viz = torch.clamp(videos[0], -1, 1) / 1.0  # Original video for reference
                    
                    gif_utils.save_video_as_gif(pred_noise_viz, f'./noise_display/pred_noise_{self.global_step}.gif')
                    gif_utils.save_video_as_gif(real_noise_viz, f'./noise_display/real_noise_{self.global_step}.gif')
                    gif_utils.save_video_as_gif(x_0_viz, f'./noise_display/original_video_{self.global_step}.gif')
                    print(f"✅ Saved noise display GIFs at step {self.global_step}")
                except Exception as e:
                    logger.warning(f"Failed to save noise display GIFs: {e}")
                    print(f"❌ Noise display error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Save checkpoint (now less frequent - every two epochs via config)
            if self.global_step % self.config.SAVE_EVERY == 0 and self.global_step > 0:
                progress_bar.set_description(f"Epoch {self.epoch} - Saving checkpoint...")
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
                progress_bar.set_description(f"Epoch {self.epoch}")
            
            # Flush TensorBoard less frequently now (every 100 steps instead of 3)
            if self.global_step % 100 == 0:
                self.writer.flush()
            
            self.global_step += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = epoch_grad_norm / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss, 'grad_norm': avg_grad_norm}
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total epochs: {self.config.NUM_EPOCHS}")
        logger.info(f"Batch size: {self.config.BATCH_SIZE}")
        logger.info(f"Logging frequency: every {self.config.LOG_EVERY} steps (~1 per epoch)")
        logger.info(f"Sampling frequency: every {self.config.SAMPLE_EVERY} steps (~1 per epoch)")
        logger.info(f"Checkpoint frequency: every {self.config.SAVE_EVERY} steps (~every 2 epochs)")
        
        # Log configuration
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
                
                # Update epoch progress with metrics
                epoch_progress.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'time': f"{epoch_time:.1f}s"
                })
                
                # Log comprehensive epoch metrics
                self.writer.add_scalar('epoch/loss', metrics['loss'], epoch)
                self.writer.add_scalar('epoch/time', epoch_time, epoch)
                self.writer.add_scalar('epoch/grad_norm', metrics['grad_norm'], epoch)
                self.writer.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                # Also log step-based metrics at epoch end for consistency
                self.writer.add_scalar('train/loss', metrics['loss'], self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('train/grad_norm', metrics['grad_norm'], self.global_step)
                
                # Generate and log samples at the end of each epoch
                logger.info(f"🎬 Generating samples for epoch {epoch}...")
                samples = self.generate_samples()
                self.save_samples_as_gifs(samples, self.global_step)
                
                # Log training throughput
                samples_processed = len(self.dataloader) * self.config.BATCH_SIZE
                throughput = samples_processed / epoch_time  # samples per second
                self.writer.add_scalar('epoch/throughput_samples_per_sec', throughput, epoch)
                self.writer.add_scalar('epoch/samples_processed', samples_processed, epoch)
                
                # Log model parameter histograms (every 10 epochs to reduce overhead)
                if epoch % 10 == 0:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(f'parameters/{name}', param.data, epoch)
                            self.writer.add_histogram(f'gradients/{name}', param.grad.data, epoch)
                
                # Log training statistics
                self.writer.add_scalar('stats/epoch', epoch, epoch)
                self.writer.add_scalar('stats/global_step', self.global_step, epoch)
                
                self.writer.flush()  # Flush epoch metrics immediately
                
                # Log comprehensive training statistics (every 5 epochs)
                if epoch % 5 == 0:
                    self.create_training_summary(epoch)
                
                # Update epoch progress bar with completion info
                epoch_progress.set_postfix({
                    'avg_loss': f"{metrics['loss']:.4f}",
                    'time': f"{epoch_time:.1f}s",
                    'step': self.global_step
                })
                
                # Save epoch checkpoint every two epochs
                if epoch % 2 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
                    logger.info(f"💾 Saved checkpoint for epoch {epoch}")
                
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
            self.writer.close()
            logger.info("Training completed")

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
        shuffle=True
    )
    print(f"✅ Data loader ready: {len(dataloader)} batches")
    
    # Create model with progress indication
    print("🤖 Creating diffusion model...")
    model = create_diffusion_model(config)
    model.to(config.DEVICE)
    print(f"✅ Model created and moved to {config.DEVICE}")
    
    # Count parameters
    from models import count_parameters
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
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
    
    # No learning rate scheduler - using constant learning rate
    scheduler = None
    print("✅ Optimizer ready (constant learning rate)")
    
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
