"""
Training utilities and functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
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
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Log AMP status
        if getattr(config, 'USE_AMP', False) and not torch.cuda.is_available():
            logger.info("AMP requested but CUDA not available - disabling AMP")
        
        # AMP dtype
        self.amp_dtype = getattr(config, 'AMP_DTYPE', torch.float16)
        
        # Create directories
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.SAMPLES_DIR, exist_ok=True)  # Create samples directory
        
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
    
    def load_checkpoint(self, filename: str):
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
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
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
                with autocast('cuda'):
                    samples = self.model.p_sample(shape)
            else:
                samples = self.model.p_sample(shape)
                
            # Clamp to [0, 1] range
            samples = torch.clamp(samples, 0, 1)
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
            
            # Convert from CHW to HWC format and scale to [0, 255]
            video_frames = []
            for frame_idx in range(frames):
                frame = sample[:, frame_idx]  # (channels, height, width)
                frame = np.transpose(frame, (1, 2, 0))  # Convert to (height, width, channels)
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255]
                video_frames.append(frame)
            
            # Save as GIF
            gif_filename = f"sample_step_{step}_idx_{sample_idx}.gif"
            gif_path = os.path.join(self.config.SAMPLES_DIR, gif_filename)
            
            # Save with imageio
            imageio.mimsave(gif_path, video_frames, fps=8, loop=0)
            logger.info(f"💾 Saved GIF: {gif_path}")
    
    def log_samples(self, samples: torch.Tensor, step: int):
        """
        Log samples to tensorboard and save as GIF files
        
        Args:
            samples (torch.Tensor): Generated samples
            step (int): Current training step
        """
        # Save samples as GIF files first
        self.save_samples_as_gifs(samples, step)
        
        # Log the first few frames of each sample as images to tensorboard
        batch_size, channels, frames, height, width = samples.shape
        
        # Take every 4th frame to show progression
        frame_indices = list(range(0, frames, max(1, frames // 7)))[:7]
        
        for sample_idx in range(min(batch_size, 4)):  # Log up to 4 samples
            sample = samples[sample_idx]  # (channels, frames, height, width)
            
            # Create a grid of frames
            frame_grid = []
            for frame_idx in frame_indices:
                frame = sample[:, frame_idx]  # (channels, height, width)
                # Convert to HWC format for tensorboard
                frame = frame.permute(1, 2, 0)  # (height, width, channels)
                frame_grid.append(frame)
            
            # Concatenate frames horizontally
            if frame_grid:
                frames_concat = torch.cat(frame_grid, dim=1)  # (height, width*num_frames, channels)
                # Log to tensorboard
                self.writer.add_image(
                    f'generated_samples/sample_{sample_idx}',
                    frames_concat,
                    step,
                    dataformats='HWC'
                )
        
        # Flush after logging all samples
        self.writer.flush()
    
    def log_training_stats(self, epoch: int):
        """
        Log comprehensive training statistics to TensorBoard
        
        Args:
            epoch (int): Current epoch number
        """
        # Log model parameter statistics
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                # Log parameter statistics
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                self.writer.add_scalar(f'param_stats/{name}_mean', param_mean, epoch)
                self.writer.add_scalar(f'param_stats/{name}_std', param_std, epoch)
            total_params += param.numel()
        
        # Log training configuration stats
        self.writer.add_scalar('config/total_parameters', total_params, epoch)
        self.writer.add_scalar('config/trainable_parameters', trainable_params, epoch)
        self.writer.add_scalar('config/batch_size', self.config.BATCH_SIZE, epoch)
        self.writer.add_scalar('config/learning_rate', self.config.LEARNING_RATE, epoch)
        
        # Log memory usage if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            self.writer.add_scalar('memory/allocated_gb', memory_allocated, epoch)
            self.writer.add_scalar('memory/reserved_gb', memory_reserved, epoch)
        
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
        Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}
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
            unit="batch",
            ncols=120  # Wider display for more info
        )
        
        for batch_idx, (videos, texts) in enumerate(progress_bar):
            # Move data to device
            videos = videos.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            if self.use_amp and self.scaler is not None:
                with autocast('cuda'):
                    loss, predicted_noise, noise = self.model(videos)
                
                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf detected at step {self.global_step}")
                    print(f"Input videos shape: {videos.shape}")
                    print(f"Input videos range: [{videos.min():.3f}, {videos.max():.3f}]")
                    print(f"Input has NaN: {torch.isnan(videos).any()}")
                    print(f"Predicted noise has NaN: {torch.isnan(predicted_noise).any()}")
                    print(f"Noise has NaN: {torch.isnan(noise).any()}")
                    # Replace NaN loss with small constant to continue training
                    loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                
                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()
                
                # Gradient clipping with scaled gradients
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                
                # Check for NaN gradients
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"NaN/Inf gradient norm detected at step {self.global_step}, zeroing gradients")
                    self.optimizer.zero_grad()
                    # Still need to complete the scaler cycle
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    continue
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward pass (for MPS, CPU, or when AMP disabled)
                loss, predicted_noise, noise = self.model(videos)
                
                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf detected at step {self.global_step}")
                    print(f"Input videos shape: {videos.shape}")
                    print(f"Input videos range: [{videos.min():.3f}, {videos.max():.3f}]")
                    print(f"Input has NaN: {torch.isnan(videos).any()}")
                    print(f"Predicted noise has NaN: {torch.isnan(predicted_noise).any()}")
                    print(f"Noise has NaN: {torch.isnan(noise).any()}")
                    # Replace NaN loss with small constant to continue training
                    loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                
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
            
            # Logging
            if self.global_step % self.config.LOG_EVERY == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('train/grad_norm', grad_norm.item(), self.global_step)
                # Samples generated and logged to tensorboard
                # Update progress bar with detailed step info
                progress_bar.set_description(
                    f"Epoch {self.epoch} [Step {self.global_step}] - "
                    f"Loss: {loss.item():.4f}"
                )
            
            # Generate and log samples
            if self.global_step % self.config.SAMPLE_EVERY == 0 and self.global_step > 0:
                progress_bar.set_description(f"Epoch {self.epoch} - Generating samples...")
                samples = self.generate_samples()
                self.log_samples(samples, self.global_step)
                progress_bar.set_description(f"Epoch {self.epoch}")
            
            # Save checkpoint
            if self.global_step % self.config.SAVE_EVERY == 0 and self.global_step > 0:
                progress_bar.set_description(f"Epoch {self.epoch} - Saving checkpoint...")
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
                progress_bar.set_description(f"Epoch {self.epoch}")
            
            # Flush TensorBoard regularly for real-time updates (every 3 steps)
            if self.global_step % 3 == 0:
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
                    self.log_training_stats(epoch)
                    self.create_training_summary(epoch)
                
                # Update epoch progress bar with completion info
                epoch_progress.set_postfix({
                    'avg_loss': f"{metrics['loss']:.4f}",
                    'time': f"{epoch_time:.1f}s",
                    'step': self.global_step
                })
                
                # Save epoch checkpoint
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
                self.save_checkpoint('latest_checkpoint.pt')  # Always keep latest
        
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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4
    )
    
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
