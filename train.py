import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from src.models.pipeline import create_pipeline, Text2SignDiffusionPipeline
from src.data.dataset import create_dataloader
from src.models.text_encoder import SimpleTextEncoder


class Trainer:
    """
    Trainer class for the text-to-sign language diffusion model.
    """
    
    def __init__(
        self,
        pipeline: Text2SignDiffusionPipeline,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./checkpoints",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        project_name: str = "text2sign-diffusion"
    ):
        """
        Initialize the trainer.
        
        Args:
            pipeline: Diffusion pipeline
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            learning_rate: Learning rate
            device: Device to train on
            save_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            project_name: W&B project name
        """
        self.pipeline = pipeline
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard setup
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard_logs"))
            print(f"TensorBoard logs will be saved to: {self.save_dir / 'tensorboard_logs'}")
        else:
            self.tb_writer = None
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.pipeline.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # Tracking
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_train_loss = float('inf')
        self.loss_history = []
        self.lr_history = []
        
        # Checkpoint settings
        self.checkpoint_every_steps = 500  # Save checkpoint every N steps
        self.log_every_steps = 10  # Log metrics every N steps
        self.generate_samples_every_steps = 1000  # Generate samples every N steps
        self.detailed_log_every_steps = 100  # Detailed logging every N steps
        
        # Weights & Biases
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(project=project_name)
            wandb.watch(self.pipeline.unet)
        
        # Log network architecture and initial metrics to TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            self._log_model_graph()
            self._log_model_parameters()
            self._log_hyperparameters(learning_rate)
    
    def _log_model_graph(self):
        """Log the model architecture to TensorBoard."""
        try:
            # Create dummy input for graph logging (updated for new dimensions)
            dummy_video = torch.randn(1, 3, 28, 128, 128).to(self.device)
            dummy_timestep = torch.randint(0, 1000, (1,)).to(self.device)
            dummy_text_embeds = torch.randn(1, 256).to(self.device)  # Updated for smaller embedding
            
            # Log UNet architecture
            self.tb_writer.add_graph(
                self.pipeline.unet, 
                (dummy_video, dummy_timestep, dummy_text_embeds)
            )
            print("✅ Model architecture logged to TensorBoard")
        except Exception as e:
            print(f"⚠️ Could not log model graph to TensorBoard: {e}")
    
    def _log_model_parameters(self):
        """Log model parameter statistics to TensorBoard."""
        if not self.use_tensorboard or self.tb_writer is None:
            return
        
        try:
            # Count parameters
            unet_params = sum(p.numel() for p in self.pipeline.unet.parameters())
            text_params = sum(p.numel() for p in self.pipeline.text_encoder.parameters())
            total_params = unet_params + text_params
            
            # Log parameter counts
            self.tb_writer.add_scalar('Model/UNet_Parameters', unet_params, 0)
            self.tb_writer.add_scalar('Model/TextEncoder_Parameters', text_params, 0)
            self.tb_writer.add_scalar('Model/Total_Parameters', total_params, 0)
            
            # Log parameter distributions
            for name, param in self.pipeline.unet.named_parameters():
                if param.requires_grad:
                    self.tb_writer.add_histogram(f'UNet_Params/{name}', param, 0)
            
            print(f"✅ Model parameters logged: UNet={unet_params:,}, Text={text_params:,}, Total={total_params:,}")
        except Exception as e:
            print(f"⚠️ Could not log model parameters: {e}")
    
    def _log_hyperparameters(self, learning_rate: float):
        """Log hyperparameters to TensorBoard."""
        if not self.use_tensorboard or self.tb_writer is None:
            return
        
        try:
            # Create hyperparameter dict
            hparams = {
                'learning_rate': learning_rate,
                'batch_size': len(next(iter(self.train_dataloader))['videos']),
                'model_channels': getattr(self.pipeline.unet, 'model_channels', 'unknown'),
                'num_res_blocks': getattr(self.pipeline.unet, 'num_res_blocks', 'unknown'),
                'num_heads': getattr(self.pipeline.unet, 'num_heads', 'unknown'),
            }
            
            # Log hyperparameters
            self.tb_writer.add_hparams(
                hparams, 
                {'hparam/train_loss': 0.0}  # Will be updated during training
            )
            
            print("✅ Hyperparameters logged to TensorBoard")
        except Exception as e:
            print(f"⚠️ Could not log hyperparameters: {e}")
    
    def _log_memory_usage(self, step: int):
        """Log memory usage statistics."""
        if not self.use_tensorboard or self.tb_writer is None:
            return
        
        try:
            if torch.cuda.is_available():
                # CUDA memory
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                self.tb_writer.add_scalar('Memory/CUDA_Allocated_GB', allocated, step)
                self.tb_writer.add_scalar('Memory/CUDA_Reserved_GB', reserved, step)
            elif torch.backends.mps.is_available():
                # MPS memory (approximation)
                allocated = torch.mps.current_allocated_memory() / 1024**3  # GB
                self.tb_writer.add_scalar('Memory/MPS_Allocated_GB', allocated, step)
        except Exception as e:
            pass  # Memory logging is optional
    
    def _log_samples_to_tensorboard(self, videos: torch.Tensor, texts: List[str], tag: str, step: int):
        """Log video samples to TensorBoard."""
        if not self.use_tensorboard or self.tb_writer is None:
            return
        
        try:
            # Take first few samples from batch
            num_samples = min(4, videos.size(0))
            sample_videos = videos[:num_samples]
            sample_texts = texts[:num_samples]
            
            # Convert videos to format suitable for TensorBoard
            # Expected shape: (N, T, C, H, W) -> (N, C, T, H, W)
            if sample_videos.dim() == 5:
                sample_videos = sample_videos.permute(0, 2, 1, 3, 4)
            
            # Normalize to [0, 1] range
            sample_videos = torch.clamp(sample_videos, 0, 1)
            
            # Log videos
            for i, (video, text) in enumerate(zip(sample_videos, sample_texts)):
                self.tb_writer.add_video(
                    f"{tag}/sample_{i}", 
                    video.unsqueeze(0),  # Add batch dimension
                    global_step=step,
                    fps=5
                )
                self.tb_writer.add_text(
                    f"{tag}/text_{i}", 
                    text, 
                    global_step=step
                )
        except Exception as e:
            print(f"⚠️ Could not log samples to TensorBoard: {e}")
    
    def _log_generated_samples_to_tensorboard(self, generated_videos: torch.Tensor, prompts: List[str], step: int):
        """Log generated samples to TensorBoard."""
        if not self.use_tensorboard or self.tb_writer is None:
            return
        
        try:
            # Convert generated videos for logging
            if generated_videos.dim() == 5:
                generated_videos = generated_videos.permute(0, 2, 1, 3, 4)
            
            generated_videos = torch.clamp(generated_videos, 0, 1)
            
            for i, (video, prompt) in enumerate(zip(generated_videos, prompts)):
                self.tb_writer.add_video(
                    f"generated/sample_{i}", 
                    video.unsqueeze(0),
                    global_step=step,
                    fps=5
                )
                self.tb_writer.add_text(
                    f"generated/prompt_{i}", 
                    prompt, 
                    global_step=step
                )
        except Exception as e:
            print(f"⚠️ Could not log generated samples to TensorBoard: {e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.pipeline.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Move data to device
            videos = batch['videos'].to(self.device)
            texts = batch['texts']
            
            # Build vocabulary for simple text encoder if needed
            if isinstance(self.pipeline.text_encoder, SimpleTextEncoder):
                self.pipeline.text_encoder.build_vocab(texts)
            
            # Forward pass
            loss = self.pipeline.train_step(videos, texts)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.step += 1
            
            # Track loss history
            self.loss_history.append(loss.item())
            self.lr_history.append(self.optimizer.param_groups[0]['lr'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / num_batches,
                'lr': self.optimizer.param_groups[0]['lr'],
                'step': self.step
            })
            
            # Regular logging to TensorBoard
            if self.use_tensorboard and self.tb_writer is not None and self.step % self.log_every_steps == 0:
                self.tb_writer.add_scalar('Loss/Train_Step', loss.item(), self.step)
                self.tb_writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.step)
                
                # Log moving averages
                if len(self.loss_history) >= 10:
                    recent_loss = np.mean(self.loss_history[-10:])
                    self.tb_writer.add_scalar('Loss/Train_Moving_Avg_10', recent_loss, self.step)
                
                if len(self.loss_history) >= 100:
                    recent_loss = np.mean(self.loss_history[-100:])
                    self.tb_writer.add_scalar('Loss/Train_Moving_Avg_100', recent_loss, self.step)
            
            # Detailed logging (gradients, memory, etc.)
            if self.use_tensorboard and self.tb_writer is not None and self.step % self.detailed_log_every_steps == 0:
                # Log gradient norms
                total_norm = 0
                param_count = 0
                for name, p in self.pipeline.named_parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                        
                        # Log individual parameter gradient norms
                        if 'unet' in name:
                            self.tb_writer.add_scalar(f'Gradients/UNet/{name.replace(".", "_")}', param_norm.item(), self.step)
                        elif 'text_encoder' in name:
                            self.tb_writer.add_scalar(f'Gradients/TextEncoder/{name.replace(".", "_")}', param_norm.item(), self.step)
                
                total_norm = total_norm ** (1. / 2)
                self.tb_writer.add_scalar('Gradients/Total_Norm', total_norm, self.step)
                self.tb_writer.add_scalar('Gradients/Average_Norm', total_norm / max(param_count, 1), self.step)
                
                # Log memory usage
                self._log_memory_usage(self.step)
                
                # Log training samples
                self._log_samples_to_tensorboard(videos, texts, "training", self.step)
                
                # Log parameter updates (weights)
                for name, param in self.pipeline.unet.named_parameters():
                    if param.requires_grad and 'weight' in name:
                        self.tb_writer.add_histogram(f'Weights/UNet/{name.replace(".", "_")}', param, self.step)
            
            # Save checkpoint every N steps
            if self.step % self.checkpoint_every_steps == 0:
                checkpoint_name = f"checkpoint_step_{self.step}.pt"
                self.save_checkpoint(checkpoint_name)
                print(f"💾 Checkpoint saved at step {self.step}")
                
                # Keep only last 5 step checkpoints to save space
                self._cleanup_step_checkpoints()
            
            # Generate samples occasionally
            if self.step % self.generate_samples_every_steps == 0:
                try:
                    sample_prompts = ["Hello", "How are you?", "Thank you", "Good morning"]
                    print(f"🎨 Generating samples at step {self.step}...")
                    samples = self.generate_samples(
                        sample_prompts, 
                        save_path=str(self.save_dir / f"samples_step_{self.step}.pt")
                    )
                    print(f"✅ Generated samples saved for step {self.step}")
                except Exception as e:
                    print(f"⚠️ Failed to generate samples at step {self.step}: {e}")
            
            # Update best training loss
            if loss.item() < self.best_train_loss:
                self.best_train_loss = loss.item()
                if self.use_tensorboard and self.tb_writer is not None:
                    self.tb_writer.add_scalar('Loss/Best_Train_Loss', self.best_train_loss, self.step)
            
            # Log to wandb
            if WANDB_AVAILABLE and wandb is not None and hasattr(wandb, 'run') and wandb.run is not None:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'step': self.step
                })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_dataloader is None:
            return {}
        
        self.pipeline.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Store samples for logging
        sample_videos = None
        sample_texts = None
        
        for batch in tqdm(self.val_dataloader, desc="Validating"):
            videos = batch['videos'].to(self.device)
            texts = batch['texts']
            
            # Store first batch for logging
            if sample_videos is None:
                sample_videos = videos
                sample_texts = texts
            
            # Build vocabulary for simple text encoder if needed
            if isinstance(self.pipeline.text_encoder, SimpleTextEncoder):
                self.pipeline.text_encoder.build_vocab(texts)
            
            loss = self.pipeline.train_step(videos, texts)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Log validation metrics to TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_scalar('Loss/Validation', avg_loss, self.step)
            
            # Log validation samples
            if sample_videos is not None and sample_texts is not None:
                self._log_samples_to_tensorboard(sample_videos, sample_texts, "validation", self.step)
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, filename: Optional[str] = None):
        """Save model checkpoint with enhanced metadata."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}_step_{self.step}.pt"
        
        filepath = self.save_dir / filename
        
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.pipeline.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_train_loss': self.best_train_loss,
            'loss_history': self.loss_history[-1000:],  # Keep last 1000 losses
            'lr_history': self.lr_history[-1000:],      # Keep last 1000 learning rates
            'training_config': {
                'model_channels': getattr(self.pipeline.unet, 'model_channels', None),
                'num_res_blocks': getattr(self.pipeline.unet, 'num_res_blocks', None),
                'num_heads': getattr(self.pipeline.unet, 'num_heads', None),
                'device': self.device,
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved to {filepath}")
        
        # Log checkpoint event to TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_scalar('Checkpoints/Saved', 1, self.step)
    
    def _cleanup_step_checkpoints(self):
        """Keep only the most recent step checkpoints to save disk space."""
        try:
            # Find all step checkpoints
            step_checkpoints = list(self.save_dir.glob("checkpoint_step_*.pt"))
            
            # Sort by step number
            step_checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            # Keep only the last 5 checkpoints
            if len(step_checkpoints) > 5:
                for checkpoint in step_checkpoints[:-5]:
                    checkpoint.unlink()
                    print(f"🗑️ Removed old checkpoint: {checkpoint.name}")
        except Exception as e:
            print(f"⚠️ Failed to cleanup checkpoints: {e}")
    
    def save_epoch_checkpoint(self):
        """Save a special epoch checkpoint."""
        filename = f"checkpoint_epoch_{self.epoch}.pt"
        self.save_checkpoint(filename)
        
        # Also save as latest.pt for easy resuming
        latest_path = self.save_dir / "latest.pt"
        checkpoint_path = self.save_dir / filename
        try:
            import shutil
            shutil.copy2(checkpoint_path, latest_path)
            print(f"📋 Latest checkpoint updated: {latest_path}")
        except Exception as e:
            print(f"⚠️ Failed to update latest checkpoint: {e}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.pipeline.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Checkpoint loaded from {filepath}")
    
    def generate_samples(self, prompts: List[str], save_path: Optional[str] = None):
        """Generate sample videos."""
        self.pipeline.eval()
        
        with torch.no_grad():
            generated = self.pipeline(
                prompts=prompts,
                num_frames=16,
                height=64,
                width=64,
                num_inference_steps=50
            )
        
        # Log to TensorBoard
        if self.use_tensorboard and self.tb_writer is not None and isinstance(generated, torch.Tensor):
            self._log_generated_samples_to_tensorboard(generated, prompts, self.step)
        
        if save_path:
            torch.save(generated, save_path)
        
        return generated
    
    def close_tensorboard(self):
        """Close TensorBoard writer."""
        if self.tb_writer is not None:
            self.tb_writer.close()
            print("TensorBoard writer closed.")
    
    def train(self, num_epochs: int, save_every: int = 5, validate_every: int = 5):
        """Main training loop with enhanced checkpointing and logging."""
        print(f"🚀 Starting training for {num_epochs} epochs")
        print(f"📊 Enhanced TensorBoard logging enabled")
        print(f"💾 Checkpoints: Every {save_every} epochs + every {self.checkpoint_every_steps} steps")
        print(f"🎨 Sample generation: Every {self.generate_samples_every_steps} steps")
        print(f"📈 Detailed logging: Every {self.detailed_log_every_steps} steps")
        print(f"Device: {self.device}")
        
        try:
            dataset_len = len(self.train_dataloader.dataset)  # type: ignore
            print(f"Training samples: {dataset_len}")
            print(f"Batches per epoch: {len(self.train_dataloader)}")
        except:
            print("Training samples: Unknown")
            
        if self.val_dataloader:
            try:
                val_dataset_len = len(self.val_dataloader.dataset)  # type: ignore
                print(f"Validation samples: {val_dataset_len}")
            except:
                print("Validation samples: Unknown")
        
        # Log initial state
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_scalar('Training/Epochs_Total', num_epochs, 0)
            self.tb_writer.add_scalar('Training/Started', 1, 0)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start_step = self.step
            
            print(f"\n{'='*60}")
            print(f"📅 Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Log epoch metrics
            if self.use_tensorboard and self.tb_writer is not None:
                self.tb_writer.add_scalar('Loss/Train_Epoch', train_metrics['train_loss'], epoch)
                self.tb_writer.add_scalar('Training/Epoch_Progress', epoch / num_epochs, epoch)
                
                # Log steps per epoch
                steps_this_epoch = self.step - epoch_start_step
                self.tb_writer.add_scalar('Training/Steps_Per_Epoch', steps_this_epoch, epoch)
            
            # Validate
            if self.val_dataloader and epoch % validate_every == 0:
                print(f"🔍 Running validation...")
                val_metrics = self.validate()
                
                # Check if best model
                if val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    self.save_checkpoint("best_model.pt")
                    print(f"🏆 New best validation loss: {self.best_loss:.4f}")
                    
                    if self.use_tensorboard and self.tb_writer is not None:
                        self.tb_writer.add_scalar('Loss/Best_Validation_Loss', self.best_loss, self.step)
                
                print(f"📊 Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Log to wandb
                if WANDB_AVAILABLE and wandb is not None and hasattr(wandb, 'run') and wandb.run is not None:
                    wandb.log({
                        'val/loss': val_metrics['val_loss'],
                        'epoch': epoch
                    })
                
                # Log epoch metrics to TensorBoard
                if self.use_tensorboard and self.tb_writer is not None:
                    self.tb_writer.add_scalars('Loss/Epoch_Comparison', {
                        'train': train_metrics['train_loss'],
                        'validation': val_metrics['val_loss']
                    }, epoch)
                    self.tb_writer.add_scalar('Loss/Val_Epoch', val_metrics['val_loss'], epoch)
            else:
                print(f"📊 Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}")
                
                # Log only training metrics to TensorBoard
                if self.use_tensorboard and self.tb_writer is not None:
                    self.tb_writer.add_scalar('Loss/Train_Epoch_Only', train_metrics['train_loss'], epoch)
            
            # Save epoch checkpoint
            if epoch % save_every == 0:
                self.save_epoch_checkpoint()
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if self.use_tensorboard and self.tb_writer is not None:
                self.tb_writer.add_scalar('Learning_Rate_Epoch', new_lr, epoch)
            
            if old_lr != new_lr:
                print(f"📉 Learning rate updated: {old_lr:.2e} → {new_lr:.2e}")
            
            # Generate samples at epoch milestones
            if epoch % (save_every * 2) == 0:
                try:
                    sample_prompts = [
                        "Hello", "How are you?", "Thank you", "Good morning",
                        "The aileron is controlled by lateral movement",
                        "Sign language is beautiful"
                    ]
                    print(f"🎨 Generating epoch samples...")
                    samples = self.generate_samples(
                        sample_prompts, 
                        save_path=str(self.save_dir / f"samples_epoch_{epoch}.pt")
                    )
                    print(f"✅ Generated samples saved for epoch {epoch}")
                except Exception as e:
                    print(f"⚠️ Failed to generate samples: {e}")
            
            # Flush TensorBoard logs
            if self.use_tensorboard and self.tb_writer is not None:
                self.tb_writer.flush()
        
        # Save final model
        print(f"\n🏁 Training completed!")
        self.save_checkpoint("final_model.pt")
        
        # Log training completion
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_scalar('Training/Completed', 1, self.step)
            self.tb_writer.add_text('Training/Summary', 
                f'Training completed after {num_epochs} epochs and {self.step} steps. '
                f'Best validation loss: {self.best_loss:.4f}', self.step)
        
        # Close TensorBoard writer
        self.close_tensorboard()
        
        print(f"📊 TensorBoard logs saved to: {self.save_dir / 'tensorboard_logs'}")
        print(f"💾 Checkpoints saved to: {self.save_dir}")
        print("To view TensorBoard, run: tensorboard --logdir=./checkpoints/tensorboard_logs")
        print("\n🎉 Training session complete!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Text-to-Sign Language Diffusion Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--max_frames", type=int, default=28, help="Maximum frames per video")
    parser.add_argument("--frame_size", type=int, default=128, help="Frame size (height and width)")
    parser.add_argument("--model_channels", type=int, default=32, help="Base model channels")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Checkpoint save directory")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--use_tensorboard", action="store_true", default=True, help="Use TensorBoard (default: True)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--text_encoder", type=str, default="simple", choices=["simple", "clip", "t5"])
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    
    # Enhanced checkpoint and logging arguments
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--validate_every", type=int, default=5, help="Run validation every N epochs")
    parser.add_argument("--checkpoint_every_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--log_every_steps", type=int, default=10, help="Log metrics every N steps")
    parser.add_argument("--generate_samples_every_steps", type=int, default=1000, help="Generate samples every N steps")
    parser.add_argument("--detailed_log_every_steps", type=int, default=100, help="Detailed logging every N steps")
    
    args = parser.parse_args()
    
    # Set device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        max_frames=args.max_frames,
        frame_size=(args.frame_size, args.frame_size)
    )
    
    # Create pipeline
    print("Creating pipeline...")
    pipeline = create_pipeline(
        model_channels=args.model_channels,
        text_encoder_type=args.text_encoder,
        scheduler_type=args.scheduler,
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        pipeline=pipeline,
        train_dataloader=train_dataloader,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard
    )
    
    # Update checkpoint and logging frequencies
    trainer.checkpoint_every_steps = args.checkpoint_every_steps
    trainer.log_every_steps = args.log_every_steps
    trainer.generate_samples_every_steps = args.generate_samples_every_steps
    trainer.detailed_log_every_steps = args.detailed_log_every_steps
    
    print(f"📊 Logging configuration:")
    print(f"   - Regular metrics every {args.log_every_steps} steps")
    print(f"   - Detailed logging every {args.detailed_log_every_steps} steps")
    print(f"   - Checkpoints every {args.checkpoint_every_steps} steps")
    print(f"   - Sample generation every {args.generate_samples_every_steps} steps")
    print(f"   - Epoch checkpoints every {args.save_every} epochs")
    print(f"   - Validation every {args.validate_every} epochs")
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        validate_every=args.validate_every
    )


if __name__ == "__main__":
    main()
