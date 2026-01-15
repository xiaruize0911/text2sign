"""
Trainer for Text-to-Sign Language DDIM Diffusion Model
Includes TensorBoard logging and tqdm progress bars
"""

import os
import math
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image

from config import TrainingConfig, ModelConfig, DDIMConfig
from dataset import get_dataloader
from models import UNet3D, TextEncoder, create_text_encoder
from schedulers import DDIMScheduler
from utils import EMA
from utils.metrics_logger import MetricsLogger, ExperimentTracker


class Trainer:
    """Trainer for text-to-sign language diffusion model"""
    
    def __init__(
        self,
        model: UNet3D,
        text_encoder: TextEncoder,
        scheduler: DDIMScheduler,
        train_config: TrainingConfig,
        model_config: ModelConfig,
        ddim_config: DDIMConfig,
    ):
        self.model = model
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.train_config = train_config
        self.model_config = model_config
        self.ddim_config = ddim_config
        
        self.device = torch.device(train_config.device)
        self.use_clip_text_encoder = getattr(model_config, "use_clip_text_encoder", False) or getattr(text_encoder, "use_clip", False)
        
        # Move models to device
        self.model = self.model.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        
        # Move scheduler tensors to device
        self._move_scheduler_to_device()
        
        # Optimizer
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.text_encoder.parameters()),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        
        # Learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if train_config.use_amp else None
        
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"text2sign_{timestamp}"
        self.checkpoint_dir = os.path.join(train_config.checkpoint_dir, self.run_name)
        self.log_dir = os.path.join(train_config.log_dir, self.run_name)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Research-grade metrics logging
        experiment_config = {
            "model": model_config.__dict__,
            "training": train_config.__dict__,
            "ddim": ddim_config.__dict__,
        }
        self.metrics_logger = MetricsLogger(
            log_dir=self.log_dir,
            experiment_name=self.run_name,
            config=experiment_config,
        )
        
        # Register experiment
        self.experiment_tracker = ExperimentTracker(base_dir="text_to_sign/experiments")
        self.experiment_tracker.register_experiment(
            experiment_name=self.run_name,
            config=experiment_config,
            description=f"Text2Sign training with EMA={getattr(train_config, 'use_ema', True)}, "
                       f"beta_schedule={ddim_config.beta_schedule}, "
                       f"lr={train_config.learning_rate}"
        )
        
        # EMA (Exponential Moving Average) for better sample quality
        self.use_ema = getattr(train_config, 'use_ema', True)
        if self.use_ema:
            ema_decay = getattr(train_config, 'ema_decay', 0.9999)
            ema_update_every = getattr(train_config, 'ema_update_every', 10)
            self.ema = EMA(
                self.model,
                decay=ema_decay,
                update_every=ema_update_every,
                device=self.device,
            )
            print(f"✅ EMA initialized with decay={ema_decay}, update_every={ema_update_every}")
        else:
            self.ema = None
            print("⚠️  EMA disabled")
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
    
    def _move_scheduler_to_device(self):
        """Move scheduler tensors to device"""
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler.alphas_cumprod_prev = self.scheduler.alphas_cumprod_prev.to(self.device)
        self.scheduler.sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod.to(self.device)
        self.scheduler.sqrt_one_minus_alphas_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod.to(self.device)
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler with improved warmup and decay"""
        # Better estimate of total steps
        total_steps = self.train_config.num_epochs * 1000  # Will be refined during training
        warmup_steps = min(self.train_config.warmup_steps, total_steps // 10)  # Cap at 10%
        
        # Gentler warmup from 0.1% to 100% of LR
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        # Cosine decay to 1% of original LR (not 0 for stability)
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.train_config.learning_rate * 0.01,
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.text_encoder.train()
        
        # Get data
        video = batch["video"].to(self.device)  # (B, T, C, H, W)
        tokens = None  # Not used with CLIP text encoder
        
        # Reshape video to (B, C, T, H, W)
        video = video.permute(0, 2, 1, 3, 4)
        
        batch_size = video.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.ddim_config.num_train_timesteps, (batch_size,),
            device=self.device
        )
        
        # Sample noise
        noise = torch.randn_like(video)
        
        # Add noise to video
        noisy_video = self.scheduler.add_noise(video, noise, timesteps)
        
        # Get text embeddings
        with autocast(enabled=self.train_config.use_amp):
            text_embeddings = self.text_encoder(tokens, text=batch.get("text"))
            
            # Predict noise
            noise_pred = self.model(noisy_video, timesteps, text_embeddings)
            
            # Compute loss
            if self.ddim_config.prediction_type == "epsilon":
                target = noise
            elif self.ddim_config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(video, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type: {self.ddim_config.prediction_type}")
            
            loss = F.mse_loss(noise_pred, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.text_encoder.parameters()),
                self.train_config.max_grad_norm,
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.text_encoder.parameters()),
                self.train_config.max_grad_norm,
            )
            self.optimizer.step()
        
        self.lr_scheduler.step()
        
        # Update EMA weights
        if self.ema is not None:
            self.ema.update()
        
        # Calculate gradient statistics for logging
        total_grad_norm = 0.0
        max_grad_norm = 0.0
        num_params_with_grad = 0
        
        for p in list(self.model.parameters()) + list(self.text_encoder.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_grad_norm += param_norm ** 2
                max_grad_norm = max(max_grad_norm, param_norm)
                num_params_with_grad += 1
        
        total_grad_norm = total_grad_norm ** 0.5
        avg_grad_norm = total_grad_norm / max(num_params_with_grad, 1)
        
        metrics = {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm_total": total_grad_norm,
            "grad_norm_avg": avg_grad_norm,
            "grad_norm_max": max_grad_norm,
        }
        
        # Log to metrics logger
        self.metrics_logger.log_step(
            step=self.global_step,
            metrics=metrics,
            phase="train"
        )
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_dataloader) -> float:
        """Run validation"""
        self.model.eval()
        self.text_encoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_dataloader, desc="Validating", leave=False):
            video = batch["video"].to(self.device)
            tokens = None
            
            video = video.permute(0, 2, 1, 3, 4)
            batch_size = video.shape[0]
            
            timesteps = torch.randint(
                0, self.ddim_config.num_train_timesteps, (batch_size,),
                device=self.device
            )
            
            noise = torch.randn_like(video)
            noisy_video = self.scheduler.add_noise(video, noise, timesteps)
            
            text_embeddings = self.text_encoder(tokens, text=batch.get("text"))
            noise_pred = self.model(noisy_video, timesteps, text_embeddings)
            
            if self.ddim_config.prediction_type == "epsilon":
                target = noise
            else:
                target = self.scheduler.get_velocity(video, noise, timesteps)
            
            loss = F.mse_loss(noise_pred, target)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def generate_samples(
        self,
        prompts: list,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate video samples from text prompts"""
        self.model.eval()
        self.text_encoder.eval()
        
        # Use EMA weights for better quality
        if self.ema is not None:
            self.ema.apply_shadow()
        
        batch_size = len(prompts)
        
        # Tokenize prompts
        tokens = None

        # Get text embeddings
        text_embeddings = self.text_encoder(
            torch.zeros(len(prompts), self.model_config.max_text_length, dtype=torch.long, device=self.device),
            text=prompts,
        )
        
        # For classifier-free guidance, also get unconditional embeddings
        if guidance_scale > 1.0:
            uncond_embeddings = self.text_encoder(
                torch.zeros(batch_size, self.model_config.max_text_length, dtype=torch.long, device=self.device),
                text=[""] * batch_size,
            )
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Set inference timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Initialize latents
        latents = torch.randn(
            batch_size,
            self.model_config.in_channels,
            self.model_config.num_frames,
            self.model_config.image_size,
            self.model_config.image_size,
            device=self.device,
        )
        
        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Generating", leave=False):
            latent_model_input = latents
            
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latents] * 2)
            
            # Convert timestep to int for scheduler.step, keep tensor for model
            t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timestep = torch.tensor([t_int] * latent_model_input.shape[0], device=self.device)
            
            noise_pred = self.model(latent_model_input, timestep, text_embeddings)
            
            # Apply classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DDIM step - use int timestep for scheduler
            latents, _ = self.scheduler.step(noise_pred, t_int, latents, eta=eta)
        
        # Restore original weights if using EMA
        if self.ema is not None:
            self.ema.restore()
        
        # Denormalize [-1, 1] -> [0, 1]
        videos = (latents + 1) / 2
        videos = videos.clamp(0, 1)
        
        return videos
    
    def save_videos_as_gif(self, videos: torch.Tensor, path: str, fps: int = 8):
        """Save videos as GIF files"""
        # videos: (B, C, T, H, W)
        videos = videos.cpu().numpy()
        
        for i, video in enumerate(videos):
            # (C, T, H, W) -> (T, H, W, C)
            frames = video.transpose(1, 2, 3, 0)
            frames = (frames * 255).astype(np.uint8)
            
            # Create PIL images
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Save as GIF
            gif_path = f"{path}_{i}.gif"
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=1000 // fps,
                loop=0,
            )
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "text_encoder_state_dict": self.text_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "ema_state_dict": self.ema.state_dict() if self.ema is not None else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "model_config": self.model_config,
            "train_config": self.train_config,
            "ddim_config": self.ddim_config,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, resume_training: bool = True):
        """Load model checkpoint
        
        Args:
            path: Path to checkpoint file
            resume_training: If True, restore optimizer and scheduler state for resuming training.
                           If False, only load model weights (for inference or fine-tuning).
        """
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
        print(f"  Loaded model weights")
        
        if resume_training:
            # Load optimizer and scheduler state
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print(f"  Loaded optimizer state")
            except Exception as e:
                print(f"  Warning: Could not load optimizer state: {e}")
                print(f"  Starting with fresh optimizer")
            
            try:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                print(f"  Loaded LR scheduler state")
            except Exception as e:
                print(f"  Warning: Could not load LR scheduler state: {e}")
                print(f"  Starting with fresh LR scheduler")
            
            if self.scaler and checkpoint.get("scaler_state_dict"):
                try:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    print(f"  Loaded gradient scaler state")
                except Exception as e:
                    print(f"  Warning: Could not load scaler state: {e}")
            
            # Load EMA state if available
            if self.ema is not None and checkpoint.get("ema_state_dict"):
                try:
                    self.ema.load_state_dict(checkpoint["ema_state_dict"])
                    print(f"  ✅ Loaded EMA state")
                except Exception as e:
                    print(f"  Warning: Could not load EMA state: {e}")
        
        # Always restore training progress
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0) + 1  # Start from next epoch
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        self.best_loss = checkpoint["best_loss"]
    
    def train(self):
        """Main training loop with comprehensive research-grade logging"""
        print("\n" + "="*70)
        print("🚀 Starting Training with Research-Grade Logging")
        print("="*70)
        print(f"Experiment: {self.run_name}")
        print(f"Logs: {self.log_dir}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print("="*70 + "\n")
        
        # Create dataloaders
        train_dataloader = get_dataloader(
            data_dir=self.train_config.data_dir,
            batch_size=self.train_config.batch_size,
            image_size=self.model_config.image_size,
            num_frames=self.model_config.num_frames,
            num_workers=self.train_config.num_workers,
            train=True,
        )
        
        val_dataloader = get_dataloader(
            data_dir=self.train_config.data_dir,
            batch_size=self.train_config.batch_size,
            image_size=self.model_config.image_size,
            num_frames=self.model_config.num_frames,
            num_workers=self.train_config.num_workers,
            train=False,
        )
        
        # Sample prompts for visualization
        sample_prompts = [
            "Hello",
            "Thank you",
            "I love you",
            "Good morning",
        ]
        
        print(f"Starting training for {self.train_config.num_epochs} epochs")
        print(f"Training samples: {len(train_dataloader.dataset)}")
        print(f"Validation samples: {len(val_dataloader.dataset)}")
        print(f"TensorBoard logs: {self.log_dir}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        
        for epoch in range(self.epoch, self.train_config.num_epochs):
            self.epoch = epoch
            
            # Start epoch tracking
            self.metrics_logger.start_epoch(epoch + 1)
            
            epoch_loss = 0.0
            epoch_losses = []  # Track all losses for statistics
            epoch_grad_norms = []
            num_batches = 0
            
            # Progress bar for epoch
            pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.train_config.num_epochs}",
                leave=True,
            )
            
            for batch in pbar:
                metrics = self.train_step(batch)
                
                epoch_loss += metrics["loss"]
                epoch_losses.append(metrics["loss"])
                epoch_grad_norms.append(metrics.get("grad_norm_total", 0))
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{metrics['lr']:.2e}",
                })
                
                # Log to TensorBoard
                if self.global_step % self.train_config.log_every == 0:
                    self.writer.add_scalar("train/loss", metrics["loss"], self.global_step)
                    self.writer.add_scalar("train/lr", metrics["lr"], self.global_step)
                    self.writer.add_scalar("train/grad_norm_total", metrics.get("grad_norm_total", 0), self.global_step)
                    self.writer.add_scalar("train/grad_norm_avg", metrics.get("grad_norm_avg", 0), self.global_step)
                    self.writer.add_scalar("train/grad_norm_max", metrics.get("grad_norm_max", 0), self.global_step)
                
                # Generate samples
                if self.global_step % self.train_config.sample_every == 0:
                    try:
                        print(f"\n🎨 Generating samples at step {self.global_step}...")
                        videos = self.generate_samples(
                            sample_prompts[:2],
                            num_inference_steps=50,  # Use 50 steps for quality samples
                            guidance_scale=3.0,
                        )
                        
                        # Save samples
                        sample_dir = os.path.join(self.checkpoint_dir, "samples")
                        os.makedirs(sample_dir, exist_ok=True)
                        self.save_videos_as_gif(
                            videos,
                            os.path.join(sample_dir, f"step_{self.global_step}"),
                        )
                        
                        # Log to TensorBoard (first frame of first video)
                        if videos.shape[0] > 0:
                            frame = videos[0, :, 0]  # (C, H, W)
                            self.writer.add_image("samples/generated", frame, self.global_step)
                        
                        print(f"✅ Samples saved successfully!")
                        
                        # Log sample generation success
                        self.metrics_logger.log_step(
                            step=self.global_step,
                            metrics={"sample_generated": 1.0},
                            phase="generation"
                        )
                    except Exception as e:
                        print(f"❌ Error generating samples at step {self.global_step}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # Log sample generation failure
                        self.metrics_logger.log_step(
                            step=self.global_step,
                            metrics={"sample_generated": 0.0, "sample_error": str(e)[:100]},
                            phase="generation"
                        )
            
            # End of epoch - calculate comprehensive statistics
            avg_train_loss = epoch_loss / max(num_batches, 1)
            
            # Calculate training statistics
            train_loss_std = np.std(epoch_losses) if epoch_losses else 0.0
            train_loss_min = np.min(epoch_losses) if epoch_losses else 0.0
            train_loss_max = np.max(epoch_losses) if epoch_losses else 0.0
            avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
            max_grad_norm = np.max(epoch_grad_norms) if epoch_grad_norms else 0.0
            
            # Validation
            val_loss = self.validate(val_dataloader)
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Epoch metrics for research paper
            epoch_metrics = {
                "train_loss_mean": avg_train_loss,
                "train_loss_std": train_loss_std,
                "train_loss_min": train_loss_min,
                "train_loss_max": train_loss_max,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "grad_norm_avg": avg_grad_norm,
                "grad_norm_max": max_grad_norm,
                "num_batches": num_batches,
                "samples_processed": num_batches * self.train_config.batch_size,
            }
            
            # Add EMA info if available
            if self.ema is not None:
                epoch_metrics["ema_step_counter"] = self.ema.step_counter
            
            # Log to metrics logger
            self.metrics_logger.end_epoch(epoch + 1, epoch_metrics)
            
            # Log to TensorBoard
            self.writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
            self.writer.add_scalar("epoch/train_loss_std", train_loss_std, epoch)
            self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
            self.writer.add_scalar("epoch/grad_norm_avg", avg_grad_norm, epoch)
            self.writer.add_scalar("epoch/learning_rate", current_lr, epoch)
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.train_config.save_every == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1}.pt"
                )
                self.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                self.save_checkpoint(best_path)
                print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Final checkpoint
        final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        self.save_checkpoint(final_path)
        
        # Save comprehensive training summary
        summary = self.metrics_logger.save_summary()
        
        print("\n" + "="*70)
        print("🎉 Training Complete!")
        print("="*70)
        print(f"Final model: {final_path}")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"Total training time: {summary['total_duration_hours']:.2f} hours")
        print(f"Total steps: {summary['total_steps']}")
        print(f"Total epochs: {summary['total_epochs']}")
        print("\nMetrics saved to:")
        print(f"  - Steps CSV: {self.metrics_logger.step_csv_path}")
        print(f"  - Epochs CSV: {self.metrics_logger.epoch_csv_path}")
        print(f"  - Summary JSON: {self.metrics_logger.json_dir / f'{self.run_name}_summary.json'}")
        print(f"  - TensorBoard: {self.log_dir}")
        print("="*70 + "\n")
        
        self.writer.close()
        self.metrics_logger.close()


def create_trainer(
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    ddim_config: Optional[DDIMConfig] = None,
) -> Trainer:
    """Create trainer with default or custom configs"""
    if model_config is None:
        model_config = ModelConfig()
    if train_config is None:
        train_config = TrainingConfig()
    if ddim_config is None:
        ddim_config = DDIMConfig()
    
    # Create models
    model = UNet3D(
        in_channels=model_config.in_channels,
        model_channels=model_config.model_channels,
        out_channels=model_config.in_channels,
        num_res_blocks=model_config.num_res_blocks,
        attention_resolutions=model_config.attention_resolutions,
        channel_mult=model_config.channel_mult,
        num_heads=model_config.num_heads,
        context_dim=model_config.context_dim,
    )
    
    text_encoder = create_text_encoder(
        model_config,
        use_clip=getattr(model_config, "use_clip_text_encoder", False),
    )
    
    scheduler = DDIMScheduler(
        num_train_timesteps=ddim_config.num_train_timesteps,
        beta_start=ddim_config.beta_start,
        beta_end=ddim_config.beta_end,
        beta_schedule=ddim_config.beta_schedule,
        clip_sample=ddim_config.clip_sample,
        prediction_type=ddim_config.prediction_type,
    )
    
    return Trainer(
        model=model,
        text_encoder=text_encoder,
        scheduler=scheduler,
        train_config=train_config,
        model_config=model_config,
        ddim_config=ddim_config,
    )


if __name__ == "__main__":
    # Test trainer creation
    trainer = create_trainer()
    print("Trainer created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"Text encoder parameters: {sum(p.numel() for p in trainer.text_encoder.parameters()):,}")
