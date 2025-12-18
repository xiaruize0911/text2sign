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
        """Create learning rate scheduler with warmup"""
        total_steps = self.train_config.num_epochs * 1000  # Approximate
        warmup_steps = self.train_config.warmup_steps
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6,
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
        
        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
    
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
            
            timestep = torch.tensor([t] * latent_model_input.shape[0], device=self.device)
            
            noise_pred = self.model(latent_model_input, timestep, text_embeddings)
            
            # Apply classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DDIM step
            latents, _ = self.scheduler.step(noise_pred, t, latents, eta=eta)
        
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
        
        # Always restore training progress
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0) + 1  # Start from next epoch
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        self.best_loss = checkpoint["best_loss"]
    
    def train(self):
        """Main training loop"""
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
            epoch_loss = 0.0
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
                
                # Generate samples
                if self.global_step % self.train_config.sample_every == 0:
                    try:
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
                    except Exception as e:
                        print(f"Error generating samples: {e}")
            
            # End of epoch
            avg_train_loss = epoch_loss / max(num_batches, 1)
            
            # Validation
            val_loss = self.validate(val_dataloader)
            
            # Log epoch metrics
            self.writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
            self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
            
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
        print(f"Training complete! Final model saved to {final_path}")
        
        self.writer.close()


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
