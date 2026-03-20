"""
Trainer for Text-to-Sign Language DDIM Diffusion Model
Includes TensorBoard logging and tqdm progress bars
"""

import os
import sys
import math
import glob
import signal
import shutil
from contextlib import nullcontext
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
try:
    from torch.amp import GradScaler, autocast
    AMP_USES_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_USES_DEVICE_TYPE = False
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image

from config import TrainingConfig, ModelConfig, DDIMConfig
from dataset import get_dataloader, SimpleTokenizer
from models import UNet3D, TextEncoder, create_text_encoder
from schedulers import DDIMScheduler
from utils import EMA
from utils.metrics_logger import MetricsLogger, ExperimentTracker
from torchvision.utils import make_grid


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
        run_name: Optional[str] = None,
    ):
        self.model = model
        self.text_encoder = text_encoder
        self.base_model = model
        self.base_text_encoder = text_encoder
        self.scheduler = scheduler
        self.train_config = train_config
        self.model_config = model_config
        self.ddim_config = ddim_config
        
        self.device = torch.device(train_config.device)
        self.use_clip_text_encoder = getattr(model_config, "use_clip_text_encoder", False) or getattr(text_encoder, "use_clip", False)
        self.channels_last_3d = self.device.type == "cuda" and getattr(train_config, "channels_last_3d", True)
        self.compile_enabled = False
        self.amp_dtype = None
        self.precision_mode = "fp32"

        self._configure_backend_flags()
        
        # Initialize tokenizer if not using CLIP
        if not self.use_clip_text_encoder:
            self.tokenizer = SimpleTokenizer(
                vocab_size=model_config.vocab_size,
                max_length=model_config.max_text_length,
            )
        else:
            self.tokenizer = None
        
        # Move models to device
        self.base_model = self.base_model.to(self.device)
        self.base_text_encoder = self.base_text_encoder.to(self.device)
        self.model = self.base_model
        self.text_encoder = self.base_text_encoder

        if self.channels_last_3d:
            self.base_model = self.base_model.to(memory_format=torch.channels_last_3d)
            self.model = self.base_model
            print("🧠 Using channels_last_3d memory format for the UNet")

        self.amp_dtype, self.precision_mode = self._resolve_amp_dtype()
        if self.amp_dtype is None:
            print(f"🔢 Precision mode: {self.precision_mode} (AMP disabled)")
        else:
            print(f"🔢 Precision mode: {self.precision_mode}")
        
        # Move scheduler tensors to device
        self._move_scheduler_to_device()

        self._maybe_compile_modules()

        self.trainable_parameters = [
            p for p in list(self.base_model.parameters()) + list(self.base_text_encoder.parameters())
            if p.requires_grad
        ]
        if not self.trainable_parameters:
            raise ValueError("No trainable parameters found for optimization.")
        
        # Optimizer with Fused AdamW optimization if available
        # Check for PyTorch 2.0+ fused version
        optim_kwargs = {
            "lr": train_config.learning_rate,
            "weight_decay": train_config.weight_decay,
            "eps": 1e-6, # Increased epsilon for better stability in FP16/Mixed Precision
        }
        
        # Detect if 'fused' argument exists (available in PyTorch 2.0+)
        import inspect
        if "fused" in inspect.signature(AdamW).parameters and self.device.type == "cuda":
            optim_kwargs["fused"] = True
            print("⚡ Using Fused AdamW optimizer for increased speed")
            
        self.optimizer = AdamW(
            self.trainable_parameters,
            **optim_kwargs
        )
        
        # Learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Mixed precision
        self.scaler = self._create_grad_scaler()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Performance optimization: Disable debugging tools for speed
        torch.autograd.set_detect_anomaly(False)
        
        # Run name and directories
        if run_name:
            self.run_name = run_name
        else:
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
                self.base_model,
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
        self._interrupted = False

    def _configure_backend_flags(self):
        """Enable backend-level training accelerators when available."""
        if self.device.type != "cuda":
            return

        torch.backends.cudnn.benchmark = True
        print("🚀 CuDNN benchmarking enabled for speed")

        allow_tf32 = getattr(self.train_config, "allow_tf32", True)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

        if allow_tf32:
            print("⚡ TF32 matmul/cuDNN enabled")
        else:
            print("⚙️ TF32 disabled")

    def _resolve_amp_dtype(self) -> Tuple[Optional[torch.dtype], str]:
        """Resolve the effective AMP dtype from the requested precision mode."""
        requested = getattr(self.train_config, "precision", "auto").lower()

        if not getattr(self.train_config, "use_amp", True):
            return None, "fp32"

        if self.device.type != "cuda":
            return None, "fp32"

        if requested == "fp32":
            return None, "fp32"

        if requested == "auto":
            requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

        if requested == "bf16":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, "bf16"
            print("⚠️ bf16 requested but not supported on this GPU; falling back to fp16")
            return torch.float16, "fp16"

        if requested == "fp16":
            return torch.float16, "fp16"

        return None, "fp32"

    def _create_grad_scaler(self) -> Optional[GradScaler]:
        """Create a gradient scaler only when it is actually helpful."""
        if self.amp_dtype != torch.float16:
            return None

        try:
            return GradScaler(self.device.type)
        except TypeError:
            return GradScaler()

    def _autocast_context(self):
        """Return the correct autocast context for the configured precision mode."""
        if self.amp_dtype is None:
            return nullcontext()
        if AMP_USES_DEVICE_TYPE:
            return autocast(device_type=self.device.type, dtype=self.amp_dtype)
        return autocast(dtype=self.amp_dtype)

    def _maybe_compile_modules(self):
        """Compile the heaviest trainable modules when the runtime supports it."""
        if not getattr(self.train_config, "enable_compile", False):
            print("⚙️ torch.compile disabled")
            return

        if not hasattr(torch, "compile"):
            print("⚠️ torch.compile not available in this PyTorch build")
            return

        if self.device.type != "cuda":
            print("⚙️ Skipping torch.compile on non-CUDA device")
            return

        compile_kwargs = {
            "mode": getattr(self.train_config, "compile_mode", "reduce-overhead"),
            "fullgraph": getattr(self.train_config, "compile_fullgraph", False),
            "dynamic": getattr(self.train_config, "compile_dynamic", False),
        }

        try:
            self.model = torch.compile(self.base_model, **compile_kwargs)
            if not self.use_clip_text_encoder:
                self.text_encoder = torch.compile(self.base_text_encoder, **compile_kwargs)
            self.compile_enabled = True
            print(f"🔥 torch.compile enabled (mode={compile_kwargs['mode']})")
        except Exception as exc:
            self.compile_enabled = False
            self.model = self.base_model
            self.text_encoder = self.base_text_encoder
            print(f"⚠️ torch.compile failed, continuing without it: {exc}")
    
    def _move_scheduler_to_device(self):
        """Move scheduler tensors to device"""
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler.alphas_cumprod_prev = self.scheduler.alphas_cumprod_prev.to(self.device)
        self.scheduler.sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod.to(self.device)
        self.scheduler.sqrt_one_minus_alphas_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod.to(self.device)
    
    def _get_snr(self, timesteps):
        """Compute SNR for given timesteps with numerical stability"""
        # Ensure timesteps are within range
        timesteps = torch.clamp(timesteps, 0, self.ddim_config.num_train_timesteps - 1)
        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        # Add epsilon to prevent division by zero at alphas_cumprod=1
        return alphas_cumprod / (1.0 - alphas_cumprod + 1e-9)
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler with improved warmup and decay"""
        # Improved: Dynamic estimation of steps per epoch
        # Look for GIF files in data_dir to get a better count
        try:
            num_samples = len(glob.glob(os.path.join(self.train_config.data_dir, "*.gif")))
            if num_samples == 0:
                num_samples = 1000 # Fallback
        except:
            num_samples = 1000
            
        # Actual steps per epoch considering gradient accumulation
        steps_per_epoch = max(1, (num_samples // (self.train_config.batch_size * self.train_config.gradient_accumulation_steps)))
        total_steps = self.train_config.num_epochs * steps_per_epoch
        
        # Warmup: 5 epochs or 500 steps, whichever is more reasonable
        warmup_steps = min(500, total_steps // 10)
        
        print(f"📈 Scheduler: total_steps={total_steps}, warmup_steps={warmup_steps}, steps_per_epoch={steps_per_epoch}")
        
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
            T_max=max(1, total_steps - warmup_steps),
            eta_min=self.train_config.learning_rate * 1e-2,
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    def _apply_conditioning_mode(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply conditioning ablations without duplicating data-loader logic."""
        mode = getattr(self.model_config, "text_conditioning_mode", "normal")
        if mode == "none":
            return torch.zeros_like(text_embeddings)
        if mode == "random" and text_embeddings.shape[0] > 1:
            perm = torch.randperm(text_embeddings.shape[0], device=text_embeddings.device)
            return text_embeddings[perm]
        return text_embeddings
    
    def train_step(self, batch: Dict[str, torch.Tensor], accumulate_only: bool = False) -> Dict[str, float]:
        """Single training step with gradient accumulation support and Min-SNR weighting"""
        self.model.train()
        self.text_encoder.train()
        
        # Get data with non_blocking transfer for speed
        video = batch["video"].to(self.device, non_blocking=True)  # (B, T, C, H, W)
        tokens = batch.get("tokens")
        if tokens is not None:
            tokens = tokens.to(self.device, non_blocking=True)
        
        # Reshape video to (B, C, T, H, W)
        video = video.permute(0, 2, 1, 3, 4)
        
        batch_size = video.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.ddim_config.num_train_timesteps, (batch_size,),
            device=self.device
        )
        
        # Sample noise with Noise Offset (0.1) for better global structure and contrast
        # Helps prevent "washed out" videos
        noise = torch.randn_like(video)
        noise_offset = 0.1 * torch.randn(batch_size, video.shape[1], 1, 1, 1, device=self.device)
        input_noise = noise + noise_offset
        
        # Add noise to video
        noisy_video = self.scheduler.add_noise(video, input_noise, timesteps)
        
        # Get text embeddings
        with self._autocast_context():
            if self.use_clip_text_encoder:
                text_embeddings = self.text_encoder(tokens=tokens, text=batch.get("text"))
            else:
                text_embeddings = self.text_encoder(tokens)
            text_embeddings = self._apply_conditioning_mode(text_embeddings)
            
            # Predict noise
            noise_pred = self.model(noisy_video, timesteps, text_embeddings)
            
            # Compute target
            if self.ddim_config.prediction_type == "epsilon":
                target = noise # Target original noise, not the offsetted one
            elif self.ddim_config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(video, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type: {self.ddim_config.prediction_type}")
            
            # Min-SNR-5 weighting for faster convergence and better quality
            # Based on "Efficient Diffusion Training via Min-SNR Weighting Strategy"
            snr = self._get_snr(timesteps)
            # Add small epsilon to snr in denominator to prevent NaN if snr=0
            # Also clamp SNR to avoid extreme values that might explode weights
            snr = torch.clamp(snr, min=1e-7, max=1000.0)
            mse_loss_weights = torch.stack([snr, 5.0 * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
            
            loss = F.mse_loss(noise_pred, target, reduction="none")
            loss = loss.mean(dim=(1, 2, 3, 4)) * mse_loss_weights
            loss = loss.mean()
            
            # REMOVED: loss = torch.clamp(loss, max=10.0)
            # Clamping the loss value masks instabilities and can zero out gradients when 
            # the model is in a high-loss state and needs to learn the most.
            # Gradient clipping (max_grad_norm) is sufficient for stability.
            
            # Scale loss for gradient accumulation
            if self.train_config.gradient_accumulation_steps > 1:
                loss = loss / self.train_config.gradient_accumulation_steps
        
        # Check for NaN loss to prevent corruption
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  Detected NaN/Inf loss at step {self.global_step}. Skipping optimization step.")
            
            # Diagnostic info
            if torch.isnan(noise_pred).any():
                print(f"   - NaN detected in noise_pred")
            if torch.isinf(noise_pred).any():
                print(f"   - Inf detected in noise_pred")
            if torch.isnan(target).any():
                print(f"   - NaN detected in target")
            if torch.isnan(text_embeddings).any():
                print(f"   - NaN detected in text_embeddings")
            if torch.isnan(mse_loss_weights).any() or torch.isinf(mse_loss_weights).any():
                print(f"   - NaN/Inf detected in mse_loss_weights")
                
            self.optimizer.zero_grad(set_to_none=True) # Clear any previously accumulated gradients
            return {
                "loss": 1.0, # Return 1.0 instead of 0.0 to avoid misleadingly 'best' scores
                "nan": True,
                "lr": self.optimizer.param_groups[0]["lr"] # Return current LR for progress bar
            }
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # Optimization step
        metrics = {"loss": loss.item() * self.train_config.gradient_accumulation_steps}
        
        if not accumulate_only:
            # Unscale gradients for clipping and manual checking
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                
            # Check for NaN gradients before stepping
            has_nan_grad = False
            for p in self.trainable_parameters:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"\n⚠️  Detected NaN/Inf gradients at step {self.global_step}. Skipping optimization step.")
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    # Still need to update the scaler to maintain internal state
                    # This will decrease the scale factor for the next iteration
                    self.scaler.update() 
                return {
                    "loss": metrics["loss"],
                    "nan": True,
                    "lr": self.optimizer.param_groups[0]["lr"]
                }

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters,
                self.train_config.max_grad_norm,
            )
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Calculate gradient statistics for logging BEFORE zero_grad
            total_grad_norm = 0.0
            max_grad_norm = 0.0
            num_params_with_grad = 0
            
            for p in self.trainable_parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
                    max_grad_norm = max(max_grad_norm, param_norm)
                    num_params_with_grad += 1
            
            total_grad_norm = total_grad_norm ** 0.5
            avg_grad_norm = total_grad_norm / max(num_params_with_grad, 1)

            # Clear gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Update EMA weights
            if self.ema is not None:
                self.ema.update()
            
            metrics.update({
                "lr": self.optimizer.param_groups[0]["lr"],
                "grad_norm_total": total_grad_norm,
                "grad_norm_avg": avg_grad_norm,
                "grad_norm_max": max_grad_norm,
            })
            
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
            if self._interrupted:
                break
                
            video = batch["video"].to(self.device)
            tokens = batch.get("tokens")
            if tokens is not None:
                tokens = tokens.to(self.device)
            
            video = video.permute(0, 2, 1, 3, 4)
            batch_size = video.shape[0]
            
            timesteps = torch.randint(
                0, self.ddim_config.num_train_timesteps, (batch_size,),
                device=self.device
            )
            
            noise = torch.randn_like(video)
            noisy_video = self.scheduler.add_noise(video, noise, timesteps)
            
            with self._autocast_context():
                if self.use_clip_text_encoder:
                    text_embeddings = self.text_encoder(tokens=None, text=batch.get("text"))
                else:
                    text_embeddings = self.text_encoder(tokens)
                text_embeddings = self._apply_conditioning_mode(text_embeddings)
                
                noise_pred = self.model(noisy_video, timesteps, text_embeddings)
            
            if self.ddim_config.prediction_type == "epsilon":
                target = noise
            else:
                target = self.scheduler.get_velocity(video, noise, timesteps)
            
            loss = F.mse_loss(noise_pred, target)
            total_loss += loss.item()
            num_batches += 1
        
        avg_val_loss = total_loss / max(num_batches, 1)
        
        # Ensure we don't return NaN or suspiciously zero loss
        if np.isnan(avg_val_loss) or np.isinf(avg_val_loss) or avg_val_loss < 1e-6:
            return 1.0 # Fallback to a safe high loss
            
        return avg_val_loss
    
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
        try:
            # Use EMA weights for better quality
            if self.ema is not None:
                self.ema.apply_shadow()
            
            batch_size = len(prompts)
            
            # Add length prefix if enabled
            model_prompts = prompts
            if getattr(self.model_config, 'use_length_prefix', False):
                model_prompts = []
                for p in prompts:
                    word_count = len(p.split())
                    safe_count = min(word_count, 30)
                    model_prompts.append(f"[LEN_{safe_count}] {p}")
            
            # Get text embeddings
            if self.use_clip_text_encoder:
                text_embeddings = self.text_encoder(tokens=None, text=model_prompts)
            else:
                tokens = self.tokenizer(model_prompts).to(self.device)
                text_embeddings = self.text_encoder(tokens)
            text_embeddings = self._apply_conditioning_mode(text_embeddings)
            
            # For classifier-free guidance, also get unconditional embeddings
            if guidance_scale > 1.0:
                if self.use_clip_text_encoder:
                    uncond_embeddings = self.text_encoder(tokens=None, text=[""] * batch_size)
                else:
                    uncond_tokens = self.tokenizer([""] * batch_size).to(self.device)
                    uncond_embeddings = self.text_encoder(uncond_tokens)
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
            if self.channels_last_3d:
                latents = latents.contiguous(memory_format=torch.channels_last_3d)
            
            # Denoising loop
            with self._autocast_context():
                for t in tqdm(self.scheduler.timesteps, desc="Generating", leave=False):
                    if self._interrupted:
                        break
                        
                    latent_model_input = latents
                    
                    if guidance_scale > 1.0:
                        latent_model_input = torch.cat([latents] * 2)
                    
                    # Convert timestep to int for scheduler.step, keep tensor for model
                    t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                    timestep = torch.tensor(
                        [t_int] * latent_model_input.shape[0],
                        device=self.device,
                        dtype=torch.long,
                    )
                    
                    noise_pred = self.model(latent_model_input, timestep, text_embeddings)
                    
                    # Apply classifier-free guidance
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # DDIM step - use int timestep for scheduler
                    latents, _ = self.scheduler.step(noise_pred, t_int, latents, eta=eta)
        finally:
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
        """Save model checkpoint atomically using temporary files"""
        checkpoint = {
            "model_state_dict": self.base_model.state_dict(),
            "text_encoder_state_dict": self.base_text_encoder.state_dict(),
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
        
        # Atomic save: save to .tmp then rename
        tmp_path = path + ".tmp"
        
        # Backup existing best model if saving best_model.pt
        if os.path.basename(path) == "best_model.pt" and os.path.exists(path):
            backup_path = path.replace(".pt", "_prev.pt")
            shutil.copy2(path, backup_path)
            
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        print(f"✅ Atomically saved checkpoint to {path}")

    def _move_optimizer_state_to_device(self):
        """Move optimizer state tensors to the active training device."""
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device, non_blocking=True)
    
    def load_checkpoint(self, path: str, resume_training: bool = True):
        """Load model checkpoint
        
        Args:
            path: Path to checkpoint file
            resume_training: If True, restore optimizer and scheduler state for resuming training.
                           If False, only load model weights (for inference or fine-tuning).
        """
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location="cpu")
        
        # Load model weights
        self.base_model.load_state_dict(checkpoint["model_state_dict"])
        self.base_text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
        print(f"  Loaded model weights")
        
        if resume_training:
            # Load optimizer and scheduler state
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self._move_optimizer_state_to_device()
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
            train_ratio=self.train_config.train_ratio,
            split_mode=self.train_config.split_mode,
            random_seed=self.train_config.split_seed,
            tokenizer=self.text_encoder.tokenizer if hasattr(self.text_encoder, "tokenizer") else self.tokenizer,
            use_length_prefix=getattr(self.model_config, 'use_length_prefix', False),
            pin_memory=self.train_config.dataloader_pin_memory and self.device.type == "cuda",
            persistent_workers=self.train_config.dataloader_persistent_workers,
            prefetch_factor=self.train_config.dataloader_prefetch_factor,
        )
        
        val_dataloader = get_dataloader(
            data_dir=self.train_config.data_dir,
            batch_size=self.train_config.batch_size,
            image_size=self.model_config.image_size,
            num_frames=self.model_config.num_frames,
            num_workers=self.train_config.num_workers,
            train=False,
            train_ratio=self.train_config.train_ratio,
            split_mode=self.train_config.split_mode,
            random_seed=self.train_config.split_seed,
            tokenizer=self.text_encoder.tokenizer if hasattr(self.text_encoder, "tokenizer") else self.tokenizer,
            use_length_prefix=getattr(self.model_config, 'use_length_prefix', False),
            pin_memory=self.train_config.dataloader_pin_memory and self.device.type == "cuda",
            persistent_workers=self.train_config.dataloader_persistent_workers,
            prefetch_factor=self.train_config.dataloader_prefetch_factor,
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
        
        # Setup signal handler for graceful shutdown
        self._interrupted = False
        def signal_handler(sig, frame):
            if self._interrupted:
                print("\n🛑 Second interrupt received! Force quitting...")
                sys.exit(1)
            print("\n🛑 Interrupt received! Will save after current step. Press Ctrl+C again to force quit.")
            self._interrupted = True
        
        old_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            for epoch in range(self.epoch, self.train_config.num_epochs):
                if self._interrupted:
                    break
                
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
                
                for i, batch in enumerate(pbar):
                    if self._interrupted:
                        break
                        
                    # Gradient accumulation logic
                    is_last_batch = (i + 1) == len(train_dataloader)
                    is_accumulating = ((i + 1) % self.train_config.gradient_accumulation_steps != 0) and not is_last_batch
                    
                    # Perform training step
                    metrics = self.train_step(batch, accumulate_only=is_accumulating)
                    
                    # Update epoch statistics (only on steps where we don't accumulate, or every step for loss)
                    epoch_loss += metrics["loss"]
                    epoch_losses.append(metrics["loss"])
                    
                    if not is_accumulating:
                        epoch_grad_norms.append(metrics.get("grad_norm_total", 0))
                        num_batches += 1
                        self.global_step += 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            "loss": f"{metrics['loss']:.4f}",
                            "lr": f"{metrics.get('lr', 0):.2e}",
                            "step": self.global_step
                        })
                        
                        # Log to TensorBoard
                        if self.global_step % self.train_config.log_every == 0:
                            self.writer.add_scalar("train/loss", metrics["loss"], self.global_step)
                            self.writer.add_scalar("train/lr", metrics.get("lr", 0), self.global_step)
                            self.writer.add_scalar("train/grad_norm_total", metrics.get("grad_norm_total", 0), self.global_step)
                            self.writer.add_scalar("train/grad_norm_avg", metrics.get("grad_norm_avg", 0), self.global_step)
                            self.writer.add_scalar("train/grad_norm_max", metrics.get("grad_norm_max", 0), self.global_step)
                            self.writer.flush()
                        
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
                                
                                # Log to TensorBoard (grid of frames for the first sequence)
                                if videos.shape[0] > 0:
                                    # Move to CPU and ensure [0, 1] range for visualization
                                    vis_videos = videos.detach().cpu()
                                    
                                    # Log sequence as a grid (T, C, H, W) for make_grid
                                    # videos[0] is (C, T, H, W) -> permute to (T, C, H, W)
                                    frames = vis_videos[0].permute(1, 0, 2, 3)
                                    grid = make_grid(frames, nrow=4, normalize=False)
                                    self.writer.add_image("samples/generated_sequence", grid, self.global_step)
                                    
                                    # Also log as a video if possible (B, T, C, H, W)
                                    # permute (B, C, T, H, W) -> (B, T, C, H, W)
                                    vid_tensor = vis_videos.permute(0, 2, 1, 3, 4)
                                    try:
                                        self.writer.add_video("samples/generated_video", vid_tensor, self.global_step, fps=self.model_config.fps if hasattr(self.model_config, "fps") else 8)
                                    except Exception as video_err:
                                        # Fallback if moviepy/ffmpeg not available
                                        pass
                                    
                                    # Also log first frames of the batch
                                    first_frames = vis_videos[:, :, 0] # (B, C, H, W)
                                    batch_grid = make_grid(first_frames, nrow=2, normalize=False)
                                    self.writer.add_image("samples/generated_batch", batch_grid, self.global_step)
                                
                                print(f"✅ Samples saved successfully and logged to TensorBoard!")
                                
                                # Log sample generation success
                                self.writer.add_scalar("generation/success", 1.0, self.global_step)
                                self.writer.flush()
                            except Exception as e:
                                print(f"❌ Error generating samples at step {self.global_step}: {e}")
                                import traceback
                                traceback.print_exc()
                                
                                # Log sample generation failure
                                self.writer.add_scalar("generation/success", 0.0, self.global_step)
                                self.writer.flush()
                
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
                self.writer.flush()
                
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
        
        except Exception as e:
            print(f"❌ Training crashed with error: {e}")
            import traceback
            traceback.print_exc()
            # Save an emergency checkpoint
            emergency_path = os.path.join(self.checkpoint_dir, "emergency_checkpoint.pt")
            self.save_checkpoint(emergency_path)
            
        finally:
            # Restore signal handler
            signal.signal(signal.SIGINT, old_handler)
            
            # Final checkpoint if we weren't just crashed
            if not self._interrupted:
                final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
                self.save_checkpoint(final_path)
            else:
                interrupted_path = os.path.join(self.checkpoint_dir, "interrupted_checkpoint.pt")
                self.save_checkpoint(interrupted_path)
            
            # Save comprehensive training summary
            summary = self.metrics_logger.save_summary()
            
            print("\n" + "="*70)
            print("🎉 Training Shutdown Complete")
            print("="*70)
            print(f"Total training time: {summary['total_duration_hours']:.2f} hours")
            print("="*70 + "\n")
            
            self.writer.close()
            self.metrics_logger.close()


def create_trainer(
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    ddim_config: Optional[DDIMConfig] = None,
    run_name: Optional[str] = None,
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
        use_transformer=getattr(model_config, 'use_transformer', True),
        transformer_depth=getattr(model_config, 'transformer_depth', 1),
        use_gradient_checkpointing=getattr(model_config, 'use_gradient_checkpointing', False),
        num_frames=getattr(model_config, 'num_frames', 16),
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
        run_name=run_name,
    )


if __name__ == "__main__":
    # Test trainer creation
    # Ensure precision is set to medium/high for stability on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        
    trainer = create_trainer()
    print("Trainer created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"Text encoder parameters: {sum(p.numel() for p in trainer.text_encoder.parameters()):,}")
