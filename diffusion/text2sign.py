"""
Diffusion model for text-to-sign language video generation.

Implements DDPM training and DDIM sampling following:
- Ho et al. (2020): Denoising Diffusion Probabilistic Models
- Song et al. (2020): Denoising Diffusion Implicit Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm

from schedulers.noise_schedulers import create_noise_scheduler

class DiffusionModel(nn.Module):
    """
    Diffusion model for text-conditioned video generation.
    
    Args:
        model: Backbone model (ViViT, TinyFusion, etc.)
        timesteps: Number of diffusion timesteps for training
        inference_timesteps: Number of steps for sampling (can be < timesteps for faster inference)
        noise_scheduler: Type of noise schedule ("linear", "cosine", etc.)
        device: Device to run on
        text_encoder: Text encoder for conditioning
        **scheduler_kwargs: Additional scheduler-specific parameters
    """
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        inference_timesteps: Optional[int] = None,
        noise_scheduler: str = "linear",
        device: torch.device = torch.device("cpu"),
        text_encoder=None,
        **scheduler_kwargs,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.inference_timesteps = inference_timesteps or timesteps
        self.device = device
        self.noise_scheduler_type = noise_scheduler
        self.text_encoder = text_encoder
        
        # Initialize noise scheduler
        self.noise_scheduler = create_noise_scheduler(noise_scheduler, timesteps, **scheduler_kwargs)
        self.betas = self.noise_scheduler.get_schedule().to(device)
        
        # Compute alpha schedule
        self.alphas, self.alphas_cumprod, self.alphas_cumprod_prev = self.noise_scheduler.compute_alpha_schedule(self.betas)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        
        # Precompute frequently used values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Posterior variance for reverse sampling
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-6)
        
        print(f"✅ Diffusion model initialized with {noise_scheduler} scheduler (T={timesteps})")
        if text_encoder is not None:
            print(f"✅ Text-conditioned mode enabled")
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Adds noise to clean data according to the schedule:
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        
        Args:
            x_start: Clean video data (x_0)
            t: Timestep tensor (batch_size,)
            noise: Optional noise tensor, generated if None
            
        Returns:
            Tuple of (noise, noisy_data)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        alpha_bar_t = self.alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1)))
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        x_noisy = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

        return noise, x_noisy
    
    def p_sample_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prev_t: torch.Tensor,
        text: Optional[str] = None,
        deterministic: bool = False,
        eta: float = 1.0,
    ) -> torch.Tensor:
        """
        Single DDIM reverse diffusion step following Song et al. (2020).
        
        DDIM sampling formula:
        x_{t-1} = sqrt(α̅_{t-1}) * x_0 + sqrt(1 - α̅_{t-1} - σ_t²) * ε_θ(x_t, t) + σ_t * z
        
        where:
        - x_0 = (x_t - sqrt(1 - α̅_t) * ε_θ(x_t, t)) / sqrt(α̅_t)  [predicted clean sample]
        - σ_t = η * sqrt((1 - α̅_{t-1}) / (1 - α̅_t)) * sqrt(1 - α̅_t / α̅_{t-1})  [stochasticity]
        - η = 0 gives deterministic DDIM, η = 1 gives DDPM-like stochastic sampling
        """
        with torch.no_grad():
            # Encode text conditioning
            text_emb = None
            if text is not None and self.text_encoder is not None:
                batch_size = x.shape[0]
                text_batch = [text] * batch_size if isinstance(text, str) else list(text)
                if len(text_batch) != batch_size:
                    repeats = (batch_size + len(text_batch) - 1) // max(len(text_batch), 1)
                    text_batch = (text_batch * repeats)[:batch_size]
                text_emb = self.text_encoder(text_batch).to(x.device)

            # Predict noise ε_θ(x_t, t)
            predicted_noise = self.model(x, t, text_emb)
            if predicted_noise.shape != x.shape:
                predicted_noise = F.interpolate(
                    predicted_noise, size=x.shape[2:], mode="trilinear", align_corners=False
                )

            # Reshape for broadcasting
            shape = (-1,) + (1,) * (x.ndim - 1)
            
            # Get α̅_t
            alpha_prod_t = self.alphas_cumprod[t].view(shape)
            sqrt_alpha_prod_t = self.sqrt_alphas_cumprod[t].view(shape)
            sqrt_one_minus_alpha_prod_t = self.sqrt_one_minus_alphas_cumprod[t].view(shape)

            # Get α̅_{t-1}, handling the final step (prev_t = -1 → α̅ = 1)
            prev_t_clamped = prev_t.clamp(min=0)
            alpha_prod_prev = self.alphas_cumprod[prev_t_clamped].view(shape)
            alpha_prod_prev[prev_t < 0] = 1.0
            sqrt_alpha_prod_prev = torch.sqrt(alpha_prod_prev)

            # Predict x_0 from x_t
            pred_x0 = (x - sqrt_one_minus_alpha_prod_t * predicted_noise) / sqrt_alpha_prod_t
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Compute stochasticity σ_t = η * sqrt(β̃_t) where β̃_t is posterior variance
            eta_value = 0.0 if deterministic else float(max(eta, 0.0))
            posterior_variance = (1.0 - alpha_prod_prev) / (1.0 - alpha_prod_t) * (1.0 - alpha_prod_t / alpha_prod_prev)
            posterior_variance = torch.clamp(posterior_variance, min=1e-12)
            sigma_t = eta_value * torch.sqrt(posterior_variance)
            sigma_t = torch.where(prev_t.view(shape) < 0, torch.zeros_like(sigma_t), sigma_t)

            # Direction pointing to x_t coefficient
            direction_coeff = torch.sqrt(torch.clamp(1.0 - alpha_prod_prev - sigma_t**2, min=1e-12))

            # Sample noise for stochastic sampling
            noise = torch.randn_like(x) if eta_value > 0 and (prev_t >= 0).any() else torch.zeros_like(x)

            # DDIM update: x_{t-1} = sqrt(α̅_{t-1}) * x_0 + sqrt(1 - α̅_{t-1} - σ_t²) * ε + σ_t * z
            prev_sample = sqrt_alpha_prod_prev * pred_x0 + direction_coeff * predicted_noise + sigma_t * noise

            return prev_sample
        
    @torch.no_grad()
    def p_sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device = None,
        text: Optional[str] = None,
        deterministic: bool = False,
        num_inference_steps: Optional[int] = None,
        eta: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Complete reverse diffusion sampling process using DDIM.
        
        Args:
            shape: Output shape (batch_size, channels, frames, height, width)
            device: Device for generation
            text: Text conditioning prompt
            deterministic: Use deterministic DDIM (η=0) if True
            num_inference_steps: Number of sampling steps (can be < timesteps for faster sampling)
            eta: DDIM stochasticity parameter (0=deterministic, 1=stochastic like DDPM)
            
        Returns:
            Generated video tensor in range [-1, 1]
        """
        if device is None:
            device = self.device

        if self.text_encoder is not None and text is None:
            print("⚠️ Warning: Text encoder exists but no text provided. Using unconditional sampling.")

        # Initialize from pure noise: x_T ~ N(0, I)
        x = torch.randn(shape, device=device)

        # Configure inference timesteps
        if num_inference_steps is None:
            num_inference_steps = self.inference_timesteps
        num_inference_steps = max(1, min(num_inference_steps, self.timesteps))

        # Generate timestep schedule
        if num_inference_steps == self.timesteps:
            # Full schedule: [T-1, T-2, ..., 1, 0]
            timesteps = torch.arange(self.timesteps - 1, -1, -1, device=device, dtype=torch.long)
        else:
            # Sparse schedule for accelerated sampling
            timesteps = torch.linspace(self.timesteps - 1, 0, steps=num_inference_steps, device=device)
            timesteps = torch.round(timesteps).long().clamp(0, self.timesteps - 1)
            timesteps = torch.unique_consecutive(timesteps)
            # Ensure we end at t=0
            if timesteps[-1].item() != 0:
                timesteps = torch.cat([timesteps, timesteps.new_tensor([0])])

        # Determine eta (stochasticity coefficient)
        if eta is None:
            eta_value = 0.0 if deterministic or len(timesteps) < self.timesteps else 1.0
        else:
            eta_value = max(0.0, float(eta))

        deterministic_mode = deterministic or eta_value == 0.0
        total_steps = len(timesteps)
        
        # Iterative denoising process
        sampling_desc = f"Sampling ({'DDIM' if deterministic_mode else f'DDIM η={eta_value:.2f}'})"
        for idx, timestep in enumerate(tqdm(timesteps, desc=sampling_desc, ncols=100)):
            t = torch.full((shape[0],), timestep.item(), device=device, dtype=torch.long)
            prev_timestep = timesteps[idx + 1].item() if idx + 1 < total_steps else -1
            prev_t = torch.full((shape[0],), prev_timestep, device=device, dtype=torch.long)

            x = self.p_sample_step(x, t, prev_t, text, deterministic=deterministic_mode, eta=eta_value)

        return torch.clamp(x, -1.0, 1.0)

    def sample(
        self,
        text: str,
        batch_size: int = 1,
        num_frames: int = 28,
        height: int = 128,
        width: int = 128,
        deterministic: bool = False,
        num_inference_steps: Optional[int] = None,
        eta: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Text-conditioned video generation convenience method.
        
        Args:
            text: Text prompt for sign language generation
            batch_size: Number of videos to generate
            num_frames: Number of frames per video
            height: Video height in pixels
            width: Video width in pixels
            deterministic: Use deterministic DDIM sampling
            num_inference_steps: Number of sampling steps
            eta: DDIM stochasticity coefficient

        Returns:
            Generated video tensor of shape (batch_size, 3, num_frames, height, width) in range [-1, 1]
        """
        shape = (batch_size, 3, num_frames, height, width)
        return self.p_sample(
            shape,
            device=self.device,
            text=text,
            deterministic=deterministic,
            num_inference_steps=num_inference_steps,
            eta=eta,
        )
    
    def forward(self, x: torch.Tensor, text: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass implementing the DDPM loss: L = E[||ε - ε_θ(x_t, t)||²]
        
        Args:
            x: Clean video data (batch_size, channels, frames, height, width)
            text: Text conditioning
            
        Returns:
            Tuple of (loss, predicted_noise, actual_noise)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Forward diffusion: q(x_t | x_0)
        noise, x_noisy = self.q_sample(x, t)
        
        # Encode text conditioning
        text_emb = None
        if text is not None and self.text_encoder is not None:
            text_emb = self.text_encoder(text)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t, text_emb)
        
        # MSE loss between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss, predicted_noise, noise

def create_diffusion_model(config) -> DiffusionModel:
    """
    Create a diffusion model based on the configuration
    
    Args:
        config: Configuration object containing model settings
        
    Returns:
        DiffusionModel: Configured diffusion model
    """
    # Create text encoder
    text_encoder = None
    if hasattr(config, 'TEXT_ENCODER_MODEL'):
        try:
            from models.text_encoder import create_text_encoder
            text_encoder = create_text_encoder(config)
            print(f"✅ Text encoder created: {type(text_encoder).__name__}")
        except Exception as e:
            print(f"⚠️ Failed to create text encoder: {e}")
            print("Model will operate unconditionally")
    
    if config.MODEL_ARCHITECTURE == "vit3d":
        from models.architectures.vit3d import ViT3D
        backbone = ViT3D(**config.get_model_config())
    elif config.MODEL_ARCHITECTURE == "unet3d":
        from models.architectures.unet3d import UNet3D
        backbone = UNet3D(**config.get_model_config())
    elif config.MODEL_ARCHITECTURE == "dit3d":
        from models.architectures.dit3d import DiT3D_models
        # Get the specific DiT3D model from the registry
        model_name = getattr(config, 'DIT_MODEL_SIZE', 'DiT3D-S/2')
        if model_name not in DiT3D_models:
            raise ValueError(f"Unknown DiT3D model: {model_name}")
        model_config = config.get_model_config()
        backbone = DiT3D_models[model_name](**model_config)
    elif config.MODEL_ARCHITECTURE == "vivit":
        from models.architectures.vivit import ViViT
        backbone = ViViT(**config.get_model_config())
    elif config.MODEL_ARCHITECTURE == "tinyfusion":
        from models.architectures.tinyfusion import create_tinyfusion_model

        backbone = create_tinyfusion_model(**config.get_model_config())
    else:
        raise ValueError(f"Unknown model architecture: {config.MODEL_ARCHITECTURE}")
    
    # Prepare scheduler parameters based on scheduler type
    scheduler_kwargs = {}
    if config.NOISE_SCHEDULER == "cosine":
        scheduler_kwargs['s'] = getattr(config, 'COSINE_S', 0.008)
        scheduler_kwargs['max_beta'] = getattr(config, 'COSINE_MAX_BETA', 0.999)
    elif config.NOISE_SCHEDULER == "linear":
        scheduler_kwargs['beta_start'] = config.BETA_START
        scheduler_kwargs['beta_end'] = config.BETA_END
    elif config.NOISE_SCHEDULER in ["quadratic", "sigmoid"]:
        scheduler_kwargs['beta_start'] = config.BETA_START
        scheduler_kwargs['beta_end'] = config.BETA_END
    
    model = DiffusionModel(
        model=backbone,
        timesteps=config.TIMESTEPS,
        inference_timesteps=getattr(config, 'INFERENCE_TIMESTEPS', config.TIMESTEPS),
        noise_scheduler=config.NOISE_SCHEDULER,
        device=config.DEVICE,
        text_encoder=text_encoder,
        **scheduler_kwargs
    )
    
    return model
