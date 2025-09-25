"""
Diffusion model implementation for text-to-sign language video generation.

This module implements DDPM and DDIM sampling methods with text conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Sequence, Union
from tqdm import tqdm

from schedulers.noise_schedulers import create_noise_scheduler

class DiffusionModel(nn.Module):
    """
    Text-conditioned diffusion model for sign language video generation.
    
    Supports both DDPM (stochastic) and DDIM (deterministic) sampling methods.
    
    Args:
        model (nn.Module): Backbone model (UNet3D, ViT3D, DiT3D, or ViViT)
        timesteps (int): Number of diffusion timesteps for training
        noise_scheduler (str): Type of noise schedule ('linear', 'cosine', etc.)
        device (torch.device): Device for model computation
        text_encoder: Text encoder for conditioning
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        timesteps: int = 1000,
        noise_scheduler: str = "linear",
        device: torch.device = torch.device("cpu"),
        text_encoder=None,
        **scheduler_kwargs
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.noise_scheduler_type = noise_scheduler
        self.text_encoder = text_encoder
        
        self.noise_scheduler = create_noise_scheduler(noise_scheduler, timesteps, **scheduler_kwargs)
        self.betas = self.noise_scheduler.get_schedule().to(device)
        
        # Precompute alpha values for efficient sampling
        self.alphas, self.alphas_cumprod, self.alphas_cumprod_prev = self.noise_scheduler.compute_alpha_schedule(self.betas)
        
        # Move to device
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        
        # Precompute frequently used square roots
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior calculation in reverse sampling
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # Clamp posterior variance to avoid numerical issues
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-6)
        
        # Minimal initialization summary
        if text_encoder is not None:
            print(f"✅ Diffusion model initialized: {noise_scheduler} scheduler, {timesteps} steps, text-conditioned")
        else:
            print(f"✅ Diffusion model initialized: {noise_scheduler} scheduler, {timesteps} steps, unconditional")
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Implements the forward diffusion process that gradually adds noise to clean data:
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        
        Where α̅_t is the cumulative product of alphas up to timestep t.
        
        Args:
            x_start (torch.Tensor): Clean video data (x_0)
            t (torch.Tensor): Timestep tensor with shape (batch_size,)
            noise (torch.Tensor, optional): Noise to add. If None, random noise is generated
            
        Returns:
            torch.Tensor: Noisy data at timestep t (x_t)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        alpha_bar_t = self.alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1)))
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        # Apply the noise schedule: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        x_noisy = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        return noise, x_noisy

    def p_sample_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text: Optional[Union[str, Sequence[str]]] = None,
        deterministic: bool = False,
        eta: Optional[float] = None,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a single reverse diffusion step.

        Supports both stochastic (DDPM) and deterministic (DDIM) sampling depending on
        ``deterministic`` / ``eta`` parameters.

        Args:
            x (torch.Tensor): Noisy latents at timestep ``t``.
            t (torch.Tensor): Current timestep indices for each batch item.
            text (Union[str, Sequence[str]], optional): Text conditioning prompts. If a single
                string is provided it will be broadcast to the batch. Ignored when
                ``text_embeddings`` is supplied.
            deterministic (bool): When ``True`` perform deterministic DDIM update (eta == 0).
            eta (float, optional): Controls the amount of stochasticity. ``eta == 1`` matches DDPM,
                ``eta == 0`` matches DDIM. When ``None`` falls back to 1.0 for stochastic sampling
                unless ``deterministic`` is set.
            text_embeddings (torch.Tensor, optional): Pre-computed text embeddings with shape
                ``(batch, embed_dim)``. When provided it is used directly to avoid re-encoding text.

        Returns:
            torch.Tensor: Latents approximating ``x_{t-1}``.
        """
        with torch.no_grad():
            batch_size = x.shape[0]

            # Resolve text conditioning
            if text_embeddings is not None:
                text_emb = text_embeddings.to(x.device)
            elif text is not None and self.text_encoder is not None:
                if isinstance(text, str):
                    text_batch = [text] * batch_size
                else:
                    text_batch = list(text)
                    if len(text_batch) != batch_size:
                        raise ValueError(
                            f"Provided {len(text_batch)} text prompts for batch size {batch_size}."
                        )
                text_emb = self.text_encoder(text_batch).to(x.device)
            else:
                text_emb = None

            predicted_noise = self.model(x, t, text_emb)

            # Gather schedules for the current timestep
            alpha_bar_t = self.alphas_cumprod[t].view(batch_size, 1, 1, 1, 1)
            prev_t = torch.clamp(t - 1, min=0)
            alpha_bar_prev = self.alphas_cumprod[prev_t].view(batch_size, 1, 1, 1, 1)
            alpha_bar_prev = torch.where(
                (t > 0).view(batch_size, 1, 1, 1, 1),
                alpha_bar_prev,
                torch.ones_like(alpha_bar_prev),
            )

            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))

            # Predict the clean sample x_0
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            nonzero_mask = (t > 0).view(batch_size, 1, 1, 1, 1)

            if deterministic or (eta is not None and eta == 0.0):
                # DDIM deterministic update
                sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
                sqrt_one_minus_alpha_bar_prev = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=1e-8))
                x_prev = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * predicted_noise
                return torch.where(nonzero_mask, x_prev, x0_pred)

            # Default to stochastic DDPM sampling when eta is unspecified
            eta = 1.0 if eta is None else eta

            beta_t = self.betas[t].view(batch_size, 1, 1, 1, 1)
            alpha_t = self.alphas[t].view(batch_size, 1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            one_minus_alpha_bar_t = torch.clamp(1.0 - alpha_bar_t, min=1e-8)

            posterior_mean_coef1 = beta_t * torch.sqrt(alpha_bar_prev) / one_minus_alpha_bar_t
            posterior_mean_coef2 = (1.0 - alpha_bar_prev) * sqrt_alpha_t / one_minus_alpha_bar_t
            posterior_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x

            posterior_variance = self.posterior_variance[t].view(batch_size, 1, 1, 1, 1)
            sigma = torch.sqrt(posterior_variance) * eta
            noise = torch.randn_like(x)

            x_prev = posterior_mean + sigma * noise
            return x_prev

    @torch.no_grad()
    def p_sample(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        text: Optional[Union[str, Sequence[str]]] = None,
        deterministic: bool = False,
        num_inference_steps: Optional[int] = None,
        eta: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Complete reverse diffusion process (sampling from noise to data).

        Args:
            shape (tuple): Output tensor shape ``(batch, channels, frames, height, width)``.
            device (torch.device, optional): Target device. Defaults to model device.
            text (Union[str, Sequence[str]], optional): Conditioning prompts. When a single
                string is provided it is broadcast across the batch.
            deterministic (bool): Use deterministic DDIM sampling (no injected noise).
            num_inference_steps (int, optional): Number of denoising steps. Defaults to
                ``Config.INFERENCE_TIMESTEPS`` when available.
            eta (float, optional): Stochasticity parameter forwarded to ``p_sample_step``.

        Returns:
            torch.Tensor: Generated latent video sample in ``[-1, 1]`` range.
        """
        if device is None:
            device = self.device

        batch_size = shape[0]

        # Prepare text conditioning once for efficiency
        text_embeddings = None
        if self.text_encoder is not None:
            if text is None:
                raise ValueError("Text prompt must be provided for a conditioned model")
            if isinstance(text, str):
                text_batch = [text] * batch_size
            else:
                text_batch = list(text)
                if len(text_batch) != batch_size:
                    raise ValueError(
                        f"Expected {batch_size} text prompts, received {len(text_batch)}"
                    )
            text_embeddings = self.text_encoder(text_batch).to(device)

        # Determine number of denoising steps
        if num_inference_steps is None:
            from config import Config
            num_inference_steps = getattr(Config, "INFERENCE_TIMESTEPS", self.timesteps)

        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        # Build timestep schedule (descending)
        if num_inference_steps < self.timesteps:
            timesteps = torch.linspace(
                self.timesteps - 1,
                0,
                num_inference_steps,
                device=device,
            ).round().long()
        else:
            timesteps = torch.arange(self.timesteps - 1, -1, -1, device=device)

        # Allocate initial Gaussian noise x_T ~ N(0, I)
        x = torch.randn(shape, device=device)

        # Choose eta default (0 for deterministic schedule shorter than training steps)
        if eta is None:
            eta = 0.0 if deterministic or num_inference_steps < self.timesteps else 1.0

        use_deterministic = deterministic or (num_inference_steps < self.timesteps and eta == 0.0)

        for i, timestep in enumerate(tqdm(timesteps, desc="Sampling", leave=False)):
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            x = self.p_sample_step(
                x,
                t,
                text=None if text_embeddings is not None else text,
                deterministic=use_deterministic,
                eta=eta,
                text_embeddings=text_embeddings,
            )

            # Only log at start, middle, and end for performance
            if i == 0 or i == len(timesteps) // 2 or i == len(timesteps) - 1:
                x_min, x_max = x.min().item(), x.max().item()
                print(
                    f"Sampling step {i + 1}/{len(timesteps)} (t={int(timestep)}): "
                    f"x range [{x_min:.3f}, {x_max:.3f}]"
                )

        return x
    
    def sample(self, text: str, batch_size: int = 1, num_frames: int = 28, height: int = 128, width: int = 128, 
               deterministic: bool = True, num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        Convenience method for fast text-conditioned video generation
        
        Args:
            text (str): Text prompt for sign language generation
            batch_size (int): Number of videos to generate
            num_frames (int): Number of frames in the video
            height (int): Video height
            width (int): Video width
            deterministic (bool): Whether to use DDIM deterministic sampling (default: True for speed)
            num_inference_steps (int, optional): Number of denoising steps (default: fast inference config)
            
        Returns:
            torch.Tensor: Generated video tensor with shape (batch_size, 3, num_frames, height, width)
        """
        shape = (batch_size, 3, num_frames, height, width)
        
        # Use fast inference by default
        if num_inference_steps is None:
            from config import Config
            num_inference_steps = getattr(Config, 'INFERENCE_TIMESTEPS', 50)
            
        return self.p_sample(
            shape, 
            device=self.device, 
            text=text, 
            deterministic=deterministic,
            num_inference_steps=num_inference_steps
        )
    
    def forward(self, x: torch.Tensor, text: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass for diffusion model
        
        Args:
            x (torch.Tensor): Clean video data with shape (batch_size, channels, frames, height, width)
            text (str, optional): Text conditioning
            
        Returns:
            tuple: (loss, predicted_noise, actual_noise)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps for each item in the batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Apply noise scheduler: q(x_t | x_0) - forward diffusion process
        noise, x_noisy = self.q_sample(x, t)
        
        # Encode text if provided
        text_emb = None
        text_batch = text
        if len(text) != batch_size:
            raise ValueError(f"Text batch size {len(text)} does not match input batch size {batch_size}")
        text_emb = self.text_encoder(text_batch)  # (batch_size, embed_dim)
        predicted_noise = self.model(x_noisy, t, text_emb)        
        # Calculate denoising loss (MSE between predicted and actual noise)
        # This is the standard DDPM training objective: L = E[||ε - ε_θ(x_t, t)||²]
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
        except Exception as e:
            print(f"⚠️ Text encoder initialization failed, using unconditional model: {e}")
            text_encoder = None
    
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
        noise_scheduler=config.NOISE_SCHEDULER,
        device=config.DEVICE,
        text_encoder=text_encoder,
        **scheduler_kwargs
    )
    
    return model
