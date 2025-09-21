"""
Diffusion model implementation
This module contains the forward and reverse diffusion processes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm

# Import noise schedulers
from schedulers.noise_schedulers import create_noise_scheduler

class DiffusionModel(nn.Module):
    """
    Diffusion model for video generation
    
    Args:
        model (nn.Module): The backbone model (UNet3D or ViT3D)
        timesteps (int): Number of diffusion timesteps
        beta_start (float): Starting beta value
        beta_end (float): Ending beta value
        device (torch.device): Device to run the model on
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
        
        print(f"🔧 Initializing {noise_scheduler} noise scheduler...")
        self.noise_scheduler = create_noise_scheduler(noise_scheduler, timesteps, **scheduler_kwargs)
        
        # Create noise schedule
        self.betas = self.noise_scheduler.get_schedule().to(device)
        
        # Precompute alpha values for efficiency during training
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
        
        # Log scheduler statistics
        print(f"✅ {noise_scheduler.capitalize()} scheduler initialized:")
        print(f"   Timesteps: {timesteps}")
        print(f"   Beta range: [{self.betas.min():.6f}, {self.betas.max():.6f}]")
        print(f"   Alpha_cumprod range: [{self.alphas_cumprod.min():.6f}, {self.alphas_cumprod.max():.6f}]")
        
        if text_encoder is not None:
            print(f"✅ Text encoder integrated: {type(text_encoder).__name__}")
        else:
            print("⚠️  No text encoder provided - model will operate unconditionally")
        
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
    
    def p_sample_step(self, x: torch.Tensor, t: torch.Tensor, text: Optional[str] = None, 
                      deterministic: bool = False, eta: float = 0.0) -> torch.Tensor:
        """
        Single reverse diffusion step with DDIM support for faster sampling
        
        Args:
            x (torch.Tensor): Noisy data at timestep t
            t (torch.Tensor): Current timestep  
            text (str, optional): Text conditioning
            deterministic (bool): If True, use DDIM deterministic sampling
            eta (float): DDIM stochasticity parameter (0=deterministic, 1=DDPM)
            
        Returns:
            torch.Tensor: Denoised data at timestep t-1
        """
        with torch.no_grad():
            # Encode text if provided
            text_emb = None
            if text is not None and self.text_encoder is not None:
                batch_size = x.shape[0]
                # Repeat text for batch
                text_batch = [text] * batch_size
                text_emb = self.text_encoder(text_batch)  # (batch_size, embed_dim)
                
                # Ensure text embedding is on the correct device
                text_emb = text_emb.to(x.device)

            predicted_noise = self.model(x, t, text_emb)

            # Get alpha values for current and previous timesteps
            alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            alpha_bar_prev = torch.where(
                t > 0, 
                self.alphas_cumprod[t - 1], 
                torch.ones_like(self.alphas_cumprod[t])
            ).view(-1, 1, 1, 1, 1)
            
            # Predict x_0 from current x_t and predicted noise
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            
            if deterministic or eta == 0.0:
                # DDIM deterministic sampling (much faster)
                x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + \
                        torch.sqrt(1 - alpha_bar_prev) * predicted_noise
            else:
                # Standard DDPM with optional DDIM interpolation
                # Gather per-sample terms
                alpha_t = self.alphas[t].view(-1, 1, 1, 1, 1)
                
                # DDPM mean calculation
                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
                )

                # Calculate variance
                if eta == 1.0:
                    # Use original DDPM posterior variance
                    posterior_var = self.posterior_variance[t].view(-1, 1, 1, 1, 1)
                    sigma = torch.sqrt(posterior_var)
                else:
                    # DDIM interpolated variance
                    sigma_ddpm = torch.sqrt(self.posterior_variance[t]).view(-1, 1, 1, 1, 1)
                    sigma = eta * sigma_ddpm

                # Create noise tensor
                eps = torch.randn_like(x)

                # Build mask to zero-out noise where t == 0 (final step deterministic)
                t_mask = (t > 0).view(-1, 1, 1, 1, 1).to(x.dtype)

                # Apply noise only where appropriate
                x_prev = mean + sigma * eps * t_mask
                
            return x_prev
        
    @torch.no_grad()
    def p_sample(self, shape: Tuple[int, ...], device: torch.device = None, text: Optional[str] = None, 
                deterministic: bool = False, num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        Complete reverse diffusion process (sampling from noise to data)
        
        Args:
            shape (tuple): Shape of the sample to generate (batch_size, channels, frames, height, width)
            device (torch.device): Device to generate on
            text (str, optional): Text conditioning for generation
            deterministic (bool): If True, use deterministic sampling (no noise in intermediate steps)
            num_inference_steps (int, optional): Number of denoising steps (default: use config setting)
            
        Returns:
            torch.Tensor: Generated video sample
        """
        if device is None:
            device = self.device

        # Assert that text is provided if a text encoder exists
        if self.text_encoder is not None:
            assert text is not None, "Text prompt must be provided for a conditioned model"
        
        # Use reduced timesteps for faster inference
        if num_inference_steps is None:
            from config import Config
            num_inference_steps = getattr(Config, 'INFERENCE_TIMESTEPS', 50)
        
        # Create timestep schedule for fast sampling
        if num_inference_steps < self.timesteps:
            # DDIM-style sampling: skip timesteps uniformly
            timestep_schedule = torch.linspace(self.timesteps - 1, 0, num_inference_steps, dtype=torch.long)
        else:
            # Full sampling
            timestep_schedule = torch.arange(self.timesteps - 1, -1, -1)
            
        # Start with random noise (x_T ~ N(0, I))
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion process: gradually denoise using scheduled timesteps
        for i, timestep in enumerate(tqdm(timestep_schedule, desc="Fast Sampling")):
            t = torch.full((shape[0],), timestep, device=device, dtype=torch.long)
            # Use deterministic DDIM sampling for faster inference with fewer steps
            use_deterministic = (num_inference_steps < self.timesteps) or deterministic
            x = self.p_sample_step(x, t, text, deterministic=use_deterministic)

            # Log sampling progress occasionally
            if i % max(1, len(timestep_schedule) // 10) == 0:
                print(f"Sampling step {i+1}/{len(timestep_schedule)} (t={timestep}): x range [{x.min().item():.3f}, {x.max().item():.3f}]")

        # Return sample without final clamping (regular sampling procedure)
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
        # print(f'text is {text}')
        # Repeat text for batch
        text_batch = text
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
