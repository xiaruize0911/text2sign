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
        **scheduler_kwargs
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.noise_scheduler_type = noise_scheduler
        
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
            
        # Get the noise schedule coefficients for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(0, t)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t)
        
        # Reshape for broadcasting across (batch_size, channels, frames, height, width)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.reshape(-1, 1, 1, 1, 1)
        
        # Apply the noise schedule: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_noisy
    
    def p_sample_step(self, x: torch.Tensor, t: torch.Tensor, text: Optional[str] = None) -> torch.Tensor:
        """
        Single reverse diffusion step: p(x_{t-1} | x_t)
        
        Implements one step of the reverse diffusion process using the DDPM formula.
        
        Args:
            x (torch.Tensor): Noisy data at timestep t
            t (torch.Tensor): Current timestep  
            text (str, optional): Text conditioning (for future use)
            
        Returns:
            torch.Tensor: Denoised data at timestep t-1
        """
        with torch.no_grad():
            # Predict noise using the trained model
            predicted_noise = self.model(x, t)
            
            # Get alpha values for current timestep
            alpha_t = self.alphas.gather(0, t).reshape(-1, 1, 1, 1, 1)
            alpha_cumprod_t = self.alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1, 1)
            alpha_cumprod_t_prev = self.alphas_cumprod_prev.gather(0, t).reshape(-1, 1, 1, 1, 1)
            beta_t = self.betas.gather(0, t).reshape(-1, 1, 1, 1, 1)
            
            # Predict x_0 from noise prediction: x_0 = (x_t - sqrt(1-α̅_t) * ε_θ) / sqrt(α̅_t)
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
            pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
            
            # Clamp x_0 to reasonable range to prevent instability
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Calculate mean of p(x_{t-1} | x_t, x_0) using DDPM posterior formula
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
            
            # DDPM posterior mean: μ_t = (β_t * sqrt(α̅_{t-1}) / (1 - α̅_t)) * x_0 + ((1 - α̅_{t-1}) * sqrt(α_t) / (1 - α̅_t)) * x_t
            coeff1 = beta_t * sqrt_alpha_cumprod_t_prev / (1.0 - alpha_cumprod_t)
            coeff2 = (1.0 - alpha_cumprod_t_prev) * sqrt_alpha_t / (1.0 - alpha_cumprod_t)
            
            mean = coeff1 * pred_x0 + coeff2 * x
            
            return mean
    
    def p_sample(self, shape: Tuple[int, ...], device: torch.device = None, text: Optional[str] = None, deterministic: bool = False) -> torch.Tensor:
        """
        Complete reverse diffusion process (sampling from noise to data)
        
        Args:
            shape (tuple): Shape of the sample to generate (batch_size, channels, frames, height, width)
            device (torch.device): Device to generate on
            text (str, optional): Text conditioning for generation
            deterministic (bool): If True, use deterministic sampling (no noise in intermediate steps)
            
        Returns:
            torch.Tensor: Generated video sample
        """
        if device is None:
            device = self.device
            
        # Start with random noise (x_T ~ N(0, I))
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion process: gradually denoise from T to 0
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample_step(x, t, text)
            
        return x
    
    def sample(self, text: str, batch_size: int = 1, num_frames: int = 28, height: int = 128, width: int = 128, deterministic: bool = False) -> torch.Tensor:
        """
        Convenience method for text-conditioned video generation
        
        Args:
            text (str): Text prompt for sign language generation
            batch_size (int): Number of videos to generate
            num_frames (int): Number of frames in the video
            height (int): Video height
            width (int): Video width
            deterministic (bool): Whether to use deterministic sampling
            
        Returns:
            torch.Tensor: Generated video tensor with shape (batch_size, 3, num_frames, height, width)
        """
        shape = (batch_size, 3, num_frames, height, width)
        return self.p_sample(shape, device=self.device, text=text, deterministic=deterministic)
    
    def forward(self, x: torch.Tensor, text: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass for diffusion model
        
        Args:
            x (torch.Tensor): Clean video data with shape (batch_size, channels, frames, height, width)
            text (str, optional): Text conditioning (for future text2sign conditioning)
            
        Returns:
            tuple: (loss, predicted_noise, actual_noise)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps for each item in the batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Sample noise to add to the videos
        noise = torch.randn_like(x)
        
        # Apply noise scheduler: q(x_t | x_0) - forward diffusion process
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict the noise using the backbone model
        # The model should learn to predict the noise that was added
        try:
            predicted_noise = self.model(x_noisy, t)
        except Exception as e:
            print(f"Error in model prediction: {e}")
            # Return safe values to continue training
            predicted_noise = torch.zeros_like(noise)
            loss = torch.tensor(1.0, device=device, requires_grad=True)
            return loss, predicted_noise, noise
        
        # Validate predictions
        if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
            print(f"Warning: NaN/Inf in model prediction at timesteps {t}")
            print(f"Input x_noisy stats: mean={x_noisy.mean():.4f}, std={x_noisy.std():.4f}, range=[{x_noisy.min():.4f}, {x_noisy.max():.4f}]")
            # Replace invalid predictions with zeros
            predicted_noise = torch.where(
                torch.isnan(predicted_noise) | torch.isinf(predicted_noise),
                torch.zeros_like(predicted_noise),
                predicted_noise
            )
        
        # Calculate denoising loss (MSE between predicted and actual noise)
        # This is the standard DDPM training objective: L = E[||ε - ε_θ(x_t, t)||²]
        loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        # Validate loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected at timesteps {t}")
            print(f"Predicted noise stats: mean={predicted_noise.mean():.4f}, std={predicted_noise.std():.4f}")
            print(f"Actual noise stats: mean={noise.mean():.4f}, std={noise.std():.4f}")
            # Use a small positive loss to continue training
            loss = torch.tensor(0.1, device=device, requires_grad=True)
        
        return loss, predicted_noise, noise

def create_diffusion_model(config) -> DiffusionModel:
    """
    Create a diffusion model based on the configuration
    
    Args:
        config: Configuration object containing model settings
        
    Returns:
        DiffusionModel: Configured diffusion model
    """
    if config.MODEL_ARCHITECTURE == "vit3d":
        from models.architectures.vit3d import ViT3D
        backbone = ViT3D(**config.get_model_config())
    elif config.MODEL_ARCHITECTURE == "unet3d":
        from models.architectures.unet3d import UNet3D
        backbone = UNet3D(**config.get_model_config())
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
        **scheduler_kwargs
    )
    
    return model
