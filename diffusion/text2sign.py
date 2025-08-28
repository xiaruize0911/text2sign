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
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Create beta schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        # Precompute alpha values for efficiency
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Clamp alphas_cumprod to avoid numerical issues
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=1e-6)
        
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior calculation
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # Clamp posterior variance to avoid division by zero
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-6)
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: add noise to clean data
        
        Args:
            x_start (torch.Tensor): Clean data
            t (torch.Tensor): Timestep
            noise (torch.Tensor, optional): Noise to add. If None, random noise is generated
            
        Returns:
            torch.Tensor: Noisy data
        """
        if noise is None:
            # Use deterministic seed based on input for reproducible results
            seed = hash(tuple(x_start.flatten()[:100].tolist())) % 2**32
            torch.manual_seed(seed)
            noise = torch.randn_like(x_start)
        """
        Forward diffusion process: add noise to clean data
        
        Args:
            x_start (torch.Tensor): Clean data
            t (torch.Tensor): Timestep
            noise (torch.Tensor, optional): Noise to add. If None, random noise is generated
            
        Returns:
            torch.Tensor: Noisy data
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(0, t)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t)
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.reshape(-1, 1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample_step(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Single reverse diffusion step
        
        Args:
            x (torch.Tensor): Noisy data at timestep t
            t (torch.Tensor): Current timestep
            
        Returns:
            torch.Tensor: Denoised data at timestep t-1
        """
        # Predict noise
        predicted_noise = self.model(x, t)
        
        # Get alpha values for current timestep
        alpha = self.alphas.gather(0, t).reshape(-1, 1, 1, 1, 1)
        alpha_cumprod = self.alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1, 1)
        beta = self.betas.gather(0, t).reshape(-1, 1, 1, 1, 1)
        
        # Calculate predicted x_0
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        
        # Predict original sample
        pred_original_sample = (x - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod
        
        # Calculate previous sample mean
        sqrt_alpha = torch.sqrt(alpha)
        pred_sample_direction = (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * predicted_noise
        prev_sample = (x - pred_sample_direction) / sqrt_alpha
        
        # Add noise for non-final steps
        if t.min() > 0:
            variance = self.posterior_variance.gather(0, t).reshape(-1, 1, 1, 1, 1)
            noise = torch.randn_like(x)
            prev_sample += torch.sqrt(variance) * noise
        
        return prev_sample
    
    def p_sample(self, shape: Tuple[int, ...], device: torch.device = None) -> torch.Tensor:
        """
        Complete reverse diffusion process (sampling)
        
        Args:
            shape (tuple): Shape of the sample to generate
            device (torch.device): Device to generate on
            
        Returns:
            torch.Tensor: Generated sample
        """
        if device is None:
            device = self.device
            
        # Start with random noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion process
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample_step(x, t)
            
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass
        
        Args:
            x (torch.Tensor): Clean data
            
        Returns:
            tuple: (loss, predicted_noise, noise)
        """
        batch_size = x.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Forward diffusion
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t)
        
        # Check for NaN in predictions
        if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
            print(f"NaN/Inf in model prediction at step {t}")
            print(f"Input x_noisy range: [{x_noisy.min():.3f}, {x_noisy.max():.3f}]")
            # Return zero loss to continue training
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            predicted_noise = torch.zeros_like(noise)
        else:
            # Calculate loss (MSE between predicted and actual noise)
            loss = F.mse_loss(predicted_noise, noise)
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss detected!")
            print(f"Predicted noise range: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")
            print(f"Noise range: [{noise.min():.3f}, {noise.max():.3f}]")
            print(f"x_noisy range: [{x_noisy.min():.3f}, {x_noisy.max():.3f}]")
            print(f"t values: {t}")
            # Return a small loss to continue training
            loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
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
    
    model = DiffusionModel(
        model=backbone,
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=config.DEVICE
    )
    
    return model
