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
from model import UNet3D

class DiffusionModel(nn.Module):
    """
    Diffusion model for video generation
    
    Args:
        model (nn.Module): The UNet3D model
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
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior calculation
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
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
            torch.Tensor: Slightly less noisy data
        """
        # Predict noise
        predicted_noise = self.model(x, t)
        
        # Get coefficients
        alpha_t = self.alphas.gather(0, t).reshape(-1, 1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1, 1)
        beta_t = self.betas.gather(0, t).reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1, 1)
        
        # Compute predicted x_0
        pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        # Compute mean of reverse process
        pred_prev_mean = (
            torch.sqrt(self.alphas_cumprod_prev.gather(0, t).reshape(-1, 1, 1, 1, 1)) * beta_t * pred_x0 +
            torch.sqrt(alpha_t) * (1.0 - self.alphas_cumprod_prev.gather(0, t).reshape(-1, 1, 1, 1, 1)) * x
        ) / (1.0 - alpha_cumprod_t)
        
        # Add noise if not at final step
        if t.min() > 0:
            posterior_variance_t = self.posterior_variance.gather(0, t).reshape(-1, 1, 1, 1, 1)
            noise = torch.randn_like(x)
            return pred_prev_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return pred_prev_mean
    
    def p_sample(self, shape: Tuple[int, ...], return_all_timesteps: bool = False) -> torch.Tensor:
        """
        Full reverse diffusion process (sampling)
        
        Args:
            shape (tuple): Shape of the tensor to generate
            return_all_timesteps (bool): Whether to return intermediate steps
            
        Returns:
            torch.Tensor: Generated data
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        imgs = []
        if return_all_timesteps:
            imgs = [x]
            
        # Reverse diffusion process with progress bar
        timesteps_range = list(reversed(range(self.timesteps)))
        progress_bar = tqdm(
            timesteps_range, 
            desc="Sampling", 
            leave=False,
            unit="step"
        )
        
        for i in progress_bar:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample_step(x, t)
            
            # Update progress bar with current timestep
            progress_bar.set_postfix({'timestep': i})
            
            if return_all_timesteps:
                imgs.append(x)
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)  # (batch_size, timesteps, channels, frames, height, width)
        else:
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
        
        # Calculate loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss, predicted_noise, noise

def create_diffusion_model(config) -> DiffusionModel:
    """
    Create a diffusion model with the given configuration
    
    Args:
        config: Configuration object
        
    Returns:
        DiffusionModel: Configured diffusion model
    """
    # Create UNet3D model
    unet = UNet3D(
        in_channels=config.UNET_CHANNELS,
        out_channels=config.UNET_CHANNELS,
        dim=config.UNET_DIM,
        dim_mults=config.UNET_DIM_MULTS,
        time_dim=config.UNET_TIME_DIM
    )
    
    # Create diffusion model
    diffusion_model = DiffusionModel(
        model=unet,
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=config.DEVICE
    )
    
    return diffusion_model

def test_diffusion():
    """Test function to verify the diffusion model works correctly"""
    from config import Config
    
    print("Testing diffusion model...")
    
    # Create model
    model = create_diffusion_model(Config)
    model.to(Config.DEVICE)
    
    # Test forward pass (training)
    batch_size = 2
    channels, frames, height, width = Config.INPUT_SHAPE
    x = torch.randn(batch_size, channels, frames, height, width).to(Config.DEVICE)
    
    loss, pred_noise, noise = model(x)
    print(f"Training loss: {loss.item():.4f}")
    print(f"Predicted noise shape: {pred_noise.shape}")
    print(f"Actual noise shape: {noise.shape}")
    
    # Test sampling
    with torch.no_grad():
        samples = model.p_sample((2, channels, frames, height, width))
        print(f"Generated samples shape: {samples.shape}")
        print(f"Generated samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    print("Diffusion model test completed successfully!")

if __name__ == "__main__":
    test_diffusion()
