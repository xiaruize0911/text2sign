"""
DDIM (Denoising Diffusion Implicit Models) Scheduler
Implements both training and sampling procedures
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np


class DDIMScheduler:
    """
    DDIM Scheduler for diffusion models
    
    Supports both DDPM training and DDIM deterministic/stochastic sampling
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
    ):
        """
        Args:
            num_train_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Type of beta schedule ("linear" or "cosine")
            clip_sample: Whether to clip predicted samples
            prediction_type: What the model predicts ("epsilon" or "v_prediction")
            thresholding: Whether to use dynamic thresholding
            dynamic_thresholding_ratio: Ratio for dynamic thresholding
            sample_max_value: Max value for clipping
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        
        # Compute betas
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = self._squaredcos_cap_v2_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # For sampling
        self.num_inference_steps = None
        self.timesteps = None
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _squaredcos_cap_v2_schedule(self, timesteps: int) -> torch.Tensor:
        """Squared cosine schedule used in improved DDPM"""
        return self._cosine_beta_schedule(timesteps)
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = "cpu"):
        """
        Set the timesteps for inference
        
        Args:
            num_inference_steps: Number of steps for inference
            device: Device to put tensors on
        """
        self.num_inference_steps = num_inference_steps
        
        # DDIM uses uniform spacing
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
    
    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        """Compute variance for given timestep"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        
        return variance
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples for training
        
        Args:
            original_samples: Clean samples x_0
            noise: Noise to add
            timesteps: Timesteps for each sample
        
        Returns:
            Noisy samples x_t
        """
        # Move coefficients to correct device and dtype
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(original_samples.device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(original_samples.device)
        
        sqrt_alpha_prod = sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one DDIM denoising step
        
        Args:
            model_output: Output from the model (predicted noise or v)
            timestep: Current timestep
            sample: Current noisy sample x_t
            eta: Stochasticity factor (0 = deterministic DDIM, 1 = DDPM)
            generator: Random generator for reproducibility
        
        Returns:
            Tuple of (predicted x_{t-1}, predicted x_0)
        """
        # Get previous timestep
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        
        # Compute predicted x_0
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Clip predicted x_0
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute variance
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** 0.5
        
        # Compute direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output
        
        # Compute x_{t-1}
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0
        if eta > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape, 
                generator=generator, 
                device=device, 
                dtype=model_output.dtype
            )
            prev_sample = prev_sample + std_dev_t * noise
        
        return prev_sample, pred_original_sample
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity for v-prediction
        
        v = sqrt(alpha_t) * noise - sqrt(1 - alpha_t) * sample
        """
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(sample.device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(sample.device)
        
        sqrt_alpha_prod = sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[timesteps]
        
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        
        return velocity


# Import F for F.pad
import torch.nn.functional as F


def get_ddim_scheduler(config) -> DDIMScheduler:
    """Create DDIM scheduler from config"""
    return DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        clip_sample=config.clip_sample,
        prediction_type=config.prediction_type,
    )


if __name__ == "__main__":
    # Test the scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )
    
    # Test adding noise
    x = torch.randn(2, 3, 16, 64, 64)
    noise = torch.randn_like(x)
    timesteps = torch.tensor([100, 500])
    
    noisy_x = scheduler.add_noise(x, noise, timesteps)
    print(f"Original shape: {x.shape}")
    print(f"Noisy shape: {noisy_x.shape}")
    
    # Test sampling
    scheduler.set_timesteps(50)
    print(f"Inference timesteps: {scheduler.timesteps[:10]}...")
    
    # Test step
    model_output = torch.randn_like(x)
    prev_sample, pred_x0 = scheduler.step(model_output, 500, noisy_x, eta=0.0)
    print(f"Previous sample shape: {prev_sample.shape}")
    print(f"Predicted x0 shape: {pred_x0.shape}")
