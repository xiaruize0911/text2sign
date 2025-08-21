import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional


class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) scheduler for training and inference.
    
    This implements the noise scheduling and sampling algorithms from the DDPM paper.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon"
    ):
        """
        Initialize the DDPM scheduler.
        
        Args:
            num_train_timesteps: Number of diffusion steps
            beta_start: Starting value of beta
            beta_end: Ending value of beta
            beta_schedule: Type of beta schedule ("linear", "scaled_linear", "cosine")
            prediction_type: What the model predicts ("epsilon", "v_prediction")
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        
        # Generate beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # Used in DDPM paper for ImageNet
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior computation
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to the original samples according to the noise schedule.
        
        Args:
            original_samples: Original clean samples
            noise: Random noise to add
            timesteps: Timesteps for each sample in the batch
            
        Returns:
            Noisy samples
        """
        # Make sure alphas_cumprod are on the same device
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device=original_samples.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device=original_samples.device)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
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
        sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the sample at the previous timestep by reversing the SDE.
        
        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current discrete timestep in the diffusion chain
            sample: Current instance of sample being created by diffusion process
            
        Returns:
            Previous sample
        """
        t = timestep
        
        # Make sure all tensors are on the same device
        device = sample.device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        
        # 1. Compute predicted original sample (x_0) from model output
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])
        
        # Ensure proper broadcasting
        while len(sqrt_alpha_cumprod.shape) < len(sample.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - sqrt_one_minus_alpha_cumprod * model_output) / sqrt_alpha_cumprod
        elif self.prediction_type == "v_prediction":
            pred_original_sample = sqrt_alpha_cumprod * sample - sqrt_one_minus_alpha_cumprod * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # 2. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = self.posterior_mean_coef1[t]
        current_sample_coeff = self.posterior_mean_coef2[t]
        
        # Ensure proper broadcasting
        while len(pred_original_sample_coeff.shape) < len(sample.shape):
            pred_original_sample_coeff = pred_original_sample_coeff.unsqueeze(-1)
            current_sample_coeff = current_sample_coeff.unsqueeze(-1)
        
        # 3. Compute predicted previous sample μ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # 4. Add noise
        variance = 0
        if t > 0:
            noise = torch.randn_like(sample)
            posterior_variance = self.posterior_variance[t]
            while len(posterior_variance.shape) < len(sample.shape):
                posterior_variance = posterior_variance.unsqueeze(-1)
            variance = torch.sqrt(posterior_variance) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample
    
    def set_timesteps(self, num_inference_steps: int):
        """
        Set the timesteps used for the diffusion chain during inference.
        
        Args:
            num_inference_steps: Number of diffusion steps during inference
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def scale_model_input(self, sample: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        Scale the denoising model input. For DDPM, this is a no-op.
        
        Args:
            sample: Input sample
            timestep: Current timestep
            
        Returns:
            Scaled sample
        """
        return sample


class DDIMScheduler(DDPMScheduler):
    """
    DDIM (Denoising Diffusion Implicit Models) scheduler for faster sampling.
    
    Extends DDPM scheduler with deterministic sampling capability.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = 0.0  # 0.0 for deterministic, 1.0 for stochastic (DDPM)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Predict the sample at the previous timestep using DDIM.
        
        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current discrete timestep in the diffusion chain
            sample: Current instance of sample being created by diffusion process
            eta: Stochasticity parameter (0.0 = deterministic, 1.0 = stochastic)
            
        Returns:
            Previous sample
        """
        # Make sure all tensors are on the same device
        device = sample.device
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=device)
        
        beta_prod_t = 1 - alpha_prod_t
        
        # 1. Predict original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = torch.sqrt(alpha_prod_t) * sample - torch.sqrt(beta_prod_t) * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # 2. Direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - eta**2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)) * model_output
        
        # 3. Compute x_t without "random noise"
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            # Add stochastic noise
            variance = eta**2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + torch.sqrt(variance) * noise
        
        return prev_sample


def test_scheduler():
    """Test the diffusion scheduler."""
    # Test DDPM scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Test adding noise
    batch_size = 2
    frames = 8
    height = 32
    width = 32
    
    original_samples = torch.randn(batch_size, 3, frames, height, width)
    noise = torch.randn_like(original_samples)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)
    print(f"Original shape: {original_samples.shape}")
    print(f"Noisy shape: {noisy_samples.shape}")
    
    # Test denoising step
    model_output = torch.randn_like(noisy_samples)
    prev_sample = scheduler.step(model_output, int(timesteps[0].item()), noisy_samples[0:1])
    print(f"Denoised shape: {prev_sample.shape}")
    
    # Test DDIM scheduler
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000)
    ddim_scheduler.set_timesteps(50)  # 50 inference steps
    print(f"DDIM timesteps: {ddim_scheduler.timesteps[:10]}")  # First 10 timesteps
    
    print("Scheduler test passed!")


if __name__ == "__main__":
    test_scheduler()
