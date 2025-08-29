"""
Noise schedulers for diffusion models
This module implements various noise scheduling strategies for the forward diffusion process
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class NoiseScheduler(ABC):
    """
    Abstract base class for noise schedulers
    """
    
    @abstractmethod
    def __init__(self, timesteps: int, **kwargs):
        """Initialize the noise scheduler"""
        pass
    
    @abstractmethod
    def get_schedule(self) -> torch.Tensor:
        """Return the beta schedule"""
        pass
    
    def compute_alpha_schedule(self, betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute alpha-related values from beta schedule
        
        Args:
            betas (torch.Tensor): Beta schedule
            
        Returns:
            tuple: (alphas, alphas_cumprod, alphas_cumprod_prev)
        """
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Clamp to avoid numerical issues
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-6, max=1.0 - 1e-6)
        
        # Previous timestep values (pad with 1.0 for t=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        return alphas, alphas_cumprod, alphas_cumprod_prev


class LinearNoiseScheduler(NoiseScheduler):
    """
    Linear noise scheduler (standard DDPM)
    
    Args:
        timesteps (int): Number of diffusion timesteps
        beta_start (float): Starting beta value
        beta_end (float): Ending beta value
    """
    
    def __init__(self, timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
    
    def get_schedule(self) -> torch.Tensor:
        """
        Create linear beta schedule
        
        Returns:
            torch.Tensor: Linear beta schedule
        """
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)


class CosineNoiseScheduler(NoiseScheduler):
    """
    Cosine noise scheduler for improved training dynamics
    
    Based on "Improved Denoising Diffusion Probabilistic Models" by Nichol & Dhariwal
    https://arxiv.org/abs/2102.09672
    
    The cosine schedule provides:
    - More gradual noise addition in early timesteps
    - Better preservation of image structure
    - Improved training stability
    - Better sample quality
    
    Args:
        timesteps (int): Number of diffusion timesteps
        s (float): Small offset to prevent β from being too small near t=0
        max_beta (float): Maximum value for beta to prevent instability
    """
    
    def __init__(self, timesteps: int, s: float = 0.008, max_beta: float = 0.999):
        self.timesteps = timesteps
        self.s = s
        self.max_beta = max_beta
    
    def get_schedule(self) -> torch.Tensor:
        """
        Create cosine beta schedule
        
        The cosine schedule is computed as:
        f(t) = cos((t/T + s) / (1 + s) * π/2)²
        α̅_t = f(t) / f(0)
        β_t = 1 - α̅_t / α̅_{t-1}
        
        Returns:
            torch.Tensor: Cosine beta schedule
        """
        def f(t):
            """Cosine function for alpha_cumprod calculation"""
            return torch.cos((t / self.timesteps + self.s) / (1 + self.s) * np.pi / 2) ** 2
        
        # Compute alpha_cumprod values
        timesteps_tensor = torch.arange(self.timesteps + 1, dtype=torch.float32)
        alphas_cumprod = f(timesteps_tensor) / f(torch.tensor(0.0))
        
        # Compute betas from alpha_cumprod
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        
        # Clip to prevent numerical instability
        betas = torch.clamp(betas, min=0.0, max=self.max_beta)
        
        return betas


class QuadraticNoiseScheduler(NoiseScheduler):
    """
    Quadratic noise scheduler for alternative dynamics
    
    Args:
        timesteps (int): Number of diffusion timesteps
        beta_start (float): Starting beta value
        beta_end (float): Ending beta value
    """
    
    def __init__(self, timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
    
    def get_schedule(self) -> torch.Tensor:
        """
        Create quadratic beta schedule
        
        Returns:
            torch.Tensor: Quadratic beta schedule
        """
        t = torch.linspace(0, 1, self.timesteps)
        betas = self.beta_start + (self.beta_end - self.beta_start) * t ** 2
        return betas


class SigmoidNoiseScheduler(NoiseScheduler):
    """
    Sigmoid noise scheduler for smooth transitions
    
    Args:
        timesteps (int): Number of diffusion timesteps
        beta_start (float): Starting beta value
        beta_end (float): Ending beta value
        steepness (float): Steepness of the sigmoid curve
    """
    
    def __init__(self, timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02, steepness: float = 10.0):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.steepness = steepness
    
    def get_schedule(self) -> torch.Tensor:
        """
        Create sigmoid beta schedule
        
        Returns:
            torch.Tensor: Sigmoid beta schedule
        """
        t = torch.linspace(-self.steepness/2, self.steepness/2, self.timesteps)
        sigmoid_values = torch.sigmoid(t)
        betas = self.beta_start + (self.beta_end - self.beta_start) * sigmoid_values
        return betas


def create_noise_scheduler(scheduler_type: str, timesteps: int, **kwargs) -> NoiseScheduler:
    """
    Factory function to create noise schedulers
    
    Args:
        scheduler_type (str): Type of scheduler ('linear', 'cosine', 'quadratic', 'sigmoid')
        timesteps (int): Number of timesteps
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        NoiseScheduler: Configured noise scheduler
        
    Raises:
        ValueError: If scheduler_type is not supported
    """
    schedulers = {
        'linear': LinearNoiseScheduler,
        'cosine': CosineNoiseScheduler,
        'quadratic': QuadraticNoiseScheduler,
        'sigmoid': SigmoidNoiseScheduler
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Supported types: {list(schedulers.keys())}")
    
    return schedulers[scheduler_type](timesteps, **kwargs)


def compare_schedulers(timesteps: int = 300) -> None:
    """
    Utility function to compare different noise schedulers
    
    Args:
        timesteps (int): Number of timesteps to compare
    """
    import matplotlib.pyplot as plt
    
    schedulers = {
        'Linear': LinearNoiseScheduler(timesteps),
        'Cosine': CosineNoiseScheduler(timesteps),
        'Quadratic': QuadraticNoiseScheduler(timesteps),
        'Sigmoid': SigmoidNoiseScheduler(timesteps)
    }
    
    plt.figure(figsize=(15, 5))
    
    # Plot beta schedules
    plt.subplot(1, 3, 1)
    for name, scheduler in schedulers.items():
        betas = scheduler.get_schedule()
        plt.plot(betas.numpy(), label=f'{name}')
    plt.title('Beta Schedules')
    plt.xlabel('Timestep')
    plt.ylabel('Beta')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot alpha_cumprod schedules
    plt.subplot(1, 3, 2)
    for name, scheduler in schedulers.items():
        betas = scheduler.get_schedule()
        _, alphas_cumprod, _ = scheduler.compute_alpha_schedule(betas)
        plt.plot(alphas_cumprod.numpy(), label=f'{name}')
    plt.title('Alpha Cumprod Schedules')
    plt.xlabel('Timestep')
    plt.ylabel('Alpha Cumprod')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot noise levels (sqrt(1 - alpha_cumprod))
    plt.subplot(1, 3, 3)
    for name, scheduler in schedulers.items():
        betas = scheduler.get_schedule()
        _, alphas_cumprod, _ = scheduler.compute_alpha_schedule(betas)
        noise_levels = torch.sqrt(1 - alphas_cumprod)
        plt.plot(noise_levels.numpy(), label=f'{name}')
    plt.title('Noise Levels')
    plt.xlabel('Timestep')
    plt.ylabel('Sqrt(1 - Alpha Cumprod)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_scheduler_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Scheduler comparison saved as 'noise_scheduler_comparison.png'")


if __name__ == "__main__":
    # Test the schedulers
    print("🧪 Testing noise schedulers...")
    
    timesteps = 300
    
    # Test each scheduler
    schedulers_to_test = ['linear', 'cosine', 'quadratic', 'sigmoid']
    
    for scheduler_type in schedulers_to_test:
        print(f"\nTesting {scheduler_type} scheduler:")
        scheduler = create_noise_scheduler(scheduler_type, timesteps)
        betas = scheduler.get_schedule()
        alphas, alphas_cumprod, alphas_cumprod_prev = scheduler.compute_alpha_schedule(betas)
        
        print(f"  Beta range: [{betas.min():.6f}, {betas.max():.6f}]")
        print(f"  Alpha_cumprod range: [{alphas_cumprod.min():.6f}, {alphas_cumprod.max():.6f}]")
        print(f"  Schedule shape: {betas.shape}")
        
        # Validate monotonicity for alpha_cumprod (should decrease)
        is_decreasing = torch.all(alphas_cumprod[1:] <= alphas_cumprod[:-1])
        print(f"  Alpha_cumprod decreasing: {is_decreasing}")
    
    print("\n✅ All scheduler tests completed!")
    
    # Create comparison plot
    try:
        compare_schedulers(timesteps)
        print("✅ Scheduler comparison plot created!")
    except ImportError:
        print("ℹ️  Matplotlib not available, skipping comparison plot")
