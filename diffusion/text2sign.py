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
        inference_timesteps: Optional[int] = None,
        noise_scheduler: str = "linear",
        device: torch.device = torch.device("cpu"),
        text_encoder=None,
        use_timestep_weighting: bool = True,
        weight_min_snr: float = 0.1,
        weight_max_snr: float = 10.0,
        **scheduler_kwargs,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.inference_timesteps = inference_timesteps or timesteps
        self.device = device
        self.noise_scheduler_type = noise_scheduler
        self.text_encoder = text_encoder
        self.use_timestep_weighting = use_timestep_weighting
        self.weight_min_snr = weight_min_snr
        self.weight_max_snr = weight_max_snr
        
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
        Single reverse diffusion step: p(x_{t'} | x_t) where t' < t

        Supports both classic DDPM sampling (t' = t-1) and skip-aware schedules used
        during fast inference. The method predicts the clean sample x₀, then computes
        the posterior mean and variance corresponding to the transition from timestep
        ``t`` to ``prev_t``.

        Args:
            x (torch.Tensor): Noisy data at timestep ``t``
            t (torch.Tensor): Current timestep indices (shape: batch,)
            prev_t (torch.Tensor): Previous timestep indices (shape: batch, values in [-1, t))
            text (str or Sequence[str], optional): Text conditioning
            deterministic (bool): If True, perform deterministic (DDIM/eta=0) sampling
            eta (float): DDIM stochasticity coefficient. 0.0 yields deterministic sampling,
                1.0 matches ancestral DDPM noise injection. Values outside [0, 1] are clamped.

        Returns:
            torch.Tensor: Denoised data at timestep ``prev_t``
        """
        with torch.no_grad():
            # Encode text if provided
            text_emb = None
            if text is not None and self.text_encoder is not None:
                batch_size = x.shape[0]
                if isinstance(text, str):
                    text_batch = [text] * batch_size
                else:
                    text_batch = list(text)
                    if len(text_batch) == 1 and batch_size > 1:
                        text_batch = text_batch * batch_size
                    elif len(text_batch) != batch_size:
                        # Gracefully repeat/truncate to match batch size
                        repeats = (batch_size + len(text_batch) - 1) // max(len(text_batch), 1)
                        text_batch = (text_batch * repeats)[:batch_size]

                text_emb = self.text_encoder(text_batch)
                text_emb = text_emb.to(x.device)

            predicted_noise = self.model(x, t, text_emb)

            # Prepare broadcast shapes
            shape = (-1,) + (1,) * (x.ndim - 1)

            prev_t_clamped = prev_t.clamp(min=0)
            prev_mask = (prev_t >= 0).view(shape)

            alpha_prod_t = self.alphas_cumprod[t].view(shape)
            alpha_prod_prev = self.alphas_cumprod[prev_t_clamped].view(shape)
            alpha_prod_prev = torch.where(prev_mask, alpha_prod_prev, torch.ones_like(alpha_prod_prev))

            sqrt_alpha_prod_t = torch.sqrt(torch.clamp(alpha_prod_t, min=1e-10))
            sqrt_one_minus_alpha_prod_t = torch.sqrt(torch.clamp(1.0 - alpha_prod_t, min=1e-10))

            # Predict the original clean sample x0
            pred_x0 = (x - sqrt_one_minus_alpha_prod_t * predicted_noise) / sqrt_alpha_prod_t
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            sqrt_alpha_prod_prev = torch.sqrt(torch.clamp(alpha_prod_prev, min=1e-10))
            one_minus_alpha_prod_prev = torch.clamp(1.0 - alpha_prod_prev, min=0.0)
            one_minus_alpha_prod_t = torch.clamp(1.0 - alpha_prod_t, min=1e-10)

            eta_value = max(0.0, float(eta))
            if deterministic:
                eta_value = 0.0

            sigma = eta_value * torch.sqrt(
                torch.clamp(
                    (one_minus_alpha_prod_prev / one_minus_alpha_prod_t)
                    * (1.0 - alpha_prod_t / torch.clamp(alpha_prod_prev, min=1e-10)),
                    min=0.0,
                )
            )

            if eta_value == 0.0:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)

            coef_eps = torch.sqrt(torch.clamp(one_minus_alpha_prod_prev - sigma ** 2, min=0.0))

            prev_sample = (
                sqrt_alpha_prod_prev * pred_x0
                + coef_eps * predicted_noise
                + sigma * noise
            )

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
        Complete reverse diffusion process (sampling from noise to data)
        
        Args:
            shape (tuple): Shape of the sample to generate (batch_size, channels, frames, height, width)
            device (torch.device): Device to generate on
            text (str, optional): Text conditioning for generation
            deterministic (bool): If True, use deterministic sampling (no noise in intermediate steps)
            num_inference_steps (int, optional): Number of inference timesteps. Defaults to
                self.inference_timesteps.
            eta (float, optional): DDIM stochasticity coefficient. When None, defaults to 0.0 for
                fast (skipped) sampling and 1.0 when using the full training schedule.
            
        Returns:
            torch.Tensor: Generated video sample
        """
        if device is None:
            device = self.device

        # Assert that text is provided if a text encoder exists
        if self.text_encoder is not None and text is None:
            print("Warning: Text encoder exists but no text provided. Using unconditional sampling.")

        # Start with random noise (x_T ~ N(0, I))
        x = torch.randn(shape, device=device)

        # Determine inference timesteps
        if num_inference_steps is None:
            num_inference_steps = self.inference_timesteps
        num_inference_steps = max(1, min(num_inference_steps, self.timesteps))

        if num_inference_steps == self.timesteps:
            timesteps = torch.arange(self.timesteps - 1, -1, -1, device=device, dtype=torch.long)
        else:
            timesteps = torch.linspace(
                self.timesteps - 1,
                0,
                steps=num_inference_steps,
                device=device,
            )
            timesteps = torch.round(timesteps).long()
            timesteps = torch.clamp(timesteps, 0, self.timesteps - 1)
            timesteps = torch.unique_consecutive(timesteps)
            if timesteps[-1].item() != 0:
                timesteps = torch.cat([timesteps, timesteps.new_tensor([0])])

        # Select eta schedule
        if eta is None:
            if deterministic:
                eta_value = 0.0
            elif len(timesteps) < self.timesteps:
                # Default to deterministic DDIM when skipping many steps to avoid oversized noise
                eta_value = 0.0
            else:
                eta_value = 1.0
        else:
            eta_value = max(0.0, float(eta))

        deterministic_mode = deterministic or eta_value == 0.0
        total_steps = len(timesteps)

        # Improve progress bar formatting
        sampling_desc = f"Sampling ({'deterministic' if deterministic_mode else 'stochastic'}, eta={eta_value:.2f})"
        
        # Reverse diffusion process using the selected timestep schedule
        for idx, timestep in enumerate(tqdm(timesteps, desc=sampling_desc, ncols=100)):
            t = torch.full((shape[0],), timestep.item(), device=device, dtype=torch.long)
            if idx + 1 < total_steps:
                prev_timestep = timesteps[idx + 1].item()
            else:
                prev_timestep = -1
            prev_t = torch.full((shape[0],), prev_timestep, device=device, dtype=torch.long)

            x = self.p_sample_step(x, t, prev_t, text, deterministic=deterministic_mode, eta=eta_value)

            # Log sampling progress more strategically - beginning, end, and major steps
            log_step = (idx == 0 or idx == total_steps-1 or  # First and last step
                       (total_steps > 20 and idx % (total_steps // 10) == 0))  # ~10 updates for long runs
                       
            if log_step:
                print(
                    f"[{idx+1}/{total_steps}] Step t={timestep.item():4d}: "
                    f"range [{x.min().item():.3f}, {x.max().item():.3f}], "
                    f"mean={x.mean().item():.3f}, std={x.std().item():.3f}"
                )

        # Clamp final sample to training data range [-1,1]
        x = torch.clamp(x, -1.0, 1.0)
        return x

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
        Convenience method for text-conditioned video generation
        
        Args:
            text (str): Text prompt for sign language generation
            batch_size (int): Number of videos to generate
            num_frames (int): Number of frames in the video
            height (int): Video height
            width (int): Video width
            deterministic (bool): Whether to use deterministic sampling
            num_inference_steps (int, optional): Number of sampling steps
            eta (float, optional): DDIM stochasticity coefficient passed through to ``p_sample``

        Returns:
            torch.Tensor: Generated video tensor with shape (batch_size, 3, num_frames, height, width)
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
        base_loss = F.mse_loss(predicted_noise, noise, reduction='none')
        
        # Apply timestep-aware loss weighting if enabled
        if self.use_timestep_weighting:
            # Low timesteps (cleaner images) are harder to denoise and need more emphasis
            # Weight inversely proportional to SNR (Signal-to-Noise Ratio)
            alpha_bar_t = self.alphas_cumprod[t].view(-1, *([1] * (base_loss.ndim - 1)))
            snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)  # Signal-to-Noise Ratio
            
            # Compute loss weight: higher weight for low SNR (low timesteps)
            loss_weight = 1.0 / torch.clamp(snr, min=self.weight_min_snr, max=self.weight_max_snr)
            
            # Apply weight and reduce
            weighted_loss = base_loss * loss_weight
            loss = weighted_loss.mean()
        else:
            # Standard unweighted loss
            loss = base_loss.mean()
        
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
        use_timestep_weighting=getattr(config, 'USE_TIMESTEP_WEIGHTING', True),
        weight_min_snr=getattr(config, 'TIMESTEP_WEIGHT_MIN_SNR', 0.1),
        weight_max_snr=getattr(config, 'TIMESTEP_WEIGHT_MAX_SNR', 10.0),
        **scheduler_kwargs
    )
    
    return model
