import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Any
import numpy as np
from tqdm import tqdm

from ..models.unet3d import UNet3D
from ..models.scheduler import DDPMScheduler, DDIMScheduler
from ..models.text_encoder import CLIPTextEncoder, T5TextEncoder, SimpleTextEncoder


class Text2SignDiffusionPipeline(nn.Module):
    """
    Complete diffusion pipeline for text-to-sign language video generation.
    """
    
    def __init__(
        self,
        unet: UNet3D,
        scheduler: Union[DDPMScheduler, DDIMScheduler],
        text_encoder: Union[CLIPTextEncoder, T5TextEncoder, SimpleTextEncoder],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the diffusion pipeline.
        
        Args:
            unet: 3D UNet denoising model
            scheduler: Noise scheduler (DDPM or DDIM)
            text_encoder: Text encoder for conditioning
            device: Device to run the pipeline on
        """
        super().__init__()
        
        self.unet = unet
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.device = device
        
        # Move models to device
        self.unet = self.unet.to(device)
        self.text_encoder = self.text_encoder.to(device)
    
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """
        Encode text prompts to embeddings.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            Text embeddings tensor
        """
        with torch.no_grad():
            text_embeddings = self.text_encoder(prompts)
        return text_embeddings
    
    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Prepare random latents for diffusion.
        
        Args:
            batch_size: Batch size
            num_channels: Number of channels
            num_frames: Number of frames
            height: Frame height
            width: Frame width
            dtype: Data type
            generator: Random generator
            
        Returns:
            Random latents tensor
        """
        shape = (batch_size, num_channels, num_frames, height, width)
        latents = torch.randn(shape, generator=generator, dtype=dtype, device=self.device)
        
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        
        return latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        num_frames: int = 28,
        height: int = 128,
        width: int = 128,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Generate sign language videos from text prompts.
        
        Args:
            prompts: Text prompts
            num_frames: Number of frames to generate
            height: Frame height
            width: Frame width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompts for guidance
            generator: Random generator
            return_dict: Whether to return a dictionary
            
        Returns:
            Generated videos
        """
        # Handle single prompt
        if isinstance(prompts, str):
            prompts = [prompts]
        
        batch_size = len(prompts)
        
        # Handle negative prompts
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        
        # Encode text prompts
        text_embeddings = self.encode_text(prompts)
        
        # Encode negative prompts for classifier-free guidance
        if guidance_scale > 1.0:
            negative_embeddings = self.encode_text(negative_prompt)
            # Concatenate for classifier-free guidance
            text_embeddings = torch.cat([negative_embeddings, text_embeddings])
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Prepare latents
        num_channels = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=num_channels,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator
        )
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Generating")):
            # Convert timestep to tensor if needed
            if isinstance(t, torch.Tensor):
                timestep_tensor = t.expand(latents.shape[0]).to(latents.device)
            else:
                timestep_tensor = torch.full((latents.shape[0],), t, device=latents.device, dtype=torch.long)
            
            # Expand latents if we are doing classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, int(t.item()) if isinstance(t, torch.Tensor) else t)
            
            # Expand timestep for classifier-free guidance
            if guidance_scale > 1.0:
                timestep_input = torch.cat([timestep_tensor] * 2)
            else:
                timestep_input = timestep_tensor
            
            # Predict noise residual
            noise_pred = self.unet(latent_model_input, timestep_input, text_embeddings)
            
            # Perform classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, int(t.item()) if isinstance(t, torch.Tensor) else t, latents)
        
        # Post-process latents to videos
        videos = self.decode_latents(latents)
        
        if return_dict:
            return {"videos": videos, "latents": latents}
        else:
            return videos
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to videos.
        
        Args:
            latents: Latent tensors
            
        Returns:
            Decoded videos
        """
        # For now, we assume latents are already in video space
        # In a full implementation, you might have a VAE decoder here
        videos = torch.clamp(latents, -1, 1)
        
        # Convert from [-1, 1] to [0, 1]
        videos = (videos + 1) / 2
        
        return videos
    
    def train_step(
        self,
        videos: torch.Tensor,
        texts: List[str]
    ) -> torch.Tensor:
        """
        Perform a single training step.
        
        Args:
            videos: Ground truth videos of shape (B, C, T, H, W)
            texts: Text prompts
            
        Returns:
            Loss tensor
        """
        batch_size = videos.shape[0]
        device = videos.device
        
        # Encode text
        text_embeddings = self.encode_text(texts)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (batch_size,), device=device
        ).long()
        
        # Add noise to videos
        noise = torch.randn_like(videos)
        noisy_videos = self.scheduler.add_noise(videos, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(noisy_videos, timesteps, text_embeddings)
        
        # Compute loss
        if self.scheduler.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.prediction_type == "v_prediction":
            # Simple v-prediction target (you may need to adjust this)
            sqrt_alpha_prod = torch.sqrt(self.scheduler.alphas_cumprod[timesteps])
            sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.scheduler.alphas_cumprod[timesteps])
            
            # Reshape for broadcasting
            while len(sqrt_alpha_prod.shape) < len(videos.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
            target = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * videos
        else:
            raise ValueError(f"Unknown prediction type: {self.scheduler.prediction_type}")
        
        loss = nn.functional.mse_loss(noise_pred, target)
        
        return loss


def create_pipeline(
    model_channels: int = 32,
    num_res_blocks: int = 1,
    attention_resolutions: List[int] = [32, 64],
    channel_mult: List[int] = [1, 2, 3],
    num_heads: int = 2,
    text_encoder_type: str = "simple",
    scheduler_type: str = "ddpm",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Text2SignDiffusionPipeline:
    """
    Create a complete diffusion pipeline.
    
    Args:
        model_channels: Base number of channels in UNet
        num_res_blocks: Number of ResNet blocks per level
        attention_resolutions: Resolutions to apply attention at
        channel_mult: Channel multipliers for each level
        num_heads: Number of attention heads
        text_encoder_type: Type of text encoder ("simple", "clip", "t5")
        scheduler_type: Type of scheduler ("ddpm", "ddim")
        device: Device to run on
        
    Returns:
        Configured diffusion pipeline
    """
    # Create UNet
    unet = UNet3D(
        in_channels=3,
        out_channels=3,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
        num_heads=num_heads,
        text_embed_dim=256
    )
    
    # Create scheduler
    if scheduler_type == "ddpm":
        scheduler = DDPMScheduler(num_train_timesteps=1000)
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=1000)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Create text encoder
    if text_encoder_type == "simple":
        text_encoder = SimpleTextEncoder(vocab_size=5000, embed_dim=256)
    elif text_encoder_type == "clip":
        text_encoder = CLIPTextEncoder()
    elif text_encoder_type == "t5":
        text_encoder = T5TextEncoder()
    else:
        raise ValueError(f"Unknown text encoder type: {text_encoder_type}")
    
    # Create pipeline
    pipeline = Text2SignDiffusionPipeline(
        unet=unet,
        scheduler=scheduler,
        text_encoder=text_encoder,
        device=device
    )
    
    return pipeline


def test_pipeline():
    """Test the diffusion pipeline."""
    print("Testing diffusion pipeline...")
    
    # Create a small pipeline for testing
    pipeline = create_pipeline(
        model_channels=64,  # Smaller for testing
        num_res_blocks=1,
        attention_resolutions=[8, 16],
        channel_mult=[1, 2, 4],
        num_heads=4,
        text_encoder_type="simple",
        device="cpu"  # Use CPU for testing
    )
    
    # Test text encoding
    prompts = ["Hello", "How are you?"]
    text_embeddings = pipeline.encode_text(prompts)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Test training step
    videos = torch.randn(2, 3, 28, 128, 128)  # Updated to new format
    loss = pipeline.train_step(videos, prompts)
    print(f"Training loss: {loss.item()}")
    
    # Test generation (smaller for speed)
    with torch.no_grad():
        generated = pipeline(
            prompts=["Hello"],
            num_frames=8,  # Fewer frames for testing
            height=64,     # Smaller size for testing
            width=64,
            num_inference_steps=2  # Very few steps for testing
        )
    
    print(f"Generated videos shape: {generated.shape if isinstance(generated, torch.Tensor) else generated['videos'].shape}")
    print("Pipeline test passed!")


if __name__ == "__main__":
    test_pipeline()
