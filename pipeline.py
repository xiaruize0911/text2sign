"""
Pipeline for text-to-sign language GIF generation
End-to-end inference with a trained model
"""

import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import ModelConfig, DDIMConfig, GenerationConfig
from models import UNet3D, TextEncoder
from schedulers import DDIMScheduler
from dataset import SimpleTokenizer


class Text2SignPipeline:
    """
    End-to-end pipeline for text-to-sign language GIF generation
    """
    
    def __init__(
        self,
        model: UNet3D,
        text_encoder: TextEncoder,
        scheduler: DDIMScheduler,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        device: Union[str, torch.device] = "cuda",
    ):
        self.model = model.to(device)
        self.text_encoder = text_encoder.to(device)
        self.scheduler = scheduler
        self.model_config = model_config
        self.generation_config = generation_config
        self.device = device
        
        # Move scheduler tensors to device
        self._move_scheduler_to_device()
        
        # Tokenizer
        self.tokenizer = SimpleTokenizer(
            vocab_size=model_config.vocab_size,
            max_length=model_config.max_text_length,
        )
        
        # Set models to eval mode
        self.model.eval()
        self.text_encoder.eval()
    
    def _move_scheduler_to_device(self):
        """Move scheduler tensors to device"""
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler.alphas_cumprod_prev = self.scheduler.alphas_cumprod_prev.to(self.device)
        self.scheduler.sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod.to(self.device)
        self.scheduler.sqrt_one_minus_alphas_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod.to(self.device)
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: Union[str, torch.device] = "cuda",
    ) -> "Text2SignPipeline":
        """
        Load pipeline from a saved checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load models on
        
        Returns:
            Text2SignPipeline instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get configs from checkpoint
        model_config = checkpoint.get("model_config", ModelConfig())
        ddim_config = checkpoint.get("ddim_config", DDIMConfig())
        generation_config = GenerationConfig()
        
        # Handle dataclass or dict
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
        if isinstance(ddim_config, dict):
            ddim_config = DDIMConfig(**ddim_config)
        
        # Detect actual transformer_depth from model weights (config may be wrong)
        state_dict = checkpoint["model_state_dict"]
        actual_transformer_depth = 1
        for key in state_dict.keys():
            if 'spatial_blocks.' in key:
                idx = int(key.split('spatial_blocks.')[1].split('.')[0])
                actual_transformer_depth = max(actual_transformer_depth, idx + 1)
        
        config_depth = getattr(model_config, 'transformer_depth', 1)
        if config_depth != actual_transformer_depth:
            print(f"  Note: Config says transformer_depth={config_depth}, but weights have depth={actual_transformer_depth}")
            print(f"  Using actual depth from weights: {actual_transformer_depth}")
        
        # Create models with all transformer parameters from config
        model = UNet3D(
            in_channels=model_config.in_channels,
            model_channels=model_config.model_channels,
            out_channels=model_config.in_channels,
            num_res_blocks=model_config.num_res_blocks,
            attention_resolutions=model_config.attention_resolutions,
            channel_mult=model_config.channel_mult,
            num_heads=model_config.num_heads,
            context_dim=model_config.context_dim,
            use_transformer=getattr(model_config, 'use_transformer', True),
            transformer_depth=actual_transformer_depth,  # Use detected depth from weights
            use_gradient_checkpointing=getattr(model_config, 'use_gradient_checkpointing', False),
        )
        
        text_encoder = TextEncoder(
            vocab_size=model_config.vocab_size,
            max_length=model_config.max_text_length,
            embed_dim=model_config.text_embed_dim,
        )
        
        scheduler = DDIMScheduler(
            num_train_timesteps=ddim_config.num_train_timesteps,
            beta_start=ddim_config.beta_start,
            beta_end=ddim_config.beta_end,
            beta_schedule=ddim_config.beta_schedule,
            clip_sample=ddim_config.clip_sample,
            prediction_type=ddim_config.prediction_type,
        )
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
        
        return cls(
            model=model,
            text_encoder=text_encoder,
            scheduler=scheduler,
            model_config=model_config,
            generation_config=generation_config,
            device=device,
        )
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",  # "pil", "tensor", "numpy"
    ) -> Union[List[List[Image.Image]], torch.Tensor, np.ndarray]:
        """
        Generate sign language video from text prompt
        
        Args:
            prompt: Text prompt or list of prompts
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            eta: Stochasticity parameter (0 = deterministic DDIM)
            generator: Random generator for reproducibility
            output_type: Type of output ("pil", "tensor", "numpy")
        
        Returns:
            Generated videos in requested format
        """
        # Handle single prompt
        if isinstance(prompt, str):
            prompt = [prompt]
        
        batch_size = len(prompt)
        
        # Use default values if not specified
        if num_inference_steps is None:
            num_inference_steps = self.generation_config.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.generation_config.guidance_scale
        if eta is None:
            eta = self.generation_config.eta
        
        # Tokenize prompts
        tokens = self.tokenizer(prompt).to(self.device)
        
        # Get text embeddings
        text_embeddings = self.text_encoder(tokens)
        
        # For classifier-free guidance
        if guidance_scale > 1.0:
            uncond_tokens = self.tokenizer([""] * batch_size).to(self.device)
            uncond_embeddings = self.text_encoder(uncond_tokens)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Set inference timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Initialize latents
        latents_shape = (
            batch_size,
            self.model_config.in_channels,
            self.model_config.num_frames,
            self.model_config.image_size,
            self.model_config.image_size,
        )
        
        if generator is not None:
            latents = torch.randn(latents_shape, generator=generator, device=self.device)
        else:
            latents = torch.randn(latents_shape, device=self.device)
        
        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Generating sign language", leave=True):
            latent_model_input = latents
            
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latents] * 2)
            
            timestep = torch.tensor([t] * latent_model_input.shape[0], device=self.device)
            
            # Predict noise
            noise_pred = self.model(latent_model_input, timestep, text_embeddings)
            
            # Apply classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DDIM step
            latents, _ = self.scheduler.step(noise_pred, t, latents, eta=eta, generator=generator)
        
        # Denormalize
        videos = (latents + 1) / 2
        videos = videos.clamp(0, 1)
        
        # Convert to output type
        if output_type == "tensor":
            return videos
        elif output_type == "numpy":
            return videos.cpu().numpy()
        else:  # "pil"
            return self._tensor_to_pil(videos)
    
    def _tensor_to_pil(self, videos: torch.Tensor) -> List[List[Image.Image]]:
        """Convert tensor videos to PIL images"""
        # videos: (B, C, T, H, W)
        videos = videos.cpu().numpy()
        
        all_videos = []
        for video in videos:
            # (C, T, H, W) -> (T, H, W, C)
            frames = video.transpose(1, 2, 3, 0)
            frames = (frames * 255).astype(np.uint8)
            
            pil_frames = [Image.fromarray(frame) for frame in frames]
            all_videos.append(pil_frames)
        
        return all_videos
    
    def save_gif(
        self,
        frames: List[Image.Image],
        path: str,
        fps: Optional[int] = None,
    ):
        """
        Save frames as GIF
        
        Args:
            frames: List of PIL images
            path: Output path
            fps: Frames per second
        """
        if fps is None:
            fps = self.generation_config.fps
        
        duration = 1000 // fps
        
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
    
    def generate_and_save(
        self,
        prompt: Union[str, List[str]],
        output_dir: str,
        prefix: str = "sign",
        **kwargs,
    ) -> List[str]:
        """
        Generate and save GIFs
        
        Args:
            prompt: Text prompt(s)
            output_dir: Directory to save GIFs
            prefix: Filename prefix
            **kwargs: Arguments passed to __call__
        
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        videos = self(prompt, **kwargs)
        
        saved_paths = []
        for i, (frames, text) in enumerate(zip(videos, prompt)):
            # Create filename from prompt
            safe_text = "".join(c if c.isalnum() else "_" for c in text[:30])
            filename = f"{prefix}_{i}_{safe_text}.gif"
            filepath = os.path.join(output_dir, filename)
            
            self.save_gif(frames, filepath)
            saved_paths.append(filepath)
            print(f"Saved: {filepath}")
        
        return saved_paths


def create_pipeline(
    model_config: Optional[ModelConfig] = None,
    ddim_config: Optional[DDIMConfig] = None,
    generation_config: Optional[GenerationConfig] = None,
    device: str = "cuda",
) -> Text2SignPipeline:
    """
    Create a new pipeline with untrained models
    (useful for testing)
    """
    if model_config is None:
        model_config = ModelConfig()
    if ddim_config is None:
        ddim_config = DDIMConfig()
    if generation_config is None:
        generation_config = GenerationConfig()
    
    model = UNet3D(
        in_channels=model_config.in_channels,
        model_channels=model_config.model_channels,
        out_channels=model_config.in_channels,
        num_res_blocks=model_config.num_res_blocks,
        attention_resolutions=model_config.attention_resolutions,
        channel_mult=model_config.channel_mult,
        num_heads=model_config.num_heads,
        context_dim=model_config.context_dim,
    )
    
    text_encoder = TextEncoder(
        vocab_size=model_config.vocab_size,
        max_length=model_config.max_text_length,
        embed_dim=model_config.text_embed_dim,
    )
    
    scheduler = DDIMScheduler(
        num_train_timesteps=ddim_config.num_train_timesteps,
        beta_start=ddim_config.beta_start,
        beta_end=ddim_config.beta_end,
        beta_schedule=ddim_config.beta_schedule,
        clip_sample=ddim_config.clip_sample,
        prediction_type=ddim_config.prediction_type,
    )
    
    return Text2SignPipeline(
        model=model,
        text_encoder=text_encoder,
        scheduler=scheduler,
        model_config=model_config,
        generation_config=generation_config,
        device=device,
    )


if __name__ == "__main__":
    # Test pipeline
    print("Creating pipeline...")
    pipeline = create_pipeline(device="cpu")
    
    print("Testing generation...")
    videos = pipeline(
        ["Hello", "Thank you"],
        num_inference_steps=5,
        guidance_scale=3.0,
    )
    
    print(f"Generated {len(videos)} videos")
    print(f"Each video has {len(videos[0])} frames")
    print(f"Frame size: {videos[0][0].size}")
