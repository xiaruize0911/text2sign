"""
Main script for Text-to-Sign Language DDIM Diffusion Model
Supports training and generation with command-line interface
"""

import argparse
import os
import sys
from datetime import datetime

import torch


def train(args):
    """Run training"""
    from config import ModelConfig, TrainingConfig, DDIMConfig, apply_model_size_preset
    from ablation_configs import apply_ablation_preset
    from trainer import create_trainer
    
    print("=" * 60)
    print("Text-to-Sign Language DDIM Diffusion Model - Training")
    print("=" * 60)
    
    # Create configs
    model_config = ModelConfig()
    train_config = TrainingConfig()
    ddim_config = DDIMConfig()

    ablation_description = None
    model_size_description = None
    if args.ablation_preset:
        model_config, train_config, ablation_description = apply_ablation_preset(
            model_config,
            train_config,
            args.ablation_preset,
        )
    if args.model_size:
        model_config, train_config, model_size_description = apply_model_size_preset(
            model_config,
            train_config,
            args.model_size,
        )
    
    # Update configs with command-line arguments
    if args.data_dir:
        train_config.data_dir = args.data_dir
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.epochs:
        train_config.num_epochs = args.epochs
    if args.lr:
        train_config.learning_rate = args.lr
    if args.image_size:
        model_config.image_size = args.image_size
    if args.num_frames:
        model_config.num_frames = args.num_frames
    if args.model_channels:
        model_config.model_channels = args.model_channels
    if args.num_heads:
        model_config.num_heads = args.num_heads
    if args.timesteps:
        ddim_config.num_train_timesteps = args.timesteps
    if args.beta_schedule:
        ddim_config.beta_schedule = args.beta_schedule
    if args.prediction_type:
        ddim_config.prediction_type = args.prediction_type
    if args.num_workers:
        train_config.num_workers = args.num_workers
    if args.grad_accum_steps:
        train_config.gradient_accumulation_steps = args.grad_accum_steps
    if args.max_optimizer_steps is not None:
        train_config.max_run_optimizer_steps = args.max_optimizer_steps
    if args.split_mode:
        train_config.split_mode = args.split_mode
    if args.text_conditioning_mode:
        model_config.text_conditioning_mode = args.text_conditioning_mode
    if args.use_length_prefix:
        model_config.use_length_prefix = True
    if args.clip_trainable_layers is not None:
        model_config.clip_trainable_layers = args.clip_trainable_layers
    if args.precision:
        train_config.precision = args.precision
    if args.checkpoint_dir:
        train_config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        train_config.log_dir = args.log_dir
    if args.save_every:
        train_config.save_every = args.save_every
    if args.log_every:
        train_config.log_every = args.log_every
    if args.sample_every:
        train_config.sample_every = args.sample_every
    if args.sample_steps is not None:
        train_config.sample_inference_steps = args.sample_steps
    if args.sample_guidance_scale is not None:
        train_config.sample_guidance_scale = args.sample_guidance_scale
    if args.prefetch_factor is not None:
        train_config.dataloader_prefetch_factor = args.prefetch_factor
    if args.no_amp:
        train_config.use_amp = False
        train_config.precision = "fp32"
    if args.no_compile:
        train_config.enable_compile = False
    if args.compile_mode:
        train_config.compile_mode = args.compile_mode
    if args.compile_fullgraph:
        train_config.compile_fullgraph = True
    if args.compile_dynamic:
        train_config.compile_dynamic = True
    if args.no_tf32:
        train_config.allow_tf32 = False
    if args.no_channels_last:
        train_config.channels_last_3d = False
    if args.no_gradient_checkpointing:
        model_config.use_gradient_checkpointing = False
    if args.cpu:
        train_config.device = "cpu"
        train_config.enable_compile = False
        train_config.use_amp = False
        train_config.precision = "fp32"
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Data directory: {train_config.data_dir}")
    print(f"  Image size: {model_config.image_size}")
    print(f"  Num frames: {model_config.num_frames}")
    print(f"  Model channels: {model_config.model_channels}")
    print(f"  Channel mult: {model_config.channel_mult}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Grad accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"  Max optimizer steps this run: {train_config.max_run_optimizer_steps}")
    print(f"  Epochs: {train_config.num_epochs}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Device: {train_config.device}")
    print(f"  Mixed precision: {train_config.use_amp}")
    print(f"  Precision mode: {train_config.precision}")
    print(f"  torch.compile: {train_config.enable_compile}")
    print(f"  Compile mode: {train_config.compile_mode}")
    print(f"  TF32: {train_config.allow_tf32}")
    print(f"  channels_last_3d: {train_config.channels_last_3d}")
    print(f"  Gradient checkpointing: {model_config.use_gradient_checkpointing}")
    print(f"  Using CLIP text encoder: {model_config.use_clip_text_encoder}")
    print(f"  Conditioning mode: {model_config.text_conditioning_mode}")
    print(f"  Train/val split: {train_config.split_mode}")
    print(f"  CLIP trainable layers: {model_config.clip_trainable_layers}")
    print(f"  Length prefix conditioning: {model_config.use_length_prefix}")
    print(f"  Sample inference steps: {train_config.sample_inference_steps}")
    print(f"  Sample guidance scale: {train_config.sample_guidance_scale}")
    if args.model_size:
        print(f"  Model size preset: {args.model_size}")
        print(f"  Model size detail: {model_size_description}")
    if args.ablation_preset:
        print(f"  Ablation preset: {args.ablation_preset}")
        print(f"  Ablation detail: {ablation_description}")
    print(f"  Timesteps: {ddim_config.num_train_timesteps}")
    print(f"  Beta schedule: {ddim_config.beta_schedule}")
    print()
    
    # Extract run name if resuming
    run_name = None
    if args.resume:
        # Expected path: .../checkpoints/run_name/checkpoint.pt
        parent_dir = os.path.basename(os.path.dirname(args.resume))
        if parent_dir and "text2sign_" in parent_dir:
            run_name = parent_dir
            print(f"Detected run name from checkpoint: {run_name}")
    
    # Create trainer
    trainer = create_trainer(model_config, train_config, ddim_config, run_name=run_name)
    
    # Load checkpoint if provided
    if args.resume:
        print(f"\n{'='*60}")
        print("RESUMING TRAINING")
        print(f"{'='*60}")
        trainer.load_checkpoint(args.resume, resume_training=True)
        print(f"\nResuming from:")
        print(f"  Epoch: {trainer.epoch}")
        print(f"  Global step: {trainer.global_step}")
        print(f"  Best loss: {trainer.best_loss:.6f}")
        print(f"{'='*60}\n")
    
    # Start training
    trainer.train()


def validate(args):
    """Run model validation"""
    from validate import ModelValidator
    
    print("=" * 60)
    print("Text-to-Sign Language DDIM Diffusion Model - Validation")
    print("=" * 60)
    
    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    # Create validator and run
    validator = ModelValidator(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=device,
        num_samples=args.num_samples,
        benchmark_repeats=args.benchmark_repeats,
        enable_backtranslation=not args.skip_backtranslation,
        eval_num_inference_steps=args.steps,
        eval_guidance_scale=args.guidance_scale,
        fvd_backbone=args.fvd_backbone,
    )
    
    results = validator.run_full_validation(args.output_dir)
    
    print("\n✅ Validation complete!")
    return results


def generate(args):
    """Run generation"""
    from pipeline import Text2SignPipeline
    
    print("=" * 60)
    print("Text-to-Sign Language DDIM Diffusion Model - Generation")
    print("=" * 60)
    
    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load pipeline
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"\nLoading model from: {args.checkpoint}")
    print(f"Device: {device}")
    
    pipeline = Text2SignPipeline.from_pretrained(args.checkpoint, device=device)
    
    # Get prompts
    if args.prompt:
        prompts = args.prompt
    elif args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        prompts = []
        print("\nEnter prompts (one per line, empty line to finish):")
        while True:
            prompt = input("> ").strip()
            if not prompt:
                break
            prompts.append(prompt)
    
    if not prompts:
        print("No prompts provided!")
        sys.exit(1)
    
    print(f"\nGenerating {len(prompts)} sign language videos...")
    print(f"  Inference steps: {args.steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  ETA: {args.eta}")
    print()
    
    # Generate
    output_dir = args.output_dir or f"text_to_sign/generated/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    saved_paths = pipeline.generate_and_save(
        prompts,
        output_dir=output_dir,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
    )
    
    print(f"\nGeneration complete!")
    print(f"Saved {len(saved_paths)} GIFs to: {output_dir}")


def test(args):
    """Test model and data loading"""
    print("=" * 60)
    print("Text-to-Sign Language DDIM Diffusion Model - Test")
    print("=" * 60)
    
    # Test dataset
    print("\n1. Testing dataset...")
    from dataset import SignLanguageDataset, get_dataloader
    
    try:
        dataset = SignLanguageDataset(
            data_dir=args.data_dir,
            image_size=64,
            num_frames=16,
            train=True,
        )
        print(f"   Dataset loaded: {len(dataset)} samples")
        
        sample = dataset[0]
        print(f"   Sample video shape: {sample['video'].shape}")
        print(f"   Sample text: {sample['text'][:50]}...")
        
        dataloader = get_dataloader(args.data_dir, batch_size=2)
        batch = next(iter(dataloader))
        print(f"   Batch video shape: {batch['video'].shape}")
        print(f"   Batch tokens shape: {batch['tokens'].shape}")
    except Exception as e:
        print(f"   Dataset test failed: {e}")
        return
    
    # Test model
    print("\n2. Testing model...")
    from models import UNet3D, TextEncoder
    from config import ModelConfig
    
    config = ModelConfig()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    try:
        # Smaller model for testing
        model = UNet3D(
            in_channels=3,
            model_channels=32,
            channel_mult=(1, 2),
            attention_resolutions=(8,),
            num_heads=4,
            context_dim=128,
        ).to(device)
        
        text_encoder = TextEncoder(
            vocab_size=config.vocab_size,
            max_length=config.max_text_length,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
        ).to(device)
        
        print(f"   UNet parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Text encoder parameters: {sum(p.numel() for p in text_encoder.parameters()):,}")
        
        # Test forward pass
        x = torch.randn(1, 3, 16, 64, 64).to(device)
        t = torch.tensor([500]).to(device)
        tokens = torch.randint(0, config.vocab_size, (1, config.max_text_length)).to(device)
        
        context = text_encoder(tokens)
        output = model(x, t, context)
        
        print(f"   Forward pass successful!")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"   Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test scheduler
    print("\n3. Testing scheduler...")
    from schedulers import DDIMScheduler
    
    try:
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
        )
        
        # Test noise addition
        noisy = scheduler.add_noise(x, torch.randn_like(x), t)
        print(f"   Noise addition successful!")
        
        # Test step
        scheduler.set_timesteps(50, device=device)
        # Use a valid timestep from the scheduler's inference steps
        test_t = scheduler.timesteps[0].item()
        prev_sample, pred_x0 = scheduler.step(output, test_t, noisy)
        print(f"   DDIM step successful!")
        
    except Exception as e:
        print(f"   Scheduler test failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("All tests passed! Ready for training.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-Sign Language DDIM Diffusion Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
    python main.py train --data-dir text_to_sign/training_data --epochs 100

  # Generate sign language GIFs
  python main.py generate --checkpoint checkpoints/best_model.pt --prompt "Hello"

  # Validate model quality
    python main.py validate --checkpoint checkpoints/best_model.pt --data-dir text_to_sign/training_data

  # Test data and model
    python main.py test --data-dir text_to_sign/training_data
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-dir", type=str, default=None,
                             help="Path to training data directory")
    train_parser.add_argument("--batch-size", type=int, default=None,
                             help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=None,
                             help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=None,
                             help="Learning rate")
    train_parser.add_argument("--image-size", type=int, default=None,
                             help="Image size to resize to")
    train_parser.add_argument("--num-frames", type=int, default=None,
                             help="Number of frames per video")
    train_parser.add_argument("--model-channels", type=int, default=None,
                             help="Base model channels")
    train_parser.add_argument("--model-size", type=str, default=None,
                             choices=["small", "base", "large"],
                             help="Named model/training size preset")
    train_parser.add_argument("--num-heads", type=int, default=None,
                             help="Number of attention heads")
    train_parser.add_argument("--timesteps", type=int, default=None,
                             help="Number of diffusion timesteps")
    train_parser.add_argument("--beta-schedule", type=str, default=None,
                             choices=["linear", "cosine"],
                             help="Beta schedule type")
    train_parser.add_argument("--prediction-type", type=str, default=None,
                             choices=["epsilon", "v_prediction"],
                             help="What the model predicts")
    train_parser.add_argument("--num-workers", type=int, default=None,
                             help="Number of data loading workers")
    train_parser.add_argument("--grad-accum-steps", type=int, default=None,
                             help="Gradient accumulation steps")
    train_parser.add_argument("--max-optimizer-steps", type=int, default=None,
                             help="Stop after this many optimizer steps in the current invocation")
    train_parser.add_argument("--ablation-preset", type=str, default=None,
                             choices=["frozen_clip", "no_text", "random_text", "clip_finetuned_last2"],
                             help="Named conditioning ablation preset")
    train_parser.add_argument("--split-mode", type=str, default=None,
                             choices=["signer_disjoint", "random"],
                             help="Dataset split protocol")
    train_parser.add_argument("--text-conditioning-mode", type=str, default=None,
                             choices=["normal", "none", "random"],
                             help="Conditioning ablation mode")
    train_parser.add_argument("--clip-trainable-layers", type=int, default=None,
                             help="Unfreeze the last N CLIP encoder layers")
    train_parser.add_argument("--use-length-prefix", action="store_true",
                             help="Prefix prompts with a discretized word-count token")
    train_parser.add_argument("--checkpoint-dir", type=str, default=None,
                             help="Directory to save checkpoints")
    train_parser.add_argument("--log-dir", type=str, default=None,
                             help="Directory for TensorBoard logs")
    train_parser.add_argument("--save-every", type=int, default=None,
                             help="Save checkpoint every N epochs")
    train_parser.add_argument("--log-every", type=int, default=None,
                             help="Log to TensorBoard every N steps")
    train_parser.add_argument("--sample-every", type=int, default=None,
                             help="Generate samples every N steps")
    train_parser.add_argument("--sample-steps", type=int, default=None,
                             help="DDIM steps for periodic training-time samples")
    train_parser.add_argument("--sample-guidance-scale", type=float, default=None,
                             help="CFG scale for periodic training-time samples")
    train_parser.add_argument("--precision", type=str, default=None,
                             choices=["auto", "fp16", "bf16", "fp32"],
                             help="Precision mode for training")
    train_parser.add_argument("--no-compile", action="store_true",
                             help="Disable torch.compile for the UNet")
    train_parser.add_argument("--compile-mode", type=str, default=None,
                             choices=["default", "reduce-overhead", "max-autotune"],
                             help="torch.compile optimization mode")
    train_parser.add_argument("--compile-fullgraph", action="store_true",
                             help="Enable fullgraph torch.compile mode")
    train_parser.add_argument("--compile-dynamic", action="store_true",
                             help="Enable dynamic-shape torch.compile mode")
    train_parser.add_argument("--no-tf32", action="store_true",
                             help="Disable TF32 matmul/cuDNN acceleration on CUDA")
    train_parser.add_argument("--no-channels-last", action="store_true",
                             help="Disable channels_last_3d memory format")
    train_parser.add_argument("--prefetch-factor", type=int, default=None,
                             help="DataLoader prefetch factor when num_workers > 0")
    train_parser.add_argument("--no-gradient-checkpointing", action="store_true",
                             help="Disable model gradient checkpointing")
    train_parser.add_argument("--resume", type=str, default=None,
                             help="Path to checkpoint to resume from")
    train_parser.add_argument("--no-amp", action="store_true",
                             help="Disable mixed precision training")
    train_parser.add_argument("--cpu", action="store_true",
                             help="Force CPU training")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate sign language GIFs")
    gen_parser.add_argument("--checkpoint", type=str, required=True,
                           help="Path to model checkpoint")
    gen_parser.add_argument("--prompt", type=str, nargs="+",
                           help="Text prompt(s) for generation")
    gen_parser.add_argument("--prompt-file", type=str,
                           help="File with prompts (one per line)")
    gen_parser.add_argument("--output-dir", type=str, default="Samples",
                           help="Directory to save generated GIFs")
    gen_parser.add_argument("--steps", type=int, default=50,
                           help="Number of inference steps")
    gen_parser.add_argument("--guidance-scale", type=float, default=7.5,
                           help="Classifier-free guidance scale")
    gen_parser.add_argument("--eta", type=float, default=0.0,
                           help="DDIM eta (0 = deterministic)")
    gen_parser.add_argument("--cpu", action="store_true",
                           help="Force CPU generation")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test data and model")
    test_parser.add_argument("--data-dir", type=str, default="/teamspace/studios/this_studio/text_to_sign/training_data",
                            help="Path to training data directory")
    test_parser.add_argument("--cpu", action="store_true",
                            help="Force CPU testing")
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate model quality")
    val_parser.add_argument("--checkpoint", type=str, required=True,
                           help="Path to model checkpoint")
    val_parser.add_argument("--data-dir", type=str, default="/teamspace/studios/this_studio/text_to_sign/training_data",
                           help="Path to training data directory")
    val_parser.add_argument("--output-dir", type=str, default="validation_output",
                           help="Directory to save validation results")
    val_parser.add_argument("--num-samples", type=int, default=50,
                           help="Number of samples for evaluation")
    val_parser.add_argument("--benchmark-repeats", type=int, default=5,
                           help="Number of repeated runs for timing benchmark")
    val_parser.add_argument("--steps", type=int, default=20,
                           help="Number of inference steps for validation-time generation")
    val_parser.add_argument("--guidance-scale", type=float, default=5.0,
                           help="CFG scale for validation-time generation")
    val_parser.add_argument("--fvd-backbone", type=str, default="videomae",
                           choices=["videomae", "r3d_18"],
                           help="Video feature backbone for FVD")
    val_parser.add_argument("--skip-backtranslation", action="store_true",
                           help="Skip GloFE back-translation evaluation")
    val_parser.add_argument("--cpu", action="store_true",
                           help="Force CPU validation")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "test":
        test(args)
    elif args.command == "validate":
        validate(args)


if __name__ == "__main__":
    main()
