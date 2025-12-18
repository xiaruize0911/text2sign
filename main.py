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
    from config import ModelConfig, TrainingConfig, DDIMConfig
    from trainer import create_trainer
    
    print("=" * 60)
    print("Text-to-Sign Language DDIM Diffusion Model - Training")
    print("=" * 60)
    
    # Create configs
    model_config = ModelConfig(
        image_size=args.image_size,
        num_frames=args.num_frames,
        model_channels=args.model_channels,
        num_heads=args.num_heads,
    )
    
    train_config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        sample_every=args.sample_every,
        use_amp=not args.no_amp,
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
    )
    
    ddim_config = DDIMConfig(
        num_train_timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Data directory: {train_config.data_dir}")
    print(f"  Image size: {model_config.image_size}")
    print(f"  Num frames: {model_config.num_frames}")
    print(f"  Model channels: {model_config.model_channels}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Epochs: {train_config.num_epochs}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Device: {train_config.device}")
    print(f"  Mixed precision: {train_config.use_amp}")
    print(f"  Timesteps: {ddim_config.num_train_timesteps}")
    print(f"  Beta schedule: {ddim_config.beta_schedule}")
    print()
    
    # Create trainer
    trainer = create_trainer(model_config, train_config, ddim_config)
    
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
        prev_sample, pred_x0 = scheduler.step(output, 500, noisy)
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
  python main.py train --data-dir text2sign/training_data --epochs 100

  # Generate sign language GIFs
  python main.py generate --checkpoint checkpoints/best_model.pt --prompt "Hello"

  # Validate model quality
  python main.py validate --checkpoint checkpoints/best_model.pt --data-dir text2sign/training_data

  # Test data and model
  python main.py test --data-dir text2sign/training_data
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-dir", type=str, default="text2sign/training_data",
                             help="Path to training data directory")
    train_parser.add_argument("--batch-size", type=int, default=4,
                             help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=100,
                             help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4,
                             help="Learning rate")
    train_parser.add_argument("--image-size", type=int, default=64,
                             help="Image size to resize to")
    train_parser.add_argument("--num-frames", type=int, default=16,
                             help="Number of frames per video")
    train_parser.add_argument("--model-channels", type=int, default=128,
                             help="Base model channels")
    train_parser.add_argument("--num-heads", type=int, default=8,
                             help="Number of attention heads")
    train_parser.add_argument("--timesteps", type=int, default=1000,
                             help="Number of diffusion timesteps")
    train_parser.add_argument("--beta-schedule", type=str, default="linear",
                             choices=["linear", "cosine"],
                             help="Beta schedule type")
    train_parser.add_argument("--prediction-type", type=str, default="epsilon",
                             choices=["epsilon", "v_prediction"],
                             help="What the model predicts")
    train_parser.add_argument("--num-workers", type=int, default=4,
                             help="Number of data loading workers")
    train_parser.add_argument("--checkpoint-dir", type=str, default="text_to_sign/checkpoints",
                             help="Directory to save checkpoints")
    train_parser.add_argument("--log-dir", type=str, default="text_to_sign/logs",
                             help="Directory for TensorBoard logs")
    train_parser.add_argument("--save-every", type=int, default=5,
                             help="Save checkpoint every N epochs")
    train_parser.add_argument("--log-every", type=int, default=100,
                             help="Log to TensorBoard every N steps")
    train_parser.add_argument("--sample-every", type=int, default=1000,
                             help="Generate samples every N steps")
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
    test_parser.add_argument("--data-dir", type=str, default="text2sign/training_data",
                            help="Path to training data directory")
    test_parser.add_argument("--cpu", action="store_true",
                            help="Force CPU testing")
    
    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate model quality")
    val_parser.add_argument("--checkpoint", type=str, required=True,
                           help="Path to model checkpoint")
    val_parser.add_argument("--data-dir", type=str, default="../text2sign/training_data",
                           help="Path to training data directory")
    val_parser.add_argument("--output-dir", type=str, default="validation_output",
                           help="Directory to save validation results")
    val_parser.add_argument("--num-samples", type=int, default=50,
                           help="Number of samples for evaluation")
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
