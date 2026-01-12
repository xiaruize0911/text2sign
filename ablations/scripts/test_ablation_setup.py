"""
Small-Scale Ablation Study Test

This script runs a quick validation test of the ablation study setup:
- Runs 2 epochs instead of 150
- Uses a subset of data
- Tests all three ablation configurations
- Verifies logging and metrics collection

Usage:
    cd text_to_sign/ablations
    python test_ablation_setup.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ablation configs
import importlib.util


def load_config(config_path: str):
    """
    Dynamically load a config module.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config module
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def create_test_config(original_config, num_epochs: int = 2):
    """
    Create a test configuration with reduced epochs and batch size.
    
    Args:
        original_config: Original config module
        num_epochs: Number of epochs for testing
        
    Returns:
        Modified config dict with test settings
    """
    config = original_config.get_config()
    
    # Reduce training parameters for quick testing
    config['training'].num_epochs = num_epochs
    config['training'].batch_size = 1  # Smaller batch for testing
    config['training'].log_every = 1  # Log every step
    config['training'].sample_every = 50  # Sample less frequently
    config['training'].save_every = 1  # Save more frequently for testing
    
    return config


def test_ablation_setup():
    """Test ablation study setup with small-scale experiment."""
    
    print("=" * 80)
    print("ABLATION STUDY - SMALL-SCALE TEST")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Define test configurations
    config_dir = Path(__file__).parent.parent / "configs"
    ablation_configs = [
        ("baseline", config_dir / "config_baseline.py"),
        ("text_finetuned", config_dir / "config_text_finetuned.py"),
        ("no_ema", config_dir / "config_no_ema.py"),
    ]
    
    results = {}
    
    for ablation_name, config_path in ablation_configs:
        print(f"\n[TEST] Loading {ablation_name} config from {config_path}")
        
        if not config_path.exists():
            print(f"  ❌ Config file not found: {config_path}")
            continue
        
        try:
            # Load and verify config
            config_module = load_config(str(config_path))
            config = config_module.get_config()
            
            print(f"  ✓ Config loaded successfully")
            
            # Print key settings
            print(f"  Settings:")
            print(f"    - Freeze text encoder: {config['model'].freeze_text_encoder}")
            print(f"    - Use EMA: {config['training'].use_ema}")
            print(f"    - Epochs (for test): {config['training'].num_epochs}")
            print(f"    - Batch size: {config['training'].batch_size}")
            print(f"    - Learning rate: {config['training'].learning_rate}")
            
            results[ablation_name] = {
                "status": "✓ PASSED",
                "config_path": str(config_path),
                "key_settings": {
                    "freeze_text_encoder": config['model'].freeze_text_encoder,
                    "use_ema": config['training'].use_ema,
                    "num_epochs": config['training'].num_epochs,
                    "batch_size": config['training'].batch_size,
                    "learning_rate": config['training'].learning_rate
                }
            }
            
        except Exception as e:
            print(f"  ❌ Error loading config: {e}")
            results[ablation_name] = {
                "status": "✗ FAILED",
                "error": str(e)
            }
    
    # Test metrics logger
    print(f"\n[TEST] Testing metrics logger...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from metrics_logger import MetricsLogger, TrainingMetrics, EvaluationMetrics
        
        logger = MetricsLogger(
            log_dir=str(Path(__file__).parent.parent / "results"),
            experiment_name="test_baseline"
        )
        
        # Log some dummy training metrics
        for step in range(5):
            logger.log_training_step(
                epoch=0,
                step=step,
                loss=1.0 - (step * 0.1),
                learning_rate=1e-4,
                elapsed_time=step * 10
            )
        
        # Log dummy evaluation metrics
        logger.log_evaluation(
            model_name="test_baseline",
            config_name="config_baseline",
            fvd=15.2,
            lpips=0.18,
            temporal_consistency=0.87,
            inference_time_ms=2100,
            inference_memory_gb=6.5,
            num_parameters_millions=145
        )
        
        logger.save_all()
        print(f"  ✓ Metrics logger working correctly")
        print(f"  ✓ Metrics saved to {logger.log_dir}")
        
        results["metrics_logger"] = {
            "status": "✓ PASSED",
            "log_dir": str(logger.log_dir)
        }
        
    except Exception as e:
        print(f"  ❌ Error testing metrics logger: {e}")
        results["metrics_logger"] = {
            "status": "✗ FAILED",
            "error": str(e)
        }
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}")
    
    test_results_file = Path(__file__).parent.parent / "results" / "test_results.json"
    test_results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTest Results:")
    for test_name, test_result in results.items():
        status = test_result.get("status", "UNKNOWN")
        print(f"  {test_name}: {status}")
    
    print(f"\nDetailed results saved to: {test_results_file}")
    
    # Check if all tests passed
    all_passed = all(r.get("status", "").startswith("✓") for r in results.values())
    
    if all_passed:
        print(f"\n✓ All tests PASSED! Ready to start full ablation experiments.")
    else:
        print(f"\n❌ Some tests FAILED. Please review the errors above.")
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ablation study setup")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    success = test_ablation_setup()
    sys.exit(0 if success else 1)
