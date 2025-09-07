"""
Test script to verify TensorBoard logger functionality
"""

import torch
import tempfile
import os
from config import Config
from utils.tensorboard_logger import TensorBoardLogger, create_tensorboard_logger

def test_tensorboard_logger():
    """Test basic TensorBoard logger functionality"""
    
    print("🔍 Testing TensorBoard Logger...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = os.path.join(temp_dir, "test_logs")
        
        # Create logger
        logger = TensorBoardLogger(log_dir, Config)
        
        try:
            # Test 1: Basic scalar logging
            print("✅ Testing scalar logging...")
            training_metrics = {
                'loss': 0.5,
                'learning_rate': 0.001,
                'grad_norm': 1.2,
                'batch_size': 4
            }
            logger.log_training_step(training_metrics, step=100)
            
            # Test 2: Diffusion metrics
            print("✅ Testing diffusion metrics...")
            diffusion_metrics = {
                'noise_mse': 0.1,
                'noise_mae': 0.08,
                'snr': 15.5,
                'timestep_distribution': [100, 200, 300, 400, 500]
            }
            logger.log_diffusion_metrics(diffusion_metrics, step=100)
            
            # Test 3: Epoch summary
            print("✅ Testing epoch summary...")
            epoch_metrics = {
                'loss': 0.45,
                'time': 120.5,
                'grad_norm': 1.1,
                'learning_rate': 0.001,
                'samples_per_second': 8.5
            }
            logger.log_epoch_summary(epoch_metrics, epoch=10)
            
            # Test 4: System metrics
            print("✅ Testing system metrics...")
            logger.log_system_metrics(epoch=10)
            
            # Test 5: Configuration logging
            print("✅ Testing configuration logging...")
            logger.log_configuration(Config, epoch=0)
            
            # Test 6: Fake model for architecture logging
            print("✅ Testing model architecture logging...")
            fake_model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1)
            )
            logger.log_model_architecture(fake_model, epoch=0)
            
            # Test 7: Generated samples (fake data)
            print("✅ Testing sample logging...")
            fake_samples = torch.randn(2, 3, 8, 64, 64)  # (B, C, T, H, W)
            logger.log_generated_samples(fake_samples, step=100, tag="test_samples")
            
            # Test 8: Noise visualization (fake data)
            print("✅ Testing noise visualization...")
            fake_pred_noise = torch.randn(1, 3, 8, 64, 64)
            fake_actual_noise = torch.randn(1, 3, 8, 64, 64)
            fake_video = torch.randn(1, 3, 8, 64, 64)
            logger.log_noise_visualization(fake_pred_noise, fake_actual_noise, fake_video, step=100)
            
            # Flush and get summary
            logger.flush()
            summary = logger.get_logging_summary()
            
            print("📊 Logging Summary:")
            for key, value in summary.items():
                print(f"   {key}: {value}")
            
            print("✅ All TensorBoard logger tests passed!")
            print(f"📁 Test logs created in: {log_dir}")
            
            # Check if files were created
            if os.path.exists(log_dir):
                files = os.listdir(log_dir)
                if files:
                    print(f"📄 Generated files: {files}")
                else:
                    print("⚠️  No files generated in log directory")
            else:
                print("❌ Log directory was not created")
        
        except Exception as e:
            print(f"❌ Logger test failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            logger.close()

def test_logger_integration():
    """Test logger integration with config"""
    
    print("\n🔗 Testing logger integration with config...")
    
    try:
        # Test config-based logger creation
        logger = create_tensorboard_logger(Config)
        
        print(f"✅ Logger created with log dir: {getattr(logger.writer, 'log_dir', 'unknown')}")
        
        # Test configuration validation
        Config.validate_config()
        print("✅ Configuration validation passed")
        
        # Test logging categories
        categories = Config.TENSORBOARD_LOG_CATEGORIES
        print(f"✅ Found {len(categories)} logging categories:")
        for i, category in enumerate(categories[:5], 1):  # Show first 5
            print(f"   {i}. {category}")
        if len(categories) > 5:
            print(f"   ... and {len(categories) - 5} more")
        
        logger.close()
        print("✅ Logger integration test passed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tensorboard_logger()
    test_logger_integration()
    print("\n🎯 TensorBoard logger testing complete!")
