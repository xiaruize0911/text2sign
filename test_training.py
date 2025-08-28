#!/usr/bin/env python3
"""
Short training test to check for NaN issues
"""

from config import Config
from methods import setup_training

def test_training():
    # Modify config for short test
    config = Config()
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 2
    config.LOG_EVERY = 1
    config.SAMPLE_EVERY = 10
    config.SAVE_EVERY = 10

    print('Starting short training test...')
    print(f'Device: {config.DEVICE}')
    print(f'Model: {config.MODEL_ARCHITECTURE}')
    print(f'Batch size: {config.BATCH_SIZE}')
    print(f'Learning rate: {config.LEARNING_RATE}')
    print(f'Use AMP: {config.USE_AMP}')

    try:
        trainer = setup_training(config)
        trainer.train()
        print('✅ Training test completed successfully!')
    except Exception as e:
        print(f'❌ Training failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training()
