#!/bin/bash

# Quick start training script
# This script provides an easy way to start training with reasonable defaults

# Set default values
DATA_DIR="./training_data"
BATCH_SIZE=4
EPOCHS=50
FRAME_SIZE=128
MAX_FRAMES=28
MODEL_CHANNELS=128
LEARNING_RATE=1e-4
SAVE_DIR="./checkpoints"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --small)
            # Small model for testing
            BATCH_SIZE=2
            FRAME_SIZE=64
            MAX_FRAMES=16
            MODEL_CHANNELS=64
            shift
            ;;
        --large)
            # Large model for better quality
            BATCH_SIZE=8
            FRAME_SIZE=128
            MAX_FRAMES=28
            MODEL_CHANNELS=256
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data_dir DIR     Training data directory (default: ./training_data)"
            echo "  --batch_size N     Batch size (default: 4)"
            echo "  --epochs N         Number of epochs (default: 50)"
            echo "  --gpu ID           GPU ID to use (optional)"
            echo "  --small            Use small model settings for testing"
            echo "  --large            Use large model settings for better quality"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set GPU if specified
if [ ! -z "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist!"
    echo "Please provide a valid data directory with --data_dir"
    exit 1
fi

# Check if there are any .gif files in the data directory
if [ ! -f "$DATA_DIR"/*.gif ]; then
    echo "Warning: No .gif files found in '$DATA_DIR'"
    echo "Make sure your training data is in the correct location"
fi

echo "Starting training with the following settings:"
echo "  Data directory: $DATA_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Frame size: ${FRAME_SIZE}x${FRAME_SIZE}"
echo "  Max frames: $MAX_FRAMES"
echo "  Model channels: $MODEL_CHANNELS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Save directory: $SAVE_DIR"
echo ""

# Create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Run training
python train.py \
    --data_dir "$DATA_DIR" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $EPOCHS \
    --max_frames $MAX_FRAMES \
    --frame_size $FRAME_SIZE \
    --model_channels $MODEL_CHANNELS \
    --save_dir "$SAVE_DIR" \
    --text_encoder simple \
    --scheduler ddpm
