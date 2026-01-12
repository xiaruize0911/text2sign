#!/bin/bash
# Quick start script for training with optimizations

set -e

echo "=========================================="
echo "Text-to-Sign Training - Quick Start"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "config.py" ]; then
    echo "❌ Error: config.py not found. Please run from text_to_sign directory"
    exit 1
fi

echo "✅ Configuration verified"
echo ""

# Print optimization summary
echo "Active Optimizations:"
echo "--------------------"
python3 -c "
from config import get_config
c = get_config()
print(f'  ✅ EMA: {c[\"training\"].use_ema} (decay={c[\"training\"].ema_decay})')
print(f'  ✅ Beta Schedule: {c[\"ddim\"].beta_schedule}')
print(f'  ✅ Warmup Steps: {c[\"training\"].warmup_steps}')
print(f'  ✅ Learning Rate: {c[\"training\"].learning_rate}')
print(f'  ✅ Epochs: {c[\"training\"].num_epochs}')
print(f'  ✅ Batch Size: {c[\"training\"].batch_size} (effective: {c[\"training\"].batch_size * c[\"training\"].gradient_accumulation_steps})')
"
echo ""

# Check for existing checkpoints
if [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
    echo "⚠️  Found existing checkpoints:"
    ls -lh checkpoints/ | tail -n +2 | head -n 5
    echo ""
    read -p "Do you want to resume from latest checkpoint? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        LATEST=$(ls -t checkpoints/*.pt 2>/dev/null | head -n 1)
        if [ -n "$LATEST" ]; then
            echo "Will resume from: $LATEST"
            RESUME_ARG="--resume $LATEST"
        fi
    fi
fi

echo ""
echo "🚀 Starting training..."
echo ""

# Start training
if [ -n "$RESUME_ARG" ]; then
    python3 main.py train $RESUME_ARG
else
    python3 main.py train
fi
