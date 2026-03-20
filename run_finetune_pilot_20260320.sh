#!/usr/bin/env bash
set -euo pipefail

cd /teamspace/studios/this_studio/text_to_sign
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u main.py train \
  --resume /teamspace/studios/this_studio/text_to_sign/text_to_sign/checkpoints/text2sign_20260319_015042/best_model.pt \
  --model-size large \
  --num-frames 32 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --epochs 20 \
  --lr 1e-5 \
  --num-workers 2 \
  --split-mode signer_disjoint \
  --precision auto \
  --no-compile \
  --clip-trainable-layers 2 \
  --sample-every 8 \
  --sample-steps 8 \
  --sample-guidance-scale 5.0 \
  --save-every 1 \
  --max-optimizer-steps 32 \
  --checkpoint-dir text_to_sign/checkpoints_finetune \
  --log-dir text_to_sign/logs_finetune
