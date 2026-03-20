#!/usr/bin/env bash
set -euo pipefail

LOCK_FILE=/tmp/text2sign_step2_1_resume.lock
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another Step 2.1 resume job is already running. Exiting to avoid duplicate launches."
  exit 1
fi

cd /teamspace/studios/this_studio/text_to_sign
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u main.py train \
  --data-dir /teamspace/studios/this_studio/text_to_sign/training_data \
  --model-size large \
  --epochs 60 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --lr 5e-5 \
  --split-mode signer_disjoint \
  --precision auto \
  --no-compile \
  --num-workers 0 \
  --save-every 5 \
  --log-every 100 \
  --sample-every 4096 \
  --resume text_to_sign/checkpoints/text2sign_20260319_015042/checkpoint_epoch_10.pt \
  > /teamspace/studios/this_studio/text_to_sign/step2_1_training_l4_b2_resume.log 2>&1
