#!/usr/bin/env bash
set -euo pipefail

ROOT="/teamspace/studios/this_studio/text_to_sign"
CKPT="${1:-$ROOT/text_to_sign/checkpoints_textboost_full10_small/text2sign_20260114_145727/best_model.pt}"
OUTDIR="$ROOT/conditional_loss_audit_20260321"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$OUTDIR/full10_posteval_summary_${STAMP}.md"

mkdir -p "$OUTDIR"

SHORT_JSON="$OUTDIR/full10_short_high_t_${STAMP}.json"
LARGE_JSON="$OUTDIR/full10_large_high_t_${STAMP}.json"

cd "$ROOT"

echo "[posteval] checkpoint: $CKPT"
echo "[posteval] running short high-t audit..."
python conditional_loss_audit.py \
  --checkpoint "$CKPT" \
  --data-dir "$ROOT/training_data" \
  --output "$SHORT_JSON" \
  --num-samples 8 \
  --batch-size 4 \
  --repeats 1 \
  --max-word-count 5 \
  --unique-prompts \
  --diverse-batches \
  --min-timestep 70 \
  --max-timestep 100 \
  --cpu

echo "[posteval] running larger high-t audit..."
python conditional_loss_audit.py \
  --checkpoint "$CKPT" \
  --data-dir "$ROOT/training_data" \
  --output "$LARGE_JSON" \
  --num-samples 24 \
  --batch-size 4 \
  --repeats 1 \
  --max-word-count 5 \
  --unique-prompts \
  --diverse-batches \
  --min-timestep 70 \
  --max-timestep 100 \
  --cpu

STAMP="$STAMP" SHORT_JSON="$SHORT_JSON" LARGE_JSON="$LARGE_JSON" SUMMARY="$SUMMARY" python - <<'PY'
import json
from pathlib import Path
import os

outdir = Path('/teamspace/studios/this_studio/text_to_sign/conditional_loss_audit_20260321')
stamp = os.environ.get('STAMP', '')
short_json = Path(os.environ['SHORT_JSON'])
large_json = Path(os.environ['LARGE_JSON'])
summary = Path(os.environ['SUMMARY'])

s = json.loads(short_json.read_text())
l = json.loads(large_json.read_text())

def g(d, *keys):
    x = d
    for k in keys:
        x = x[k]
    return x

text = f"""# Full-10 post-training experiment summary

Checkpoint: `{s['checkpoint']}`

## Short audit (num_samples={s['num_samples']})
- top1 accuracy mean: {g(s,'prompt_ranking','top1_accuracy','mean'):.4f}
- mean correct rank: {g(s,'prompt_ranking','mean_correct_rank','mean'):.4f}
- none-minus-normal mean: {g(s,'none_minus_normal','mean'):.6f}
- random-minus-normal mean: {g(s,'random_minus_normal','mean'):.6f}

## Larger audit (num_samples={l['num_samples']})
- top1 accuracy mean: {g(l,'prompt_ranking','top1_accuracy','mean'):.4f}
- mean correct rank: {g(l,'prompt_ranking','mean_correct_rank','mean'):.4f}
- none-minus-normal mean: {g(l,'none_minus_normal','mean'):.6f}
- random-minus-normal mean: {g(l,'random_minus_normal','mean'):.6f}

## Artifacts
- `{short_json}`
- `{large_json}`
"""
summary.write_text(text)
print(f"[posteval] wrote summary: {summary}")
PY

echo "[posteval] done"
