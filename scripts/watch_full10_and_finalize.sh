#!/usr/bin/env bash
set -euo pipefail

ROOT="/teamspace/studios/this_studio"
T2S="$ROOT/text_to_sign"
SUMMARY_JSON="$T2S/text_to_sign/logs_textboost_full10_small/text2sign_20260114_145727/json/text2sign_20260114_145727_summary.json"
WATCH_LOG="$T2S/post_full10_watcher_20260321.log"
EXP_LOG="$T2S/post_full10_experiments_20260321.log"

mkdir -p "$T2S/conditional_loss_audit_20260321"
: > "$WATCH_LOG"
: > "$EXP_LOG"

echo "[watcher] started $(date -Is)" >> "$WATCH_LOG"

while true; do
  if [[ -f "$SUMMARY_JSON" ]]; then
    if python - <<PY
import json
from pathlib import Path
p = Path(r'''$SUMMARY_JSON''')
d = json.loads(p.read_text())
raise SystemExit(0 if d.get('total_epochs', 0) >= 10 else 1)
PY
    then
      echo "[watcher] detected total_epochs>=10 at $(date -Is)" >> "$WATCH_LOG"
      break
    fi
  fi
  echo "[watcher] waiting $(date -Is)" >> "$WATCH_LOG"
  sleep 300
done

bash "$T2S/scripts/post_full10_experiments.sh" \
  "$T2S/text_to_sign/checkpoints_textboost_full10_small/text2sign_20260114_145727/best_model.pt" \
  >> "$EXP_LOG" 2>&1

python "$T2S/scripts/update_paper_from_full10.py" >> "$EXP_LOG" 2>&1

echo "[watcher] finalize complete $(date -Is)" >> "$WATCH_LOG"
