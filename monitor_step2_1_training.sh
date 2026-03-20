#!/usr/bin/env bash
set -euo pipefail

cd /teamspace/studios/this_studio/text_to_sign

INTERVAL_SECONDS="${1:-300}"
LOG_FILE=/teamspace/studios/this_studio/text_to_sign/step2_1_training_l4_b2_resume.log
MONITOR_LOG=/teamspace/studios/this_studio/text_to_sign/step2_1_training_l4_b2_monitor.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting monitor loop (interval=${INTERVAL_SECONDS}s)" >> "${MONITOR_LOG}"

while true; do
  {
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitor snapshot"
    echo "--- GPU ---"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
    echo "--- Training processes ---"
    pgrep -af 'python .*main.py train' || true
    echo "--- Log tail ---"
    tail -n 40 "${LOG_FILE}" 2>/dev/null || echo "Log file not available yet"
    echo
  } >> "${MONITOR_LOG}"

  sleep "${INTERVAL_SECONDS}"
done