#!/usr/bin/env bash
set -euo pipefail
cd /root/differential-attention-project
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=.
for variant in standard differential headwise tokenwise token_headwise; do
  echo "[$(date -Is)] starting $variant"
  python scripts/run_experiment1.py --config "configs/experiment1_${variant}.json"
  echo "[$(date -Is)] finished $variant"
done
