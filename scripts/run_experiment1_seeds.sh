#!/usr/bin/env bash
set -euo pipefail
cd /root/differential-attention-project
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=.
for seed in 20260921 20260922; do
  for variant in standard differential headwise tokenwise token_headwise; do
    log="logs/experiment1_seed${seed}_${variant}.log"
    echo "[$(date -Is)] starting seed=${seed} variant=${variant}" | tee -a "$log"
    python scripts/run_experiment1.py --config "configs/experiment1_seed${seed}_${variant}.json" >> "$log" 2>&1
    echo "[$(date -Is)] finished seed=${seed} variant=${variant}" | tee -a "$log"
  done
done
