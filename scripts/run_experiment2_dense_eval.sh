#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results figures pids
for seed in 20260920 20260921 20260922; do
  for variant in standard differential headwise tokenwise token_headwise; do
    log="logs/experiment2_dense_seed${seed}_${variant}.log"
    echo "[$(date -Is)] evaluating ${seed} ${variant}" | tee "$log"
    /root/miniconda3/bin/python3 scripts/evaluate_experiment2_dense.py --config "configs/experiment2_seed${seed}_${variant}.json" 2>&1 | tee -a "$log"
    echo "[$(date -Is)] finished ${seed} ${variant}" | tee -a "$log"
  done
done
/root/miniconda3/bin/python3 scripts/summarize_experiment2_dense.py 2>&1 | tee logs/experiment2_dense_aggregate.log
