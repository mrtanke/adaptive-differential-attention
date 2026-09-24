#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints results pids
for seed in 20260920 20260921 20260922; do
  for variant in standard differential headwise tokenwise token_headwise; do
    config="configs/experiment2_seed${seed}_${variant}.json"
    log="logs/experiment2_seed${seed}_${variant}.log"
    echo "[$(date -Is)] starting ${seed} ${variant}" | tee -a "$log"
    /root/miniconda3/bin/python3 scripts/run_experiment2.py --config "$config" 2>&1 | tee -a "$log"
    echo "[$(date -Is)] finished ${seed} ${variant}" | tee -a "$log"
  done
done
/root/miniconda3/bin/python3 scripts/summarize_experiment2.py 2>&1 | tee logs/experiment2_aggregate.log
