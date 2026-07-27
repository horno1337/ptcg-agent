#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_root="$repo_root/tools/checkpoints/md-v2-scaled"
trainer="$repo_root/tools/research/train_qu_v2a.py"
initial="$repo_root/tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-checkpoint.pt"
target="c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
python_bin="${PTCG_TRAIN_PYTHON:-$HOME/.venvs/ptcg-rl/bin/python}"

for scale in 1500 4000 8000 full; do
    "$python_bin" "$trainer" \
        --manifest "$run_root/corpus/scale-$scale.json" \
        --out-dir "$run_root/model-$scale" \
        --cache-dir "$run_root/cache" \
        --epochs 10 \
        --batch-size 128 \
        --shuffle-buffer 4096 \
        --learning-rate 0.00005 \
        --weight-decay 0.00001 \
        --value-coefficient 0 \
        --gradient-clip 1 \
        --seed 20260726 \
        --device cuda \
        --win-weight 1 \
        --draw-weight 1 \
        --loss-weight 1 \
        --game-normalized \
        --initial-checkpoint "$initial" \
        --target-deck-sha256 "$target" \
        --target-select-type 0 \
        --freeze-public-backbone \
        --kl-coefficient 0 \
        --defer-test \
        --require-gpu \
        --min-available-gib 5.5 \
        --min-gpu-free-gib 5.5
done
