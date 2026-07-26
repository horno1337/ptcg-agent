#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run_phase() {
    local label="$1"
    local opponents="$2"
    local seed="$3"
    local run_dir="$repo_root/tools/checkpoints/qu-v2c-canary-v1/$label"
    mkdir -p "$run_dir"
    local pids=()
    for shard in 0 1 2 3 4 5 6 7; do
        local log="$run_dir/shard-${shard}.log"
        local report="$run_dir/shard-${shard}.json"
        local pid_file="$run_dir/shard-${shard}.pid"
        if [[ -e "$report" || -e "$pid_file" ]]; then
            echo "refusing to overwrite $label shard $shard" >&2
            return 2
        fi
        env \
            OPENBLAS_NUM_THREADS=1 \
            OMP_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            python tools/eval_ab.py 50000 agent/weights.npz \
                --base agent/weights.npz \
                --candidate-policy qu-v2c-canary \
                --opp "$opponents" \
                --opp-policy mixed \
                --seed "$seed" \
                --num-shards 8 \
                --shard-index "$shard" \
                --max-selects 5000 \
                --time-bank 600 \
                --json-out "$report" \
                --quiet \
            >"$log" 2>&1 </dev/null &
        local pid=$!
        pids+=("$pid")
        printf '%s\n' "$pid" >"$pid_file"
        echo "started $label shard $shard pid=$pid"
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "$label had a failed shard" >&2
        return 3
    fi
}

run_phase "local-ab-main-50000" "pool:8" 260727
run_phase "local-ab-holdout-50000" "pool:8:16" 260728
