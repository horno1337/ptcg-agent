#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="$repo_root/tools/checkpoints/qu-v2c-canary-v1/local-ab-640"
mkdir -p "$run_dir"
cd "$repo_root"

for shard in 0 1 2 3 4 5 6 7; do
    log="$run_dir/shard-${shard}.log"
    report="$run_dir/shard-${shard}.json"
    pid_file="$run_dir/shard-${shard}.pid"
    if [[ -e "$report" || -e "$pid_file" ]]; then
        echo "refusing to overwrite shard $shard state" >&2
        exit 2
    fi
    nohup env \
        OPENBLAS_NUM_THREADS=1 \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        python tools/eval_ab.py 640 agent/weights.npz \
            --base agent/weights.npz \
            --candidate-policy qu-v2c-canary \
            --opp pool:8 \
            --opp-policy mixed \
            --seed 260726 \
            --num-shards 8 \
            --shard-index "$shard" \
            --max-selects 5000 \
            --time-bank 600 \
            --json-out "$report" \
            --quiet \
        >"$log" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >"$pid_file"
    echo "started shard $shard pid=$pid"
done

# Keep the supervisor alive. The execution environment deliberately reaps
# orphaned descendants, so a bare detached launch is not an overnight job.
wait
