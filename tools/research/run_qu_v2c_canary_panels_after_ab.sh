#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

main_dir="tools/checkpoints/qu-v2c-canary-v1/local-ab-main-50000"
holdout_dir="tools/checkpoints/qu-v2c-canary-v1/local-ab-holdout-50000"
root_dir="tools/checkpoints/qu-v2c-roots-v1/qu-v2c-canary-override-candidates-45"
discovery="tools/checkpoints/qu-v2c-canary-v1/override-45-discovery-r32.json"
confirmation="tools/checkpoints/qu-v2c-canary-v1/override-45-confirmation-r32.json"

for _ in $(seq 1 720); do
    main_count=$(find "$main_dir" -maxdepth 1 -name 'shard-*.json' | wc -l)
    holdout_count=$(find "$holdout_dir" -maxdepth 1 -name 'shard-*.json' | wc -l)
    if [[ "$main_count" -eq 8 && "$holdout_count" -eq 8 ]]; then
        break
    fi
    sleep 60
done

main_count=$(find "$main_dir" -maxdepth 1 -name 'shard-*.json' | wc -l)
holdout_count=$(find "$holdout_dir" -maxdepth 1 -name 'shard-*.json' | wc -l)
if [[ "$main_count" -ne 8 || "$holdout_count" -ne 8 ]]; then
    echo "timed out waiting for both extended A/B phases" >&2
    exit 2
fi

python tools/research/label_qu_v2c_exact_panels.py \
    --root-dir "$root_dir" \
    --rollouts 32 \
    --json-out "$discovery"

python tools/research/label_qu_v2c_exact_panels.py \
    --root-dir "$root_dir" \
    --rollouts 32 \
    --json-out "$confirmation"
