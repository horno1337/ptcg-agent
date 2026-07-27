#!/usr/bin/env bash
# Outcome-blind refresh for the locked MD-v1/Grim vs Qu-v2B/Alakazam read.
# This downloads replay files and reports counts only. Do not aggregate rewards
# until both arms reach the preregistered 160-game cohort.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$ROOT/tools/checkpoints/md-v1/ladder-cross-deck-v2"

python "$ROOT/tools/download_episodes.py" \
    --submission-id 55013396 --refresh --per-sub 200 --workers 4 \
    --out "$RUN/md"
python "$ROOT/tools/download_episodes.py" \
    --submission-id 55013385 --refresh --per-sub 200 --workers 4 \
    --out "$RUN/qu"

md_count="$(find "$RUN/md" -maxdepth 1 -type f -name '[0-9]*.json' | wc -l)"
qu_count="$(find "$RUN/qu" -maxdepth 1 -type f -name '[0-9]*.json' | wc -l)"
printf 'outcome-blind replay files: md=%s/160 qu=%s/160\n' \
    "$md_count" "$qu_count"
