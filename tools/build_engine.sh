#!/usr/bin/env bash
# Build the official cabt engine into engine/libcg.so.
#
# The engine source is competition-use-only (not redistributable) and lives
# OUTSIDE this repo; point ENGINE_SRC at the ptcgProgram folder. Nothing under
# engine/ is committed.
set -euo pipefail

ENGINE_SRC="${ENGINE_SRC:-$HOME/Desktop/ptcg_engine/ptcgProgram 22}"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/engine"

if [[ ! -f "$ENGINE_SRC/Export.cpp" ]]; then
    echo "error: Export.cpp not found in ENGINE_SRC=$ENGINE_SRC" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
# -include climits: MSVC pulls it in transitively, GCC does not (INT_MAX use
# in EffectProc.h). Injected here so the licensed source stays unmodified.
g++ -std=c++20 -O2 -shared -fPIC -fvisibility=hidden -include climits \
    "$ENGINE_SRC/Export.cpp" -o "$OUT_DIR/libcg.so"
echo "built $OUT_DIR/libcg.so"
