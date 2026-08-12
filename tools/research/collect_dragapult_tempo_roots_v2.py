"""Expanded fresh learner-root collection for outcome-based Dragapult work."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import collect_dragapult_tempo_roots as C


C.RUN = C.ROOT / "tools/checkpoints/dragapult-tempo-roots-v2-20260812"
C.LOCK = C.RUN / "lock.json"
C.ATTEMPT = C.RUN / "attempt.json"
C.PUBLIC = C.RUN / "public-roots.jsonl"
C.PRIVILEGED = C.RUN / "privileged-roots.jsonl"
C.RESULT = C.RUN / "result.json"
C.GAMES = 256
C.SEED = 2_026_081_245
C.MAX_ROOTS_PER_LOSS = 6


if __name__ == "__main__":
    raise SystemExit(C.main())
