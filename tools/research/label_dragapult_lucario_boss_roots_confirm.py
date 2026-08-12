"""Episode-disjoint confirmation for the Lucario Boss/attack discovery."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import label_dragapult_lucario_boss_roots as D  # noqa: E402


P, C = D.P, D.C
P.RUN = ROOT / "tools/checkpoints/dragapult-lucario-boss-roots-v1-20260812/confirmation"
P.LOCK = P.RUN / "lock.json"
P.ATTEMPT = P.RUN / "attempt.json"
P.RESULT = P.RUN / "result.json"
P.ROOTS = 5
P.ROLLOUTS = 16
P.SELECTION_DESCRIPTION = (
    "all unexposed exact Mega-Lucario loss roots where Boss and an attack are "
    "both legal and SHA256(seed:episode_id)[0] >= 128, ordered by root_id"
)


def selected_roots() -> list[str]:
    excluded = D.excluded_roots()
    rows = []
    for row in P.read_jsonl(C.PUBLIC):
        source = row["source"]
        if (
            row["root_id"] in excluded
            or not source["opponent_key"].startswith("Mega Lucario/")
            or not {"boss", "attack"}.issubset(row["option_families"])
        ):
            continue
        payload = f"{P.SPLIT_SEED}:{source['episode_id']}".encode()
        if hashlib.sha256(payload).digest()[0] >= 128:
            rows.append(row["root_id"])
    rows.sort()
    if len(rows) != P.ROOTS:
        raise P.PanelError(f"Lucario Boss confirmation inventory drifted: {len(rows)}")
    return rows


P.selected_roots = selected_roots


if __name__ == "__main__":
    raise SystemExit(P.main())
