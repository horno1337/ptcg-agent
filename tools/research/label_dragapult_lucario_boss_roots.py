"""Causal panels for exact-07bed Lucario roots with Boss and attack legal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import collect_dragapult_tempo_roots as C  # noqa: E402
from tools.research import label_dragapult_tempo_roots as P  # noqa: E402


SOURCE = ROOT / "tools/checkpoints/dragapult-tempo-roots-v2-20260812"
C.RUN = SOURCE
C.LOCK = SOURCE / "lock.json"
C.ATTEMPT = SOURCE / "attempt.json"
C.PUBLIC = SOURCE / "public-roots.jsonl"
C.PRIVILEGED = SOURCE / "privileged-roots.jsonl"
C.RESULT = SOURCE / "result.json"

P.RUN = ROOT / "tools/checkpoints/dragapult-lucario-boss-roots-v1-20260812/discovery"
P.LOCK = P.RUN / "lock.json"
P.ATTEMPT = P.RUN / "attempt.json"
P.RESULT = P.RUN / "result.json"
P.ROOTS = 7
P.ROLLOUTS = 16
P.SPLIT_SEED = 2_026_081_230
P.SELECTION_DESCRIPTION = (
    "all unexposed exact Mega-Lucario loss roots where Boss and an attack are "
    "both legal and SHA256(seed:episode_id)[0] < 128, ordered by root_id"
)


def excluded_roots() -> set[str]:
    result = set()
    for path in (
        SOURCE / "discovery-panels-v1/result.json",
        SOURCE / "discovery-panels-v2/result.json",
    ):
        value = json.loads(path.read_text())
        result.update(row["root_id"] for row in value.get("panels", ()))
    return result


def selected_roots() -> list[str]:
    excluded = excluded_roots()
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
        if hashlib.sha256(payload).digest()[0] < 128:
            rows.append(row["root_id"])
    rows.sort()
    if len(rows) != P.ROOTS:
        raise P.PanelError(f"Lucario Boss discovery inventory drifted: {len(rows)}")
    return rows


P.selected_roots = selected_roots


if __name__ == "__main__":
    raise SystemExit(P.main())
