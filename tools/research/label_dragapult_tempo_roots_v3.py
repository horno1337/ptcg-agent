"""Replacement stratified panels after rejecting the v2 wrapper contract bug."""

from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import collect_dragapult_tempo_roots as C
from tools.research import label_dragapult_tempo_roots as P


C.RUN = C.ROOT / "tools/checkpoints/dragapult-tempo-roots-v2-20260812"
C.LOCK = C.RUN / "lock.json"
C.ATTEMPT = C.RUN / "attempt.json"
C.PUBLIC = C.RUN / "public-roots.jsonl"
C.PRIVILEGED = C.RUN / "privileged-roots.jsonl"
C.RESULT = C.RUN / "result.json"

INVALID_LOCK = C.RUN / "discovery-panels-v1/lock.json"
P.RUN = C.RUN / "discovery-panels-v2"
P.LOCK = P.RUN / "lock.json"
P.ATTEMPT = P.RUN / "attempt.json"
P.RESULT = P.RUN / "result.json"
P.ROOTS = 48
P.ROLLOUTS = 8
P.SPLIT_SEED = 2_026_081_247
P.SELECTION_DESCRIPTION = (
    "round-robin lowest root_id by opponent archetype/current-policy family, "
    "excluding every root exposed by invalid four-repetition v1"
)


def selected_roots() -> list[str]:
    excluded = set(json.loads(INVALID_LOCK.read_text())["selection"]["root_ids"])
    groups = defaultdict(list)
    for row in P.read_jsonl(C.PUBLIC):
        if (
            row["root_id"] in excluded
            or not P.discovery_episode(int(row["source"]["episode_id"]))
        ):
            continue
        archetype = row["source"]["opponent_key"].split("/", 1)[0]
        family = row["current_policy"]["family"]
        groups[(archetype, family)].append(row["root_id"])
    queues = {
        key: deque(sorted(values)) for key, values in sorted(groups.items())
    }
    selected = []
    while len(selected) < P.ROOTS and any(queues.values()):
        for key in sorted(queues):
            if queues[key] and len(selected) < P.ROOTS:
                selected.append(queues[key].popleft())
    if len(selected) < P.ROOTS:
        raise P.PanelError(f"only {len(selected)} replacement discovery roots")
    return selected


P.selected_roots = selected_roots


if __name__ == "__main__":
    raise SystemExit(P.main())
