"""Lock the fresh current-Lucario discovery cohort before rollout outcomes."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "tools/checkpoints/md-v2-lucario-roots-v1"
COLLECTION = RUN / "collection"
OUTPUT = RUN / "rollout-lock.json"
CANDIDATE_ROOTS = 500


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Round-robin over games, seats, and turn buckets in a stable order."""
    remaining = sorted(rows, key=lambda row: row["root_id"])
    games: Counter[str] = Counter()
    seats: Counter[int] = Counter()
    turns: Counter[str] = Counter()
    selected: list[Mapping[str, Any]] = []

    def bucket(row: Mapping[str, Any]) -> str:
        turn = int(row["prompt"]["turn"] or 0)
        return "early" if turn <= 3 else "middle" if turn <= 6 else "late"

    while len(selected) < CANDIDATE_ROOTS:
        chosen = min(
            remaining,
            key=lambda row: (
                games[str(row["source"]["episode_id"])],
                seats[int(row["source"]["learner_seat"])],
                turns[bucket(row)],
                row["root_id"],
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        games[str(chosen["source"]["episode_id"])] += 1
        seats[int(chosen["source"]["learner_seat"])] += 1
        turns[bucket(chosen)] += 1
    return selected


def main() -> int:
    paths = {
        "collection_lock": RUN / "collection-lock.json",
        "collection_summary": COLLECTION / "summary.json",
        "public_roots": COLLECTION / "public-roots.jsonl",
        "privileged_roots": COLLECTION / "privileged-roots.jsonl",
        "candidate": ROOT / "tools/checkpoints/md-v1/md-v1-weights.npz",
        "base": ROOT / "agent/weights.npz",
        "meta": (
            ROOT
            / "tools/checkpoints/md-v1-current-threats-v1/recent-threat-meta.json"
        ),
        "labeler": ROOT / "tools/research/label_md_v2_lucario_panels.py",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    public = load_jsonl(paths["public_roots"])
    privileged = load_jsonl(paths["privileged_roots"])
    if len(public) != len(privileged) or len(public) < CANDIDATE_ROOTS:
        raise SystemExit("collection does not contain enough paired roots")
    if [r["root_id"] for r in public] != [r["root_id"] for r in privileged]:
        raise SystemExit("public/privileged collection order diverged")
    if len({r["identity"]["public_root_fingerprint"] for r in public}) != len(public):
        raise SystemExit("collection contains duplicate public states")
    selected = select(public)
    games = Counter(str(r["source"]["episode_id"]) for r in selected)
    seats = Counter(int(r["source"]["learner_seat"]) for r in selected)
    if len(games) < 30 or set(seats) != {0, 1}:
        raise SystemExit("selected cohort fails locked diversity requirements")
    payload = {
        "schema": "ptcg.md-v2.current-lucario-rollout-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_rollout_outcomes": True,
        "outcome_data_used_for_ordering": False,
        "selection": {
            "query_distribution": (
                "fresh MD-v1/Grim losses vs exact current Lucario; loss is "
                "not an action-value label"
            ),
            "candidate_roots": CANDIDATE_ROOTS,
            "ordered_root_ids": [row["root_id"] for row in selected],
            "ordered_root_ids_sha256": value_sha256(
                [row["root_id"] for row in selected]),
            "distinct_games": len(games),
            "roots_by_seat": dict(sorted(seats.items())),
            "ordering": (
                "stable root-id tie break, minimizing roots already selected "
                "from the same game, then seat and turn-bucket counts"
            ),
        },
        "discovery": {
            "rollouts_per_action": 32,
            "comparison": "MD-v1 root action minus frozen Qu-v2B root action",
            "significance_filter": "abs(delta) > 1.96 * paired_SE",
            "apply_before_independent_confirmation": True,
        },
        "confirmation": {
            "rollouts_per_action": 32,
            "roots": "every discovery-significant root in locked order",
            "independent_native_engine_draws": True,
            "minimum_sign_agreement": 0.85,
            "minimum_sign_confirmed_roots": 200,
            "minimum_distinct_confirmed_games": 30,
            "both_learner_seats_required": True,
        },
        "decision_rule": {
            "pass": (
                "all completion/diversity floors pass and sign agreement is "
                "at least 0.85"
            ),
            "on_fail": (
                "stop; do not lower thresholds, re-slice roots, or train MD-v2"
            ),
            "training_authorized_before_pass": False,
        },
        "artifacts": {
            label: {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
            }
            for label, path in paths.items()
        },
    }
    payload["lock_sha256"] = value_sha256(payload)
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite lock: {OUTPUT}")
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(payload["lock_sha256"])
    print(json.dumps({
        "roots": len(selected), "games": len(games),
        "roots_by_seat": dict(seats),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
