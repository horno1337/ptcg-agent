"""Pre-register the current-frequency MD-v1 versus Alakazam deck gate."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model, policy  # noqa: E402
from tools.rl_env import build_paired_schedule  # noqa: E402
from tools.research import eval_md_v1_recent_weighted_field as EVAL  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v1-recent-weighted-field-v1"
OUTPUT = RUN / "lock.json"
GAMES = 320
SEED = 20260727


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


def main() -> int:
    paths = {
        "field": RUN / "field.json",
        "snapshot_builder": (
            ROOT / "tools/research/snapshot_recent_weighted_field.py"
        ),
        "evaluator": (
            ROOT / "tools/research/eval_md_v1_recent_weighted_field.py"
        ),
        "md_v1": ROOT / "tools/checkpoints/md-v1/md-v1-weights.npz",
        "qu_v2b": ROOT / "agent/weights.npz",
        "grim_deck": ROOT / "decks/md_v1_grimmsnarl.csv",
        "alakazam_deck": ROOT / "decks/deck.csv",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    field_payload, field = EVAL.load_field(paths["field"])
    base = model.load(str(paths["qu_v2b"]))
    if base is None or not getattr(base, "is_qu_v2", False):
        raise SystemExit("frozen Qu-v2B failed to load")
    opponents, _ = EVAL.build_opponents(field, base)
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    matchup_games = Counter(
        opponents[row.opponent_index].key for row in schedule)
    if len(schedule) != GAMES or {row.learner_seat for row in schedule} != {0, 1}:
        raise SystemExit("weighted schedule is not complete and seat-balanced")
    grim = [
        int(line) for line in paths["grim_deck"].read_text().splitlines()
        if line.strip()
    ]
    alakazam = policy.load_deck()
    payload = {
        "schema": "ptcg.md-v1.recent-weighted-field-gate-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_engine_outcomes": True,
        "motivation": (
            "historical pool:8 overweights extinct Mega Lucario variants and "
            "omits live Mewtwo/Garchomp; its completed result is not rescored"
        ),
        "protocol": {
            "games_per_arm": GAMES,
            "schedule_seed": SEED,
            "arms": {
                "md-v1-grimmsnarl": (
                    "MD-v1 at ST_MAIN, frozen Qu-v2B at all other prompts"
                ),
                "qu-v2b-alakazam": "frozen Qu-v2B on shipped Alakazam",
            },
            "opponent_pilot": "frozen Qu-v2B for every field deck",
            "identical_weighted_matchup_and_seat_schedule": True,
            "invalid_games_allowed": 0,
            "score": "(wins + 0.5 * official draws) / scheduled games",
            "delta": "MD-v1/Grim score minus Qu-v2B/Alakazam score",
            "delta_ci95": (
                "unpooled independent normal interval over bounded per-game "
                "scores; fixed before outcomes"
            ),
            "decision": {
                "md-v1-grimmsnarl": "delta CI95 lower bound > 0",
                "qu-v2b-alakazam": "delta CI95 upper bound < 0",
                "otherwise": "inconclusive",
            },
            "no_posthoc_reweighting_or_stratum_deletion": True,
        },
        "field_contract": {
            "source_episode_files": field_payload["source"]["episode_files"],
            "source_registered_seats": (
                field_payload["source"]["registered_seats"]),
            "minimum_archetype_share": (
                field_payload["selection"]["minimum_observed_seat_share"]),
            "included_share": field_payload["selection"]["included_share"],
            "representative_rule": (
                field_payload["selection"]["representative_rule"]),
            "weight_rule": field_payload["selection"]["weight_rule"],
            "scheduled_games_by_archetype": dict(sorted(matchup_games.items())),
        },
        "schedule": {
            "episodes": [asdict(row) for row in schedule],
            "sha256": value_sha256([asdict(row) for row in schedule]),
        },
        "learner_decks": {
            "grim_sha256": value_sha256(grim),
            "alakazam_sha256": value_sha256(alakazam),
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["field_contract"], indent=2, sort_keys=True))
    print(f"wrote {OUTPUT}")
    print(payload["lock_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
