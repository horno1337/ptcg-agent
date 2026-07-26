"""Pre-register the MD-v2 rollout cohort before any counterfactual outcome
is observed.

Selects an over-provisioned, diversity-priority, deduplicated set of MD-v1
vs. frozen-Qu-v2B disagreement roots from a discovery report, then binds the
exact ordered root list, discovery report hash, model weight hashes, and the
full decision rule (rollout count, sign-agreement floor, minimum confirmed
roots/games/teams) into one self-hashed lock. No terminal outcome, rollout
result, or advantage value is read or referenced anywhere in this module --
selection uses only discovery-stage action identities (which of MD/Qu/logged
differ), never which one is better.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import md_v1 as MD  # noqa: E402


SCHEMA = "ptcg.md-v2.rollout-cohort-lock.v1"

# Over-provisioning: sized against the 200-confirmed-root floor, hedging
# both mechanical (engine/hop-cap) attrition and the 0.85 sign-agreement
# filter, given discovery supply of ~18.6k disagreement roots across 60
# teams comfortably supports it.
CANDIDATE_ROOT_COUNT = 500

ROLLOUTS_PER_COMPARISON = 32
MIN_SIGN_AGREEMENT = 0.85
MIN_CONFIRMED_ROOTS = 200
MIN_CONFIRMED_GAMES = 30
MIN_CONFIRMED_TEAMS = 5


class CohortLockError(RuntimeError):
    """The rollout cohort cannot be locked from this discovery report."""


def _value_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_signature(prompt: Mapping[str, Any]) -> str:
    payload = json.dumps(
        prompt["semantic_options"], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _turn_bucket(turn: int) -> str:
    if turn <= 3:
        return "early_1_3"
    if turn <= 6:
        return "middle_4_6"
    return "late_7_plus"


def _category(prompt: Mapping[str, Any]) -> str:
    md_vs_logged = prompt["secondary_md_vs_logged_disagreement"]
    qu_vs_logged = prompt["secondary_qu_vs_logged_disagreement"]
    if not md_vs_logged and qu_vs_logged:
        return "md_matches_logged_qu_differs"
    if md_vs_logged and not qu_vs_logged:
        return "qu_matches_logged_md_differs"
    if md_vs_logged and qu_vs_logged:
        return "three_way_distinct"
    return "inconsistent"


@dataclass(frozen=True)
class Candidate:
    root_id: str
    game_key: str
    team_name: str
    turn: int
    turn_bucket: str
    option_count: int
    category: str
    representative_prompt: Mapping[str, Any]


def _candidates(report: Mapping[str, Any]) -> list[Candidate]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for prompt in report["prompts"]:
        if not prompt["primary_md_vs_qu_disagreement"]:
            continue
        grouped.setdefault(_root_signature(prompt), []).append(prompt)
    candidates = []
    for signature, rows in grouped.items():
        representative = rows[0]
        turn = int(representative["turn"] or 0)
        candidates.append(Candidate(
            root_id=signature,
            game_key=str(representative["episode_id"]),
            team_name=str(representative["team_name"] or "?"),
            turn=turn,
            turn_bucket=_turn_bucket(turn),
            option_count=len(representative["semantic_options"]),
            category=_category(representative),
            representative_prompt=representative,
        ))
    if len({c.root_id for c in candidates}) != len(candidates):
        raise CohortLockError("duplicate root ids after deduplication")
    return candidates


def _diversity_score(
    candidate: Candidate,
    teams: Counter[str],
    games: Counter[str],
    turn_buckets: Counter[str],
    categories: Counter[str],
) -> tuple[int, ...]:
    return (
        int(not teams[candidate.team_name]),
        int(not games[candidate.game_key]),
        int(not turn_buckets[candidate.turn_bucket]),
        int(not categories[candidate.category]),
        -teams[candidate.team_name],
        -games[candidate.game_key],
        candidate.root_id,
    )


def select_cohort(
    report: Mapping[str, Any],
    *,
    candidate_root_count: int = CANDIDATE_ROOT_COUNT,
) -> tuple[list[Candidate], dict[str, Any]]:
    eligible = _candidates(report)
    if len(eligible) < candidate_root_count:
        raise CohortLockError(
            f"only {len(eligible)} disagreement roots available; "
            f"{candidate_root_count} required for the over-provisioned pool")
    teams: Counter[str] = Counter()
    games: Counter[str] = Counter()
    turn_buckets: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    remaining = list(eligible)
    selected: list[Candidate] = []
    while len(selected) < candidate_root_count:
        chosen = max(
            remaining,
            key=lambda c: _diversity_score(
                c, teams, games, turn_buckets, categories),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        teams[chosen.team_name] += 1
        games[chosen.game_key] += 1
        turn_buckets[chosen.turn_bucket] += 1
        categories[chosen.category] += 1
    diagnostics = {
        "eligible_disagreement_roots": len(eligible),
        "selected_teams": len(teams),
        "selected_games": len(games),
        "selected_turn_buckets": dict(sorted(turn_buckets.items())),
        "selected_categories": dict(sorted(categories.items())),
    }
    return selected, diagnostics


def build_lock(
    report_path: Path,
    *,
    candidate_root_count: int = CANDIDATE_ROOT_COUNT,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected, diagnostics = select_cohort(
        report, candidate_root_count=candidate_root_count)
    if diagnostics["selected_teams"] < MIN_CONFIRMED_TEAMS:
        raise CohortLockError(
            "candidate pool does not even span the minimum team count")
    ordered_roots = [
        {
            "root_id": c.root_id,
            "game_key": c.game_key,
            "team_name": c.team_name,
            "turn": c.turn,
            "turn_bucket": c.turn_bucket,
            "option_count": c.option_count,
            "category": c.category,
            "replay_path": c.representative_prompt["replay_path"],
            "replay_sha256": c.representative_prompt["replay_sha256"],
            "acting_seat": c.representative_prompt["acting_seat"],
            "source_step": c.representative_prompt["source_step"],
            "answer_step": c.representative_prompt["answer_step"],
            "logged_indices": c.representative_prompt["logged_indices"],
            "qu_v2b_indices": c.representative_prompt["qu_v2b_indices"],
            "md_v1_indices": c.representative_prompt["md_v1_indices"],
        }
        for c in selected
    ]
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_rollout_generation": True,
        "outcome_data_used_in_selection": False,
        "research_only": True,
        "actor_training_authorized": False,
        "source_discovery_report": {
            "path": str(report_path),
            "file_sha256": _file_sha256(report_path),
            "report_sha256": _value_sha256(report),
        },
        "weights": report["weights"],
        "target_deck_sha256": MD.TARGET_DECK_SHA256,
        "candidate_pool": {
            "root_count": len(ordered_roots),
            "ordered_roots": ordered_roots,
            "ordered_root_ids_sha256": _value_sha256(
                [row["root_id"] for row in ordered_roots]),
        },
        "selection_diagnostics": diagnostics,
        "panel_protocol": {
            "rollouts_per_comparison": ROLLOUTS_PER_COMPARISON,
            "comparisons": [
                "md_vs_qu",
                "md_vs_logged_when_distinct",
                "qu_vs_logged_when_distinct_optional",
            ],
            "confirmation": "independent second panel, sign agreement only",
        },
        "decision_rule": {
            "minimum_sign_agreement": MIN_SIGN_AGREEMENT,
            "minimum_confirmed_roots": MIN_CONFIRMED_ROOTS,
            "minimum_confirmed_games": MIN_CONFIRMED_GAMES,
            "minimum_confirmed_teams": MIN_CONFIRMED_TEAMS,
            "lower_floor_after_inspecting_results": False,
        },
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["lock_sha256"] = _value_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CohortLockError(f"cohort lock output already exists: {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return payload


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    recorded = value.pop("lock_sha256", None)
    if value.get("schema") != SCHEMA or recorded != _value_sha256(value):
        raise CohortLockError("rollout cohort lock mismatch")
    value["lock_sha256"] = recorded
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-report", type=Path, required=True)
    parser.add_argument(
        "--candidate-root-count", type=int, default=CANDIDATE_ROOT_COUNT)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = atomic_write(
            args.json_out,
            build_lock(
                args.discovery_report,
                candidate_root_count=args.candidate_root_count),
        )
    except (CohortLockError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload["selection_diagnostics"], indent=2))
    print(f"Locked {payload['candidate_pool']['root_count']} candidate roots")
    print(f"Lock: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
