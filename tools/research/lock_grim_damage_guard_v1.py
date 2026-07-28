"""Pre-register the Grimmsnarl damage-routing guard experiment.

This phase lock is deliberately created before extracting any Adrena-Brain or
Shadow Bullet actions from the reserved evaluation dates.  July 17--24 may be
used to understand the prompt schema and design one fixed rule.  July 25--26
stay sealed until a second rule lock binds the implementation and exact
offline decision thresholds.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "tools/checkpoints/md-v2-allthrough26/corpus.json"
DEFAULT_OUTPUT = ROOT / "tools/checkpoints/grim-damage-guard-v1/phase-lock.json"

TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
DISCOVERY_DATES = tuple(f"2026-07-{day:02d}" for day in range(17, 25))
EVALUATION_DATES = ("2026-07-25", "2026-07-26")
MUNKIDORI = 112
GRIMMSNARL_EX = 648
SHADOW_BULLET = 937


class DamageGuardLockError(ValueError):
    """The source corpus cannot satisfy the pre-registered protocol."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_exact_mirror(game: Mapping[str, Any]) -> bool:
    seats = game.get("seats")
    return (
        isinstance(seats, list)
        and len(seats) == 2
        and all(
            isinstance(seat, Mapping)
            and seat.get("registered_deck_sha256") == TARGET_DECK_SHA256
            for seat in seats
        )
    )


def build_lock(
    corpus: Mapping[str, Any],
    *,
    corpus_path: Path,
    corpus_file_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    if corpus.get("clean") is not True:
        raise DamageGuardLockError("source corpus is not clean")
    manifest_sha = corpus.get("manifest_sha256")
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise DamageGuardLockError("source manifest hash is missing")
    games = corpus.get("games")
    if not isinstance(games, list):
        raise DamageGuardLockError("source corpus games are missing")

    expected_dates = set(DISCOVERY_DATES + EVALUATION_DATES)
    date_counts: Counter[str] = Counter()
    uid_sets = {"discovery": set(), "evaluation": set()}
    content_sets = {"discovery": set(), "evaluation": set()}
    for game in games:
        if not isinstance(game, Mapping) or not _is_exact_mirror(game):
            continue
        date = game.get("md_v2_date")
        if date not in expected_dates:
            continue
        cohort = "discovery" if date in DISCOVERY_DATES else "evaluation"
        uid = game.get("game_uid")
        content = game.get("content_sha256")
        if not isinstance(uid, str) or not isinstance(content, str):
            raise DamageGuardLockError("mirror game is missing an immutable id")
        date_counts[str(date)] += 1
        uid_sets[cohort].add(uid)
        content_sets[cohort].add(content)

    if uid_sets["discovery"] & uid_sets["evaluation"]:
        raise DamageGuardLockError("game uid leakage across guard cohorts")
    if content_sets["discovery"] & content_sets["evaluation"]:
        raise DamageGuardLockError("content leakage across guard cohorts")

    cohort_counts = {
        "discovery": sum(date_counts[date] for date in DISCOVERY_DATES),
        "evaluation": sum(date_counts[date] for date in EVALUATION_DATES),
    }
    if any(count <= 0 for count in cohort_counts.values()):
        raise DamageGuardLockError("one or more guard cohorts are empty")
    payload: dict[str, Any] = {
        "schema": "ptcg.grim-damage-guard.phase-lock.v1",
        "created_at": created_at,
        "locked_before_reserved_action_extraction": True,
        "scope": {
            "registered_deck_sha256": TARGET_DECK_SHA256,
            "matchup": "exact-deck Grimmsnarl mirror only",
            "guard_family": (
                "Adrena-Brain source/count/target plus Shadow Bullet bench target"
            ),
            "card_ids": {
                "munkidori": MUNKIDORI,
                "marnies_grimmsnarl_ex": GRIMMSNARL_EX,
            },
            "attack_ids": {"shadow_bullet": SHADOW_BULLET},
            "public_information_only": True,
            "ambiguous_or_mismatched_prompt": "return None to frozen policy",
        },
        "cohorts": {
            "discovery": {
                "dates": list(DISCOVERY_DATES),
                "games": cohort_counts["discovery"],
                "per_date_games": {
                    date: date_counts[date] for date in DISCOVERY_DATES
                },
                "permitted_use": (
                    "prompt-schema discovery, trigger counts, rule design, and "
                    "expert-action summaries"
                ),
            },
            "evaluation": {
                "dates": list(EVALUATION_DATES),
                "games": cohort_counts["evaluation"],
                "per_date_games": {
                    date: date_counts[date] for date in EVALUATION_DATES
                },
                "sealed_until": (
                    "a rule lock binds the implementation hash, trigger "
                    "eligibility, base comparator, and all pass/fail thresholds"
                ),
                "no_rule_revision_after_opening": True,
            },
        },
        "prospective_gameplay_gate": {
            "matchup": "exact-deck Grimmsnarl mirror",
            "candidate": "fixed accepted MD-v2 plus damage guard",
            "control": "the same fixed accepted MD-v2 without the guard",
            "games": 640,
            "seat_balanced": True,
            "identical_schedule": True,
            "score": "(wins + 0.5 * official draws) / scheduled games",
            "pass": {
                "point_estimate": "> 0.50",
                "wilson_lower_bound": "> 0.45",
                "invalid_actions": 0,
                "exceptions": 0,
                "legality_repairs": 0,
            },
            "one_change": "damage-routing guard only",
        },
        "source": {
            "corpus_path": str(corpus_path),
            "corpus_file_sha256": corpus_file_sha256,
            "corpus_manifest_sha256": manifest_sha,
            "corpus_content_sha256": corpus.get("corpus_content_sha256"),
        },
        "methodology": {
            "reserved_dates_are_not_discovery_data": True,
            "no_posthoc_date_or_prompt_subtype_deletion": True,
            "offline_action_agreement_is_diagnostic_not_promotion_evidence": True,
            "local_gameplay_proposes_and_ladder_disposes": True,
            "base_md_v2_is_packaged_and_tested_before_guard_gameplay": True,
        },
    }
    payload["lock_sha256"] = value_sha256(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite phase lock: {args.output}")
    raw = args.corpus.read_bytes()
    corpus = json.loads(raw)
    payload = build_lock(
        corpus,
        corpus_path=args.corpus.resolve(),
        corpus_file_sha256=hashlib.sha256(raw).hexdigest(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["cohorts"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(payload["lock_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
