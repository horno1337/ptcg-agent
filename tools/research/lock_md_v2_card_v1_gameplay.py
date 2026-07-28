"""Bind the passed ST_CARD candidate and its locked 640-game mirror schedule."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import md_v2_card as CARD  # noqa: E402
from agent import model, policy, qu_v2_features, safety  # noqa: E402
from tools import eval_ab, rl_env  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as GAME  # noqa: E402
from tools.research import eval_md_v2_card_v1_validation as VALID  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v2-card-v1"
SOURCE_LOCK = RUN / "lock.json"
VALIDATION_RESULT = RUN / "validation-result.json"
PROVENANCE = RUN / "model/candidate-qu-v2a-training-manifest.json"
CANDIDATE_WEIGHTS = RUN / "model/candidate-qu-v2a-weights.npz"
CANDIDATE_CHECKPOINT = RUN / "model/candidate-qu-v2a-checkpoint.pt"
RUNTIME_CARD_WEIGHTS = ROOT / "agent/md_v2_card_weights.npz"
MD_V2_MAIN_WEIGHTS = (
    ROOT
    / "tools/checkpoints/md-v2-allthrough26/model/"
    "candidate-qu-v2a-weights.npz"
)
QU_V2B_WEIGHTS = ROOT / "agent/weights.npz"
DECK = ROOT / "decks/md_v1_grimmsnarl.csv"
OUTPUT = RUN / "gameplay-lock.json"


class LockError(RuntimeError):
    """The passed candidate cannot be bound to the gameplay contract."""


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LockError(f"missing artifact: {resolved}")
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {
        "path": rendered,
        "sha256": COMMON.file_sha256(resolved),
    }


def _load_validation_result() -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(
            VALIDATION_RESULT,
            schema=VALID.RESULT_SCHEMA,
            hash_key="result_sha256",
        )
    except COMMON.EvaluationError as error:
        raise LockError(str(error)) from error


def build_lock() -> dict[str, Any]:
    try:
        source_lock, source_paths = VALID.load_lock(SOURCE_LOCK)
        provenance = VALID._verify_training_output(
            source_lock,
            source_paths,
            PROVENANCE,
            CANDIDATE_WEIGHTS,
        )
        validation = _load_validation_result()
        deck = COMMON.read_deck(DECK)
    except (
        VALID.EvaluationError,
        COMMON.EvaluationError,
    ) as error:
        raise LockError(str(error)) from error
    if (
        validation.get("lock_sha256") != source_lock["lock_sha256"]
        or validation.get("decision", {}).get("passed") is not True
        or validation.get("promotion_authority") is not False
        or provenance.get("selection", {}).get("best_epoch") != 10
        or COMMON.file_sha256(CANDIDATE_WEIGHTS) != CARD.WEIGHTS_SHA256
        or COMMON.file_sha256(RUNTIME_CARD_WEIGHTS) != CARD.WEIGHTS_SHA256
        or COMMON.file_sha256(MD_V2_MAIN_WEIGHTS)
            != source_lock["artifacts"]["md_v2_main_weights"]["sha256"]
        or COMMON.file_sha256(QU_V2B_WEIGHTS)
            != source_lock["artifacts"]["qu_v2b_weights"]["sha256"]
    ):
        raise LockError("candidate validation/runtime identity drifted")
    main_sha = COMMON.file_sha256(MD_V2_MAIN_WEIGHTS)
    qu_sha = COMMON.file_sha256(QU_V2B_WEIGHTS)
    opponents = GAME.build_baseline_opponents(deck, main_sha, qu_sha)
    schedule = COMMON.build_schedule_contract(
        opponents, games=GAME.GAMES, seed=GAME.SEED
    )
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 320 or seats.count(1) != 320:
        raise LockError("gameplay schedule is not exactly seat balanced")
    artifacts = {
        "source_lock": _record(SOURCE_LOCK),
        "validation_result": _record(VALIDATION_RESULT),
        "candidate_weights": _record(CANDIDATE_WEIGHTS),
        "candidate_checkpoint": _record(CANDIDATE_CHECKPOINT),
        "candidate_training_provenance": _record(PROVENANCE),
        "runtime_card_weights": _record(RUNTIME_CARD_WEIGHTS),
        "md_v2_main_weights": _record(MD_V2_MAIN_WEIGHTS),
        "qu_v2b_weights": _record(QU_V2B_WEIGHTS),
        "grim_deck": _record(DECK),
        "runtime_card": _record(Path(CARD.__file__)),
        "validation_evaluator": _record(Path(VALID.__file__)),
        "evaluator": _record(Path(GAME.__file__)),
        "lock_builder": _record(Path(__file__)),
        "common_gameplay": _record(Path(COMMON.__file__)),
        "eval_ab": _record(Path(eval_ab.__file__)),
        "rl_env": _record(Path(rl_env.__file__)),
        "model": _record(Path(model.__file__)),
        "features": _record(Path(qu_v2_features.__file__)),
        "policy": _record(Path(policy.__file__)),
        "safety": _record(Path(safety.__file__)),
    }
    payload = {
        "schema": GAME.LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_engine_outcomes": True,
        "source_preregistration_sha256": source_lock["lock_sha256"],
        "validation_result_sha256": validation["result_sha256"],
        "candidate": {
            "name": "MD-v2 ST_CARD mirror specialist v1",
            "best_epoch": provenance["selection"]["best_epoch"],
            "best_validation_objective": provenance["selection"][
                "best_validation_objective"
            ],
            "weights_sha256": CARD.WEIGHTS_SHA256,
            "route": (
                "exact own deck plus ST_CARD plus public opposing Grimmsnarl "
                "signature only; MD-v2 ST_MAIN and Qu-v2B fallback unchanged"
            ),
            "public_signature": {
                "zones": ["active", "bench"],
                "card_ids": [646, 647, 648],
            },
        },
        "baseline": {
            "name": "unchanged MD-v2",
            "route": "MD-v2 ST_MAIN plus frozen Qu-v2B elsewhere",
            "main_weights_sha256": main_sha,
            "qu_v2b_weights_sha256": qu_sha,
        },
        "protocol": {
            "games": GAME.GAMES,
            "seed": GAME.SEED,
            "deck": "exact target Grimmsnarl list in every seat",
            "comparison": (
                "candidate layered runtime versus unchanged MD-v2 layered "
                "runtime"
            ),
            "seat_balance": "exactly 320 games per candidate seat",
            "one_schedule_one_attempt": True,
            "decision_rule": {
                "point_estimate": "> 0.50",
                "wilson_ci95_lower": "> 0.45",
                "cleanliness": (
                    "zero invalids, truncations, errors, exceptions, "
                    "fallbacks, repairs, or off-deck routes; candidate card "
                    "route must execute and baseline card route must not"
                ),
                "accept": "all rules pass",
            },
        },
        "artifacts": artifacts,
        "schedule": schedule,
        "promotion_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise LockError(f"refusing to overwrite {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.partial")
    if temporary.exists():
        raise LockError(f"partial output already exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        payload = build_lock()
        _write_new(args.json_out, payload)
    except (LockError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "lock_sha256": payload["lock_sha256"],
        "candidate": payload["candidate"],
        "schedule_sha256": payload["schedule"]["sha256"],
    }, sort_keys=True))
    print(f"wrote {args.json_out.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
