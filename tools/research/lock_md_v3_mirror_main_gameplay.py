"""Bind the validation-selected mirror-main candidate and direct-mirror gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import md_v2_card, model, policy, qu_v2_features, safety  # noqa: E402
from tools import eval_ab, rl_env  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import eval_md_v3_mirror_main_gameplay as GAME  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v3-mirror-main-v1"
OUTPUT = RUN / "gameplay-lock.json"


class LockError(RuntimeError):
    pass


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LockError(f"missing artifact: {resolved}")
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": COMMON.file_sha256(resolved)}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockError(f"cannot load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise LockError(f"{path} is not an object")
    return payload


def build() -> dict[str, Any]:
    source_lock = COMMON.load_self_hashed_json(
        RUN / "lock.json",
        schema="ptcg.md-v3.mirror-main-lock.v1",
        hash_key="lock_sha256",
    )
    validation = COMMON.load_self_hashed_json(
        RUN / "validation-result.json",
        schema="ptcg.md-v3.mirror-main-validation-result.v1",
        hash_key="result_sha256",
    )
    provenance = _load_json(
        RUN / "model-65/candidate-qu-v2a-training-manifest.json")
    if (
        validation.get("passed") is not True
        or validation.get("selected_arm") != "65"
        or validation.get("lock", {}).get("lock_sha256")
            != source_lock["lock_sha256"]
        or provenance.get("configuration", {}).get("matchup_weighting", {}).get(
            "base") != 4.051771558000324
        or provenance.get("configuration", {}).get("weights", {}).get("win") != 1.0
        or provenance.get("configuration", {}).get("weights", {}).get("draw") != 1.0
        or provenance.get("configuration", {}).get("weights", {}).get("loss") != 1.0
        or provenance.get("selection", {}).get("best_epoch") != 4
        or provenance.get("test_status") != "deferred"
    ):
        raise LockError("selected candidate identity or training contract drifted")

    candidate_checkpoint = RUN / "model-65/candidate-qu-v2a-checkpoint.pt"
    if (
        COMMON.file_sha256(candidate_checkpoint)
        != validation["candidate_checkpoint_sha256"]["65"]
    ):
        raise LockError("selected checkpoint differs from validation")
    candidate_weights = RUN / "model-65/candidate-qu-v2a-weights.npz"
    frozen_main = (
        ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
        "candidate-qu-v2a-weights.npz"
    )
    card = ROOT / "agent/md_v2_card_weights.npz"
    qu = ROOT / "agent/weights.npz"
    deck_path = ROOT / "decks/md_v1_grimmsnarl.csv"
    deck = COMMON.read_deck(deck_path)
    artifacts = {
        "source_lock": _record(RUN / "lock.json"),
        "validation_result": _record(RUN / "validation-result.json"),
        "validation_amendment": _record(
            RUN / "validation-implementation-amendment.json"),
        "realized_mass_audit": _record(
            RUN / "realized-mass-prevalidation-audit.json"),
        "candidate_main_weights": _record(candidate_weights),
        "candidate_checkpoint": _record(candidate_checkpoint),
        "candidate_training_provenance": _record(
            RUN / "model-65/candidate-qu-v2a-training-manifest.json"),
        "frozen_main_weights": _record(frozen_main),
        "card_weights": _record(card),
        "qu_weights": _record(qu),
        "grim_deck": _record(deck_path),
        "evaluator": _record(Path(GAME.__file__)),
        "lock_builder": _record(Path(__file__)),
        "layered_controller": _record(Path(LAYERED.__file__)),
        "common_gameplay": _record(Path(COMMON.__file__)),
        "eval_ab": _record(Path(eval_ab.__file__)),
        "rl_env": _record(Path(rl_env.__file__)),
        "model": _record(Path(model.__file__)),
        "features": _record(Path(qu_v2_features.__file__)),
        "card_runtime": _record(Path(md_v2_card.__file__)),
        "policy": _record(Path(policy.__file__)),
        "safety": _record(Path(safety.__file__)),
    }
    opponents = GAME.build_control_opponents(
        deck,
        artifacts["frozen_main_weights"]["sha256"],
        artifacts["card_weights"]["sha256"],
        artifacts["qu_weights"]["sha256"],
    )
    schedule = COMMON.build_schedule_contract(
        opponents, games=GAME.GAMES, seed=GAME.SEED)
    seats = [row["learner_seat"] for row in schedule["episodes"]]
    if seats.count(0) != 640 or seats.count(1) != 640:
        raise LockError("schedule is not exactly seat balanced")
    payload = {
        "schema": GAME.LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_engine_outcomes": True,
        "source_preregistration_sha256": source_lock["lock_sha256"],
        "validation_result_sha256": validation["result_sha256"],
        "candidate": {
            "selected_arm": "65",
            "realized_train_st_main_mass": 0.6194802339324573,
            "main_weights_sha256": artifacts[
                "candidate_main_weights"]["sha256"],
            "runtime": (
                "selected ST_MAIN plus frozen MD-v3 ST_CARD plus frozen Qu-v2B"
            ),
        },
        "control": {
            "runtime": "byte-frozen MD-v3 package",
            "main_weights_sha256": artifacts["frozen_main_weights"]["sha256"],
            "card_weights_sha256": artifacts["card_weights"]["sha256"],
            "qu_weights_sha256": artifacts["qu_weights"]["sha256"],
        },
        "protocol": {
            "games": GAME.GAMES,
            "seed": GAME.SEED,
            "deck": "exact target Grimmsnarl list in every seat",
            "seat_balance": "exactly 640 games per candidate seat",
            "one_schedule_one_attempt": True,
            "decision_rule": (
                "valid and candidate Wilson CI95 lower bound strictly above 0.50"
            ),
            "cleanliness": (
                "zero invalids, truncations, errors, controller exceptions, "
                "fallbacks, legality repairs, or off-deck routes"
            ),
        },
        "artifacts": artifacts,
        "schedule": schedule,
        "promotion_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def main() -> int:
    if OUTPUT.exists():
        print(f"error: refusing to overwrite {OUTPUT}", file=sys.stderr)
        return 2
    try:
        payload = build()
        OUTPUT.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (LockError, COMMON.EvaluationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "lock": str(OUTPUT),
        "lock_sha256": payload["lock_sha256"],
        "schedule_sha256": payload["schedule"]["sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
