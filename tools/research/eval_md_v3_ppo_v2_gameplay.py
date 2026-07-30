"""Run the locked 2,560-game PPO-v2 versus frozen MD-v3 mirror gate.

This evaluator has no promotion authority.  It executes exactly one
prospectively locked, exactly seat-balanced Grimmsnarl mirror schedule.  A
passing result permits a separately preregistered recent-field gate; it does
not authorize packaging or upload.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import (  # noqa: E402
    EpisodeSpec,
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


LOCK_SCHEMA = "ptcg.md-v3.st-main-ppo-v2-gameplay-lock.v1"
RESULT_SCHEMA = "ptcg.md-v3.st-main-ppo-v2-gameplay-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v3.st-main-ppo-v2-gameplay-attempt.v1"
TRAINING_LOCK_SCHEMA = "ptcg.md-v3.st-main-ppo-v2-training-lock.v1"
TRAINING_RESULT_SCHEMA = "ptcg.md-v3.st-main-ppo-v2-training-result.v1"
GAMES = 2_560
SEED = 2_026_073_102
DECISION_RULE = (
    "valid and candidate Wilson CI95 lower bound strictly above 0.50"
)
CLEANLINESS_RULE = (
    "zero invalids, truncations, errors, controller exceptions, fallbacks, "
    "legality repairs, or off-deck routes"
)
SCORE_METRIC = "(wins + 0.5 * draws) / 2560"


class GameplayError(RuntimeError):
    """A bound input, protocol requirement, or one-attempt rule drifted."""


def expected_protocol() -> dict[str, Any]:
    """Return the immutable direct-gate protocol embedded in the training lock."""
    return {
        "games": GAMES,
        "seed": SEED,
        "matchup": "complete frozen MD-v3 exact Grimmsnarl mirror",
        "candidate_seat_balance": {"0": 1280, "1": 1280},
        "fixed_terminal_update": 16,
        "one_schedule_one_attempt": True,
        "score": SCORE_METRIC,
        "confidence_interval": "two-sided Wilson score interval at 95%",
        "pass": "Wilson CI95 lower bound strictly greater than 0.50",
        "zero_faults": True,
        "field_gate_only_after_pass": True,
    }


def validate_protocol(protocol: Any) -> Mapping[str, Any]:
    """Require the exact prospectively declared protocol, without additions."""
    if not isinstance(protocol, Mapping) or dict(protocol) != expected_protocol():
        raise GameplayError("direct gameplay protocol drifted")
    return protocol


def policy_id(main_sha: str, card_sha: str, qu_sha: str) -> str:
    del main_sha, card_sha, qu_sha
    # The prospective schedule binds this stable policy label.  The gameplay
    # lock separately content-binds all three frozen weight files.
    return "complete-frozen-md-v3"


def _noop(obs: dict, rng: Any) -> list[int]:
    del obs, rng
    return [0]


def build_control_opponents(
    deck: Sequence[int],
    main_sha: str,
    card_sha: str,
    qu_sha: str,
    controller: LAYERED.LayeredMirrorCardController | None = None,
) -> list[OpponentSpec]:
    """Build the sole complete-frozen-MD-v3 mirror opponent."""
    return [OpponentSpec(
        key="grimmsnarl/frozen-md-v3",
        deck=tuple(int(card) for card in deck),
        move=controller.opponent_move if controller is not None else _noop,
        policy_id=policy_id(main_sha, card_sha, qu_sha),
        schedule_group="frozen-md-v3",
    )]


def build_schedule_contract(
    opponents: Sequence[OpponentSpec],
) -> tuple[list[EpisodeSpec], dict[str, Any]]:
    """Build the deterministic schedule and its compact preregistration row."""
    if len(opponents) != 1:
        raise GameplayError("direct gate requires exactly one control opponent")
    schedule = build_paired_schedule(opponents, GAMES, seed=SEED)
    manifest = schedule_manifest(schedule, opponents)
    seats = {
        str(seat): sum(row.learner_seat == seat for row in schedule)
        for seat in (0, 1)
    }
    opponent = opponents[0]
    contract = {
        "generator": "tools.rl_env.build_paired_schedule",
        "manifest": "tools.rl_env.schedule_manifest",
        "manifest_sha256": COMMON.canonical_sha256(manifest),
        "games": GAMES,
        "pairs": GAMES // 2,
        "candidate_seat_counts": seats,
        "opponent": {
            "key": opponent.key,
            "policy_id": opponent.policy_id,
            "schedule_group": opponent.schedule_group,
            "deck_sha256": opponent.deck_sha256,
        },
    }
    return schedule, contract


def enforce_schedule_contract(
    contract: Any,
    opponents: Sequence[OpponentSpec],
) -> list[EpisodeSpec]:
    """Rebuild and verify the compact exact schedule sealed before training."""
    if not isinstance(contract, Mapping):
        raise GameplayError("locked direct schedule is missing")
    schedule, expected = build_schedule_contract(opponents)
    if dict(contract) != expected:
        raise GameplayError("runtime schedule differs from the prelocked schedule")
    return schedule


def _paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    records = lock.get("artifacts")
    if not isinstance(records, Mapping) or not records:
        raise GameplayError("gameplay lock has no artifact map")
    result: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(label, str) or not isinstance(record, Mapping):
            raise GameplayError("invalid gameplay artifact row")
        path = COMMON.resolve_recorded_path(record.get("path"))
        expected = record.get("sha256")
        if (
            not path.is_file()
            or not isinstance(expected, str)
            or COMMON.file_sha256(path) != expected
        ):
            raise GameplayError(f"gameplay artifact drift: {label}")
        result[label] = path
    required = {
        "training_lock",
        "training_result",
        "candidate_main_weights",
        "candidate_checkpoint",
        "frozen_main_weights",
        "card_weights",
        "qu_weights",
        "grim_deck",
        "evaluator",
        "lock_builder",
        "layered_controller",
        "common_gameplay",
        "eval_ab",
        "rl_env",
        "model",
        "features",
        "card_runtime",
        "policy",
        "safety",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise GameplayError("missing gameplay artifacts: " + ", ".join(missing))
    if result["evaluator"] != Path(__file__).resolve():
        raise GameplayError("gameplay lock names another evaluator")
    return result


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify the self-hashed candidate-binding lock and its exact schedule."""
    lock = COMMON.load_self_hashed_json(
        path.expanduser().resolve(),
        schema=LOCK_SCHEMA,
        hash_key="lock_sha256",
    )
    validate_protocol(lock.get("protocol"))
    paths = _paths(lock)
    source = COMMON.load_self_hashed_json(
        paths["training_lock"],
        schema=TRAINING_LOCK_SCHEMA,
        hash_key="lock_sha256",
    )
    source_direct = source.get("direct_gameplay")
    source_artifacts = source.get("artifacts")
    if (
        not isinstance(source_direct, Mapping)
        or not isinstance(source_artifacts, Mapping)
        or lock.get("source_training_lock_sha256") != source["lock_sha256"]
        or lock.get("source_training_result_sha256")
            != lock["artifacts"]["training_result"]["sha256"]
        or lock.get("source_direct_gameplay_sha256")
            != COMMON.canonical_sha256(source_direct)
        or lock.get("protocol") != source_direct.get("protocol")
        or lock.get("schedule") != source_direct.get("schedule")
    ):
        raise GameplayError(
            "gameplay protocol/schedule is not the prelocked source contract"
        )
    for label, source_record in source_artifacts.items():
        copied = lock["artifacts"].get(f"training_input__{label}")
        if (
            not isinstance(source_record, Mapping)
            or not isinstance(copied, Mapping)
            or copied.get("sha256") != source_record.get("sha256")
            or COMMON.resolve_recorded_path(copied.get("path"))
                != COMMON.resolve_recorded_path(source_record.get("path"))
        ):
            raise GameplayError(f"source training artifact was not copied: {label}")
    aliases = {
        "frozen_main_weights": "parent_weights",
        "card_weights": "card_weights",
        "qu_weights": "qu_weights",
        "grim_deck": "deck",
    }
    if any(
        lock["artifacts"][alias].get("sha256")
            != source_artifacts[source_label].get("sha256")
        for alias, source_label in aliases.items()
    ):
        raise GameplayError("frozen runtime aliases differ from training inputs")
    try:
        training_result = json.loads(
            paths["training_result"].read_text(encoding="utf-8"),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise GameplayError(f"cannot read bound training result: {error}") from error
    selection = (
        training_result.get("selection")
        if isinstance(training_result, Mapping)
        else None
    )
    gate = (
        training_result.get("training_gate")
        if isinstance(training_result, Mapping)
        else None
    )
    result_artifacts = (
        training_result.get("artifacts")
        if isinstance(training_result, Mapping)
        else None
    )
    result_weights = (
        result_artifacts.get("weights")
        if isinstance(result_artifacts, Mapping)
        else None
    )
    result_checkpoint = (
        result_artifacts.get("checkpoint")
        if isinstance(result_artifacts, Mapping)
        else None
    )
    if (
        not isinstance(training_result, Mapping)
        or training_result.get("schema") != TRAINING_RESULT_SCHEMA
        or training_result.get("lock_sha256") != source["lock_sha256"]
        or not isinstance(selection, Mapping)
        or selection.get("eligible") is not True
        or selection.get("selected_update") != 16
        or selection.get("fixed_terminal_update") != 16
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or gate.get("terminal_selection_only") is not True
        or not isinstance(result_artifacts, Mapping)
        or not isinstance(result_weights, Mapping)
        or not isinstance(result_checkpoint, Mapping)
        or result_weights.get("sha256")
            != lock["artifacts"]["candidate_main_weights"]["sha256"]
        or result_checkpoint.get("sha256")
            != lock["artifacts"]["candidate_checkpoint"]["sha256"]
    ):
        raise GameplayError("bound terminal training result identity drifted")
    candidate = lock.get("candidate")
    control = lock.get("control")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("selected_update") != 16
        or candidate.get("fixed_terminal_update") != 16
        or candidate.get("main_weights_sha256")
            != lock["artifacts"]["candidate_main_weights"]["sha256"]
        or candidate.get("checkpoint_sha256")
            != lock["artifacts"]["candidate_checkpoint"]["sha256"]
        or not isinstance(control, Mapping)
        or control.get("main_weights_sha256")
            != lock["artifacts"]["frozen_main_weights"]["sha256"]
        or control.get("card_weights_sha256")
            != lock["artifacts"]["card_weights"]["sha256"]
        or control.get("qu_weights_sha256")
            != lock["artifacts"]["qu_weights"]["sha256"]
        or lock.get("promotion_authority") is not False
        or lock.get("upload_authority") is not False
    ):
        raise GameplayError("candidate/control binding or authority drifted")
    deck = COMMON.read_deck(paths["grim_deck"])
    opponents = build_control_opponents(
        deck,
        lock["artifacts"]["frozen_main_weights"]["sha256"],
        lock["artifacts"]["card_weights"]["sha256"],
        lock["artifacts"]["qu_weights"]["sha256"],
    )
    schedule = enforce_schedule_contract(lock["schedule"], opponents)
    seats = [row.learner_seat for row in schedule]
    if seats.count(0) != GAMES // 2 or seats.count(1) != GAMES // 2:
        raise GameplayError("locked direct schedule is not exactly seat balanced")
    return lock, paths


def diagnostics_clean(controller: Mapping[str, Any]) -> bool:
    """Require a fully exercised layered route with no fail-soft behavior."""
    return (
        controller.get("fallbacks") == 0
        and controller.get("repairs") == 0
        and controller.get("exceptions") == {}
        and controller.get("off_deck_main_routes") == 0
        and controller.get("off_deck_card_routes") == 0
        and controller.get("calls") == (
            controller.get("main_routes", 0)
            + controller.get("card_routes", 0)
            + controller.get("qu_routes", 0)
        )
        and controller.get("main_routes", 0) > 0
        and controller.get("card_routes", 0) > 0
        and controller.get("qu_routes", 0) > 0
    )


def series_clean(series: EVAL.SeriesResult) -> bool:
    """Apply the stricter zero-invalid/error direct-gate contract."""
    return (
        len(series.records) == GAMES
        and series.gate_valid
        and all(
            row.result in ("win", "draw", "loss")
            and row.terminated
            and not row.truncated
            and row.reason == "engine_terminal"
            and row.agent_error is None
            and row.engine_error in (None, [], {})
            and row.infrastructure_error is None
            for row in series.records
        )
    )


def decision(
    series: EVAL.SeriesResult,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    low, high = series.ci95
    valid = (
        series_clean(series)
        and diagnostics_clean(candidate)
        and diagnostics_clean(control)
    )
    return {
        "valid": valid,
        "passed": valid and low > 0.50,
        "score": series.score,
        "wilson_ci95": [low, high],
        "rule": DECISION_RULE,
    }


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably create a JSON file without an overwrite race."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise GameplayError(f"refusing to overwrite {resolved}")
    raw = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".partial", dir=resolved.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, resolved)
    except FileExistsError as error:
        raise GameplayError(f"refusing to overwrite {resolved}") from error
    finally:
        temporary.unlink(missing_ok=True)


def attempt_marker_path(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Path:
    """Use one stable marker per candidate-binding lock, independent of output."""
    lock_sha = lock.get("lock_sha256")
    if not isinstance(lock_sha, str) or len(lock_sha) != 64:
        raise GameplayError("gameplay lock has no valid identity")
    training_result = paths.get("training_result")
    if training_result is None:
        raise GameplayError("gameplay lock has no bound training-result path")
    return training_result.parent / f"ppo-v2-direct-{lock_sha}.attempt.json"


def run(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output: Path,
    *,
    quiet: bool,
) -> dict[str, Any]:
    """Consume the single allowed direct-gate attempt."""
    deck = COMMON.read_deck(paths["grim_deck"])
    candidate_main = COMMON._load_net(
        paths["candidate_main_weights"], "terminal update-16 PPO-v2 main",
    )
    frozen_main = COMMON._load_net(
        paths["frozen_main_weights"], "complete frozen MD-v3 main",
    )
    card = COMMON._load_net(paths["card_weights"], "frozen MD-v3 card")
    qu = COMMON._load_net(paths["qu_weights"], "frozen Qu-v2B")
    candidate = LAYERED.LayeredMirrorCardController(
        candidate_main, card, qu, "ppo-v2-terminal+frozen-card+qu", deck,
    )
    control = LAYERED.LayeredMirrorCardController(
        frozen_main, card, qu, "complete-frozen-md-v3", deck,
    )
    opponents = build_control_opponents(
        deck,
        lock["artifacts"]["frozen_main_weights"]["sha256"],
        lock["artifacts"]["card_weights"]["sha256"],
        lock["artifacts"]["qu_weights"]["sha256"],
        control,
    )
    schedule = enforce_schedule_contract(lock["schedule"], opponents)
    attempt = attempt_marker_path(lock, paths)
    if output.exists() or attempt.exists():
        raise GameplayError("refusing repeated gameplay outcome attempt")
    _write_new(attempt, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "gameplay_lock_sha256": lock["lock_sha256"],
        "schedule_sha256": lock["schedule"]["manifest_sha256"],
        "games": GAMES,
        "seed": SEED,
    })
    series = EVAL.run_series(
        "ppo-v2-terminal-vs-complete-frozen-md-v3",
        candidate,
        deck,
        opponents,
        schedule,
        max_selects=5000,
        time_bank_s=600.0,
        verbose=not quiet,
    )
    EVAL.print_result(series)
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    verdict = decision(series, candidate_diag, control_diag)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gameplay_lock_sha256": lock["lock_sha256"],
        "schedule_sha256": lock["schedule"]["manifest_sha256"],
        "metric": SCORE_METRIC,
        "cleanliness_rule": CLEANLINESS_RULE,
        "decision": verdict,
        "result": {
            "summary": series.summary(),
            "records": [asdict(row) for row in series.records],
        },
        "controllers": {
            "candidate": candidate_diag,
            "control": control_diag,
        },
        "environment": environment_manifest(
            deck, opponents, str(paths["grim_deck"]),
        ),
        "field_gate_permitted": bool(verdict["passed"]),
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _write_new(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        lock, paths = load_lock(args.lock)
        payload = run(lock, paths, args.json_out, quiet=args.quiet)
    except (GameplayError, COMMON.EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], sort_keys=True), flush=True)
    return 0 if payload["decision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
