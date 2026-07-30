"""Bind the accepted terminal PPO-v2 candidate to its prelocked direct gate.

The training preregistration already fixes the 2,560-game protocol and exact
schedule.  This post-training binder may only copy those fields byte-for-byte
after the training gate has passed and the fixed update-16 candidate identity
has been verified.  It neither runs games nor grants promotion/upload
authority.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import (  # noqa: E402
    md_v1,
    md_v2_card,
    model,
    obsview,
    policy,
    qu_v2_features,
    safety,
)
from tools import eval_ab, index_corpus, rl_env  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import eval_md_v3_ppo_v2_gameplay as GAME  # noqa: E402
from tools.research import train_md_v3_ppo_v2 as TRAINER  # noqa: E402
from tools.research import train_qu_v2a as BC  # noqa: E402


EXPECTED_UPDATES = 16
EXPECTED_TOTAL_GAMES = 12_288
EXPECTED_MINIMUM_ST_MAIN = 20_000
EXPECTED_MAXIMUM_PARENT_KL = 0.02


class LockError(RuntimeError):
    """The source preregistration or terminal candidate identity is invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise LockError(f"JSON root is not an object: {path}")
    return value


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LockError(f"missing artifact: {resolved}")
    return {
        "path": _display_path(resolved),
        "sha256": COMMON.file_sha256(resolved),
    }


def _resolve_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise LockError(f"invalid artifact record: {label}")
    try:
        path = COMMON.resolve_recorded_path(record.get("path"))
    except COMMON.EvaluationError as error:
        raise LockError(str(error)) from error
    expected = record.get("sha256")
    if (
        not path.is_file()
        or not isinstance(expected, str)
        or COMMON.file_sha256(path) != expected
    ):
        raise LockError(f"artifact drift: {label}")
    return path


def _verify_source_artifacts(
    source: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    raw = source.get("artifacts")
    if not isinstance(raw, Mapping) or not raw:
        raise LockError("training lock has no artifact map")
    paths: dict[str, Path] = {}
    records: dict[str, dict[str, str]] = {}
    for label, record in raw.items():
        if not isinstance(label, str):
            raise LockError("training lock has a non-string artifact label")
        paths[label] = _resolve_record(record, label=label)
        records[label] = {
            "path": _display_path(paths[label]),
            "sha256": COMMON.file_sha256(paths[label]),
        }
    required = {
        "parent_checkpoint",
        "parent_weights",
        "card_weights",
        "qu_weights",
        "ppo_v1_weights",
        "md_v1_weights",
        "deck",
    }
    missing = sorted(required - paths.keys())
    if missing:
        raise LockError("training lock omits artifacts: " + ", ".join(missing))
    return paths, records


def _load_torch_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise LockError(f"cannot load terminal checkpoint {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise LockError("terminal checkpoint is not a mapping")
    return value


def _validate_training_result(
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    source_paths: Mapping[str, Path],
) -> tuple[Path, Path]:
    """Return verified fixed-terminal weights/checkpoint paths."""
    selection = result.get("selection")
    gate = result.get("training_gate")
    if (
        result.get("schema") != GAME.TRAINING_RESULT_SCHEMA
        or result.get("lock_sha256") != source.get("lock_sha256")
        or result.get("candidate_only") is not True
        or not isinstance(selection, Mapping)
        or selection.get("eligible") is not True
        or selection.get("selected_update") != EXPECTED_UPDATES
        or selection.get("fixed_terminal_update") != EXPECTED_UPDATES
        or selection.get("intermediate_recovery_eligible") is not False
        or selection.get("checkpoint_cherry_picking") is not False
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or gate.get("zero_invalid_or_controller_faults") is not True
        or gate.get("minimum_st_main_per_update") != EXPECTED_MINIMUM_ST_MAIN
        or gate.get("frozen_shared_parameters_byte_unchanged") is not True
        or gate.get("maximum_parent_kl") != EXPECTED_MAXIMUM_PARENT_KL
        or gate.get("terminal_selection_only") is not True
        or result.get("total_games") != EXPECTED_TOTAL_GAMES
    ):
        raise LockError("PPO-v2 training result did not pass the fixed gate")
    updates = result.get("updates")
    if (
        not isinstance(updates, list)
        or len(updates) != EXPECTED_UPDATES
        or [row.get("update") for row in updates if isinstance(row, Mapping)]
            != list(range(1, EXPECTED_UPDATES + 1))
    ):
        raise LockError("PPO-v2 result does not contain all fixed updates")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise LockError("PPO-v2 result has no terminal artifacts")
    weights = _resolve_record(artifacts.get("weights"), label="terminal weights")
    checkpoint = _resolve_record(
        artifacts.get("checkpoint"), label="terminal checkpoint",
    )
    if (
        weights.name != "candidate-qu-v2a-weights.npz"
        or checkpoint.name != "candidate-qu-v2a-ppo-v2-checkpoint.pt"
        or weights.parent.name != "terminal-update-16-candidate"
        or checkpoint.parent != weights.parent
    ):
        raise LockError("candidate is not the fixed terminal update-16 output")
    payload = _load_torch_checkpoint(checkpoint)
    provenance = payload.get("provenance")
    state_dict = payload.get("state_dict")
    if (
        payload.get("schema") != TRAINER.CHECKPOINT_SCHEMA
        or payload.get("candidate_only") is not True
        or payload.get("completed_updates") != EXPECTED_UPDATES
        or payload.get("parent_checkpoint_sha256")
            != source["artifacts"]["parent_checkpoint"]["sha256"]
        or not isinstance(provenance, Mapping)
        or provenance.get("lock_sha256") != source["lock_sha256"]
        or provenance.get("candidate_only") is not True
        or provenance.get("recovery_only") is not False
        or provenance.get("selection_eligible") is not True
        or provenance.get("fixed_terminal_selection_update") != EXPECTED_UPDATES
        or provenance.get("selection_rule")
            != "only the fixed update-16 terminal checkpoint"
        or not isinstance(state_dict, Mapping)
        or payload.get("state_dict_sha256") != BC._state_dict_sha256(state_dict)
    ):
        raise LockError("terminal checkpoint metadata/provenance drifted")
    if source_paths["parent_checkpoint"] == checkpoint:
        raise LockError("terminal candidate unexpectedly aliases its parent")
    return weights, checkpoint


def _validate_source_direct(
    source: Mapping[str, Any],
    source_paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    direct = source.get("direct_gameplay")
    if not isinstance(direct, Mapping) or set(direct) != {"protocol", "schedule"}:
        raise LockError("training lock has no exact direct-gameplay contract")
    try:
        GAME.validate_protocol(direct.get("protocol"))
        deck = COMMON.read_deck(source_paths["deck"])
        opponents = GAME.build_control_opponents(
            deck,
            source["artifacts"]["parent_weights"]["sha256"],
            source["artifacts"]["card_weights"]["sha256"],
            source["artifacts"]["qu_weights"]["sha256"],
        )
        schedule = GAME.enforce_schedule_contract(
            direct.get("schedule", {}), opponents,
        )
    except (GAME.GameplayError, COMMON.EvaluationError) as error:
        raise LockError(str(error)) from error
    seats = [row.learner_seat for row in schedule]
    if (
        seats.count(0) != GAME.GAMES // 2
        or seats.count(1) != GAME.GAMES // 2
    ):
        raise LockError("prelocked direct schedule is not exactly seat balanced")
    return direct


def build(
    training_lock_path: Path,
    training_result_path: Path,
) -> dict[str, Any]:
    """Build, but do not write, the content-bound direct gameplay lock."""
    try:
        source = COMMON.load_self_hashed_json(
            training_lock_path.expanduser().resolve(),
            schema=GAME.TRAINING_LOCK_SCHEMA,
            hash_key="lock_sha256",
        )
    except COMMON.EvaluationError as error:
        raise LockError(str(error)) from error
    source_paths, source_records = _verify_source_artifacts(source)
    direct = _validate_source_direct(source, source_paths)
    result_path = training_result_path.expanduser().resolve()
    result = _load_json(result_path)
    candidate_weights, candidate_checkpoint = _validate_training_result(
        result, source, source_paths,
    )

    artifacts: dict[str, dict[str, str]] = {
        # Training provenance and every frozen input/code dependency are copied
        # into this self-hashed lock, not merely referenced through one file.
        "training_lock": _record(training_lock_path),
        "training_result": _record(result_path),
        **{
            f"training_input__{label}": deepcopy(record)
            for label, record in sorted(source_records.items())
        },
        "candidate_main_weights": _record(candidate_weights),
        "candidate_checkpoint": _record(candidate_checkpoint),
        "frozen_main_weights": deepcopy(source_records["parent_weights"]),
        "card_weights": deepcopy(source_records["card_weights"]),
        "qu_weights": deepcopy(source_records["qu_weights"]),
        "grim_deck": deepcopy(source_records["deck"]),
        "evaluator": _record(Path(GAME.__file__)),
        "lock_builder": _record(Path(__file__)),
        "layered_controller": _record(Path(LAYERED.__file__)),
        "common_gameplay": _record(Path(COMMON.__file__)),
        "eval_ab": _record(Path(eval_ab.__file__)),
        "rl_env": _record(Path(rl_env.__file__)),
        "index_corpus": _record(Path(index_corpus.__file__)),
        "model": _record(Path(model.__file__)),
        "features": _record(Path(qu_v2_features.__file__)),
        "card_runtime": _record(Path(md_v2_card.__file__)),
        "md_v1_runtime": _record(Path(md_v1.__file__)),
        "obsview": _record(Path(obsview.__file__)),
        "policy": _record(Path(policy.__file__)),
        "safety": _record(Path(safety.__file__)),
    }
    payload = {
        "schema": GAME.LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_direct_engine_outcomes": True,
        "source_training_lock_sha256": source["lock_sha256"],
        "source_training_result_sha256": artifacts[
            "training_result"]["sha256"],
        "source_direct_gameplay_sha256": COMMON.canonical_sha256(direct),
        "candidate": {
            "selected_update": EXPECTED_UPDATES,
            "fixed_terminal_update": EXPECTED_UPDATES,
            "main_weights_sha256": artifacts[
                "candidate_main_weights"]["sha256"],
            "checkpoint_sha256": artifacts["candidate_checkpoint"]["sha256"],
            "runtime": (
                "fixed terminal update-16 PPO-v2 ST_MAIN plus frozen MD-v3 "
                "ST_CARD plus frozen Qu-v2B"
            ),
        },
        "control": {
            "runtime": "complete byte-frozen MD-v3 package",
            "main_weights_sha256": artifacts["frozen_main_weights"]["sha256"],
            "card_weights_sha256": artifacts["card_weights"]["sha256"],
            "qu_weights_sha256": artifacts["qu_weights"]["sha256"],
        },
        # These are deliberately copied byte-for-byte from the prospective
        # training lock.  The evaluator reopens the source and verifies this.
        "protocol": deepcopy(direct["protocol"]),
        "schedule": deepcopy(direct["schedule"]),
        "artifacts": artifacts,
        "interpretation": {
            "one_attempt": True,
            "post_failure": "retire PPO-v2; do not run a field gate or upload",
            "post_pass": (
                "a separately locked recent-frequency field gate is permitted"
            ),
            "ladder_or_upload_authority": False,
        },
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-lock", required=True, type=Path)
    parser.add_argument("--training-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise LockError(f"refusing to overwrite {output}")
        payload = build(args.training_lock, args.training_result)
        GAME._write_new(output, payload)
    except (
        LockError,
        GAME.GameplayError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps({
        "lock": str(output),
        "lock_sha256": payload["lock_sha256"],
        "schedule_sha256": payload["schedule"]["manifest_sha256"],
        "promotion_authority": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
