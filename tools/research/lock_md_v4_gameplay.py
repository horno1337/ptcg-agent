"""Bind the accepted NumPy-deployable MD-v4 candidate to gameplay gates.

The original training lock remains the sole authority for frozen assets, deck,
and all 1,280 direct pair seeds.  The newer candidate lock/result/bundle chain
is authoritative only for the exact deployable NumPy candidate and its
unchanged offline rejection screens.

This file never runs a game and grants no promotion or upload authority.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import (  # noqa: E402
    cards,
    features as agent_features,
    md_v2_card,
    model,
    obsview,
    policy,
    qu_v2_features,
    safety,
)
from tools import eval_ab, index_corpus, rl_env  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import eval_md_v4_gameplay as GAME  # noqa: E402
from tools.research import (  # noqa: E402
    eval_md_v4_numpy_deployable as CANDIDATE_EVAL,
)
from tools.research import (  # noqa: E402
    lock_md_v4_numpy_deployable as CANDIDATE_LOCK,
)
from tools.research import lock_md_v4_training as SOURCE_LOCK  # noqa: E402
from tools.research import md_v4_runtime as RUNTIME  # noqa: E402
from tools.research import train_md_v4 as TRAIN  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v4-public-window-v1"
DEFAULT_TRAINING_LOCK = RUN / "training-evaluation-lock.json"
DEFAULT_CANDIDATE_LOCK = CANDIDATE_LOCK.OUTPUT
DEFAULT_CANDIDATE_RESULT = CANDIDATE_LOCK.RESULT
DEFAULT_CANDIDATE_BUNDLE = CANDIDATE_LOCK.CANDIDATE_BUNDLE
DEFAULT_CANDIDATE_MANIFEST = (
    DEFAULT_CANDIDATE_BUNDLE / CANDIDATE_EVAL.MANIFEST_NAME
)
DEFAULT_OUTPUT = RUN / "numpy-deployable-direct-gameplay-lock.json"


class LockError(RuntimeError):
    """The accepted candidate or direct-gate binding is invalid."""


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise LockError(f"missing or symlinked artifact: {resolved}")
    return {
        "path": _display_path(resolved),
        "sha256": COMMON.file_sha256(resolved),
    }


def _resolve_source_artifacts(
    source: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    raw = source.get("artifacts")
    if not isinstance(raw, Mapping) or not raw:
        raise LockError("source training lock has no artifact map")
    paths: dict[str, Path] = {}
    records: dict[str, dict[str, str]] = {}
    for label, row in raw.items():
        if not isinstance(label, str) or not isinstance(row, Mapping):
            raise LockError("invalid source training artifact record")
        try:
            path = COMMON.resolve_recorded_path(row.get("path"))
        except COMMON.EvaluationError as error:
            raise LockError(str(error)) from error
        if (
            not path.is_file()
            or path.is_symlink()
            or COMMON.file_sha256(path) != row.get("sha256")
        ):
            raise LockError(f"source training artifact drift: {label}")
        paths[label] = path
        records[label] = _record(path)
    required = {
        "parent_checkpoint",
        "parent_weights",
        "card_weights",
        "qu_weights",
        "target_deck",
    }
    missing = sorted(required - paths.keys())
    if missing:
        raise LockError(
            "source training lock omits artifacts: "
            + ", ".join(missing)
        )
    return paths, records


def _git_identity(code_paths: Mapping[str, Path]) -> dict[str, Any]:
    relative: list[str] = []
    for label, raw in sorted(code_paths.items()):
        path = raw.expanduser().resolve()
        try:
            relative.append(str(path.relative_to(ROOT)))
        except ValueError as error:
            raise LockError(
                f"direct-gate code artifact is outside repository: {label}"
            ) from error
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--",
                *relative,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                "--",
                *relative,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LockError(f"cannot verify direct-gate git identity: {error}") from error
    if status:
        raise LockError(
            "official direct gate requires committed clean code artifacts: "
            + status.replace("\n", "; ")
        )
    if tracked.returncode != 0:
        raise LockError(
            "official direct gate requires every code artifact tracked"
        )
    return {
        "commit": commit,
        "bound_code_paths": sorted(relative),
        "code_paths_committed_and_clean": True,
    }


def _candidate_paths(
    candidate_manifest_path: Path,
) -> dict[str, Path]:
    manifest_path = candidate_manifest_path.expanduser().resolve()
    expected_manifest = DEFAULT_CANDIDATE_MANIFEST.resolve()
    if manifest_path != expected_manifest:
        raise LockError(
            "only the canonical NumPy-deployable manifest may be bound"
        )
    return {
        "candidate_manifest": manifest_path,
        "candidate_checkpoint": (
            DEFAULT_CANDIDATE_BUNDLE / CANDIDATE_EVAL.CHECKPOINT_NAME
        ).resolve(),
        "candidate_weights": (
            DEFAULT_CANDIDATE_BUNDLE / CANDIDATE_EVAL.WEIGHTS_NAME
        ).resolve(),
    }


def build(
    training_lock_path: Path = DEFAULT_TRAINING_LOCK,
    candidate_lock_path: Path = DEFAULT_CANDIDATE_LOCK,
    candidate_result_path: Path = DEFAULT_CANDIDATE_RESULT,
    candidate_manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
) -> dict[str, Any]:
    """Build, but do not write, the prospective direct-gameplay lock."""

    resolved_training_lock = training_lock_path.expanduser().resolve()
    if resolved_training_lock != DEFAULT_TRAINING_LOCK.resolve():
        raise LockError(
            "only the canonical official MD-v4 training lock may be bound"
        )
    try:
        source = SOURCE_LOCK.load_lock(
            resolved_training_lock, verify_artifacts=True
        )
        resolved_candidate_lock = candidate_lock_path.expanduser().resolve()
        resolved_candidate_result = (
            candidate_result_path.expanduser().resolve()
        )
        if resolved_candidate_lock != DEFAULT_CANDIDATE_LOCK.resolve():
            raise LockError(
                "only the canonical NumPy-deployable lock may be bound"
            )
        if resolved_candidate_result != DEFAULT_CANDIDATE_RESULT.resolve():
            raise LockError(
                "only the canonical NumPy-deployable result may be bound"
            )
        candidate_lock = CANDIDATE_LOCK.load_lock(
            resolved_candidate_lock, verify_artifacts=True
        )
        result = COMMON.load_self_hashed_json(
            resolved_candidate_result,
            schema=CANDIDATE_EVAL.RESULT_SCHEMA,
            hash_key="result_sha256",
        )
        manifest = COMMON.load_self_hashed_json(
            candidate_manifest_path.expanduser().resolve(),
            schema=CANDIDATE_EVAL.MANIFEST_SCHEMA,
            hash_key="manifest_sha256",
        )
    except (
        SOURCE_LOCK.LockError,
        CANDIDATE_LOCK.NumpyDeployableLockError,
        COMMON.EvaluationError,
    ) as error:
        raise LockError(str(error)) from error
    source_paths, source_records = _resolve_source_artifacts(source)
    candidate_paths = _candidate_paths(candidate_manifest_path)

    code_paths = {
        "evaluator": Path(GAME.__file__),
        "lock_builder": Path(__file__),
        "runtime": Path(RUNTIME.__file__),
        "evaluator_test": ROOT / "tests/test_eval_md_v4_gameplay.py",
        "lock_test": ROOT / "tests/test_lock_md_v4_gameplay.py",
        "runtime_test": ROOT / "tests/test_md_v4_runtime.py",
        "candidate_evaluator": Path(CANDIDATE_EVAL.__file__),
        "candidate_lock_builder": Path(CANDIDATE_LOCK.__file__),
        "candidate_evaluator_test":
            ROOT / "tests/test_eval_md_v4_numpy_deployable.py",
        "candidate_lock_test":
            ROOT / "tests/test_lock_md_v4_numpy_deployable.py",
        "common_gameplay": Path(COMMON.__file__),
        "eval_ab": Path(eval_ab.__file__),
        "rl_env": Path(rl_env.__file__),
        "cabt_module": Path(rl_env.CABT.__file__),
        "index_corpus": Path(index_corpus.__file__),
        "model": Path(model.__file__),
        "features": Path(qu_v2_features.__file__),
        "agent_features": Path(agent_features.__file__),
        "card_runtime": Path(md_v2_card.__file__),
        "cards_module": Path(cards.__file__),
        "obsview": Path(obsview.__file__),
        "policy": Path(policy.__file__),
        "safety": Path(safety.__file__),
    }
    git = _git_identity(code_paths)
    artifacts: dict[str, dict[str, str]] = {
        "training_lock": _record(resolved_training_lock),
        "candidate_lock": _record(resolved_candidate_lock),
        "candidate_result": _record(resolved_candidate_result),
        "candidate_manifest": _record(
            candidate_paths["candidate_manifest"]
        ),
        "candidate_checkpoint": _record(
            candidate_paths["candidate_checkpoint"]
        ),
        "candidate_weights": _record(
            candidate_paths["candidate_weights"]
        ),
        "frozen_main_checkpoint": deepcopy(
            source_records["parent_checkpoint"]
        ),
        "frozen_main_weights": deepcopy(
            source_records["parent_weights"]
        ),
        "card_weights": deepcopy(source_records["card_weights"]),
        "qu_weights": deepcopy(source_records["qu_weights"]),
        "grim_deck": deepcopy(source_records["target_deck"]),
        **{
            label: _record(path)
            for label, path in sorted(code_paths.items())
        },
        **{
            f"training_input__{label}": deepcopy(record)
            for label, record in sorted(source_records.items())
        },
        "battle_engine": _record(Path(rl_env.CABT._LIB_PATH)),
        "cards_data": _record(ROOT / "data/cards.json"),
        "attacks_data": _record(ROOT / "data/attacks.json"),
        "meta_decks": _record(ROOT / "agent/meta_decks.json"),
    }
    paths = {
        label: COMMON.resolve_recorded_path(row["path"])
        for label, row in artifacts.items()
    }
    provisional = {"artifacts": artifacts}
    try:
        GAME.validate_candidate_bundle(
            source,
            candidate_lock,
            result,
            manifest,
            paths,
            provisional,
        )
    except (
        GAME.GameplayError,
        COMMON.EvaluationError,
    ) as error:
        raise LockError(str(error)) from error

    evaluation = source.get("evaluation")
    direct = (
        evaluation.get("direct_exact_mirror")
        if isinstance(evaluation, Mapping) else None
    )
    if not isinstance(direct, Mapping):
        raise LockError(
            "source training lock has no direct exact-mirror protocol"
        )
    pair_seeds = GAME._validate_pair_seeds(direct.get("pair_seeds"))
    if (
        direct.get("games") != GAME.DIRECT_GAMES
        or direct.get("paired_seeds") != GAME.DIRECT_PAIRS
        or direct.get("candidate_games_each_physical_seat")
            != GAME.DIRECT_PAIRS
        or direct.get("pair_seed_sha256")
            != SOURCE_LOCK.value_sha256(list(pair_seeds))
    ):
        raise LockError("source direct pair-seed contract drifted")
    deck = COMMON.read_deck(source_paths["target_deck"])
    opponents = GAME.build_control_opponents(
        deck,
        artifacts["frozen_main_weights"]["sha256"],
        artifacts["card_weights"]["sha256"],
        artifacts["qu_weights"]["sha256"],
    )
    environment = GAME.bound_environment(
        provisional, paths, deck, opponents
    )
    _, sanity_schedule = GAME.build_schedule_contract(
        pair_seeds, opponents, pairs=GAME.SANITY_PAIRS
    )
    _, direct_schedule = GAME.build_schedule_contract(
        pair_seeds, opponents, pairs=GAME.DIRECT_PAIRS
    )

    payload: dict[str, Any] = {
        "schema": GAME.LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_after_numpy_deployable_qualification_and_before_gameplay":
            True,
        "one_candidate_only": True,
        "one_schedule_one_attempt": True,
        "original_training_lock_sha256": source["lock_sha256"],
        "source_candidate_lock_sha256":
            candidate_lock["lock_sha256"],
        "source_candidate_result_sha256":
            result["result_sha256"],
        "source_candidate_manifest_sha256":
            manifest["manifest_sha256"],
        "candidate": {
            "name": "md-v4-numpy-deployable-v1",
            "selected_epoch": TRAIN.FIXED_EPOCHS,
            "checkpoint_sha256": artifacts[
                "candidate_checkpoint"
            ]["sha256"],
            "weights_sha256": artifacts["candidate_weights"]["sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "numpy_array_mapping_sha256":
                CANDIDATE_LOCK.EXPECTED_NUMPY_MAPPING_SHA256,
            "route": (
                "exact target deck ST_MAIN only; every other route is "
                "complete frozen MD-v3"
            ),
            "checkpoint_reconstructed_export_equals_candidate_npz": True,
            "embedded_parent_equals_frozen_main_npz": True,
            "research_bundle_artifact_identity_passed": True,
            "research_bundle_only": True,
            "later_exact_package_runtime_conformance_required": True,
            "offline_rejection_gates_passed": True,
            "torch_numpy_action_identity_is_a_gate": False,
            "prior_torch_numpy_action_mismatches":
                CANDIDATE_LOCK.PRIOR_ACTION_MISMATCHES,
        },
        "control": {
            "runtime": (
                "the identical layered controller with MD-v4 overlay disabled"
            ),
            "main_weights_sha256": artifacts[
                "frozen_main_weights"
            ]["sha256"],
            "card_weights_sha256": artifacts["card_weights"]["sha256"],
            "qu_weights_sha256": artifacts["qu_weights"]["sha256"],
        },
        "protocol": {
            "frozen_sanity": GAME.expected_sanity_protocol(),
            "direct_exact_mirror": deepcopy(direct),
            "native_engine_rng": deepcopy(
                GAME.NATIVE_ENGINE_RNG_CONTRACT
            ),
        },
        "schedules": {
            "sanity": sanity_schedule,
            "direct": direct_schedule,
        },
        "environment": environment,
        "git": git,
        "artifacts": artifacts,
        "decision_rule": {
            "sanity": (
                "20 clean byte-identical frozen-vs-frozen games; W/L is "
                "reported but never gates"
            ),
            "direct": GAME.DIRECT_DECISION_RULE,
            "pass": (
                "only a passing direct result permits a separately locked "
                "recent-frequency field gate"
            ),
            "failure": (
                "retire md-v4-numpy-deployable-v1; no field/temporal gate, "
                "threshold change, seed rerun, artifact substitution, or "
                "retraining"
            ),
            "native_engine_rng_seedable": False,
            "promotion_authority": False,
            "upload_authority": False,
        },
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-lock", type=Path, default=DEFAULT_TRAINING_LOCK
    )
    parser.add_argument(
        "--candidate-lock",
        type=Path,
        default=DEFAULT_CANDIDATE_LOCK,
    )
    parser.add_argument(
        "--candidate-result",
        type=Path,
        default=DEFAULT_CANDIDATE_RESULT,
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=DEFAULT_CANDIDATE_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output = args.output.expanduser().resolve()
        if output != DEFAULT_OUTPUT.resolve():
            raise LockError(
                "official direct-gameplay lock must use the single "
                f"canonical output path {DEFAULT_OUTPUT.resolve()}"
            )
        if output.exists():
            raise LockError(f"refusing to overwrite {output}")
        payload = build(
            args.training_lock,
            args.candidate_lock,
            args.candidate_result,
            args.candidate_manifest,
        )
        GAME._write_new(output, payload)
    except (
        LockError,
        GAME.GameplayError,
        COMMON.EvaluationError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps({
        "lock": str(output),
        "lock_sha256": payload["lock_sha256"],
        "candidate_weights_sha256": payload[
            "candidate"
        ]["weights_sha256"],
        "direct_schedule_sha256": payload[
            "schedules"
        ]["direct"]["episode_manifest_sha256"],
        "promotion_authority": False,
        "upload_authority": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
