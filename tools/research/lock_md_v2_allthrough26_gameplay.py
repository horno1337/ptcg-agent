"""Bind the completed MD-v2 all-through-26 model and gameplay schedules.

The corpus-level lock already fixed the thresholds and seeds before training.
This builder runs only after training finishes, verifies that the exported
model is the exact locked run, and records every runtime artifact and both
deterministic schedules before any engine outcome is generated.
"""

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

from agent import md_v1 as MD1  # noqa: E402
from agent import model, policy, qu_v2_features, safety  # noqa: E402
from tools import eval_ab, index_corpus, rl_env  # noqa: E402
from tools.research import eval_md_v2_allthrough26_gameplay as GAME  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import prepare_md_v2_allthrough26 as PREP  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v2-allthrough26"
DEFAULT_ALLTHROUGH_LOCK = RUN / "lock.json"
DEFAULT_CORPUS = RUN / "corpus.json"
DEFAULT_PROVENANCE = RUN / "model/candidate-qu-v2a-training-manifest.json"
DEFAULT_MD_V1 = ROOT / "agent/md_v1_weights.npz"
DEFAULT_QU_V2B = ROOT / "agent/weights.npz"
DEFAULT_DECK = ROOT / "decks/md_v1_grimmsnarl.csv"
DEFAULT_OUTPUT = RUN / "gameplay-lock.json"


class LockError(RuntimeError):
    """The completed training run cannot be bound to its preregistration."""


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": COMMON.file_sha256(resolved)}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise LockError(f"JSON root is not an object: {path}")
    return value


def _load_self_hashed(
    path: Path, *, schema: str, hash_key: str
) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(
            path.expanduser().resolve(), schema=schema, hash_key=hash_key
        )
    except COMMON.EvaluationError as error:
        raise LockError(str(error)) from error


def _resolve_artifact(
    provenance: Mapping[str, Any], name: str
) -> tuple[Path, str]:
    raw = provenance.get("artifacts", {}).get(name)
    if not isinstance(raw, Mapping):
        raise LockError(f"training provenance has no {name} artifact")
    path = Path(str(raw.get("path"))).expanduser().resolve()
    expected = raw.get("sha256")
    if (
        not path.is_file()
        or not isinstance(expected, str)
        or COMMON.file_sha256(path) != expected
    ):
        raise LockError(f"training artifact drift: {name}")
    return path, expected


def inspect_training(
    allthrough_lock_path: Path,
    corpus_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    lock = _load_self_hashed(
        allthrough_lock_path,
        schema="ptcg.md-v2.allthrough26-lock.v1",
        hash_key="lock_sha256",
    )
    corpus = _load_json(corpus_path.expanduser().resolve())
    if (
        corpus.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(corpus)
        or lock.get("corpus", {}).get("path")
            != str(corpus_path.expanduser().resolve())
        or lock.get("corpus", {}).get("file_sha256")
            != COMMON.file_sha256(corpus_path.expanduser().resolve())
        or lock.get("corpus", {}).get("manifest_sha256")
            != corpus.get("manifest_sha256")
    ):
        raise LockError("all-through-26 corpus binding drifted")
    provenance = _load_self_hashed(
        provenance_path,
        schema=TRAIN.TRAINING_SCHEMA,
        hash_key="manifest_sha256",
    )
    configuration = provenance.get("configuration")
    training = lock.get("training")
    input_record = provenance.get("input")
    selection = provenance.get("selection")
    if (
        provenance.get("candidate_only") is not True
        or provenance.get("test_status") != "deferred"
        or provenance.get("test") is not None
        or not isinstance(configuration, Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(input_record, Mapping)
        or input_record.get("manifest_path")
            != str(corpus_path.expanduser().resolve())
        or input_record.get("manifest_file_sha256")
            != lock["corpus"]["file_sha256"]
        or input_record.get("manifest_sha256")
            != lock["corpus"]["manifest_sha256"]
        or input_record.get("corpus_content_sha256")
            != corpus.get("corpus_content_sha256")
        or not isinstance(selection, Mapping)
    ):
        raise LockError("training provenance is not the locked deferred-test run")
    exact_configuration = {
        "architecture": list(configuration.get("architecture", [])),
        "freeze_public_backbone": configuration.get("freeze_public_backbone"),
        "target_select_type": configuration.get("target_select_type"),
        "epochs": configuration.get("epochs"),
        "batch_size": configuration.get("batch_size"),
        "shuffle_buffer": configuration.get("shuffle_buffer"),
        "learning_rate": configuration.get("learning_rate"),
        "weight_decay": configuration.get("weight_decay"),
        "gradient_clip": configuration.get("gradient_clip"),
        "value_coefficient": configuration.get("value_coefficient"),
        "kl_coefficient": configuration.get("kl_coefficient"),
        "seed": configuration.get("seed"),
    }
    expected_configuration = {
        key: training[key] for key in exact_configuration
    }
    weights = configuration.get("weights")
    if (
        exact_configuration != expected_configuration
        or configuration.get("target_deck_sha256")
            != PREP.TARGET_DECK_SHA256
        or configuration.get("defer_test") is not True
        or configuration.get("initial_checkpoint_sha256")
            != training.get("initial_checkpoint_sha256")
        or not isinstance(weights, Mapping)
        or weights.get("win") != 1.0
        or weights.get("draw") != 1.0
        or weights.get("loss") != 1.0
        or weights.get("game_normalized_by_decision_count") is not True
        or selection.get("best_epoch") not in range(1, 11)
        or not isinstance(selection.get("best_validation_objective"), float)
    ):
        raise LockError("training configuration/selection differs from the lock")
    checkpoint_path, checkpoint_sha = _resolve_artifact(
        provenance, "checkpoint"
    )
    weights_path, weights_sha = _resolve_artifact(provenance, "weights")
    return {
        "lock": lock,
        "corpus": corpus,
        "provenance": provenance,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "weights_path": weights_path,
        "weights_sha256": weights_sha,
    }


def build_lock(
    *,
    allthrough_lock_path: Path,
    corpus_path: Path,
    provenance_path: Path,
    md_v1_path: Path,
    qu_v2b_path: Path,
    deck_path: Path,
) -> dict[str, Any]:
    selected = inspect_training(
        allthrough_lock_path, corpus_path, provenance_path
    )
    for label, path in (
        ("MD-v1 weights", md_v1_path),
        ("Qu-v2B weights", qu_v2b_path),
        ("Grimmsnarl deck", deck_path),
    ):
        if not path.expanduser().resolve().is_file():
            raise LockError(f"missing {label}: {path}")
    md_v1_path = md_v1_path.expanduser().resolve()
    qu_v2b_path = qu_v2b_path.expanduser().resolve()
    deck_path = deck_path.expanduser().resolve()
    try:
        deck = COMMON.read_deck(deck_path)
    except COMMON.EvaluationError as error:
        raise LockError(str(error)) from error
    md_v1_sha = COMMON.file_sha256(md_v1_path)
    qu_sha = COMMON.file_sha256(qu_v2b_path)
    if md_v1_sha != MD1.WEIGHTS_SHA256:
        raise LockError("MD-v1 weights differ from its runtime constant")

    md_opponents = COMMON.build_primary_opponents(deck, md_v1_sha, qu_sha)
    qu_opponents = GAME.build_qu_opponents(deck, qu_sha)
    md_schedule = COMMON.build_schedule_contract(
        md_opponents, games=GAME.GAMES_PER_OPPONENT, seed=GAME.SEED
    )
    qu_schedule = COMMON.build_schedule_contract(
        qu_opponents, games=GAME.GAMES_PER_OPPONENT, seed=GAME.SEED
    )
    for label, schedule in (
        ("versus_md_v1", md_schedule),
        ("versus_qu_v2b", qu_schedule),
    ):
        seats = [row["learner_seat"] for row in schedule["episodes"]]
        if seats.count(0) != 320 or seats.count(1) != 320:
            raise LockError(f"{label} schedule is not exactly seat balanced")

    artifacts = {
        "allthrough_lock": _record(allthrough_lock_path),
        "corpus": _record(corpus_path),
        "candidate_weights": _record(selected["weights_path"]),
        "candidate_checkpoint": _record(selected["checkpoint_path"]),
        "candidate_training_provenance": _record(provenance_path),
        "md_v1_weights": _record(md_v1_path),
        "qu_v2b_weights": _record(qu_v2b_path),
        "grim_deck": _record(deck_path),
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
        "source_preregistration_sha256": selected["lock"]["lock_sha256"],
        "candidate": {
            "name": "MD-v2 all-through-July-26",
            "best_epoch": selected["provenance"]["selection"]["best_epoch"],
            "best_validation_objective": selected["provenance"]["selection"][
                "best_validation_objective"
            ],
            "weights_sha256": selected["weights_sha256"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "route": (
                "exact Grimmsnarl deck and ST_MAIN only; "
                "frozen Qu-v2B elsewhere"
            ),
        },
        "protocol": {
            "deck": "exact MD-v1 Grimmsnarl list for every seat",
            "games_per_opponent": GAME.GAMES_PER_OPPONENT,
            "seed": GAME.SEED,
            "seat_balance": "exactly 320 games per seat per comparison",
            "run_as_one_nonadaptive_test": True,
            "opponents": [
                "frozen MD-v1 ST_MAIN plus frozen Qu-v2B fallback",
                "pure frozen Qu-v2B",
            ],
            "decision_rule": {
                "versus_md_v1": (
                    "point estimate > 0.50 and Wilson CI95 lower bound > 0.45"
                ),
                "versus_qu_v2b": "point estimate > 0.50",
                "cleanliness": (
                    "zero invalids, truncations, errors, exceptions, "
                    "fail-soft fallbacks, repairs, or off-deck main routes"
                ),
                "accept": "all rules pass",
            },
        },
        "artifacts": artifacts,
        "schedules": {
            "versus_md_v1": md_schedule,
            "versus_qu_v2b": qu_schedule,
        },
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
    parser.add_argument(
        "--allthrough-lock", type=Path, default=DEFAULT_ALLTHROUGH_LOCK
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--training-provenance", type=Path, default=DEFAULT_PROVENANCE
    )
    parser.add_argument("--md-v1-weights", type=Path, default=DEFAULT_MD_V1)
    parser.add_argument("--qu-v2b-weights", type=Path, default=DEFAULT_QU_V2B)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        payload = build_lock(
            allthrough_lock_path=args.allthrough_lock,
            corpus_path=args.corpus,
            provenance_path=args.training_provenance,
            md_v1_path=args.md_v1_weights,
            qu_v2b_path=args.qu_v2b_weights,
            deck_path=args.deck,
        )
        _write_new(args.json_out, payload)
    except (LockError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "lock_sha256": payload["lock_sha256"],
        "candidate": payload["candidate"],
        "schedules": {
            key: value["sha256"]
            for key, value in payload["schedules"].items()
        },
    }, sort_keys=True))
    print(f"wrote {args.json_out.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
