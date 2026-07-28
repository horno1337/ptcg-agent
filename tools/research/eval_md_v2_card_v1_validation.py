"""Evaluate the fixed MD-v2 ST_CARD specialist on locked validation games."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import eval_md_v2_july27_temporal as CORE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v2-card-v1.lock.v1"
RESULT_SCHEMA = "ptcg.md-v2-card-v1.validation-result.v1"
RUN = ROOT / "tools/checkpoints/md-v2-card-v1"
DEFAULT_LOCK = RUN / "lock.json"
DEFAULT_RESULT = RUN / "validation-result.json"
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
TARGET_SELECT_TYPE = 1


class EvaluationError(RuntimeError):
    """The preregistration, training output, or validation stream failed."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(
            encoding="utf-8"
        ))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root is not an object: {path}")
    return value


def _resolve_artifacts(lock: Mapping[str, Any]) -> dict[str, Path]:
    records = lock.get("artifacts")
    if not isinstance(records, Mapping):
        raise EvaluationError("card-specialist lock has no artifact map")
    result: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(label, str) or not isinstance(record, Mapping):
            raise EvaluationError("invalid card-specialist artifact record")
        resolved = COMMON.resolve_recorded_path(record.get("path"))
        if (
            not resolved.is_file()
            or COMMON.file_sha256(resolved) != record.get("sha256")
        ):
            raise EvaluationError(f"card-specialist artifact drift: {label}")
        result[label] = resolved
    required = {
        "corpus",
        "source_lock",
        "initial_checkpoint",
        "qu_v2b_weights",
        "md_v2_main_weights",
        "target_deck",
        "trainer",
        "dataset_loader",
        "training_features",
        "training_model",
        "resource_preflight",
        "evaluator",
        "lock_builder",
        "temporal_core",
        "indexer",
        "runtime_features",
        "runtime_model",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise EvaluationError(f"lock omits artifacts: {missing}")
    if result["evaluator"] != Path(__file__).resolve():
        raise EvaluationError("lock names a different validation evaluator")
    return result


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    try:
        lock = COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=LOCK_SCHEMA,
            hash_key="lock_sha256",
        )
    except COMMON.EvaluationError as error:
        raise EvaluationError(str(error)) from error
    training = lock.get("training")
    validation = lock.get("validation")
    runtime = lock.get("runtime")
    if (
        not isinstance(training, Mapping)
        or not isinstance(validation, Mapping)
        or not isinstance(runtime, Mapping)
        or training.get("target_deck_sha256") != TARGET_DECK_SHA256
        or training.get("target_select_type") != TARGET_SELECT_TYPE
        or training.get("epochs") != 10
        or training.get("batch_size") != 128
        or training.get("shuffle_buffer") != 4096
        or training.get("learning_rate") != 0.00005
        or training.get("weight_decay") != 0.00001
        or training.get("gradient_clip") != 1.0
        or training.get("value_coefficient") != 0.0
        or training.get("kl_coefficient") != 0.0
        or training.get("seed") != 20260728
        or training.get("win_draw_loss_weights") != [1.0, 1.0, 1.0]
        or training.get("game_normalized") is not True
        or training.get("freeze_public_backbone") is not True
        or training.get("checkpoint_selection")
            != "lowest validation objective; earliest epoch tie"
        or validation.get("same_decision_stream_for_models") is not True
        or validation.get("pass_rule") != (
            "candidate objective is strictly lower than frozen Qu-v2B both "
            "overall and on exact-list mirror games"
        )
        or runtime.get("route") != (
            "exact own deck plus ST_CARD plus public opposing Grimmsnarl "
            "signature only"
        )
        or runtime.get("fallback") != "unchanged MD-v2 layered runtime"
    ):
        raise EvaluationError("card-specialist protocol differs from lock")
    return lock, _resolve_artifacts(lock)


def _is_exact_mirror(game: TRAIN.LockedGame) -> bool:
    return (
        game.registered_deck_sha256s[0] == TARGET_DECK_SHA256
        and game.registered_deck_sha256s[1] == TARGET_DECK_SHA256
    )


def _training_config(
    manifest_path: Path,
    out_dir: Path,
) -> TRAIN.TrainingConfig:
    return TRAIN.TrainingConfig(
        manifest_path=manifest_path,
        out_dir=out_dir,
        epochs=10,
        batch_size=128,
        shuffle_buffer=4096,
        learning_rate=0.00005,
        weight_decay=0.00001,
        gradient_clip=1.0,
        seed=20260728,
        device="cpu",
        win_weight=1.0,
        draw_weight=1.0,
        loss_weight=1.0,
        game_normalized=True,
        target_deck_sha256=TARGET_DECK_SHA256,
        target_select_type=TARGET_SELECT_TYPE,
        value_coefficient=0.0,
        kl_coefficient=0.0,
    )


def _verify_training_output(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    provenance_path: Path,
    weights_path: Path,
) -> dict[str, Any]:
    try:
        provenance = COMMON.load_self_hashed_json(
            provenance_path,
            schema=TRAIN.TRAINING_SCHEMA,
            hash_key="manifest_sha256",
        )
    except COMMON.EvaluationError as error:
        raise EvaluationError(str(error)) from error
    configuration = provenance.get("configuration")
    input_record = provenance.get("input")
    artifacts = provenance.get("artifacts")
    source_files = provenance.get("source_files_sha256")
    weights = (
        configuration.get("weights")
        if isinstance(configuration, Mapping) else None
    )
    expected_training = lock["training"]
    if (
        provenance.get("candidate_only") is not True
        or provenance.get("test_status") != "deferred"
        or provenance.get("test") is not None
        or not isinstance(configuration, Mapping)
        or not isinstance(input_record, Mapping)
        or not isinstance(artifacts, Mapping)
        or not isinstance(source_files, Mapping)
        or not isinstance(weights, Mapping)
        or configuration.get("target_deck_sha256") != TARGET_DECK_SHA256
        or configuration.get("target_select_type") != TARGET_SELECT_TYPE
        or configuration.get("architecture")
            != expected_training["architecture"]
        or configuration.get("epochs") != expected_training["epochs"]
        or configuration.get("batch_size")
            != expected_training["batch_size"]
        or configuration.get("shuffle_buffer")
            != expected_training["shuffle_buffer"]
        or configuration.get("learning_rate")
            != expected_training["learning_rate"]
        or configuration.get("weight_decay")
            != expected_training["weight_decay"]
        or configuration.get("gradient_clip")
            != expected_training["gradient_clip"]
        or configuration.get("value_coefficient")
            != expected_training["value_coefficient"]
        or configuration.get("kl_coefficient")
            != expected_training["kl_coefficient"]
        or configuration.get("freeze_public_backbone") is not True
        or configuration.get("seed") != expected_training["seed"]
        or configuration.get("defer_test") is not True
        or configuration.get("resume_latest") is not False
        or configuration.get("initial_checkpoint_sha256")
            != lock["artifacts"]["initial_checkpoint"]["sha256"]
        or weights.get("win") != 1.0
        or weights.get("draw") != 1.0
        or weights.get("loss") != 1.0
        or weights.get("sources") != {}
        or weights.get("decks") != {}
        or weights.get("game_normalized_by_decision_count") is not True
        or input_record.get("manifest_file_sha256")
            != lock["artifacts"]["corpus"]["sha256"]
        or input_record.get("manifest_sha256")
            != lock["corpus"]["manifest_sha256"]
        or input_record.get("corpus_content_sha256")
            != lock["corpus"]["corpus_content_sha256"]
        or len(input_record.get("selected_games", ()))
            != lock["corpus"]["games"]
        or source_files.get("trainer")
            != lock["artifacts"]["trainer"]["sha256"]
        or source_files.get("dataset_loader")
            != lock["artifacts"]["dataset_loader"]["sha256"]
        or source_files.get("public_features")
            != lock["artifacts"]["training_features"]["sha256"]
        or source_files.get("candidate_model")
            != lock["artifacts"]["training_model"]["sha256"]
        or source_files.get("corpus_indexer")
            != lock["artifacts"]["indexer"]["sha256"]
        or source_files.get("resource_preflight")
            != lock["artifacts"]["resource_preflight"]["sha256"]
        or source_files.get("initial_checkpoint")
            != lock["artifacts"]["initial_checkpoint"]["sha256"]
        or artifacts.get("weights", {}).get("sha256")
            != COMMON.file_sha256(weights_path)
    ):
        raise EvaluationError("training output differs from preregistration")
    return provenance


def decision(objectives: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    overall = objectives["overall"]
    mirror = objectives["exact_mirror"]
    finite = all(
        math.isfinite(value)
        for stratum in objectives.values()
        for value in stratum.values()
    )
    overall_better = overall["candidate"] < overall["qu_v2b"]
    mirror_better = mirror["candidate"] < mirror["qu_v2b"]
    return {
        "passed": finite and overall_better and mirror_better,
        "finite": finite,
        "overall_better_than_qu_v2b": overall_better,
        "exact_mirror_better_than_qu_v2b": mirror_better,
        "overall_candidate_minus_qu_v2b": (
            overall["candidate"] - overall["qu_v2b"]
        ),
        "exact_mirror_candidate_minus_qu_v2b": (
            mirror["candidate"] - mirror["qu_v2b"]
        ),
        "rule": (
            "candidate objective is strictly lower than frozen Qu-v2B both "
            "overall and on exact-list mirror games"
        ),
    }


def evaluate(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    weights_path: Path,
    provenance_path: Path,
    result_path: Path,
    quiet: bool,
) -> dict[str, Any]:
    result_path = result_path.expanduser().resolve()
    if result_path.exists():
        raise EvaluationError(f"refusing to overwrite {result_path}")
    weights_path = weights_path.expanduser().resolve()
    provenance_path = provenance_path.expanduser().resolve()
    if not weights_path.is_file() or not provenance_path.is_file():
        raise EvaluationError("fixed training output is incomplete")
    provenance = _verify_training_output(
        lock, paths, provenance_path, weights_path
    )
    try:
        plan = TRAIN.load_corpus_plan(
            paths["corpus"], required_splits=("validation",)
        )
        nets = {
            "candidate": COMMON._load_net(
                weights_path, "MD-v2 ST_CARD candidate"
            ),
            "qu_v2b": COMMON._load_net(
                paths["qu_v2b_weights"], "frozen Qu-v2B"
            ),
        }
    except (TRAIN.TrainingError, COMMON.EvaluationError) as error:
        raise EvaluationError(str(error)) from error
    config = _training_config(paths["corpus"], RUN / "validation-no-training")
    numerators = {
        stratum: {name: 0.0 for name in nets}
        for stratum in ("overall", "exact_mirror")
    }
    denominators = {
        stratum: {name: 0.0 for name in nets}
        for stratum in numerators
    }
    games = {"overall": 0, "exact_mirror": 0}
    decisions = {"overall": 0, "exact_mirror": 0}
    validation_games = plan.games["validation"]
    try:
        for game_index, game in enumerate(validation_games, start=1):
            mirror = _is_exact_mirror(game)
            game_decisions = 0
            for sample in TRAIN.iter_game_samples(
                game, config, anchor=None, cache=None
            ):
                game_decisions += 1
                runtime_sample = CORE.runtime_features(sample.features)
                for name, net in nets.items():
                    logits, _ = net.forward(runtime_sample)
                    loss = CORE.sequence_nll(logits, sample) * sample.weight
                    numerators["overall"][name] += loss
                    denominators["overall"][name] += sample.weight
                    if mirror:
                        numerators["exact_mirror"][name] += loss
                        denominators["exact_mirror"][name] += sample.weight
                decisions["overall"] += 1
                if mirror:
                    decisions["exact_mirror"] += 1
            if game_decisions:
                games["overall"] += 1
                games["exact_mirror"] += int(mirror)
            if not quiet and (
                game_index % 100 == 0 or game_index == len(validation_games)
            ):
                print(json.dumps({
                    "event": "validation_progress",
                    "completed_games": game_index,
                    "total_games": len(validation_games),
                    "st_card_decisions": decisions["overall"],
                    "exact_mirror_st_card_decisions": decisions["exact_mirror"],
                }, sort_keys=True), flush=True)
    except (
        OSError,
        ValueError,
        FloatingPointError,
        TRAIN.TrainingError,
        CORE.EvaluationError,
    ) as error:
        raise EvaluationError(f"validation scoring failed: {error}") from error
    if (
        decisions["overall"] <= 0
        or decisions["exact_mirror"] <= 0
        or any(
            value <= 0.0
            for stratum in denominators.values()
            for value in stratum.values()
        )
    ):
        raise EvaluationError("validation cohort produced an empty stratum")
    for stratum in denominators:
        if len(set(denominators[stratum].values())) != 1:
            raise EvaluationError(
                f"models received different weights in {stratum}"
            )
    objectives = {
        stratum: {
            name: numerators[stratum][name] / denominators[stratum][name]
            for name in nets
        }
        for stratum in numerators
    }
    verdict = decision(objectives)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "candidate_weights": {
            "path": str(weights_path),
            "sha256": COMMON.file_sha256(weights_path),
        },
        "training_provenance": {
            "path": str(provenance_path),
            "sha256": COMMON.file_sha256(provenance_path),
            "best_epoch": provenance["selection"]["best_epoch"],
            "best_validation_objective": provenance["selection"][
                "best_validation_objective"
            ],
        },
        "games_with_st_card": games,
        "st_card_decisions": decisions,
        "weight_denominators": denominators,
        "weighted_nll_numerators": numerators,
        "objectives": objectives,
        "decision": verdict,
        "promotion_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    try:
        CORE._atomic_new(result_path, payload)
    except CORE.EvaluationError as error:
        raise EvaluationError(str(error)) from error
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--weights",
        type=Path,
        default=RUN / "model/candidate-qu-v2a-weights.npz",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=RUN / "model/candidate-qu-v2a-training-manifest.json",
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        lock, paths = load_lock(args.lock)
        payload = evaluate(
            lock,
            paths,
            weights_path=args.weights,
            provenance_path=args.provenance,
            result_path=args.result,
            quiet=args.quiet,
        )
    except (
        EvaluationError,
        CORE.EvaluationError,
        COMMON.EvaluationError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps({
        "objectives": payload["objectives"],
        "decision": payload["decision"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
