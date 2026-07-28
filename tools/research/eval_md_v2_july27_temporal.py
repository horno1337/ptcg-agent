"""Score fixed MD-v2, MD-v1, and Qu-v2B once on the July 27 cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


LOCK_SCHEMA = "ptcg.md-v2.july27-temporal-lock.v1"
RESULT_SCHEMA = "ptcg.md-v2.july27-temporal-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v2.july27-temporal-attempt.v1"
RUN = ROOT / "tools/checkpoints/md-v2-allthrough26"
DEFAULT_LOCK = RUN / "july27-temporal-lock.json"
DEFAULT_ATTEMPT = RUN / "july27-temporal-attempt.json"
DEFAULT_RESULT = RUN / "july27-temporal-result.json"


class EvaluationError(RuntimeError):
    """The one-shot temporal contract or score stream failed closed."""


def _atomic_new(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise EvaluationError(f"refusing to overwrite {resolved}")
    raw = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".partial", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise EvaluationError(f"refusing to overwrite {resolved}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    records = lock.get("artifacts")
    if not isinstance(records, Mapping):
        raise EvaluationError("temporal lock has no artifact map")
    result: dict[str, Path] = {}
    for label, record in records.items():
        if not isinstance(label, str) or not isinstance(record, Mapping):
            raise EvaluationError("invalid temporal artifact record")
        try:
            path = COMMON.resolve_recorded_path(record.get("path"))
        except COMMON.EvaluationError as error:
            raise EvaluationError(str(error)) from error
        if (
            not path.is_file()
            or COMMON.file_sha256(path) != record.get("sha256")
        ):
            raise EvaluationError(f"temporal artifact drift: {label}")
        result[label] = path
    required = {
        "gameplay_lock",
        "raw_index",
        "cohort",
        "candidate_weights",
        "md_v1_weights",
        "qu_v2b_weights",
        "evaluator",
        "preparer",
        "trainer",
        "indexer",
        "dataset_loader",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise EvaluationError(f"temporal lock omits artifacts: {missing}")
    if result["evaluator"] != Path(__file__).resolve():
        raise EvaluationError("temporal lock names a different evaluator")
    return result


def _source_inventory(source_record: Mapping[str, Any]) -> dict[str, Any]:
    # Imported lazily to avoid a circular module import during preparation.
    from tools.research import prepare_md_v2_july27_temporal as PREP

    return PREP._inventory(Path(str(source_record.get("source"))))


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    try:
        lock = COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=LOCK_SCHEMA,
            hash_key="lock_sha256",
        )
    except COMMON.EvaluationError as error:
        raise EvaluationError(str(error)) from error
    protocol = lock.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("date") != "2026-07-27"
        or protocol.get("team_name_prefilter") is not False
        or protocol.get("same_decision_stream_for_all_models") is not True
        or protocol.get("one_shot") is not True
        or protocol.get("no_retraining_or_reselection_after_open") is not True
        or protocol.get("pass_rule")
            != "MD-v2 objective is strictly lower than both MD-v1 and Qu-v2B"
    ):
        raise EvaluationError("temporal protocol differs from the preregistration")
    paths = _paths(lock)
    try:
        from tools.research import eval_md_v2_allthrough26_gameplay as GAMEPLAY

        gameplay_lock, _ = GAMEPLAY.load_lock(paths["gameplay_lock"])
        raw_index = json.loads(paths["raw_index"].read_text(encoding="utf-8"))
    except (
        GAMEPLAY.EvaluationError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise EvaluationError(f"cannot verify temporal parent artifacts: {error}") from error
    if (
        gameplay_lock.get("lock_sha256") != lock.get("gameplay_lock_sha256")
        or gameplay_lock.get("candidate", {}).get("weights_sha256")
            != lock.get("candidate_weights_sha256")
        or lock.get("candidate_weights_sha256")
            != lock["artifacts"]["candidate_weights"]["sha256"]
        or raw_index.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(raw_index)
        or raw_index.get("manifest_sha256")
            != lock.get("raw_index_manifest_sha256")
    ):
        raise EvaluationError("temporal parent lock/raw-index binding drifted")
    try:
        cohort = json.loads(paths["cohort"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot load temporal cohort: {error}") from error
    binding = cohort.get("md_v2_july27_temporal")
    if (
        cohort.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(cohort)
        or cohort.get("manifest_sha256") != lock.get("cohort_manifest_sha256")
        or len(cohort.get("games", [])) != lock.get("cohort_games")
        or not isinstance(binding, Mapping)
        or binding.get("date") != "2026-07-27"
        or binding.get("team_name_prefilter") is not False
        or binding.get("same_cohort_for_all_models") is not True
    ):
        raise EvaluationError("temporal cohort contract/hash mismatch")
    if _source_inventory(lock["source_inventory"]) != lock["source_inventory"]:
        raise EvaluationError("official July 27 source inventory drifted")
    return lock, paths, cohort


def sequence_nll(logits: np.ndarray, sample: TRAIN.TrainingSample) -> float:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.shape != (sample.n_opts + 1,) or not np.isfinite(values).all():
        raise EvaluationError("model produced incompatible temporal logits")
    effective_max = (
        min(sample.n_max, sample.n_opts)
        if sample.n_max > 0 else sample.n_opts
    )
    sequence = list(sample.picks)
    if len(sequence) < effective_max:
        sequence.append(sample.n_opts)
    available = np.ones(sample.n_opts + 1, dtype=np.bool_)
    result = 0.0
    for step, action in enumerate(sequence):
        legal = available.copy()
        legal[sample.n_opts] = step >= sample.n_min
        legal_values = values[legal]
        maximum = float(np.max(legal_values))
        log_denominator = maximum + math.log(
            float(np.exp(legal_values - maximum).sum())
        )
        result += log_denominator - float(values[action])
        if action == sample.n_opts:
            break
        available[action] = False
    if not math.isfinite(result) or result < 0.0:
        raise EvaluationError("non-finite/negative temporal sequence NLL")
    return result


def temporal_decision(objectives: Mapping[str, float]) -> dict[str, Any]:
    candidate = objectives["md_v2"]
    md_v1 = objectives["md_v1"]
    qu_v2b = objectives["qu_v2b"]
    finite = all(math.isfinite(value) for value in objectives.values())
    passed = finite and candidate < md_v1 and candidate < qu_v2b
    return {
        "passed": passed,
        "finite": finite,
        "candidate_better_than_md_v1": candidate < md_v1,
        "candidate_better_than_qu_v2b": candidate < qu_v2b,
        "md_v2_minus_md_v1": candidate - md_v1,
        "md_v2_minus_qu_v2b": candidate - qu_v2b,
        "rule": "MD-v2 objective is strictly lower than both MD-v1 and Qu-v2B",
    }


def _config(manifest_path: Path) -> TRAIN.TrainingConfig:
    return TRAIN.TrainingConfig(
        manifest_path=manifest_path,
        out_dir=RUN / "july27-temporal-no-training",
        device="cpu",
        win_weight=1.0,
        draw_weight=1.0,
        loss_weight=1.0,
        game_normalized=True,
        target_deck_sha256=COMMON.TARGET_DECK_SHA256,
        target_select_type=0,
        value_coefficient=0.0,
        kl_coefficient=0.0,
    )


def evaluate(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    attempt_path: Path,
    result_path: Path,
    quiet: bool,
) -> dict[str, Any]:
    if attempt_path.exists() or result_path.exists():
        raise EvaluationError("the one-shot July 27 attempt is already consumed")
    try:
        plan = TRAIN.load_corpus_plan(
            paths["cohort"], required_splits=("test",)
        )
        nets = {
            "md_v2": COMMON._load_net(
                paths["candidate_weights"], "fixed MD-v2"
            ),
            "md_v1": COMMON._load_net(paths["md_v1_weights"], "frozen MD-v1"),
            "qu_v2b": COMMON._load_net(
                paths["qu_v2b_weights"], "frozen Qu-v2B"
            ),
        }
    except (TRAIN.TrainingError, COMMON.EvaluationError) as error:
        raise EvaluationError(str(error)) from error
    config = _config(paths["cohort"])
    games = plan.games["test"]
    if len(games) != lock["cohort_games"]:
        raise EvaluationError("loaded temporal game count differs from the lock")
    _atomic_new(attempt_path, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_after_all_contract/model/source checks": True,
        "written_before_first_replay_open": True,
        "consumes_only_temporal_attempt": True,
        "temporal_lock_sha256": lock["lock_sha256"],
    })

    numerators = {name: 0.0 for name in nets}
    denominators = {name: 0.0 for name in nets}
    decisions = 0
    games_with_decisions = 0
    try:
        for game_index, game in enumerate(games, start=1):
            game_decisions = 0
            for sample in TRAIN.iter_game_samples(
                game, config, anchor=None, cache=None
            ):
                game_decisions += 1
                decisions += 1
                for name, net in nets.items():
                    logits, _ = net.forward(sample.features)
                    numerators[name] += sequence_nll(logits, sample) * sample.weight
                    denominators[name] += sample.weight
            games_with_decisions += int(game_decisions > 0)
            if not quiet and (game_index % 100 == 0 or game_index == len(games)):
                print(json.dumps({
                    "event": "temporal_progress",
                    "completed_games": game_index,
                    "total_games": len(games),
                    "target_st_main_decisions": decisions,
                }, sort_keys=True), flush=True)
    except (OSError, ValueError, FloatingPointError, TRAIN.TrainingError) as error:
        raise EvaluationError(
            f"temporal scoring failed after attempt consumption: {error}"
        ) from error
    if decisions <= 0 or any(value <= 0.0 for value in denominators.values()):
        raise EvaluationError("temporal cohort produced no target ST_MAIN decisions")
    if len(set(denominators.values())) != 1:
        raise EvaluationError("models did not receive identical temporal weights")
    objectives = {
        name: numerators[name] / denominators[name] for name in nets
    }
    verdict = temporal_decision(objectives)
    payload = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "temporal_test_opened_once": True,
        "selection_or_retraining_performed": False,
        "submission_authority": False,
        "temporal_lock_sha256": lock["lock_sha256"],
        "cohort_manifest_sha256": lock["cohort_manifest_sha256"],
        "cohort_games": len(games),
        "games_with_target_st_main": games_with_decisions,
        "target_st_main_decisions": decisions,
        "shared_weight_denominator": next(iter(denominators.values())),
        "metric": "ST_MAIN game-normalized policy objective",
        "objectives": objectives,
        "weighted_nll_numerators": numerators,
        "decision": verdict,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _atomic_new(result_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--attempt-out", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--result-out", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        lock, paths, _ = load_lock(args.lock)
        payload = evaluate(
            lock,
            paths,
            attempt_path=args.attempt_out,
            result_path=args.result_out,
            quiet=args.quiet,
        )
    except (EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "objectives": payload["objectives"],
        "decision": payload["decision"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
