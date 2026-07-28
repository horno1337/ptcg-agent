"""Score fixed MD-v2, MD-v1, and Qu-v2B once on the July 28 cohort."""

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
from tools.research import eval_md_v2_allthrough26_gameplay as GAMEPLAY  # noqa: E402
from tools.research import eval_md_v2_july27_temporal as CORE  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


DATE = "2026-07-28"
LOCK_SCHEMA = "ptcg.md-v2.july28-temporal-lock.v1"
RESULT_SCHEMA = "ptcg.md-v2.july28-temporal-result.v1"
ATTEMPT_SCHEMA = "ptcg.md-v2.july28-temporal-attempt.v1"
PREREG_SCHEMA = "ptcg.md-v2.july28-temporal-preregistration.v1"
RUN = ROOT / "tools/checkpoints/md-v2-allthrough26"
DEFAULT_LOCK = RUN / "july28-temporal-lock.json"
DEFAULT_ATTEMPT = RUN / "july28-temporal-attempt.json"
DEFAULT_RESULT = RUN / "july28-temporal-result.json"


class EvaluationError(RuntimeError):
    """The one-shot July 28 contract or score stream failed closed."""


def _paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    records = lock.get("artifacts")
    if not isinstance(records, Mapping):
        raise EvaluationError("temporal lock has no artifact map")
    paths: dict[str, Path] = {}
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
        paths[label] = path
    required = {
        "preregistration",
        "gameplay_lock",
        "raw_index",
        "cohort",
        "candidate_weights",
        "md_v1_weights",
        "qu_v2b_weights",
        "evaluator",
        "preparer",
        "temporal_core",
        "trainer",
        "indexer",
        "dataset_loader",
    }
    missing = sorted(required - paths.keys())
    if missing:
        raise EvaluationError(f"temporal lock omits artifacts: {missing}")
    if paths["evaluator"] != Path(__file__).resolve():
        raise EvaluationError("temporal lock names a different evaluator")
    return paths


def load_preregistration(path: Path) -> dict[str, Any]:
    try:
        prereg = COMMON.load_self_hashed_json(
            path.expanduser().resolve(),
            schema=PREREG_SCHEMA,
            hash_key="preregistration_sha256",
        )
    except COMMON.EvaluationError as error:
        raise EvaluationError(str(error)) from error
    protocol = prereg.get("protocol")
    records = prereg.get("artifacts")
    if (
        not isinstance(protocol, Mapping)
        or not isinstance(records, Mapping)
        or protocol.get("date") != DATE
        or protocol.get("team_name_prefilter") is not False
        or protocol.get("one_shot") is not True
        or protocol.get("no_retraining_or_reselection_after_open") is not True
        or protocol.get("pass_rule")
            != "MD-v2 objective is strictly lower than both MD-v1 and Qu-v2B"
    ):
        raise EvaluationError("July 28 preregistration protocol drifted")
    for label, record in records.items():
        if not isinstance(record, Mapping):
            raise EvaluationError("invalid preregistration artifact record")
        resolved = COMMON.resolve_recorded_path(record.get("path"))
        if (
            not resolved.is_file()
            or COMMON.file_sha256(resolved) != record.get("sha256")
        ):
            raise EvaluationError(f"preregistration artifact drift: {label}")
    return prereg


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
        or protocol.get("date") != DATE
        or protocol.get("team_name_prefilter") is not False
        or protocol.get("same_decision_stream_for_all_models") is not True
        or protocol.get("one_shot") is not True
        or protocol.get("no_retraining_or_reselection_after_open") is not True
        or protocol.get("pass_rule")
            != "MD-v2 objective is strictly lower than both MD-v1 and Qu-v2B"
    ):
        raise EvaluationError("temporal protocol differs from preregistration")
    paths = _paths(lock)
    prereg = load_preregistration(paths["preregistration"])
    if (
        prereg.get("preregistration_sha256")
            != lock.get("preregistration_sha256")
    ):
        raise EvaluationError("temporal lock is not bound to the preregistration")
    try:
        gameplay_lock, _ = GAMEPLAY.load_lock(paths["gameplay_lock"])
        raw_index = json.loads(paths["raw_index"].read_text(encoding="utf-8"))
        cohort = json.loads(paths["cohort"].read_text(encoding="utf-8"))
    except (
        GAMEPLAY.EvaluationError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise EvaluationError(
            f"cannot verify temporal parent artifacts: {error}"
        ) from error
    binding = cohort.get("md_v2_daily_temporal")
    if (
        gameplay_lock.get("lock_sha256") != lock.get("gameplay_lock_sha256")
        or gameplay_lock.get("candidate", {}).get("weights_sha256")
            != lock.get("candidate_weights_sha256")
        or raw_index.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(raw_index)
        or raw_index.get("manifest_sha256")
            != lock.get("raw_index_manifest_sha256")
        or cohort.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(cohort)
        or cohort.get("manifest_sha256") != lock.get("cohort_manifest_sha256")
        or len(cohort.get("games", [])) != lock.get("cohort_games")
        or not isinstance(binding, Mapping)
        or binding.get("date") != DATE
        or binding.get("team_name_prefilter") is not False
        or binding.get("same_cohort_for_all_models") is not True
    ):
        raise EvaluationError("temporal parent/cohort binding drifted")
    from tools.research import prepare_md_v2_july28_temporal as PREP

    if (
        PREP.inventory(Path(str(lock["source_inventory"]["source"])))
        != lock["source_inventory"]
    ):
        raise EvaluationError("official July 28 source inventory drifted")
    return lock, paths, cohort


def evaluate(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    attempt_path: Path,
    result_path: Path,
    quiet: bool,
) -> dict[str, Any]:
    if attempt_path.exists() or result_path.exists():
        raise EvaluationError("the one-shot July 28 attempt is already consumed")
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
    config = TRAIN.TrainingConfig(
        manifest_path=paths["cohort"],
        out_dir=RUN / "july28-temporal-no-training",
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
    games = plan.games["test"]
    if len(games) != lock["cohort_games"]:
        raise EvaluationError("loaded temporal game count differs from the lock")
    try:
        CORE._atomic_new(attempt_path, {
            "schema": ATTEMPT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "written_after_all_contract/model/source checks": True,
            "written_before_first_replay_open": True,
            "consumes_only_temporal_attempt": True,
            "temporal_lock_sha256": lock["lock_sha256"],
        })
    except CORE.EvaluationError as error:
        raise EvaluationError(str(error)) from error

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
                runtime_sample = CORE.runtime_features(sample.features)
                for name, net in nets.items():
                    logits, _ = net.forward(runtime_sample)
                    numerators[name] += (
                        CORE.sequence_nll(logits, sample) * sample.weight
                    )
                    denominators[name] += sample.weight
            games_with_decisions += int(game_decisions > 0)
            if not quiet and (game_index % 100 == 0 or game_index == len(games)):
                print(json.dumps({
                    "event": "temporal_progress",
                    "completed_games": game_index,
                    "total_games": len(games),
                    "target_st_main_decisions": decisions,
                }, sort_keys=True), flush=True)
    except (
        CORE.EvaluationError,
        OSError,
        ValueError,
        FloatingPointError,
        TRAIN.TrainingError,
    ) as error:
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
    if not all(math.isfinite(value) for value in objectives.values()):
        raise EvaluationError("temporal objectives are not finite")
    verdict = CORE.temporal_decision(objectives)
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
    try:
        CORE._atomic_new(result_path, payload)
    except CORE.EvaluationError as error:
        raise EvaluationError(str(error)) from error
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
