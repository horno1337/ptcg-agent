"""Evaluate the one locked MD-v4 TF32 NumPy parity salvage candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_md_v4_runtime_parity_salvage as LOCK  # noqa: E402
from tools.research import md_v4_model as MODEL  # noqa: E402
from tools.research import md_v4_numpy_salvage as SALVAGE  # noqa: E402
from tools.research import train_md_v4 as TRAIN  # noqa: E402


RESULT_SCHEMA = "ptcg.md-v4.runtime-parity-salvage-result.v1"
OUTPUT = (
    LOCK.RUN / "model/runtime-parity-salvage-result.json"
)


class SalvageEvaluationError(RuntimeError):
    """The locked parity salvage evaluation cannot complete."""


def _result_hash(payload: Mapping[str, Any]) -> str:
    return LOCK.value_sha256(dict(payload))


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    result = dict(payload)
    result["result_sha256"] = _result_hash(result)
    if path.exists():
        raise SalvageEvaluationError(
            f"refusing to replace salvage result: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(LOCK.canonical_json(result) + b"\n")


def _load_candidate(
    device: torch.device,
) -> tuple[MODEL.TorchMDV4, Mapping[str, Any]]:
    parent = TRAIN._load_parent(
        TRAIN.TrainingConfig(device="cuda"), device
    )
    net = TRAIN._make_candidate(parent, device)
    try:
        checkpoint = torch.load(
            LOCK.RECOVERY, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise SalvageEvaluationError(
            f"cannot load frozen recovery: {error}"
        ) from error
    state = checkpoint.get("state_dict")
    if (
        checkpoint.get("state_dict_sha256")
            != LOCK.RECOVERY_STATE_SHA256
        or TRAIN._parameter_state_sha256(state)
            != LOCK.RECOVERY_STATE_SHA256
    ):
        raise SalvageEvaluationError("frozen recovery state drifted")
    try:
        net.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as error:
        raise SalvageEvaluationError(
            f"cannot restore frozen epoch-4 state: {error}"
        ) from error
    net.eval()
    if (
        MODEL.frozen_parent_state_sha256(net)
        != LOCK.FROZEN_PARENT_STATE_SHA256
    ):
        raise SalvageEvaluationError("frozen parent state drifted")
    return net, checkpoint


def _parity_population(
    net: MODEL.TorchMDV4,
    numpy_net: SALVAGE.NumpyMDV4ParitySalvage,
    samples,
    device: torch.device,
    *,
    expected_callbacks: int,
    expected_games: int,
    progress_every_games: int = 100,
) -> dict[str, Any]:
    callbacks = 0
    games: set[str] = set()
    max_logit_delta = 0.0
    max_value_delta = 0.0
    numeric_failures = 0
    decoded_action_mismatches = 0
    last_reported_games = 0
    net.eval()
    with torch.no_grad():
        for minibatch in TRAIN._batches(samples):
            torch_logits, torch_values = net(
                MODEL.collate(
                    [sample.features for sample in minibatch],
                    device,
                )
            )
            for row, sample in enumerate(minibatch):
                option_count = len(sample.features.option_ids)
                reference = torch_logits[
                    row, :option_count
                ].detach().cpu().numpy()
                reference_value = float(
                    torch_values[row].detach().cpu()
                )
                candidate, candidate_value = numpy_net.forward(
                    sample.features
                )
                if (
                    not np.isfinite(reference).all()
                    or not np.isfinite(candidate).all()
                    or not math.isfinite(reference_value)
                    or not math.isfinite(candidate_value)
                ):
                    raise SalvageEvaluationError(
                        "parity population produced non-finite output"
                    )
                logit_delta = float(
                    np.max(np.abs(reference - candidate))
                )
                value_delta = abs(
                    reference_value - float(candidate_value)
                )
                max_logit_delta = max(
                    max_logit_delta, logit_delta
                )
                max_value_delta = max(
                    max_value_delta, value_delta
                )
                if (
                    not np.allclose(
                        reference,
                        candidate,
                        atol=3e-5,
                        rtol=1e-5,
                    )
                    or value_delta >= 2e-5
                ):
                    numeric_failures += 1
                if (
                    TRAIN._decoded_action(reference, sample)
                    != TRAIN._decoded_action(candidate, sample)
                ):
                    decoded_action_mismatches += 1
                callbacks += 1
                games.add(sample.game_uid)
            if (
                len(games) - last_reported_games
                >= progress_every_games
                or callbacks == expected_callbacks
            ):
                print(
                    "parity progress "
                    f"games={len(games)}/{expected_games} "
                    f"callbacks={callbacks}/{expected_callbacks} "
                    f"numeric_failures={numeric_failures}",
                    flush=True,
                )
                last_reported_games = len(games)
    population_pass = (
        callbacks == expected_callbacks
        and len(games) == expected_games
    )
    return {
        "callbacks": callbacks,
        "games": len(games),
        "expected_callbacks": expected_callbacks,
        "expected_games": expected_games,
        "population_passed": population_pass,
        "maximum_absolute_logit_delta": max_logit_delta,
        "maximum_absolute_value_delta": max_value_delta,
        "numeric_failures": numeric_failures,
        "decoded_action_mismatches":
            decoded_action_mismatches,
        "logit_atol": 3e-5,
        "logit_rtol": 1e-5,
        "value_atol_strict": 2e-5,
        "passed": (
            population_pass
            and numeric_failures == 0
            and decoded_action_mismatches == 0
        ),
    }


def evaluate(
    lock_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    lock = LOCK.load_lock(lock_path, verify_artifacts=True)
    reference = lock["reference_execution"]
    if (
        not torch.cuda.is_available()
        or torch.__version__ != reference["torch_version"]
        or torch.version.cuda != reference["cuda_version"]
        or torch.backends.cudnn.version()
            != reference["cudnn_version"]
        or bool(torch.backends.cudnn.enabled)
            is not reference["cudnn_enabled"]
        or bool(torch.backends.cudnn.allow_tf32)
            is not reference["cudnn_allow_tf32"]
        or bool(torch.backends.cuda.matmul.allow_tf32)
            is not reference["cuda_matmul_allow_tf32"]
    ):
        raise SalvageEvaluationError(
            "locked CUDA/cuDNN reference environment drifted"
        )
    TRAIN._seed_everything()
    device = torch.device("cuda")
    config = TRAIN.TrainingConfig(
        lock_path=TRAIN.TRAINING_LOCK_PATH,
        device="cuda",
    )
    TRAIN._validate_config(config)
    plan = TRAIN.load_locked_corpus(config)
    cache = TRAIN.create_thin_cache(config, plan)
    _, training_lock_sha256 = TRAIN.load_training_lock(
        config, plan, cache
    )
    if (
        training_lock_sha256
        != LOCK.ORIGINAL_TRAINING_LOCK_SHA256
    ):
        raise SalvageEvaluationError("original training lock drifted")
    base_index = TRAIN.index_base_cache(config, plan)
    materialization, records = (
        TRAIN._load_materialization_payload(
            cache, plan, training_lock_sha256
        )
    )
    summary = materialization.get("summary", {}).get(
        "by_split", {}
    ).get("validation", {})
    expected_callbacks = lock["parity_gate"]["callbacks"]
    expected_games = lock["parity_gate"]["games"]
    if (
        summary.get("target_decisions") != expected_callbacks
        or summary.get("games") != expected_games
    ):
        raise SalvageEvaluationError(
            "locked validation population drifted"
        )
    net, checkpoint = _load_candidate(device)
    exported = MODEL.export_numpy_weights(net)
    numpy_net = SALVAGE.NumpyMDV4ParitySalvage(exported)
    parity = _parity_population(
        net,
        numpy_net,
        TRAIN.iter_split_samples(
            config,
            cache,
            base_index,
            records,
            training_lock_sha256,
            "validation",
            epoch=None,
        ),
        device,
        expected_callbacks=expected_callbacks,
        expected_games=expected_games,
    )
    offline = None
    if parity["passed"]:
        print("full parity passed; opening unchanged offline gate", flush=True)
        final_validation = TRAIN._run_split(
            net,
            TRAIN.iter_split_samples(
                config,
                cache,
                base_index,
                records,
                training_lock_sha256,
                "validation",
                epoch=None,
            ),
            device,
            None,
            expected_samples=expected_callbacks,
            expected_games=expected_games,
        )
        offline = TRAIN._offline_gate_report(
            checkpoint["history"],
            final_validation,
            net,
            checkpoint["initialization"],
        )
    passed = bool(
        parity["passed"]
        and isinstance(offline, Mapping)
        and offline.get("passed") is True
    )
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": lock["candidate"]["name"],
        "lock": {
            "path": str(lock_path.resolve()),
            "file_sha256": LOCK.file_sha256(lock_path),
            "lock_sha256": lock["lock_sha256"],
        },
        "recovery": lock["candidate"]["recovery"],
        "exported_array_mapping_sha256":
            MODEL._mapping_sha256(exported),
        "torch_numpy_parity": parity,
        "offline_rejection_gates": offline,
        "temporal_archive_opened": False,
        "passed": passed,
        "promotion_authority": False,
        "upload_authority": False,
    }
    _write_result(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=LOCK.OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = evaluate(
        args.lock.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    print(json.dumps({
        "passed": result["passed"],
        "parity": result["torch_numpy_parity"],
        "offline_rejection_gates":
            result["offline_rejection_gates"],
        "output": str(args.output.expanduser().resolve()),
    }, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
