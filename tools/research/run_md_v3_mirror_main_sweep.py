"""Run the locked MD-v3 mirror-weighted ST_MAIN sweep without test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import training_preflight  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


class SweepError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _load_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SweepError(f"cannot load lock: {error}") from error
    recorded = payload.pop("lock_sha256", None)
    if (
        payload.get("schema") != "ptcg.md-v3.mirror-main-lock.v1"
        or recorded != json_sha256(payload)
    ):
        raise SweepError("invalid lock")
    payload["lock_sha256"] = recorded
    return payload


def _check_file(row: Mapping[str, Any], label: str) -> Path:
    path = Path(str(row.get("path", ""))).expanduser().resolve()
    if not path.is_file() or file_sha256(path) != row.get("sha256"):
        raise SweepError(f"{label} drifted")
    return path


def _config(
    lock: Mapping[str, Any],
    *,
    arm: Mapping[str, Any],
    manifest: Path,
    initial: Path,
    output: Path,
    cache: Path,
) -> TRAIN.TrainingConfig:
    protocol = lock["training"]
    latest = output / TRAIN.LATEST_NAME
    completed = output / TRAIN.WEIGHTS_NAME
    return TRAIN.TrainingConfig(
        manifest_path=manifest,
        out_dir=output,
        cache_dir=cache,
        resume_latest=latest.is_file() and not completed.exists(),
        epochs=int(protocol["epochs"]),
        batch_size=int(protocol["batch_size"]),
        shuffle_buffer=int(protocol["shuffle_buffer"]),
        learning_rate=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
        value_coefficient=0.0,
        gradient_clip=float(protocol["gradient_clip"]),
        seed=int(protocol["seed"]),
        device="cuda",
        embedding=16,
        board_hidden=48,
        state_hidden=160,
        option_hidden=112,
        context_hidden=80,
        win_weight=1.0,
        draw_weight=1.0,
        loss_weight=1.0,
        game_normalized=True,
        qu_v2_anchor_checkpoint_path=initial,
        initial_checkpoint_path=initial,
        target_deck_sha256=str(protocol["target_deck_sha256"]),
        target_select_type=0,
        freeze_public_backbone=True,
        kl_coefficient=float(protocol["kl_coefficient"]),
        kl_weighting=str(protocol["kl_weighting"]),
        matchup_card_id=int(protocol["matchup_card_id"]),
        matchup_weight=float(arm["matchup_weight"]),
        matchup_win_weight=float(protocol["matchup_outcome_weights"]["win"]),
        matchup_draw_weight=float(protocol["matchup_outcome_weights"]["draw"]),
        matchup_loss_weight=float(protocol["matchup_outcome_weights"]["loss"]),
        min_available_bytes=training_preflight.gib(5.5),
        min_swap_free_bytes=training_preflight.gib(4.0),
        require_gpu=True,
        min_gpu_free_bytes=training_preflight.gib(5.5),
        defer_test=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()
    try:
        lock_path = args.lock.expanduser().resolve()
        lock = _load_lock(lock_path)
        manifest = _check_file(lock["artifacts"]["corpus"], "corpus")
        _check_file(lock["artifacts"]["mass_audit"], "mass audit")
        initial = _check_file(
            lock["artifacts"]["frozen_md_v3_main_checkpoint"],
            "frozen MD-v3 main checkpoint",
        )
        _check_file(lock["artifacts"]["trainer"], "trainer")
        run = lock_path.parent
        cache = run / "cache"
        summaries = []
        for arm_name in lock["training"]["execution_order"]:
            arm = lock["training"]["arms"][arm_name]
            output = run / f"model-{arm_name}"
            weights = output / TRAIN.WEIGHTS_NAME
            provenance = output / TRAIN.PROVENANCE_NAME
            if weights.is_file() and provenance.is_file():
                summaries.append({
                    "arm": arm_name,
                    "status": "already_complete",
                    "weights_sha256": file_sha256(weights),
                })
                continue
            result = TRAIN.run_training(_config(
                lock,
                arm=arm,
                manifest=manifest,
                initial=initial,
                output=output,
                cache=cache,
            ))
            summaries.append({
                "arm": arm_name,
                "status": "trained",
                "best_epoch": result["best_epoch"],
                "best_validation_objective": (
                    result["best_validation_objective"]),
                "weights_sha256": file_sha256(result["weights_path"]),
            })
            print(json.dumps(summaries[-1], sort_keys=True), flush=True)
    except (OSError, KeyError, TypeError, ValueError, SweepError,
            TRAIN.TrainingError) as error:
        parser.error(str(error))
    print(json.dumps({"arms": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
