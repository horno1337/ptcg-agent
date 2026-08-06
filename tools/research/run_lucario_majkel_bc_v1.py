"""Run the locked exact-list Majkel1337 Lucario BC experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_lucario_majkel_bc_v1 as LOCKER  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RESOURCE_OVERRIDE = LOCKER.RUN / "resource-override.json"


def _run_head(name: str, row: Mapping[str, Any], common: Mapping[str, Any]) -> dict[str, Any]:
    parent = LOCKER.MAIN_PARENT if row["parent"] == "main_parent" else LOCKER.CARD_PARENT
    output = LOCKER.RUN / name / "model"
    cache = LOCKER.RUN / name / "cache"
    config = TRAIN.TrainingConfig(
        manifest_path=LOCKER.CORPUS,
        out_dir=output,
        cache_dir=cache,
        epochs=int(common["epochs"]),
        batch_size=int(common["batch_size"]),
        shuffle_buffer=4096,
        learning_rate=float(common["learning_rate"]),
        weight_decay=float(common["weight_decay"]),
        value_coefficient=float(common["value_coefficient"]),
        gradient_clip=1.0,
        seed=int(row["seed"]),
        device=str(common["device"]),
        win_weight=float(common["winner_weight"]),
        draw_weight=float(common["draw_weight"]),
        loss_weight=float(common["loss_weight"]),
        source_weights={},
        game_normalized=bool(common["game_normalized"]),
        qu_v2_anchor_checkpoint_path=parent,
        initial_checkpoint_path=parent,
        target_deck_sha256=LOCKER.TARGET_DECK_SHA256,
        target_select_type=int(row["select_type"]),
        freeze_public_backbone=bool(common["freeze_public_backbone"]),
        kl_coefficient=float(common["kl_coefficient"]),
        kl_weighting=str(common["kl_weighting"]),
        require_gpu=bool(common["require_gpu"]),
        min_available_bytes=int(4.5 * TRAIN.training_preflight.GIB),
        min_gpu_free_bytes=int(5.0 * TRAIN.training_preflight.GIB),
        defer_test=True,
    )

    def event(event_name: str, payload: Mapping[str, Any]) -> None:
        if event_name in {"initial_checkpoint_loaded", "epoch_complete", "training_complete"}:
            print(json.dumps({"head": name, "event": event_name, **payload}, sort_keys=True), flush=True)

    result = TRAIN.run_training(config, event_hook=event)
    return {
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(result["weights_path"]),
        "checkpoint": str(result["checkpoint_path"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=("main", "card"))
    args = parser.parse_args()
    lock = json.loads(LOCKER.OUTPUT.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if claimed != LOCKER.canonical(lock):
        raise SystemExit("training lock self-hash failed")
    lock["lock_sha256"] = claimed
    override = json.loads(RESOURCE_OVERRIDE.read_text(encoding="utf-8"))
    if (
        override.get("schema") != "ptcg.lucario-majkel.bc-v1.resource-override.v1"
        or override.get("training_lock_sha256") != claimed
        or override.get("change", {}).get("original_min_available_gib") != 6.0
        or override.get("change", {}).get("override_min_available_gib") != 4.5
        or override.get("scientific_configuration_changed") is not False
        or override.get("scientific_outcomes_seen_before_override") is not False
    ):
        raise SystemExit("resource override contract failed")
    training = lock["training"]
    results = {}
    heads = (args.head,) if args.head else ("main", "card")
    for name in heads:
        results[name] = _run_head(name, training["heads"][name], training)
    print(json.dumps({
        "lock_sha256": claimed,
        "results": results,
        "test_status": "deferred",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
