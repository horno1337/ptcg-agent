"""Run the two prospectively locked Festival Lead BC heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_festival_lead_bc_v1 as LOCK  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402

RESOURCE_OVERRIDE = LOCK.RUN / "resource-override.json"


def run_head(name: str, row: Mapping[str, Any], common: Mapping[str, Any]):
    output = LOCK.RUN / name / "model"
    config = TRAIN.TrainingConfig(
        manifest_path=LOCK.CORPUS,
        out_dir=output,
        cache_dir=LOCK.RUN / name / "cache",
        epochs=int(common["epochs"]), batch_size=int(common["batch_size"]),
        shuffle_buffer=2048,
        learning_rate=float(common["learning_rate"]),
        weight_decay=float(common["weight_decay"]),
        value_coefficient=float(common["value_coefficient"]),
        gradient_clip=1.0, seed=int(row["seed"]), device=str(common["device"]),
        win_weight=float(common["winner_weight"]),
        draw_weight=float(common["draw_weight"]),
        loss_weight=float(common["loss_weight"]), source_weights={},
        game_normalized=bool(common["game_normalized"]),
        qu_v2_anchor_checkpoint_path=LOCK.PARENT,
        initial_checkpoint_path=LOCK.PARENT,
        target_deck_sha256=LOCK.TARGET_DECK_SHA256,
        target_select_type=int(row["select_type"]),
        freeze_public_backbone=bool(common["freeze_public_backbone"]),
        kl_coefficient=float(common["kl_coefficient"]),
        kl_weighting=str(common["kl_weighting"]),
        require_gpu=False,
        min_available_bytes=int(1.0 * TRAIN.training_preflight.GIB),
        min_swap_free_bytes=int(2.0 * TRAIN.training_preflight.GIB),
        min_gpu_free_bytes=0,
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
    lock = json.loads(LOCK.OUTPUT.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if claimed != LOCK.canonical(lock):
        raise SystemExit("training lock self-hash failed")
    lock["lock_sha256"] = claimed
    override = json.loads(RESOURCE_OVERRIDE.read_text(encoding="utf-8"))
    if (
        override.get("schema") != "ptcg.festival-lead.bc-v1.resource-override.v1"
        or override.get("training_lock_sha256") != claimed
        or override.get("change") != {
            "original_min_swap_free_gib": 4.0,
            "override_min_swap_free_gib": 2.0,
        }
        or override.get("scientific_configuration_changed") is not False
        or override.get("scientific_outcomes_seen_before_override") is not False
    ):
        raise SystemExit("resource override contract failed")
    heads = (args.head,) if args.head else ("main", "card")
    results = {
        name: run_head(name, lock["training"]["heads"][name], lock["training"])
        for name in heads
    }
    print(json.dumps({"lock_sha256": claimed, "results": results,
                      "test_status": "deferred"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
