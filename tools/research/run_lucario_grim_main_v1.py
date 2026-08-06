"""Run the locked Qu-v2B-anchored Lucario/Grimmsnarl ST_MAIN arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_lucario_grim_main_v1 as LOCKER  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


CACHE = LOCKER.RUN / "cache"


def run_arm(name: str, common: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    config = TRAIN.TrainingConfig(
        manifest_path=LOCKER.CORPUS,
        out_dir=LOCKER.RUN / name / "model",
        cache_dir=CACHE,
        epochs=int(arm["epochs"]),
        batch_size=int(common["batch_size"]),
        shuffle_buffer=4096,
        learning_rate=float(arm["learning_rate"]),
        weight_decay=float(common["weight_decay"]),
        value_coefficient=float(common["value_coefficient"]),
        gradient_clip=1.0,
        seed=int(arm["seed"]),
        device=str(common["device"]),
        win_weight=float(common["winner_weight"]),
        draw_weight=float(common["draw_weight"]),
        loss_weight=float(common["loss_weight"]),
        source_weights={},
        game_normalized=bool(common["game_normalized"]),
        qu_v2_anchor_checkpoint_path=LOCKER.PARENT,
        initial_checkpoint_path=LOCKER.PARENT,
        target_deck_sha256=LOCKER.TARGET_DECK_SHA256,
        target_select_type=int(common["target_select_type"]),
        freeze_public_backbone=bool(common["freeze_public_backbone"]),
        kl_coefficient=float(arm["kl_coefficient"]),
        kl_weighting=str(common["kl_weighting"]),
        matchup_card_id=int(common["matchup_card_id"]),
        matchup_weight=float(arm["matchup_weight"]),
        matchup_win_weight=float(arm["matchup_win_weight"]),
        matchup_draw_weight=float(arm["matchup_draw_weight"]),
        matchup_loss_weight=float(arm["matchup_loss_weight"]),
        require_gpu=bool(common["require_gpu"]),
        min_available_bytes=int(4.5 * TRAIN.training_preflight.GIB),
        min_gpu_free_bytes=int(5.0 * TRAIN.training_preflight.GIB),
        defer_test=True,
    )

    def event(event_name: str, payload: Mapping[str, Any]) -> None:
        if event_name in {"initial_checkpoint_loaded", "epoch_complete", "training_complete"}:
            print(json.dumps({"arm": name, "event": event_name, **payload}, sort_keys=True), flush=True)

    result = TRAIN.run_training(config, event_hook=event)
    return {
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(result["weights_path"]),
        "checkpoint": str(result["checkpoint_path"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(LOCKER.ARMS))
    args = parser.parse_args()
    lock = json.loads(LOCKER.OUTPUT.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if claimed != LOCKER.canonical(lock):
        raise SystemExit("training lock self-hash failed")
    lock["lock_sha256"] = claimed
    names = (args.arm,) if args.arm else tuple(LOCKER.ARMS)
    results = {
        name: run_arm(name, lock["common_training"], lock["arms"][name])
        for name in names
    }
    print(json.dumps({
        "lock_sha256": claimed,
        "results": results,
        "test_status": "deferred",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
