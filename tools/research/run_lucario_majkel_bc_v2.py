"""Run conditionally activated conservative joint Lucario-v2 BC."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_lucario_majkel_bc_v2 as LOCKER  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


OUT = LOCKER.RUN / "model"
CACHE = LOCKER.RUN / "cache"
V1_ISOLATION = LOCKER.V1 / "head-isolation"


def _activation_check() -> dict[str, Any]:
    results = {}
    for head in ("main", "card"):
        path = V1_ISOLATION / head / "result.json"
        if not path.is_file():
            raise SystemExit(f"conditional activation unresolved: {path} missing")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("decision", {}).get("valid") is not True:
            raise SystemExit(f"conditional activation unresolved: {head} result invalid")
        results[head] = value["decision"]
    if any(row.get("diagnostic_positive") is True for row in results.values()):
        raise SystemExit("Lucario-v2 activation condition not met: an isolated v1 head was positive")
    return results


def main() -> int:
    lock = json.loads(LOCKER.OUTPUT.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if claimed != LOCKER.canonical(lock):
        raise SystemExit("training lock self-hash failed")
    lock["lock_sha256"] = claimed
    activation = _activation_check()
    row = lock["training"]
    config = TRAIN.TrainingConfig(
        manifest_path=LOCKER.CORPUS,
        out_dir=OUT,
        cache_dir=CACHE,
        epochs=int(row["epochs"]),
        batch_size=int(row["batch_size"]),
        shuffle_buffer=4096,
        learning_rate=float(row["learning_rate"]),
        weight_decay=float(row["weight_decay"]),
        value_coefficient=float(row["value_coefficient"]),
        gradient_clip=1.0,
        seed=int(row["seed"]),
        device=str(row["device"]),
        win_weight=float(row["winner_weight"]),
        draw_weight=float(row["draw_weight"]),
        loss_weight=float(row["loss_weight"]),
        source_weights={},
        game_normalized=bool(row["game_normalized"]),
        qu_v2_anchor_checkpoint_path=LOCKER.PARENT,
        initial_checkpoint_path=LOCKER.PARENT,
        target_deck_sha256=LOCKER.TARGET_DECK_SHA256,
        target_select_type=None,
        freeze_public_backbone=bool(row["freeze_public_backbone"]),
        kl_coefficient=float(row["kl_coefficient"]),
        kl_weighting=str(row["kl_weighting"]),
        require_gpu=bool(row["require_gpu"]),
        min_available_bytes=int(4.5 * TRAIN.training_preflight.GIB),
        min_gpu_free_bytes=int(5.0 * TRAIN.training_preflight.GIB),
        defer_test=True,
    )

    def event(name: str, payload: Mapping[str, Any]) -> None:
        if name in {"initial_checkpoint_loaded", "epoch_complete", "training_complete"}:
            print(json.dumps({"event": name, **payload}, sort_keys=True), flush=True)

    result = TRAIN.run_training(config, event_hook=event)
    print(json.dumps({
        "lock_sha256": claimed,
        "activation_evidence": activation,
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(result["weights_path"]),
        "checkpoint": str(result["checkpoint_path"]),
        "test_status": "deferred",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
