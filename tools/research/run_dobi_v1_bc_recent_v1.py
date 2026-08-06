"""Run the prospectively locked conservative Dobi-v1 recent-field BC."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import lock_dobi_v1_bc_recent_v1 as LOCKER  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = LOCKER.RUN
OUT = RUN / "model"
CACHE = RUN / "cache"
RESOURCE_OVERRIDE = RUN / "resource-override.json"


def main() -> int:
    lock = json.loads(LOCKER.OUTPUT.read_text(encoding="utf-8"))
    claimed = lock.pop("lock_sha256", None)
    if LOCKER.canonical(lock) != claimed:
        raise SystemExit("training lock self-hash failed")
    resource_override = json.loads(RESOURCE_OVERRIDE.read_text(encoding="utf-8"))
    if (
        resource_override.get("schema")
        != "ptcg.dobi-v1.bc-recent-v1.resource-override.v1"
        or resource_override.get("user_authorized") is not True
        or resource_override.get("training_lock_sha256") != claimed
        or resource_override.get("change", {}).get("original_gib") != 6.0
        or resource_override.get("change", {}).get("override_gib") != 5.5
        or resource_override.get("scientific_outcomes_seen_before_override") is not False
    ):
        raise SystemExit("resource override contract failed")
    corpus = json.loads(LOCKER.CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise SystemExit("training corpus self-hash failed")
    config_row = lock["training"]
    source_weights = {"mirror_kl": 0.0}
    source_weights.update({
        row["label"]: float(config_row["crustle_multiplier"])
        for row in corpus["sources"]
        if isinstance(row.get("label"), str) and "_crustle_" in row["label"]
    })
    config = TRAIN.TrainingConfig(
        manifest_path=LOCKER.CORPUS,
        out_dir=OUT,
        cache_dir=CACHE,
        epochs=int(config_row["epochs"]),
        batch_size=int(config_row["batch_size"]),
        shuffle_buffer=4096,
        learning_rate=float(config_row["learning_rate"]),
        weight_decay=float(config_row["weight_decay"]),
        value_coefficient=float(config_row["value_coefficient"]),
        gradient_clip=1.0,
        seed=int(config_row["seed"]),
        device=str(config_row["device"]),
        win_weight=float(config_row["winner_weight"]),
        draw_weight=float(config_row["draw_weight"]),
        loss_weight=float(config_row["sampled_loss_weight"]),
        source_weights=source_weights,
        game_normalized=bool(config_row["game_normalized"]),
        qu_v2_anchor_checkpoint_path=LOCKER.PARENT,
        initial_checkpoint_path=LOCKER.PARENT,
        target_deck_sha256=str(config_row["target_deck_sha256"]),
        target_select_type=int(config_row["target_select_type"]),
        freeze_public_backbone=bool(config_row["freeze_public_backbone"]),
        kl_coefficient=float(config_row["kl_coefficient"]),
        kl_weighting=str(config_row["kl_weighting"]),
        require_gpu=bool(config_row["require_gpu"]),
        min_gpu_free_bytes=int(
            float(resource_override["change"]["override_gib"])
            * TRAIN.training_preflight.GIB
        ),
        defer_test=True,
    )

    def event(name: str, payload: Mapping[str, Any]) -> None:
        if name in {"initial_checkpoint_loaded", "epoch_complete", "training_complete"}:
            print(json.dumps({"event": name, **payload}, sort_keys=True), flush=True)

    result = TRAIN.run_training(config, event_hook=event)
    print(json.dumps({
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(result["weights_path"]),
        "checkpoint": str(result["checkpoint_path"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
