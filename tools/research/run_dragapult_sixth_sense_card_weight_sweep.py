"""Locked CARD-only 0.15-versus-0.60 loss-weight screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import train_qu_v2a as TRAIN  # noqa: E402
from tools.research.run_dragapult_sixth_sense_weight_sweep import (  # noqa: E402
    ARMS, TARGET_SHA, canonical, file_sha,
)


SOURCE_RUN = ROOT / "tools/checkpoints/dragapult-sixth-sense-weighted-bc-20260812"
RUN = ROOT / "tools/checkpoints/dragapult-sixth-sense-card-weighted-bc-20260812"
CORPUS = SOURCE_RUN / "old-corpus.json"
NEWER = SOURCE_RUN / "newer-corpus-v2.json"
LOCK = RUN / "training-lock.json"
PARENT = (ROOT / "tools/checkpoints/elite-recent-specialist-bc-20260811/"
          "candidates/dragapult-card/model/candidate-qu-v2a-checkpoint.pt")


def create_lock() -> dict:
    RUN.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        raise RuntimeError(f"refusing to overwrite {LOCK}")
    payload = {
        "schema": "ptcg.dragapult-sixth-sense-card-loss-weight-screen.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "artifacts": {
            "training_corpus": {"path": str(CORPUS.resolve()), "sha256": file_sha(CORPUS)},
            "newer_cohort": {"path": str(NEWER.resolve()), "sha256": file_sha(NEWER)},
            "elite_card_parent": {"path": str(PARENT.resolve()), "sha256": file_sha(PARENT)},
            "trainer": {"path": str(Path(TRAIN.__file__).resolve()),
                        "sha256": file_sha(Path(TRAIN.__file__).resolve())},
        },
        "training": {
            "target_deck_sha256": TARGET_SHA, "select_type": 1,
            "arms": ARMS, "win_weight": 1.0, "draw_weight": 0.3,
            "epochs": 4, "learning_rate": 3e-5, "kl": 0.3,
            "freeze_public_backbone": True, "game_normalized": True,
            "seed": 2026081233, "test_deferred": True,
        },
        "selection": {
            "primary": "newer-cohort winner CARD NLL",
            "secondary": "newer-cohort winner exact agreement",
            "guardrail": "no regression on Lucario Phantom secure-Prize roots",
            "gameplay": "exact-07bed paired field screen required",
        },
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    LOCK.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_lock() -> dict:
    value = json.loads(LOCK.read_text()); claimed = value.pop("lock_sha256", None)
    if claimed != canonical(value):
        raise RuntimeError("CARD lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        if file_sha(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"artifact drifted: {row['path']}")
    return value


def train(name: str, lock: dict) -> dict:
    output = RUN / f"candidates/{name}/model"
    cfg = TRAIN.TrainingConfig(
        manifest_path=CORPUS, out_dir=output, cache_dir=RUN / "cache",
        epochs=4, batch_size=128, shuffle_buffer=4096,
        learning_rate=3e-5, weight_decay=1e-5, value_coefficient=0.0,
        seed=int(lock["training"]["seed"]), device="cpu",
        win_weight=1.0, draw_weight=0.3, loss_weight=ARMS[name],
        game_normalized=True, qu_v2_anchor_checkpoint_path=PARENT,
        initial_checkpoint_path=PARENT, target_deck_sha256=TARGET_SHA,
        target_select_type=1, freeze_public_backbone=True,
        kl_coefficient=0.3, kl_weighting="uniform-game", require_gpu=False,
        min_available_bytes=2 * TRAIN.training_preflight.GIB,
        min_swap_free_bytes=1 * TRAIN.training_preflight.GIB,
        defer_test=True,
    )
    result = TRAIN.run_training(cfg)
    row = {"arm": name, "loss_weight": ARMS[name],
           "best_epoch": result["best_epoch"],
           "best_validation_objective": result["best_validation_objective"],
           "weights": str(result["weights_path"])}
    (RUN / f"candidates/{name}/result.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    args = parser.parse_args()
    lock = create_lock() if not LOCK.exists() else load_lock()
    print(json.dumps({"lock_sha256": lock["lock_sha256"]}))
    if not args.lock_only:
        print(json.dumps([train(name, lock) for name in ARMS], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
