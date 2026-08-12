"""Locked MAIN-only BC loss-weight screen on Sixth Sense Dragapult replays.

The screen compares the historical 0.15 losing-game weight with the proposed
0.60 weight.  It starts from the deployed elite MAIN checkpoint, freezes the
public backbone, and trains only on the older submission 55439076 cohort.
The newer 55457370 cohort is not used for fitting or checkpoint selection.
No arm is deployable without a separate exact-07bed gameplay gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import train_qu_v2a as TRAIN  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-sixth-sense-weighted-bc-20260812"
CORPUS = RUN / "old-corpus.json"
NEWER = RUN / "newer-corpus.json"
LOCK = RUN / "training-lock.json"
PARENT = (ROOT / "tools/checkpoints/elite-recent-specialist-bc-20260811/"
          "candidates/dragapult-main/model/candidate-qu-v2a-checkpoint.pt")
TARGET_SHA = "674ec3100a65337e91ee11325b087e25925f312d28c8ec5a49969220d6b9d5b8"
ARMS = {"loss015": 0.15, "loss060": 0.60}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def create_lock() -> dict[str, Any]:
    if LOCK.exists():
        raise RuntimeError(f"refusing to overwrite {LOCK}")
    for path in (CORPUS, NEWER, PARENT):
        if not path.is_file():
            raise RuntimeError(f"missing artifact: {path}")
    payload = {
        "schema": "ptcg.dragapult-sixth-sense-loss-weight-screen.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "Retaining losing-game plans at weight 0.60 improves temporal "
            "teacher imitation over the historical 0.15 weight without "
            "amplifying utility-over-attack route errors."
        ),
        "artifacts": {
            "training_corpus": {"path": str(CORPUS.resolve()), "sha256": file_sha(CORPUS)},
            "newer_external_cohort": {"path": str(NEWER.resolve()), "sha256": file_sha(NEWER)},
            "elite_main_parent": {"path": str(PARENT.resolve()), "sha256": file_sha(PARENT)},
            "trainer": {"path": str(Path(TRAIN.__file__).resolve()),
                        "sha256": file_sha(Path(TRAIN.__file__).resolve())},
        },
        "training": {
            "target_deck_sha256": TARGET_SHA,
            "select_type": 0,
            "arms": ARMS,
            "win_weight": 1.0,
            "draw_weight": 0.3,
            "game_normalized": True,
            "epochs": 4,
            "learning_rate": 3e-5,
            "weight_decay": 1e-5,
            "kl_to_elite_parent": 0.3,
            "freeze_public_backbone": True,
            "seeds": {"loss015": 202608128, "loss060": 202608128},
            "test_split": "deferred; not used to choose between arms",
        },
        "selection": {
            "primary": "newer-cohort winner-game MAIN NLL",
            "secondary": "newer-cohort winner-game exact agreement",
            "guardrails": [
                "no increase in END selection when Phantom Dive is legal",
                "no decrease in Phantom Dive selection when legal",
                "separate exact-07bed paired gameplay confirmation required",
            ],
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    LOCK.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text())
    claimed = payload.pop("lock_sha256", None)
    if claimed != canonical(payload):
        raise RuntimeError("training lock self-hash failed")
    payload["lock_sha256"] = claimed
    for row in payload["artifacts"].values():
        if file_sha(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"locked artifact drifted: {row['path']}")
    return payload


def train_arm(name: str, lock: dict[str, Any]) -> dict[str, Any]:
    output = RUN / "candidates" / name / "model"
    config = TRAIN.TrainingConfig(
        manifest_path=CORPUS,
        out_dir=output,
        cache_dir=RUN / "cache",
        epochs=4,
        batch_size=128,
        shuffle_buffer=4096,
        learning_rate=3e-5,
        weight_decay=1e-5,
        value_coefficient=0.0,
        seed=int(lock["training"]["seeds"][name]),
        device="cpu",
        win_weight=1.0,
        draw_weight=0.3,
        loss_weight=float(ARMS[name]),
        game_normalized=True,
        qu_v2_anchor_checkpoint_path=PARENT,
        initial_checkpoint_path=PARENT,
        target_deck_sha256=TARGET_SHA,
        target_select_type=0,
        freeze_public_backbone=True,
        kl_coefficient=0.3,
        kl_weighting="uniform-game",
        require_gpu=False,
        min_available_bytes=2 * TRAIN.training_preflight.GIB,
        min_swap_free_bytes=1 * TRAIN.training_preflight.GIB,
        defer_test=True,
    )
    result = TRAIN.run_training(config)
    summary = {
        "arm": name,
        "loss_weight": ARMS[name],
        "best_epoch": result["best_epoch"],
        "best_validation_objective": result["best_validation_objective"],
        "weights": str(result["weights_path"]),
        "checkpoint": str(result["checkpoint_path"]),
    }
    (RUN / "candidates" / name / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    lock = create_lock() if not LOCK.exists() else load_lock()
    print(json.dumps({"lock_sha256": lock["lock_sha256"]}))
    if args.lock_only:
        return 0
    arms = (args.arm,) if args.arm else tuple(ARMS)
    print(json.dumps([train_arm(name, lock) for name in arms], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
