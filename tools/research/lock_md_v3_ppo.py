"""Create the one-shot preregistration for the MD-v3 ST_MAIN PPO prototype."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from tools.research import train_md_v3_ppo as PPO


ROOT = Path(__file__).resolve().parents[2]


def artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": PPO.sha256_file(resolved),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error("--output already exists")
    payload = {
        "schema": PPO.LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "A small outcome-optimized, parent-anchored PPO update to only the "
            "deployed exact-deck ST_MAIN specialist can improve Grimmsnarl "
            "gameplay where winner-weighted imitation returned a gameplay null."
        ),
        "scope": {
            "changed_runtime_component": "exact-deck ST_MAIN weights only",
            "frozen_components": [
                "MD-v3 ST_CARD weights", "Qu-v2B residual weights",
                "registered Grimmsnarl deck",
            ],
            "deployment": "no upload or production mutation during prototype",
        },
        "prototype": {
            "games": 32,
            "seat_balance": "exactly 16 games per learner seat",
            "opponent": "complete frozen MD-v3 exact-deck mirror",
            "seed": 2026073007,
            "learning_rate": 1e-6,
            "ppo_epochs": 2,
            "minibatch_size": 256,
            "clip": 0.10,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.002,
            "parent_kl_coefficient": 1.0,
            "minimum_st_main_decisions": 500,
            "maximum_update_parent_kl": 0.02,
        },
        "decision_rules": {
            "prototype": (
                "advance only if all 32 games are valid; frozen opponent has "
                "zero fallback/repair/exception; at least 500 ST_MAIN decisions "
                "are collected; candidate parameters change and remain finite; "
                "mean update parent KL <= 0.02; exported NumPy inference matches "
                "Torch within 2e-4 logits and 2e-5 value; all frozen artifact "
                "hashes remain unchanged"
            ),
            "full_training": (
                "only after the prototype passes, preregister a separate fixed "
                "training schedule and one 1,280-game paired exact-mirror gate"
            ),
            "promotion": (
                "candidate must be clean and its locked 1,280-game Wilson CI95 "
                "lower bound must be strictly above 0.50; then it must pass a "
                "separately locked recent-frequency field noninferiority gate "
                "and exact-tarball safety audit before any upload is proposed"
            ),
            "ladder": (
                "no ladder upload is authorized by this lock; FIFO retirement "
                "of the older 960-scoring MD-v3 makes offline gates mandatory"
            ),
        },
        "artifacts": {
            "trainer": artifact(ROOT / "tools/research/train_md_v3_ppo.py"),
            "parent_checkpoint": artifact(
                ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
                "candidate-qu-v2a-checkpoint.pt"
            ),
            "parent_weights": artifact(
                ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
                "candidate-qu-v2a-weights.npz"
            ),
            "card_weights": artifact(ROOT / "agent/md_v2_card_weights.npz"),
            "qu_weights": artifact(ROOT / "agent/weights.npz"),
            "deck": artifact(ROOT / "decks/md_v1_grimmsnarl.csv"),
        },
    }
    payload["lock_sha256"] = PPO.json_sha256(payload)
    PPO.atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
