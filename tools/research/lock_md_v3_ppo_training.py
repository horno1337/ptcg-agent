"""Seal the full parent-anchored MD-v3 ST_MAIN PPO training experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_md_v3_ppo as BASE  # noqa: E402
from tools.research import train_md_v3_ppo as PPO  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error("--output already exists")
    payload = {
        "schema": PPO.FULL_LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Small terminal-outcome PPO updates, strongly anchored to frozen "
            "MD-v3, improve its exact-deck ST_MAIN mirror policy."
        ),
        "scope": {
            "changed_runtime_component": "exact-deck ST_MAIN weights only",
            "frozen_components": [
                "MD-v3 ST_CARD weights", "Qu-v2B residual weights",
                "registered Grimmsnarl deck", "frozen MD-v3 opponent",
            ],
            "prototype_evidence": {
                "lock_sha256": (
                    "0c4eb1f55a8618bef01ce7a31df54d306412f3eed562aa21d85a4d4d2f726eaf"
                ),
                "use": "compatibility and stability only; W/L ignored",
            },
            "deployment": "no upload authorized",
        },
        "training": {
            "updates": 4,
            "games_per_update": 384,
            "seat_balance": "exactly 192 games per learner seat per update",
            "opponent": "complete frozen MD-v3 exact-deck mirror",
            "seed": 2026073011,
            "learning_rate": 5e-6,
            "ppo_epochs": 2,
            "minibatch_size": 512,
            "clip": 0.10,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.002,
            "parent_kl_coefficient": 1.0,
            "minimum_st_main_decisions": 10000,
            "maximum_update_parent_kl": 0.02,
        },
        "decision_rules": {
            "training": (
                "training artifact is valid only if every one of four updates "
                "has 384 valid games, zero opponent fallback/repair/exception, "
                "at least 10000 ST_MAIN decisions, update parent KL <= 0.02, "
                "finite nonzero parameter delta, exact frozen artifact hashes, "
                "and Torch/NumPy parity within 2e-4 logits and 2e-5 value"
            ),
            "direct_gameplay": (
                "a valid candidate advances only by a separately locked 1280-"
                "game exact-mirror A/B versus byte-frozen MD-v3, exactly 640 "
                "candidate games per seat, with Wilson CI95 lower bound > 0.50"
            ),
            "promotion": (
                "only after direct gameplay passes: separately lock and pass "
                "recent-frequency field noninferiority, safety tests, 200-game "
                "random smoke, and cross-UID exact-tarball audit"
            ),
            "ladder": (
                "no upload from this experiment without the user naming the "
                "submission and explicitly approving FIFO retirement risk"
            ),
        },
        "artifacts": {
            "trainer": BASE.artifact(ROOT / "tools/research/train_md_v3_ppo.py"),
            "parent_checkpoint": BASE.artifact(
                ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
                "candidate-qu-v2a-checkpoint.pt"
            ),
            "parent_weights": BASE.artifact(
                ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
                "candidate-qu-v2a-weights.npz"
            ),
            "card_weights": BASE.artifact(ROOT / "agent/md_v2_card_weights.npz"),
            "qu_weights": BASE.artifact(ROOT / "agent/weights.npz"),
            "deck": BASE.artifact(ROOT / "decks/md_v1_grimmsnarl.csv"),
        },
    }
    payload["lock_sha256"] = PPO.json_sha256(payload)
    PPO.atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
