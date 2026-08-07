"""Prospectively lock the exact-list Festival Lead BC heads."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402

RUN = ROOT / "tools/checkpoints/festival-lead-bc-v1"
OUTPUT = RUN / "training-lock.json"
CORPUS = RUN / "corpus.json"
DECK = ROOT / "decks/festival_lead_majkel1337.csv"
GUIDE = ROOT / "docs/festival_lead_agent.md"
RULES = ROOT / "agent/festival_lead.py"
TESTS = ROOT / "tests/test_festival_lead.py"
PARENT = ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-checkpoint.pt"
PARENT_WEIGHTS = ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-weights.npz"
TARGET_DECK_SHA256 = "2617a1c612d86947bb078c0c5ae752007dc9320754905a4bf7d553f44d1ba667"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def main() -> int:
    if OUTPUT.exists() or any((RUN / head / "model").exists() for head in ("main", "card")):
        raise SystemExit("training lock or model outcome already exists")
    paths = {
        "corpus": CORPUS, "deck": DECK, "strategy": GUIDE,
        "rule_controller": RULES, "tests": TESTS,
        "parent_checkpoint": PARENT, "parent_weights": PARENT_WEIGHTS,
        "trainer": Path(TRAIN.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"bound artifacts missing: {missing}")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise SystemExit("corpus manifest self-hash failed")
    if (
        corpus["summary"]["valid_bc_games"] != 68
        or corpus["summary"]["split_valid_bc_games"]
            != {"train": 49, "validation": 8, "test": 11}
        or corpus["split"]["seed"] != 202608077
    ):
        raise SystemExit("Festival corpus identity or split drifted")
    payload = {
        "schema": "ptcg.festival-lead.bc-v1.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "Frozen-backbone exact-list BC can improve the context-dependent "
            "ST_MAIN and ST_CARD decisions that a deterministic combo policy "
            "does not capture, while preserving a rule fallback."
        ),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "corpus": {
            "manifest_sha256": corpus["manifest_sha256"],
            "games": 68,
            "split_games": corpus["summary"]["split_valid_bc_games"],
            "split_seed": 202608077,
            "target_deck_sha256": TARGET_DECK_SHA256,
            "test_status": "sealed until independent head selection",
        },
        "training": {
            "heads": {
                "main": {"select_type": 0, "seed": 202608078},
                "card": {"select_type": 1, "seed": 202608079},
            },
            "epochs": 4,
            "batch_size": 64,
            "learning_rate": 0.00003,
            "weight_decay": 0.00001,
            "device": "cpu",
            "freeze_public_backbone": True,
            "winner_weight": 1.0,
            "draw_weight": 0.3,
            "loss_weight": 0.15,
            "game_normalized": True,
            "value_coefficient": 0.0,
            "kl_coefficient": 0.3,
            "kl_weighting": "uniform-game",
            "selection": "lowest validation objective across exactly four epochs per head",
            "one_run_per_head": True,
        },
        "gates": {
            "behavior": "candidate improves held-out logged-action agreement over its parent",
            "gameplay": "paired-seat exact-list mirror against frozen generic Qu-v2B",
            "ppo": "deferred unless the BC hybrid first passes gameplay",
        },
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock_sha256": payload["lock_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
