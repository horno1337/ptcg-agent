"""Prospectively lock the exact-list Majkel1337 Lucario BC experiment."""

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


RUN = ROOT / "tools/checkpoints/lucario-majkel-bc-v1"
OUTPUT = RUN / "training-lock.json"
CORPUS = RUN / "corpus.json"
DECK = ROOT / "decks/lucario_majkel1337_hariyama.csv"
GUIDE = ROOT / "docs/lucario_majkel1337_alignment.md"
MAIN_PARENT = ROOT / "tools/checkpoints/dobi-v1-bc-recent-v1/dobi-v1-bc-adapter.pt"
CARD_PARENT = ROOT / "tools/checkpoints/md-v2-card-v1/model/candidate-qu-v2a-checkpoint.pt"
TARGET_DECK_SHA256 = "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"


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
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    paths = {
        "corpus": CORPUS,
        "deck": DECK,
        "guide_alignment": GUIDE,
        "main_parent": MAIN_PARENT,
        "card_parent": CARD_PARENT,
        "trainer": Path(TRAIN.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"bound artifacts missing: {missing}")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise SystemExit("corpus manifest self-hash failed")
    if corpus["summary"]["valid_bc_games"] != 507:
        raise SystemExit("unexpected Lucario corpus size")
    payload = {
        "schema": "ptcg.lucario-majkel.bc-v1.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "Exact-list winner-weighted cloning of a strong Lucario pilot can "
            "produce deck-matched ST_MAIN and ST_CARD specialists that beat the "
            "frozen generic policy while retaining clean fail-closed routing."
        ),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "corpus": {
            "manifest_sha256": corpus["manifest_sha256"],
            "games": 507,
            "split_games": corpus["summary"]["split_games"],
            "split_seed": corpus["split"]["seed"],
            "target_deck_sha256": TARGET_DECK_SHA256,
            "observed_target_decisions": {
                "ST_MAIN": 17_027,
                "ST_CARD": 12_902,
                "ST_BENCH": 645,
                "ST_ATTACK": 23,
                "ST_YES_NO": 121,
                "ST_ENERGY": 631,
            },
            "test_split_untouched_until_model_selection": True,
        },
        "training": {
            "heads": {
                "main": {"select_type": 0, "parent": "main_parent", "seed": 2026080511},
                "card": {"select_type": 1, "parent": "card_parent", "seed": 2026080512},
            },
            "epochs": 4,
            "batch_size": 128,
            "learning_rate": 1e-5,
            "weight_decay": 1e-5,
            "device": "cuda",
            "require_gpu": True,
            "freeze_public_backbone": True,
            "winner_weight": 1.0,
            "draw_weight": 0.3,
            "loss_weight": 0.25,
            "game_normalized": True,
            "value_coefficient": 0.0,
            "kl_coefficient": 0.1,
            "kl_weighting": "uniform-game",
            "selection": "lowest validation objective independently across exactly four epochs per head",
            "test_status": "deferred until both head selections are frozen",
            "guide_usage": "diagnostics and error taxonomy only; prose rules are not action labels",
        },
        "gates": {
            "behavioral": "report held-out disagreement and logged-action agreement by outcome for both heads",
            "same_deck_gameplay": (
                "combined Lucario ST_MAIN+ST_CARD package versus frozen generic "
                "policy on the identical Lucario list; fixed paired-seat gate, "
                "valid zero-fault CI95 lower bound above 50%"
            ),
            "deck_switch": (
                "only after same-deck pass; paired recent-frequency field A/B "
                "against frozen Dobi-v1/Grimmsnarl; local evidence proposes and "
                "the ladder disposes"
            ),
            "release": "exact tarball safety test, 200-game smoke, and non-owner UID audit",
        },
        "one_training_run_per_head": True,
        "no_alternate_epoch_after_gameplay": True,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "lock": str(OUTPUT),
        "lock_sha256": payload["lock_sha256"],
        "training": payload["training"],
        "gates": payload["gates"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
