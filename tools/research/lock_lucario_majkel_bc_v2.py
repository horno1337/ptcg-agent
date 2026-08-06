"""Lock conservative joint Lucario BC from the true frozen Qu-v2B parent."""

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


RUN = ROOT / "tools/checkpoints/lucario-majkel-bc-v2"
OUTPUT = RUN / "training-lock.json"
V1 = ROOT / "tools/checkpoints/lucario-majkel-bc-v1"
CORPUS = V1 / "corpus.json"
DECK = ROOT / "decks/lucario_majkel1337_hariyama.csv"
GUIDE = ROOT / "docs/lucario_majkel1337_alignment.md"
PARENT = ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-checkpoint.pt"
PARENT_WEIGHTS = ROOT / "agent/weights.npz"
V1_COMBINED_RESULT = V1 / "same-deck-vs-generic/result.json"
V1_ISOLATION_LOCK = V1 / "head-isolation/lock.json"
TARGET_DECK_SHA256 = "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"
PARENT_WEIGHTS_SHA256 = "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"
PARENT_CHECKPOINT_SHA256 = "9ba093b81a0f06f96812422084e19b66e0652006713e28ccaee81b28ec3cd65a"


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
        "qu_v2b_parent_checkpoint": PARENT,
        "qu_v2b_parent_weights": PARENT_WEIGHTS,
        "v1_combined_result": V1_COMBINED_RESULT,
        "v1_isolation_lock": V1_ISOLATION_LOCK,
        "trainer": Path(TRAIN.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"bound artifacts missing: {missing}")
    if sha256(PARENT) != PARENT_CHECKPOINT_SHA256 or sha256(PARENT_WEIGHTS) != PARENT_WEIGHTS_SHA256:
        raise SystemExit("frozen Qu-v2B parent drifted")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise SystemExit("corpus self-hash failed")
    prior = json.loads(V1_COMBINED_RESULT.read_text(encoding="utf-8"))
    if (
        prior.get("decision", {}).get("valid") is not True
        or prior.get("decision", {}).get("passed") is not False
    ):
        raise SystemExit("Lucario-v1 failure contract drifted")
    payload = {
        "schema": "ptcg.lucario-majkel.bc-v2.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_v1_isolation_outcomes": True,
        "hypothesis": (
            "A conservative joint exact-list update from true generic Qu-v2B can "
            "learn Lucario-specific play without retaining the Grimmsnarl priors "
            "or independent-head mismatch that regressed Lucario-v1."
        ),
        "prior_failure": {
            "v1_score": prior["decision"]["score"],
            "v1_ci95": prior["decision"]["wilson_ci95"],
            "design_response": (
                "replace both Grimmsnarl-specialized parents with true Qu-v2B; "
                "train one joint net; halve epochs, halve learning rate, and "
                "increase KL coefficient fivefold"
            ),
        },
        "conditional_activation": {
            "train_v2_if": "neither locked Lucario-v1 head-isolation arm is diagnostic-positive",
            "otherwise": "pause v2 and run a fresh 5120-game confirmation of the positive isolated head",
            "diagnostic_positive_definition": "valid zero-fault CI95 lower bound above 0.50",
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "corpus": {
            "manifest_sha256": corpus["manifest_sha256"],
            "games": corpus["summary"]["valid_bc_games"],
            "split_games": corpus["summary"]["split_games"],
            "target_deck_sha256": TARGET_DECK_SHA256,
            "reuse_exact_v1_split": True,
            "test_split_untouched": True,
        },
        "training": {
            "architecture": [16, 48, 160, 112, 80],
            "initial_checkpoint": "true frozen Qu-v2B production checkpoint",
            "kl_parent": "same true frozen Qu-v2B checkpoint",
            "joint_target": "all exact-deck prompts during training; deploy only ST_MAIN and ST_CARD",
            "target_select_type": None,
            "epochs": 2,
            "batch_size": 128,
            "learning_rate": 5e-6,
            "weight_decay": 1e-5,
            "seed": 2026080541,
            "device": "cuda",
            "require_gpu": True,
            "freeze_public_backbone": True,
            "winner_weight": 1.0,
            "draw_weight": 0.3,
            "loss_weight": 0.25,
            "game_normalized": True,
            "value_coefficient": 0.0,
            "kl_coefficient": 0.5,
            "kl_weighting": "uniform-game",
            "selection": "lowest validation objective across exactly two epochs",
            "test_status": "deferred until selection and gameplay gates are frozen",
            "guide_usage": "diagnostics only; no hand-written action overrides",
        },
        "gates": {
            "behavioral": "report ST_MAIN and ST_CARD disagreement versus true Qu-v2B by outcome",
            "same_deck": (
                "5120-game paired-seat exact Lucario mirror versus frozen Qu-v2B; "
                "require valid zero-fault CI95 lower bound above 50%"
            ),
            "grimmsnarl": (
                "only after same-deck pass; locked Lucario-v2 versus Dobi-v1/Grimmsnarl "
                "diagnostic and recent-field deck-switch gate"
            ),
            "release": "exact tarball safety, 200-game smoke, and non-owner UID audit",
        },
        "one_training_run": True,
        "no_alternate_epoch_after_gameplay": True,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    RUN.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "lock": str(OUTPUT),
        "lock_sha256": payload["lock_sha256"],
        "conditional_activation": payload["conditional_activation"],
        "training": payload["training"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
