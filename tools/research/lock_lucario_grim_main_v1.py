"""Lock two Qu-v2B-anchored Lucario ST_MAIN arms targeted at Grimmsnarl."""

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


RUN = ROOT / "tools/checkpoints/lucario-grim-main-v1"
OUTPUT = RUN / "training-lock.json"
LUCARIO = ROOT / "tools/checkpoints/lucario-majkel-bc-v1"
CORPUS = ROOT / "tools/checkpoints/lucario-grim-main-v3/corpus.json"
DECK = ROOT / "decks/lucario_majkel1337_hariyama.csv"
GUIDE = ROOT / "docs/lucario_majkel1337_alignment.md"
PARENT = ROOT / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-checkpoint.pt"
PARENT_WEIGHTS = ROOT / "agent/weights.npz"
DIRECT_FAILURE = LUCARIO / "card-only-vs-dobi-grim/result.json"
CARD_CONFIRM = LUCARIO / "card-only-confirm/result.json"
TARGET_DECK_SHA256 = "77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8"
GRIM_MATCHUP_CARD_ID = 648
PARENT_WEIGHTS_SHA256 = "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"
PARENT_CHECKPOINT_SHA256 = "9ba093b81a0f06f96812422084e19b66e0652006713e28ccaee81b28ec3cd65a"
ARMS = {
    "conservative": {
        "epochs": 2,
        "learning_rate": 5e-6,
        "kl_coefficient": 0.5,
        "matchup_weight": 4.0,
        "matchup_win_weight": 1.0,
        "matchup_draw_weight": 1.0,
        "matchup_loss_weight": 1.0,
        "seed": 2_026_080_571,
    },
    "focused": {
        "epochs": 3,
        "learning_rate": 1e-5,
        "kl_coefficient": 0.2,
        "matchup_weight": 6.0,
        "matchup_win_weight": 1.5,
        "matchup_draw_weight": 1.0,
        "matchup_loss_weight": 1.0,
        "seed": 2_026_080_572,
    },
}


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
        "direct_failure": DIRECT_FAILURE,
        "card_confirmation": CARD_CONFIRM,
        "trainer": Path(TRAIN.__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"bound artifacts missing: {missing}")
    if sha256(PARENT) != PARENT_CHECKPOINT_SHA256 or sha256(PARENT_WEIGHTS) != PARENT_WEIGHTS_SHA256:
        raise SystemExit("true Qu-v2B parent drifted")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise SystemExit("corpus self-hash failed")
    failure = json.loads(DIRECT_FAILURE.read_text(encoding="utf-8"))
    confirmation = json.loads(CARD_CONFIRM.read_text(encoding="utf-8"))
    if (
        failure.get("decision", {}).get("valid") is not True
        or failure.get("decision", {}).get("field_gate_eligible") is not False
        or confirmation.get("decision", {}).get("passed") is not True
    ):
        raise SystemExit("prior result contract drifted")
    payload = {
        "schema": "ptcg.lucario-grim-main-v1.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "objective": (
            "Improve exact-list Lucario specifically against Grimmsnarl; Lucario "
            "mirror strength has no advancement authority."
        ),
        "hypothesis": (
            "Majkel's successful Lucario/Grimmsnarl demonstrations can teach the "
            "missing macro turn policy when ST_MAIN starts from true Qu-v2B and "
            "is deployed only after a public Grimmsnarl signature."
        ),
        "prior_evidence": {
            "majkel_observed_grim_record": {"wins": 126, "losses": 120, "score": 126 / 246},
            "current_agent_vs_dobi_score": failure["decision"]["score"],
            "current_agent_vs_dobi_ci95": failure["decision"]["wilson_ci95"],
            "confirmed_card_only_mirror_score": confirmation["decision"]["score"],
            "interpretation": "ST_CARD is real but macro ST_MAIN behavior is the binding Grimmsnarl deficit.",
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
            "exact_target_seats": 863,
            "observed_grim_games": 246,
            "grim_wins": 126,
            "grim_losses": 120,
            "grim_split_games": {"train": 199, "validation": 27, "test": 20},
            "test_split_untouched": True,
        },
        "common_training": {
            "target_select_type": 0,
            "matchup_card_id": GRIM_MATCHUP_CARD_ID,
            "batch_size": 128,
            "weight_decay": 1e-5,
            "device": "cuda",
            "require_gpu": True,
            "freeze_public_backbone": True,
            "winner_weight": 1.0,
            "draw_weight": 0.3,
            "loss_weight": 0.25,
            "game_normalized": True,
            "value_coefficient": 0.0,
            "kl_weighting": "uniform-game",
            "initial_checkpoint": "true frozen Qu-v2B",
            "kl_parent": "same true frozen Qu-v2B",
            "selection": "lowest validation objective within each fixed arm",
            "test_status": "deferred",
        },
        "arms": ARMS,
        "runtime_scope": {
            "deck": "exact Majkel1337 Lucario list only",
            "prompt": "ST_MAIN only",
            "opponent": "public active/bench contains Marnie's Impidimp, Morgrem, or Grimmsnarl ex",
            "scope_miss": "fall through to unchanged generic Qu-v2B",
            "st_card": "evaluate confirmed Lucario ST_CARD both on and off in direct Grimmsnarl screens",
        },
        "gates": {
            "behavioral": "report each arm's ST_MAIN disagreement versus true Qu-v2B on Grimmsnarl validation decisions",
            "direct_discovery": (
                "both arms declared together; 2560 games per package against frozen "
                "Dobi-v1/Grimmsnarl; diagnostic only"
            ),
            "confirmation": "fresh 5120 games for any selected package before field testing",
            "field": "only after direct confirmation; compare against the current Grimmsnarl champion",
        },
        "one_training_run_per_arm": True,
        "no_new_arm_after_gameplay": True,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    RUN.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "lock": str(OUTPUT),
        "lock_sha256": payload["lock_sha256"],
        "objective": payload["objective"],
        "arms": payload["arms"],
        "runtime_scope": payload["runtime_scope"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
