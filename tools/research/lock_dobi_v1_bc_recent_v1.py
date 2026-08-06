"""Prospectively lock conservative recent-field BC starting from Dobi-v1."""

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


RUN = ROOT / "tools/checkpoints/dobi-v1-bc-recent-v1"
OUTPUT = RUN / "training-lock.json"
CORPUS = ROOT / (
    "tools/checkpoints/dobi-v1-ppo300k-bc-repair/training-corpus.json"
)
PARENT = RUN / "dobi-v1-bc-adapter.pt"
DOBI_WEIGHTS = ROOT / (
    "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
    "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
)
PRIOR_FIELD = ROOT / (
    "tools/checkpoints/dobi-v1-ppo300k-bc-repair/field-vs-dobi/result.json"
)
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)


def _sha256(path: Path) -> str:
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
    for path in (CORPUS, PARENT, DOBI_WEIGHTS, PRIOR_FIELD):
        if not path.is_file():
            raise SystemExit(f"bound artifact missing: {path}")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if not index_corpus.verify_manifest(corpus):
        raise SystemExit("training corpus self-hash failed")
    prior = json.loads(PRIOR_FIELD.read_text(encoding="utf-8"))
    if (
        prior.get("decision", {}).get("valid") is not True
        or prior.get("decision", {}).get("passed_noninferiority") is not False
    ):
        raise SystemExit("prior PPO-BC field result contract drifted")
    payload = {
        "schema": "ptcg.dobi-v1.bc-recent-v1.training-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "Recent winner-weighted non-mirror BC can transfer useful field "
            "behaviour without inheriting the ladder-weak PPO-300k parent."
        ),
        "prior_result_disclosed": {
            "ppo_bc_field_delta": prior["decision"]["candidate_minus_control"],
            "formal_pass": False,
            "design_response": (
                "change only the parent/anchor and reduce optimization drift; "
                "do not remove or post-hoc reweight the Fezandipiti stratum"
            ),
        },
        "artifacts": {
            "corpus": {"path": str(CORPUS.resolve()), "sha256": _sha256(CORPUS)},
            "parent": {"path": str(PARENT.resolve()), "sha256": _sha256(PARENT)},
            "dobi_weights": {
                "path": str(DOBI_WEIGHTS.resolve()),
                "sha256": _sha256(DOBI_WEIGHTS),
            },
            "prior_field_result": {
                "path": str(PRIOR_FIELD.resolve()),
                "sha256": _sha256(PRIOR_FIELD),
            },
            "trainer": {
                "path": str(Path(TRAIN.__file__).resolve()),
                "sha256": _sha256(Path(TRAIN.__file__).resolve()),
            },
        },
        "corpus": {
            "manifest_sha256": corpus["manifest_sha256"],
            "games": corpus["summary"]["games"],
            "fresh_games": corpus["summary"]["fresh_games"],
            "historical_mirror_kl_games": corpus["summary"]["historical_mirror_kl_games"],
            "split_games": corpus["summary"]["split_games"],
            "reuse_without_posthoc_game_or_stratum_removal": True,
        },
        "training": {
            "epochs": 2,
            "batch_size": 128,
            "learning_rate": 5e-6,
            "weight_decay": 1e-5,
            "seed": 20260805,
            "device": "cuda",
            "require_gpu": True,
            "target_select_type": 0,
            "target_deck_sha256": TARGET_DECK_SHA256,
            "freeze_public_backbone": True,
            "winner_weight": 1.0,
            "draw_weight": 0.3,
            "sampled_loss_weight": 0.4,
            "crustle_multiplier": 1.5,
            "mirror_source_weight": 0.0,
            "game_normalized": True,
            "value_coefficient": 0.0,
            "kl_coefficient": 0.5,
            "kl_weighting": "uniform-game",
            "initial_checkpoint": "frozen Dobi-v1 update-130 terminal",
            "kl_parent": "same frozen Dobi-v1 terminal",
            "selection": "lowest validation objective across exactly two epochs",
            "offline_metrics_are_screening_only": True,
        },
        "gates": {
            "behavioral_report": (
                "report ST_MAIN decision and game-touch disagreement versus "
                "frozen Dobi before gameplay; no posthoc minimum required"
            ),
            "mirror": (
                "5,120-game exact-list paired-seat A/B versus frozen Dobi; "
                "require valid zero-fault CI95 lower bound >=48%"
            ),
            "field": (
                "only after mirror pass; 5,120 games/arm on the locked recent "
                "non-mirror field; require valid zero-fault delta CI95 lower "
                "bound >-1.5pp and point estimate >=0"
            ),
            "ladder": "local gates propose; user separately names and approves uploads",
        },
        "one_training_run": True,
        "no_alternate_epoch_after_gameplay": True,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
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
