"""Pre-register the exact-Grimmsnarl ST_CARD specialist experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


SCHEMA = "ptcg.md-v2-card-v1.lock.v1"
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
RUN = ROOT / "tools/checkpoints/md-v2-card-v1"
OUTPUT = RUN / "lock.json"
CORPUS = ROOT / "tools/checkpoints/md-v2-allthrough26/corpus.json"
SOURCE_LOCK = ROOT / "tools/checkpoints/md-v2-allthrough26/lock.json"
INITIAL_CHECKPOINT = (
    ROOT
    / "tools/checkpoints/qu-v2b-field-v1/candidate-qu-v2a-checkpoint.pt"
)
QU_V2B_WEIGHTS = ROOT / "agent/weights.npz"
MD_V2_MAIN_WEIGHTS = (
    ROOT
    / "tools/checkpoints/md-v2-allthrough26/model/"
    "candidate-qu-v2a-weights.npz"
)
TARGET_DECK = ROOT / "decks/md_v1_grimmsnarl.csv"
ARTIFACTS = {
    "corpus": CORPUS,
    "source_lock": SOURCE_LOCK,
    "initial_checkpoint": INITIAL_CHECKPOINT,
    "qu_v2b_weights": QU_V2B_WEIGHTS,
    "md_v2_main_weights": MD_V2_MAIN_WEIGHTS,
    "target_deck": TARGET_DECK,
    "trainer": ROOT / "tools/research/train_qu_v2a.py",
    "dataset_loader": ROOT / "tools/il_dataset.py",
    "training_features": ROOT / "tools/research/qu_v2a_features.py",
    "training_model": ROOT / "tools/research/qu_v2a_model.py",
    "resource_preflight": ROOT / "tools/training_preflight.py",
    "evaluator": ROOT / "tools/research/eval_md_v2_card_v1_validation.py",
    "lock_builder": ROOT / "tools/research/lock_md_v2_card_v1.py",
    "temporal_core": (
        ROOT / "tools/research/eval_md_v2_july27_temporal.py"
    ),
    "indexer": ROOT / "tools/index_corpus.py",
    "runtime_features": ROOT / "agent/qu_v2_features.py",
    "runtime_model": ROOT / "agent/model.py",
}


class LockError(RuntimeError):
    """The preregistration inputs are missing or have drifted."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise LockError(f"JSON root is not an object: {path}")
    return value


def _record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LockError(f"missing artifact: {resolved}")
    try:
        rendered = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered = str(resolved)
    return {
        "path": rendered,
        "sha256": COMMON.file_sha256(resolved),
    }


def _load_source_lock() -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(
            SOURCE_LOCK,
            schema="ptcg.md-v2.allthrough26-lock.v1",
            hash_key="lock_sha256",
        )
    except COMMON.EvaluationError as error:
        raise LockError(str(error)) from error


def _mirror_counts(games: list[Any]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "total": 0}
    for game in games:
        if not isinstance(game, Mapping):
            raise LockError("corpus contains a malformed game")
        seats = game.get("seats")
        if (
            isinstance(seats, list)
            and len(seats) == 2
            and all(
                isinstance(seat, Mapping)
                and seat.get("registered_deck_sha256")
                    == TARGET_DECK_SHA256
                for seat in seats
            )
        ):
            split = game.get("split")
            if split not in ("train", "validation"):
                raise LockError("exact mirror is outside train/validation")
            counts[str(split)] += 1
            counts["total"] += 1
    return counts


def _verify_deck() -> None:
    try:
        cards = tuple(
            int(line)
            for line in TARGET_DECK.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValueError) as error:
        raise LockError(f"cannot read target deck: {error}") from error
    if (
        len(cards) != 60
        or tuple(sorted(cards)) != cards
        or index_corpus.deck_sha256(cards) != TARGET_DECK_SHA256
    ):
        raise LockError("target deck registration/hash differs")


def build_lock() -> dict[str, Any]:
    source_lock = _load_source_lock()
    corpus = _load_json(CORPUS)
    games = corpus.get("games")
    if (
        corpus.get("schema") != index_corpus.SCHEMA
        or not index_corpus.verify_manifest(corpus)
        or corpus.get("candidate_only") is not True
        or not isinstance(games, list)
        or len(games) != 17591
        or corpus.get("summary", {}).get("split_games")
            != {"train": 15831, "validation": 1760, "test": 0}
        or source_lock.get("corpus", {}).get("file_sha256")
            != COMMON.file_sha256(CORPUS)
        or source_lock.get("corpus", {}).get("manifest_sha256")
            != corpus.get("manifest_sha256")
        or source_lock.get("runtime", {}).get("deck_sha256")
            != TARGET_DECK_SHA256
    ):
        raise LockError("all-through-26 corpus/source lock binding drifted")
    mirrors = _mirror_counts(games)
    if mirrors != {"train": 2213, "validation": 250, "total": 2463}:
        raise LockError(f"exact-mirror cohort count drifted: {mirrors}")
    _verify_deck()
    artifacts = {
        label: _record(path) for label, path in ARTIFACTS.items()
    }
    if (
        artifacts["initial_checkpoint"]["sha256"]
        != source_lock["training"]["initial_checkpoint_sha256"]
    ):
        raise LockError("frozen Qu-v2B initial checkpoint drifted")
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_name": "MD-v2 ST_CARD mirror specialist v1",
        "scientific_status": (
            "candidate-only follow-up; no ladder or submission authority"
        ),
        "motivation_frozen_before_outcomes": {
            "live_exact_mirror_record": "1-7 descriptive signal only",
            "structural_gap": (
                "MD-v2 owns ST_MAIN only; ST_CARD still uses "
                "Alakazam-conditioned Qu-v2B"
            ),
            "observed_mirror_mechanism": (
                "fewer Munkidori activations and slower charged-Munkidori "
                "setup than exact-list opponents"
            ),
        },
        "corpus": {
            "path": str(CORPUS.resolve()),
            "file_sha256": artifacts["corpus"]["sha256"],
            "manifest_sha256": corpus["manifest_sha256"],
            "corpus_content_sha256": corpus["corpus_content_sha256"],
            "split_seed": corpus["split"]["seed"],
            "games": len(games),
            "split_games": {"train": 15831, "validation": 1760},
            "exact_list_mirror_games": mirrors,
            "discovery": (
                "all numeric replay JSONs; no team-name prefilter"
            ),
        },
        "training": {
            "architecture": [16, 48, 160, 112, 80],
            "initial_checkpoint": "frozen Qu-v2B training checkpoint",
            "target_deck_sha256": TARGET_DECK_SHA256,
            "target_select_type": 1,
            "epochs": 10,
            "batch_size": 128,
            "shuffle_buffer": 4096,
            "learning_rate": 0.00005,
            "weight_decay": 0.00001,
            "gradient_clip": 1.0,
            "seed": 20260728,
            "value_coefficient": 0.0,
            "kl_coefficient": 0.0,
            "freeze_public_backbone": True,
            "win_draw_loss_weights": [1.0, 1.0, 1.0],
            "source_weights": {},
            "deck_weights": {},
            "game_normalized": True,
            "checkpoint_selection": (
                "lowest validation objective; earliest epoch tie"
            ),
            "test_status": "deferred",
            "cache": "reuse encoding cache namespace when hashes match",
            "single_training_run": True,
            "no_retraining_or_threshold_changes_after_validation": True,
        },
        "validation": {
            "split": "locked all-through-26 validation split",
            "metric": "game-normalized ST_CARD sequence NLL",
            "models": [
                "fixed ST_CARD candidate",
                "frozen Qu-v2B",
            ],
            "strata": [
                "all exact-own-deck validation games",
                "exact-list Grimmsnarl mirror validation games",
            ],
            "same_decision_stream_for_models": True,
            "pass_rule": (
                "candidate objective is strictly lower than frozen Qu-v2B "
                "both overall and on exact-list mirror games"
            ),
            "empty_nonfinite_or_contract_drift": "fail",
            "promotion_authority": False,
        },
        "runtime": {
            "route": (
                "exact own deck plus ST_CARD plus public opposing "
                "Grimmsnarl signature only"
            ),
            "public_opposing_signature": {
                "zones": ["active", "bench"],
                "any_card_id": [646, 647, 648],
                "hidden_registration_used": False,
            },
            "fallback": "unchanged MD-v2 layered runtime",
            "md_v2_st_main_unchanged": True,
            "matchups_without_public_grim_signature_unchanged": True,
            "flag_gated_default": "off until all gates pass",
        },
        "gameplay_gate_after_validation_pass": {
            "games": 640,
            "seed": 20260811,
            "deck": "exact target Grimmsnarl list in every seat",
            "comparison": (
                "candidate runtime (MD-v2 ST_MAIN, candidate mirror ST_CARD, "
                "Qu-v2B fallback) versus unchanged MD-v2 runtime"
            ),
            "seat_balance": "exactly 320 games per candidate seat",
            "pass": {
                "candidate_point_estimate": "> 0.50",
                "candidate_wilson_95_lower": "> 0.45",
                "invalid_games": 0,
                "runtime_errors": 0,
                "fallbacks": 0,
                "legality_repairs": 0,
            },
            "one_schedule_one_attempt": True,
            "promotion_authority": False,
        },
        "prospective_ladder_replays": {
            "source": "MD-v2 and MD-v2 harvester 1",
            "allowed_use": (
                "forward descriptive validation of this already fixed "
                "mechanism/model only"
            ),
            "forbidden_use": (
                "training, retuning, route expansion, or threshold changes"
            ),
        },
        "artifacts": artifacts,
    }


def main() -> int:
    if OUTPUT.exists():
        print(f"error: refusing to overwrite {OUTPUT}", file=sys.stderr)
        return 2
    try:
        payload = build_lock()
        payload["lock_sha256"] = COMMON.canonical_sha256(payload)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, LockError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "lock": str(OUTPUT),
        "lock_sha256": payload["lock_sha256"],
        "corpus_games": payload["corpus"]["games"],
        "exact_list_mirror_games": payload["corpus"][
            "exact_list_mirror_games"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
