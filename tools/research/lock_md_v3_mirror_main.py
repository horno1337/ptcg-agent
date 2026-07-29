"""Pre-register the MD-v3 mirror-weighted ST_MAIN experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN = ROOT / "tools/checkpoints/md-v3-mirror-main-v1"
OUTPUT = RUN / "lock.json"
CORPUS = RUN / "corpus.json"
MASS_AUDIT = RUN / "mass-audit.json"
FROZEN_MAIN_CHECKPOINT = (
    ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
    "candidate-qu-v2a-checkpoint.pt"
)
FROZEN_MAIN_WEIGHTS = (
    ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
    "candidate-qu-v2a-weights.npz"
)
FROZEN_CARD_WEIGHTS = ROOT / "agent/md_v2_card_weights.npz"
FROZEN_QU_WEIGHTS = ROOT / "agent/weights.npz"
TARGET_DECK = ROOT / "decks/md_v1_grimmsnarl.csv"
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)


class LockError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockError(f"cannot load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise LockError(f"{path} is not an object")
    return payload


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LockError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def build_lock() -> dict[str, Any]:
    corpus = _load(CORPUS)
    audit = _load(MASS_AUDIT)
    counts = corpus.get("md_v3_mirror_main", {}).get("counts")
    if (
        counts is None
        or counts.get("all", {}).get("total") != 20883
        or counts.get("all", {}).get("grim_family") != 6985
        or counts.get("all", {}).get("exact_mirror") != 3460
    ):
        raise LockError("corpus counts drifted")
    if (
        audit.get("schema") != "ptcg.md-v3.mirror-main-mass-audit.v1"
        or audit.get("input", {}).get("manifest_sha256")
            != corpus.get("manifest_sha256")
    ):
        raise LockError("mass audit does not bind the corpus")
    solved = audit.get("solved_matchup_weights")
    if not isinstance(solved, Mapping) or set(solved) != {"35", "51", "65"}:
        raise LockError("mass audit does not contain the fixed arms")
    arms = {}
    for name in ("35", "51", "65"):
        row = solved[name]
        target = float(row["target_grim_family_supervision_mass"])
        achieved = float(row["achieved_train_mass"])
        multiplier = float(row["matchup_weight"])
        if abs(target - achieved) > 1e-12 or not multiplier > 0:
            raise LockError(f"invalid solved arm {name}")
        arms[name] = {
            "target_grim_family_supervision_mass": target,
            "matchup_weight": multiplier,
        }

    artifacts = {
        "corpus": _artifact(CORPUS),
        "mass_audit": _artifact(MASS_AUDIT),
        "frozen_md_v3_main_checkpoint": _artifact(FROZEN_MAIN_CHECKPOINT),
        "frozen_md_v3_main_weights": _artifact(FROZEN_MAIN_WEIGHTS),
        "frozen_md_v3_card_weights": _artifact(FROZEN_CARD_WEIGHTS),
        "frozen_qu_v2b_weights": _artifact(FROZEN_QU_WEIGHTS),
        "target_deck": _artifact(TARGET_DECK),
        "trainer": _artifact(ROOT / "tools/research/train_qu_v2a.py"),
        "sweep_runner": _artifact(
            ROOT / "tools/research/run_md_v3_mirror_main_sweep.py"),
        "validation_evaluator": _artifact(
            ROOT / "tools/research/eval_md_v3_mirror_main_validation.py"),
        "corpus_builder": _artifact(
            ROOT / "tools/research/prepare_md_v3_mirror_main.py"),
        "mass_auditor": _artifact(
            ROOT / "tools/research/audit_md_v3_mirror_main_mass.py"),
        "trainer_tests": _artifact(ROOT / "tests/test_qu_v2a_training.py"),
    }
    return {
        "schema": "ptcg.md-v3.mirror-main-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": (
            "mirror-weighted ST_MAIN adaptation of frozen MD-v3; "
            "not an upload tag"
        ),
        "hypothesis": (
            "MD-v3 is approximately median against the Grimmsnarl family "
            "because its ST_MAIN supervision underweights that matchup and "
            "trusts resource-starved losing-seat behavior equally"
        ),
        "corpus": {
            "games": counts,
            "unique_game_uids": 20883,
            "unique_contents": 20883,
            "historical_july28_uid_overlap": 0,
            "historical_july28_content_overlap": 0,
            "split": (
                "preserved Jul17-26 train/validation plus deterministic "
                "matchup-stratified Jul28 90/10"
            ),
            "test_games": 0,
        },
        "training": {
            "changed_component": "ST_MAIN specialist only",
            "frozen_components": [
                "MD-v3 ST_CARD specialist",
                "Qu-v2B fallback",
                "public feature encoder and public backbone",
            ],
            "warm_start": "frozen MD-v3 main checkpoint",
            "kl_parent": "the same frozen MD-v3 main checkpoint",
            "architecture": [16, 48, 160, 112, 80],
            "target_deck_sha256": TARGET_DECK_SHA256,
            "target_select_type": 0,
            "matchup_card_id": 648,
            "matchup_definition": (
                "actor-relative opponent registered deck contains card 648"
            ),
            "non_matchup_outcome_weights": {
                "win": 1.0, "draw": 1.0, "loss": 1.0,
            },
            "matchup_outcome_weights": {
                "win": 1.0, "draw": 0.5, "loss": 0.25,
            },
            "arms": arms,
            "execution_order": ["51", "35", "65"],
            "epochs": 5,
            "batch_size": 128,
            "shuffle_buffer": 4096,
            "learning_rate": 0.00005,
            "weight_decay": 0.00001,
            "gradient_clip": 1.0,
            "value_coefficient": 0.0,
            "kl_coefficient": 0.1,
            "kl_weighting": "uniform-game",
            "game_normalized": True,
            "freeze_public_backbone": True,
            "seed": 20260729,
            "within_arm_selection": (
                "lowest weighted validation objective across five epochs; "
                "earliest epoch on exact tie"
            ),
            "test_deferred": True,
        },
        "validation_selection": {
            "models": [
                "frozen MD-v3 main",
                "35% arm validation-selected checkpoint",
                "51% arm validation-selected checkpoint",
                "65% arm validation-selected checkpoint",
            ],
            "cohort": "the fixed 2,090-game validation split only",
            "metric": (
                "unweighted, game-normalized ST_MAIN expert policy NLL "
                "for the exact target acting seat"
            ),
            "reported_strata": [
                "Grimmsnarl-family winning seats",
                "all Grimmsnarl-family target seats",
                "exact-list mirror winning seats",
                "all exact-list mirror target seats",
                "all non-Grimmsnarl target seats",
            ],
            "arm_pass": {
                "grim_family_winner": (
                    "candidate objective strictly lower than frozen MD-v3"
                ),
                "non_grim_noninferiority": (
                    "candidate objective no more than 0.010000 above "
                    "frozen MD-v3"
                ),
            },
            "selection": (
                "among passing arms choose lowest Grimmsnarl-family "
                "winning-seat objective; exact tie priority 51, then 35, then 65"
            ),
            "failure": "no passing arm means the retrain is rejected",
            "one_candidate_advances": True,
            "gameplay_and_temporal_outcomes_not_used_for_arm_selection": True,
        },
        "post_selection_gates": {
            "direct_exact_mirror": {
                "games": 1280,
                "seed": 2026073001,
                "seat_balance": "exact",
                "candidate": (
                    "selected ST_MAIN plus frozen MD-v3 ST_CARD plus "
                    "frozen Qu-v2B"
                ),
                "control": "byte-frozen MD-v3 package",
                "pass": (
                    "candidate score Wilson CI95 lower bound strictly above 0.50"
                ),
            },
            "recent_frequency_field": {
                "games_per_arm": 1280,
                "seed": 2026073002,
                "schedule": (
                    "one identical recent-frequency matchup and seat schedule"
                ),
                "pass": (
                    "candidate-minus-control point delta > 0 and conservative "
                    "independent CI95 lower bound > -0.05"
                ),
            },
            "untouched_temporal": {
                "cohort": (
                    "first complete official day with zero overlap with "
                    "inspected ladder replays; expected 2026-07-30"
                ),
                "inventory_and_zero_overlap_locked_before opening outcomes": True,
                "candidate_fixed_before_open": True,
                "models": ["selected candidate", "frozen MD-v3"],
                "metric": (
                    "same unweighted game-normalized ST_MAIN NLL strata as "
                    "validation"
                ),
                "corroboration": (
                    "candidate Grimmsnarl-family winning-seat objective lower "
                    "than frozen MD-v3 and non-Grim objective no more than "
                    "0.020000 above frozen MD-v3"
                ),
                "no_retraining_or_reselection_after_open": True,
            },
            "all_local": {
                "invalid_games": 0,
                "exceptions_fallbacks_legality_repairs": 0,
                "threshold_changes_or_posthoc_arms": False,
            },
        },
        "ladder_rule_after_all_local_gates": {
            "slots": [
                "one byte-frozen MD-v3 paired control",
                "one candidate package",
            ],
            "minimum_resolved_games_per_arm": 80,
            "comparison": (
                "same-window resolved W/L score; rating is descriptive at "
                "low counts"
            ),
            "falsifies_replacement": (
                "candidate same-window resolved score is not strictly above "
                "the frozen MD-v3 control after both reach 80 games"
            ),
            "report_grim_family_and_exact_mirror_substrata": True,
            "upload_requires_user_named_tag_and_explicit_approval": True,
        },
        "shipping_audit_if_selected": {
            "tests": "tests/test_safety.py",
            "random_smoke_games": 200,
            "exact_extracted_tarball_non_owner_uid": True,
            "one_changed_runtime_component": "ST_MAIN weights only",
        },
        "artifacts": artifacts,
        "prelock_checks": {
            "trainer_unit_tests": "19 passed",
            "command": "python tests/test_qu_v2a_training.py",
        },
    }


def main() -> int:
    if OUTPUT.exists():
        print(f"error: refusing to overwrite {OUTPUT}", file=sys.stderr)
        return 2
    try:
        payload = build_lock()
        payload["lock_sha256"] = value_sha256(payload)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, KeyError, TypeError, ValueError, LockError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "lock": str(OUTPUT),
        "file_sha256": file_sha256(OUTPUT),
        "lock_sha256": payload["lock_sha256"],
        "arms": payload["training"]["arms"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
