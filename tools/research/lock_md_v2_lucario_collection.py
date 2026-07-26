"""Write the pre-outcome lock for fresh MD-v2 Lucario root collection."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tools/checkpoints/md-v2-lucario-roots-v1"
FILES = {
    "collector": ROOT / "tools/research/collect_md_v2_lucario_roots.py",
    "candidate": ROOT / "tools/checkpoints/md-v1/md-v1-weights.npz",
    "base": ROOT / "agent/weights.npz",
    "recent_meta_audit": (
        ROOT / "tools/checkpoints/md-v1-current-threats-v1/recent-meta-audit.json"
    ),
    "recent_threat_meta": (
        ROOT / "tools/checkpoints/md-v1-current-threats-v1/recent-threat-meta.json"
    ),
    "current_threat_screen_lock": (
        ROOT / "tools/checkpoints/md-v1-current-threats-v1/lock.json"
    ),
    "current_threat_screen_result": (
        ROOT / "tools/checkpoints/md-v1-current-threats-v1/result.json"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for name, path in FILES.items():
        if not path.is_file():
            raise SystemExit(f"missing {name}: {path}")
    lock = {
        "schema": "ptcg.md-v2.current-lucario-collection-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_before_collection_outcomes",
        "motivation": {
            "current_lucario_screen": {
                "md_v1_grim_score": 0.45,
                "qu_v2b_alakazam_score": 0.60625,
                "md_minus_alakazam_pp": -15.625,
            },
            "recent_exact_ladder_supply": {
                "games": 6,
                "grim_losses": 5,
                "eligible_unburned_losses": 4,
                "conclusion": "insufficient for the locked 30-game diversity floor",
            },
        },
        "collection": {
            "games": 320,
            "seed": 20260736,
            "learner_meta_index": 0,
            "opponent_meta_index": 1,
            "learner": "MD-v1 ST_MAIN with frozen Qu-v2B elsewhere",
            "opponent": "frozen Qu-v2B piloting current exact Lucario",
            "balanced_learner_seats": True,
            "retain": (
                "loss games only; supported one-pick MAIN roots only; "
                "MD-v1/Qu-v2B semantic disagreements only"
            ),
            "outcome_semantics": (
                "loss selects a query stratum and is not an action-value label"
            ),
        },
        "supply_read": {
            "minimum_distinct_loss_games_with_roots": 30,
            "minimum_raw_roots": 500,
            "if_short": (
                "declare insufficient supply; do not relax filters or use "
                "winning games; lock a new disjoint collection schedule"
            ),
            "if_pass": (
                "deduplicate by full public-root fingerprint, then lock a "
                "discovery cohort before any paired-rollout outcomes are read"
            ),
        },
        "downstream_protocol_not_yet_authorized": {
            "discovery_rollouts_per_action_pair": 32,
            "preliminary_significance": "abs(delta) > 1.96 * paired_SE",
            "independent_confirmation_rollouts_per_action_pair": 32,
            "minimum_sign_agreement": 0.85,
            "minimum_confirmed_roots": 200,
            "minimum_confirmed_games": 30,
            "training_authorized_by_this_collection": False,
        },
        "burned_data": {
            "first_md_v2_rollout_cohort": (
                "tools/checkpoints/qu-v2c-canary-v1/"
                "md_v2_rollout_cohort_lock.json"
            ),
            "rule": (
                "no first-attempt game or root is reusable; all games in this "
                "collection are newly simulated and disjoint"
            ),
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in FILES.items()
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "collection-lock.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing lock: {path}")
    path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(sha256(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
