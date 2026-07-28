"""Freeze the destination-only guard and its reserved-date decision rule."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_grim_damage_guard_v1 as PHASE  # noqa: E402


RUN = ROOT / "tools/checkpoints/grim-damage-guard-v1"
DEFAULT_PHASE_LOCK = PHASE.DEFAULT_OUTPUT
DEFAULT_DISCOVERY = RUN / "discovery-destination-only.json"
DEFAULT_OUTPUT = RUN / "rule-lock.json"


def build_rule_lock(
    phase_lock: dict,
    discovery: dict,
    *,
    artifacts: dict[str, dict[str, str]],
    created_at: str,
) -> dict:
    if discovery.get("reserved_dates_opened") != []:
        raise ValueError("discovery opened reserved dates")
    if discovery.get("phase_lock_sha256") != phase_lock.get("lock_sha256"):
        raise ValueError("discovery does not match phase lock")
    if discovery.get("dates_opened") != list(PHASE.DISCOVERY_DATES):
        raise ValueError("discovery date scope drifted")
    summary = discovery.get("summary") or {}
    if summary.get("guard_only_right", 0) <= summary.get("base_only_right", 0):
        raise ValueError("discovery did not support destination-only guard")

    payload = {
        "schema": "ptcg.grim-damage-guard.rule-lock.v1",
        "created_at": created_at,
        "locked_before_reserved_action_extraction": True,
        "phase_lock_sha256": phase_lock["lock_sha256"],
        "discovery_summary_sha256": discovery["summary_sha256"],
        "rule": {
            "eligible_subtypes": ["adrena_target", "shadow_target"],
            "adrena": (
                "after frozen Qu-v2B chooses source and counter count, override "
                "only when the observed moved amount has a visible KO target"
            ),
            "shadow_bullet": (
                "override only when the mandatory 30 bench damage has a visible "
                "KO target"
            ),
            "ko_ranking": (
                "two-prize Grimmsnarl ex, then powered Munkidori, then "
                "Munkidori, then Grimmsnarl line, then lower overkill"
            ),
            "source_and_count": "always frozen Qu-v2B; discovery rejected override",
            "registered_deck": PHASE.TARGET_DECK_SHA256,
            "opponent_scope": (
                "publicly revealed Marnie's Grimmsnarl line; hidden opponent "
                "registration is unavailable at runtime"
            ),
            "fail_closed": True,
            "default_runtime_flag": "off",
        },
        "cohort": {
            "dates": list(PHASE.EVALUATION_DATES),
            "games": phase_lock["cohorts"]["evaluation"]["games"],
            "matchup": "exact-deck Grimmsnarl mirror",
            "all_dates_and_both_seats_retained": True,
        },
        "thresholds": {
            "minimum_combined_triggers": 1500,
            "minimum_triggers_per_subtype": 500,
            "minimum_guard_agreement_per_subtype": 0.90,
            "paired_overall": "guard_only_right > base_only_right",
            "paired_winner_seats": "guard_only_right > base_only_right",
            "paired_each_subtype": "guard_only_right >= base_only_right",
            "invalid_guard_actions": 0,
        },
        "decision": {
            "pass": "every threshold must pass",
            "failure": (
                "reject this entire guard; no subtype, threshold, date, or "
                "ranking revision after reserved evaluation opens"
            ),
            "offline_status": "diagnostic proposal only",
            "promotion": (
                "requires the separately locked 640-game mirror A/B after "
                "the unchanged MD-v2 base is accepted"
            ),
        },
        "discovery_evidence_used_to_freeze_rule": {
            "games": discovery["games_opened"],
            "guard_triggers": summary["guard_triggers"],
            "guard_only_right": summary["guard_only_right"],
            "base_only_right": summary["base_only_right"],
        },
        "artifacts": artifacts,
    }
    payload["lock_sha256"] = PHASE.value_sha256(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-lock", type=Path, default=DEFAULT_PHASE_LOCK)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite rule lock: {args.output}")
    phase_lock = json.loads(args.phase_lock.read_bytes())
    discovery = json.loads(args.discovery.read_bytes())
    phase_without = dict(phase_lock)
    if PHASE.value_sha256({
        key: value for key, value in phase_without.items()
        if key != "lock_sha256"
    }) != phase_lock.get("lock_sha256"):
        raise SystemExit("phase lock self-hash is invalid")
    discovery_without = dict(discovery)
    if PHASE.value_sha256({
        key: value for key, value in discovery_without.items()
        if key != "summary_sha256"
    }) != discovery.get("summary_sha256"):
        raise SystemExit("discovery summary self-hash is invalid")

    paths = {
        "guard": ROOT / "agent/grim_damage_guard.py",
        "policy_integration": ROOT / "agent/policy.py",
        "frozen_qu_v2b": ROOT / "agent/weights.npz",
        "discovery_summary": args.discovery.resolve(),
        "discovery_analyzer": (
            ROOT / "tools/research/analyze_grim_damage_guard_discovery.py"
        ),
        "reserved_evaluator": (
            ROOT / "tools/research/evaluate_grim_damage_guard_reserved.py"
        ),
        "guard_tests": ROOT / "tests/test_grim_damage_guard.py",
        "evaluator_tests": (
            ROOT / "tests/test_evaluate_grim_damage_guard_reserved.py"
        ),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"missing rule-lock artifact {label}: {path}")
    artifacts = {
        label: {
            "path": str(path.relative_to(ROOT)),
            "sha256": PHASE.file_sha256(path),
        }
        for label, path in paths.items()
    }
    payload = build_rule_lock(
        phase_lock,
        discovery,
        artifacts=artifacts,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["thresholds"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(payload["lock_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
