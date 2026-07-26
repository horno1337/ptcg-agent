"""Aggregate the pre-registered MD-v1 deck-selection gate."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_md_v1_deck_gate as LOCK  # noqa: E402


SCHEMA = "ptcg.md-v1.deck-gate-result.v1"


class GateError(RuntimeError):
    pass


def _load_report(lock: dict[str, Any], panel: str) -> dict[str, Any]:
    spec = lock["panels"][panel]
    report = json.loads((ROOT / spec["output"]).read_text(encoding="utf-8"))
    args = report["args"]
    expected = {
        "games": spec["games_per_arm"],
        "seed": spec["seed"],
        "learner_deck": spec["learner_deck"],
        "opp": spec["opponent"],
        "opp_policy": spec["opponent_policy"],
        "num_shards": 1,
        "shard_index": 0,
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise GateError(f"{panel} argument drift for {key}")
    if report.get("eval_ab_sha256") != lock["artifacts"]["eval_ab"]["sha256"]:
        raise GateError(f"{panel} evaluator hash drift")
    if report.get("base_sha256") != lock["artifacts"]["qu_v2b"]["sha256"]:
        raise GateError(f"{panel} base hash drift")
    expected_candidate = lock["artifacts"][spec["candidate"]]["sha256"]
    if report.get("candidate_sha256") != expected_candidate:
        raise GateError(f"{panel} candidate hash drift")
    return report


def _summaries(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["summary"]["tag"]: row["summary"] for row in report["results"]}


def _valid(summary: dict[str, Any]) -> bool:
    controller = summary["controller"]
    return (
        summary["scheduled_games"] == 160
        and summary["gate_valid"]
        and summary["invalid"] == 0
        and controller["fallbacks"] == 0
        and not controller["exceptions"]
    )


def main() -> int:
    lock_path = ROOT / "tools/checkpoints/md-v1-deck-gate-v1/lock.json"
    output = ROOT / "tools/checkpoints/md-v1-deck-gate-v1/result.json"
    try:
        lock = LOCK.load_lock(lock_path)
        grim = _summaries(_load_report(lock, "grim_field"))
        ala = _summaries(_load_report(lock, "alakazam_field"))
        direct = _summaries(_load_report(lock, "direct_alakazam"))
        md = grim["candidate-field"]
        qu_grim = grim["base-field"]
        qu_ala = ala["candidate-field"]
        md_direct = direct["candidate-field"]

        matchup_passes = 0
        maximum_regression = 0.0
        strata = {}
        for key in sorted(set(md["by_matchup"]) | set(qu_ala["by_matchup"])):
            m = md["by_matchup"].get(key, {})
            a = qu_ala["by_matchup"].get(key, {})
            mn = sum(m.values())
            an = sum(a.values())
            ms = (m.get("win", 0) + 0.5 * m.get("draw", 0)) / mn
            ass = (a.get("win", 0) + 0.5 * a.get("draw", 0)) / an
            regression = 100.0 * (ass - ms)
            maximum_regression = max(maximum_regression, regression)
            passed = regression <= 10.0 + 1e-12
            matchup_passes += passed
            strata[key] = {
                "md_score": ms, "qu_alakazam_score": ass,
                "md_minus_alakazam_pp": 100.0 * (ms - ass),
                "within_10pp": passed,
            }

        checks = {
            "all_panels_valid": all(
                _valid(row) for row in (md, qu_grim, qu_ala, md_direct)),
            "md_above_qu_grim": md["score"] > qu_grim["score"],
            "md_above_qu_alakazam": md["score"] > qu_ala["score"],
            "direct_above_50": md_direct["score"] > 0.5,
            "at_least_6_of_8_strata_within_10pp": matchup_passes >= 6,
            "no_stratum_regresses_more_than_25pp":
                maximum_regression <= 25.0 + 1e-12,
        }
        result = {
            "schema": SCHEMA,
            "lock_sha256": lock["lock_sha256"],
            "arms": {
                "md_grim_field": md,
                "qu_grim_field": qu_grim,
                "qu_alakazam_field": qu_ala,
                "md_grim_direct_vs_qu_alakazam": md_direct,
            },
            "strata": strata,
            "checks": checks,
            "gate_passed": all(checks.values()),
        }
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            LOCK.LockError, GateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "scores": {key: value["score"] for key, value in result["arms"].items()},
        "checks": checks,
        "gate_passed": result["gate_passed"],
    }, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
