"""Aggregate the locked current-meta MD-v1 threat screen."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_md_v1_current_threat_screen as LOCK  # noqa: E402


def _read(lock: dict[str, Any], name: str) -> dict[str, Any]:
    spec = lock["panels"][name]
    report = json.loads((ROOT / spec["output"]).read_text(encoding="utf-8"))
    args = report["args"]
    expected = {
        "games": 160,
        "seed": spec["seed"],
        "learner_deck": spec["learner_deck"],
        "opp": spec["opponent"],
        "opp_policy": "reflex",
        "num_shards": 1,
        "shard_index": 0,
    }
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{name} argument drift")
    if report["eval_ab_sha256"] != lock["artifacts"]["eval_ab"]["sha256"]:
        raise ValueError(f"{name} evaluator drift")
    if report["base_sha256"] != lock["artifacts"]["qu_v2b"]["sha256"]:
        raise ValueError(f"{name} base drift")
    if report["candidate_sha256"] != lock["artifacts"][spec["candidate"]]["sha256"]:
        raise ValueError(f"{name} candidate drift")
    return next(
        row["summary"] for row in report["results"]
        if row["summary"]["tag"] == spec["read_arm"]
    )


def _valid(row: dict[str, Any]) -> bool:
    return (
        row["scheduled_games"] == 160
        and row["gate_valid"]
        and row["invalid"] == 0
        and row["controller"]["fallbacks"] == 0
        and not row["controller"]["exceptions"]
    )


def main() -> int:
    lock_path = ROOT / "tools/checkpoints/md-v1-current-threats-v1/lock.json"
    output = ROOT / "tools/checkpoints/md-v1-current-threats-v1/result.json"
    try:
        lock = LOCK.load_lock(lock_path)
        rows = {name: _read(lock, name) for name in lock["panels"]}
        threats = {}
        for threat in ("lucario", "archaludon_cinderace"):
            md = rows[f"md_{threat}"]
            ala = rows[f"alakazam_{threat}"]
            target = md["score"] < 0.5 or md["score"] <= ala["score"] - 0.05
            threats[threat] = {
                "md": md,
                "alakazam": ala,
                "md_minus_alakazam_pp": 100.0 * (md["score"] - ala["score"]),
                "root_mining_target": target,
            }
        result = {
            "schema": "ptcg.md-v1.current-threat-screen-result.v1",
            "lock_sha256": lock["lock_sha256"],
            "all_valid": all(_valid(row) for row in rows.values()),
            "threats": threats,
            "root_mining_targets": [
                name for name, row in threats.items()
                if row["root_mining_target"]
            ],
            "training_authorized": False,
        }
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            LOCK.LockError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "all_valid": result["all_valid"],
        "root_mining_targets": result["root_mining_targets"],
        "scores": {
            name: {
                "md": row["md"]["score"],
                "alakazam": row["alakazam"]["score"],
                "delta_pp": row["md_minus_alakazam_pp"],
            }
            for name, row in threats.items()
        },
    }, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
