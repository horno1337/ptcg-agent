"""Adjudicate the consumed smoke after its legacy route-count helper misfired."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import smoke_md_v2_card_v1 as SMOKE  # noqa: E402


RUN = ROOT / "tools/checkpoints/md-v2-card-v1"


def main() -> int:
    lock_path = RUN / "random-smoke-lock.json"
    result_path = RUN / "random-smoke-result.json"
    output = RUN / "random-smoke-adjudication.json"
    lock = COMMON.load_self_hashed_json(
        lock_path, schema=SMOKE.LOCK_SCHEMA, hash_key="lock_sha256"
    )
    result = COMMON.load_self_hashed_json(
        result_path, schema=SMOKE.RESULT_SCHEMA, hash_key="result_sha256"
    )
    diagnostics = result["controller"]
    summary = result["summary"]
    route_identity = (
        diagnostics["calls"]
        == diagnostics["main_routes"]
        + diagnostics["card_routes"]
        + diagnostics["qu_routes"]
    )
    written_rule_passed = (
        result["smoke_lock_sha256"] == lock["lock_sha256"]
        and summary["scheduled_games"] == 200
        and summary["invalid"] == 0
        and summary["gate_valid"] is True
        and diagnostics["exceptions"] == {}
        and diagnostics["fallbacks"] == 0
        and diagnostics["repairs"] == 0
        and diagnostics["off_deck_main_routes"] == 0
        and diagnostics["off_deck_card_routes"] == 0
        and diagnostics["card_routes"] > 0
        and route_identity
    )
    payload = {
        "schema": "ptcg.md-v2-card-v1.random-smoke-adjudication.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke_lock_sha256": lock["lock_sha256"],
        "smoke_result_sha256": result["result_sha256"],
        "outcomes_reused_without_rerun": True,
        "rule_unchanged": lock["protocol"]["pass_rule"],
        "incident": {
            "kind": "evaluator-integration",
            "legacy_predicate": "calls == main_routes + qu_routes",
            "correct_route_identity": (
                "calls == main_routes + card_routes + qu_routes"
            ),
            "observed": {
                "calls": diagnostics["calls"],
                "main_routes": diagnostics["main_routes"],
                "card_routes": diagnostics["card_routes"],
                "qu_routes": diagnostics["qu_routes"],
            },
        },
        "decision": {
            "passed": written_rule_passed,
            "route_identity_passed": route_identity,
            "original_boolean_superseded": written_rule_passed,
        },
    }
    payload["adjudication_sha256"] = COMMON.canonical_sha256(payload)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["decision"], sort_keys=True))
    return 0 if written_rule_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
