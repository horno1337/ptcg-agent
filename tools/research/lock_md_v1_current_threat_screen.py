"""Lock the current-meta MD-v1 threat screen before running it."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "ptcg.md-v1.current-threat-screen-lock.v1"
ARTIFACTS = {
    "eval_ab": ("tools/eval_ab.py",
                "f340aff9a477402d9762b1ca7291bd5ba04c09805045c7ca2567348c32d7d95b"),
    "qu_v2b": ("agent/weights.npz",
               "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"),
    "md_v1": ("tools/checkpoints/md-v1/md-v1-weights.npz",
              "5784b7ea693d2adc3a8cf0b7422b6940481042181051ef2941b982094dabe78c"),
    "recent_meta": (
        "tools/checkpoints/md-v1-current-threats-v1/recent-threat-meta.json",
        "16446c94664f57b4e24acf2dd7f1a33641b82541fa4b3ba07ee63b039a8eba38"),
    "recent_audit": (
        "tools/checkpoints/md-v1-current-threats-v1/recent-meta-audit.json",
        "0cd92977d5b69dbde41fe5c9ae67d860132f2540bd68143d6fdf56d284de88cd"),
}


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
        value, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def build_lock() -> dict[str, Any]:
    artifacts = {}
    for name, (relative, expected) in ARTIFACTS.items():
        observed = file_sha256(ROOT / relative)
        if observed != expected:
            raise LockError(f"{name} hash drift: {observed}")
        artifacts[name] = {"path": relative, "sha256": observed}
    panels = {}
    for threat, opponent, seed in (
        ("lucario", "meta:1", 20260733),
        ("archaludon_cinderace", "meta:2", 20260734),
    ):
        panels[f"md_{threat}"] = {
            "games_per_arm": 160,
            "seed": seed,
            "learner_deck": "meta:0",
            "opponent": opponent,
            "read_arm": "candidate-field",
            "candidate": "md_v1",
            "output":
                f"tools/checkpoints/md-v1-current-threats-v1/md-{threat}.json",
        }
        panels[f"alakazam_{threat}"] = {
            "games_per_arm": 160,
            "seed": seed,
            "learner_deck": "self",
            "opponent": opponent,
            "read_arm": "candidate-field",
            "candidate": "qu_v2b",
            "output":
                f"tools/checkpoints/md-v1-current-threats-v1/alakazam-{threat}.json",
        }
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_results": True,
        "artifacts": artifacts,
        "threats": {
            "lucario": {
                "meta_index": 1,
                "canonical_sha256":
                    "798608188147ee6641b079f320dfeb03a329452bc1cbc51cc3f8eedb8d53ef2d",
            },
            "archaludon_cinderace": {
                "meta_index": 2,
                "canonical_sha256":
                    "6449a1311f0c40a358f0fe63f9cc9f1cceb67f1f9b7f5a255fd7c764bff2cde2",
            },
        },
        "panels": panels,
        "common": {
            "base": "qu_v2b",
            "meta": "recent_meta",
            "opponent_policy": "reflex",
        },
        "decision_rule": {
            "all_read_arms_valid_with_zero_fallbacks": True,
            "root_mining_target_if_either": [
                "md_score < 0.50",
                "md_score <= alakazam_score - 0.05",
            ],
            "no_target_from_historical_meta2_meta5_meta7_results": True,
            "screen_does_not_authorize_training": True,
            "no_post_result_threshold_change": True,
        },
    }


def load_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("lock_sha256", None)
    if payload.get("schema") != SCHEMA or value_sha256(payload) != recorded:
        raise LockError("lock hash mismatch")
    payload["lock_sha256"] = recorded
    return payload


def main() -> int:
    output = ROOT / "tools/checkpoints/md-v1-current-threats-v1/lock.json"
    if output.exists():
        print(f"error: lock exists: {output}", file=sys.stderr)
        return 2
    try:
        payload = build_lock()
        payload["lock_sha256"] = value_sha256(payload)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, LockError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"locked {output}")
    print(f"sha256={payload['lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
