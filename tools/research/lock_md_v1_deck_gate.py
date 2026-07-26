"""Pre-register the MD-v1 160-game deck-selection gate."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "ptcg.md-v1.deck-gate-lock.v1"
EXPECTED_HASHES = {
    "eval_ab": ("tools/eval_ab.py",
                "f340aff9a477402d9762b1ca7291bd5ba04c09805045c7ca2567348c32d7d95b"),
    "qu_v2b": ("agent/weights.npz",
               "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"),
    "md_v1": ("tools/checkpoints/md-v1/md-v1-weights.npz",
              "5784b7ea693d2adc3a8cf0b7422b6940481042181051ef2941b982094dabe78c"),
    "meta_decks": ("agent/meta_decks.json",
                   "0b98313febb5893cc109fa1b2fcd644db044a5151b0331fa62e347fb83dcfdc9"),
    "engine": ("engine/libcg.so",
               "73cc06473da0645d87d3c820ddec3c078610d95c24ee1080efcd98ef50650db7"),
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
    for name, (relative, expected) in EXPECTED_HASHES.items():
        path = ROOT / relative
        observed = file_sha256(path)
        if observed != expected:
            raise LockError(f"{name} hash drift: {observed}")
        artifacts[name] = {"path": relative, "sha256": observed}
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_before_results": True,
        "artifacts": artifacts,
        "decks": {
            "alakazam": {
                "learner_spec": "self",
                "canonical_sha256":
                    "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf",
            },
            "grimmsnarl": {
                "learner_spec": "meta:4",
                "canonical_sha256":
                    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af",
            },
        },
        "panels": {
            "grim_field": {
                "games_per_arm": 160,
                "seed": 20260731,
                "learner_deck": "meta:4",
                "opponent": "pool:8",
                "opponent_policy": "reflex",
                "candidate": "md_v1",
                "base": "qu_v2b",
                "output": "tools/checkpoints/md-v1-deck-gate-v1/grim-field.json",
            },
            "alakazam_field": {
                "games_per_arm": 160,
                "seed": 20260731,
                "learner_deck": "self",
                "opponent": "pool:8",
                "opponent_policy": "reflex",
                "candidate": "qu_v2b",
                "base": "qu_v2b",
                "predeclared_read_arm": "candidate-field",
                "output": "tools/checkpoints/md-v1-deck-gate-v1/alakazam-field.json",
            },
            "direct_alakazam": {
                "games_per_arm": 160,
                "seed": 20260732,
                "learner_deck": "meta:4",
                "opponent": "meta:0",
                "opponent_policy": "reflex",
                "candidate": "md_v1",
                "base": "qu_v2b",
                "predeclared_read_arm": "candidate-field",
                "output":
                    "tools/checkpoints/md-v1-deck-gate-v1/direct-alakazam.json",
            },
        },
        "decision_rule": {
            "all_panels_must_be_valid": True,
            "runtime_exceptions_or_fallbacks_allowed": 0,
            "md_grim_field_score_strictly_above_qu_grim": True,
            "md_grim_field_score_strictly_above_qu_alakazam": True,
            "md_grim_direct_alakazam_score_strictly_above": 0.5,
            "strata": {
                "minimum_strata_with_md_no_worse_than_alakazam_minus_pp": 6,
                "no_worse_margin_pp": 10.0,
                "maximum_single_stratum_regression_pp": 25.0,
            },
            "no_post_result_extension_or_threshold_change": True,
        },
        "interpretation": {
            "pass": "MD-v1/Grimmsnarl is an authorized deck-policy challenger.",
            "adapter_only": "MD-v1 improves Grimmsnarl but does not replace Alakazam.",
            "fail": "Retain MD-v1 as a research canary; do not promote the deck.",
            "first_release_context":
                "Failure does not close the MD line; this gate establishes MD-v1's baseline.",
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
    output = ROOT / "tools/checkpoints/md-v1-deck-gate-v1/lock.json"
    if output.exists():
        print(f"error: lock already exists: {output}", file=sys.stderr)
        return 2
    try:
        payload = build_lock()
        payload["lock_sha256"] = value_sha256(payload)
        output.parent.mkdir(parents=True, exist_ok=True)
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
