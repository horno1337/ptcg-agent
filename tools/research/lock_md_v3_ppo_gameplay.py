"""Bind the accepted PPO candidate to the existing 1,280-game mirror gate."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import train_md_v3_ppo as PPO  # noqa: E402


def record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": COMMON.file_sha256(resolved),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error("--output already exists")
    source_path = ROOT / "tools/checkpoints/md-v3-mirror-main-v1/gameplay-lock.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    training_lock = ROOT / "tools/checkpoints/md-v3-ppo-v1/training-lock.json"
    training_result = ROOT / "tools/checkpoints/md-v3-ppo-v1/training/result.json"
    result = json.loads(training_result.read_text(encoding="utf-8"))
    if result.get("training_gate", {}).get("passed") is not True:
        raise SystemExit("PPO training gate did not pass")
    candidate_weights = (
        ROOT / result["artifacts"]["weights"]["path"]
    )
    candidate_checkpoint = (
        ROOT / result["artifacts"]["checkpoint"]["path"]
    )
    artifacts = dict(source["artifacts"])
    artifacts.update({
        "candidate_checkpoint": record(candidate_checkpoint),
        "candidate_main_weights": record(candidate_weights),
        "candidate_training_provenance": record(training_result),
        "validation_result": record(training_result),
        "source_lock": record(training_lock),
    })
    payload = {
        "schema": "ptcg.md-v3.mirror-main-gameplay-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "The locked terminal-outcome PPO ST_MAIN candidate beats complete "
            "frozen MD-v3 in the exact Grimmsnarl mirror."
        ),
        "candidate": {
            "route": (
                "PPO ST_MAIN plus frozen MD-v3 ST_CARD plus frozen Qu-v2B"
            ),
            "training_lock_sha256": json.loads(
                training_lock.read_text(encoding="utf-8"),
            )["lock_sha256"],
            "training_result_sha256": COMMON.file_sha256(training_result),
        },
        "protocol": source["protocol"],
        "schedule": source["schedule"],
        "artifacts": artifacts,
        "interpretation": {
            "training_rollout_wl": "ignored; sampled exploration policy",
            "decision_rule": source["protocol"]["decision_rule"],
            "post_failure": "retire PPO route; do not run field gate or upload",
            "post_pass": (
                "lock recent-frequency field noninferiority before packaging"
            ),
        },
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    PPO.atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
