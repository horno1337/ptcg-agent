"""Lock generation two of the exact-mirror league with runtime snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools.research import lock_md_v3_mirror_league_100k as V1


SCHEMA = "ptcg.md-v3.exact-mirror-league-100k-lock.v2"
RUN_ROOT = V1.ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2"
DEFAULT_LOCK = RUN_ROOT / "training-lock.json"
DEFAULT_OUTPUT = RUN_ROOT / "training"
UPDATES = V1.UPDATES
GAMES_PER_UPDATE = V1.GAMES_PER_UPDATE
TOTAL_GAMES = V1.TOTAL_GAMES
PAIRS_PER_UPDATE = V1.PAIRS_PER_UPDATE
SEATS_PER_UPDATE = V1.SEATS_PER_UPDATE
BASE_SEED = V1.BASE_SEED
SEED_STRIDE = V1.SEED_STRIDE
ROLLOUT_SEEDS = V1.ROLLOUT_SEEDS
PPO_SEEDS = V1.PPO_SEEDS


def _configure() -> None:
    V1.SCHEMA = SCHEMA
    V1.RUN_ROOT = RUN_ROOT
    V1.DEFAULT_LOCK = DEFAULT_LOCK
    V1.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    V1.ARTIFACT_PATHS = dict(V1.ARTIFACT_PATHS)
    V1.ARTIFACT_PATHS.update({
        "lock_builder": Path(__file__).resolve(),
        "wrapper": V1.ROOT / "tools/research/run_md_v3_mirror_league_100k_v2.py",
    })


def canonical_sha256(value):
    return V1.canonical_sha256(value)


def verify_bound_artifacts(lock):
    _configure()
    return V1.verify_bound_artifacts(lock)


def load_lock(path: Path, *, verify_artifacts: bool = True):
    _configure()
    return V1.load_lock(path, verify_artifacts=verify_artifacts)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_LOCK))
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    _configure()
    payload = V1.build_lock()
    payload["correction"] = {
        "supersedes_lock": "f79f6941b96543d97309b456c159cb5be7b2863d0fc5d33c3e28c84186766c05",
        "reason": "generation one snapshot ST_MAIN used a training-side feature adapter and failed closed at update 11",
        "retained_outcome": None,
        "restart": "fresh MD-v3 initialization; no generation-one checkpoint selected",
    }
    payload.pop("lock_sha256", None)
    payload["lock_sha256"] = V1.canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": V1.display_path(output), "lock_sha256": payload["lock_sha256"], "games": V1.TOTAL_GAMES}, sort_keys=True))


_configure()


if __name__ == "__main__":
    main()
