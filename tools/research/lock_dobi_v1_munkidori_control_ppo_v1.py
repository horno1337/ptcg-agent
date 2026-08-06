"""Prospectively lock the 99,840-game Dobi Munkidori-control PPO run."""

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

from tools.research import dobi_v1_mirror_league_300k_population as POP  # noqa: E402
from tools.research import lock_dobi_v1_mirror_league_300k as OLD  # noqa: E402


BASE = OLD.BASE
SCHEMA = "ptcg.dobi-v1.munkidori-control-ppo-v1-lock.v1"
RUN_ROOT = ROOT / "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1"
DEFAULT_LOCK = RUN_ROOT / "training-lock.json"
DEFAULT_OUTPUT = RUN_ROOT / "training"
UPDATES = 130
GAMES_PER_UPDATE = 768
TOTAL_GAMES = UPDATES * GAMES_PER_UPDATE
PAIRS_PER_UPDATE = 384
SEATS_PER_UPDATE = {"0": 384, "1": 384}
BASE_SEED = 2_026_080_601
SEED_STRIDE = 1_000_033
ROLLOUT_SEEDS = [BASE_SEED + i * SEED_STRIDE for i in range(UPDATES)]
PPO_SEEDS = [BASE_SEED + (UPDATES + i) * SEED_STRIDE for i in range(UPDATES)]


ARTIFACT_PATHS = {
    **OLD.ARTIFACT_PATHS,
    "lock_builder": Path(__file__).resolve(),
    "wrapper": ROOT / "tools/research/run_dobi_v1_munkidori_control_ppo_v1.py",
    "population": Path(POP.__file__).resolve(),
    "reward": ROOT / "tools/research/dobi_v1_munkidori_control_reward.py",
    "preregistration": ROOT / "tools/research/dobi-v1-munkidori-control-ppo-v1-preregistration.md",
}


def _configure() -> None:
    BASE.SCHEMA = SCHEMA
    BASE.UPDATES = UPDATES
    BASE.GAMES_PER_UPDATE = GAMES_PER_UPDATE
    BASE.TOTAL_GAMES = TOTAL_GAMES
    BASE.PAIRS_PER_UPDATE = PAIRS_PER_UPDATE
    BASE.SEATS_PER_UPDATE = dict(SEATS_PER_UPDATE)
    BASE.BASE_SEED = BASE_SEED
    BASE.SEED_STRIDE = SEED_STRIDE
    BASE.ROLLOUT_SEEDS = list(ROLLOUT_SEEDS)
    BASE.PPO_SEEDS = list(PPO_SEEDS)
    BASE.RUN_ROOT = RUN_ROOT
    BASE.DEFAULT_LOCK = DEFAULT_LOCK
    BASE.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    BASE.ARTIFACT_PATHS = dict(ARTIFACT_PATHS)
    BASE.POP = POP


def canonical_sha256(value):
    return BASE.canonical_sha256(value)


def verify_bound_artifacts(lock):
    _configure()
    return BASE.verify_bound_artifacts(lock)


def load_lock(path: Path, *, verify_artifacts: bool = True):
    _configure()
    return BASE.load_lock(path, verify_artifacts=verify_artifacts)


def build_lock():
    _configure()
    payload = BASE.build_lock()
    payload["hypothesis"] = (
        "A fixed 99,840-game exact-mirror PPO continuation from frozen Dobi-v1, "
        "using locked public Munkidori-control potential shaping, improves direct "
        "mirror gameplay without changing deck, routing, ST_CARD, or Qu-v2B."
    )
    payload["scope"] = {
        "learner": "byte-frozen Dobi-v1 Torch/NumPy-identical initialization",
        "trainable": "ST_MAIN actor and private critic only",
        "frozen": ["shared representation", "ST_CARD", "Qu-v2B residual", "deck"],
        "deployment_authorized": False,
    }
    payload["training"].update({
        "actor_learning_rate": 2.5e-6,
        "maximum_parent_kl_per_update": 0.04,
        "maximum_final_parent_kl": 0.04,
        "parent": "frozen Dobi-v1; fresh optimizer; no optimizer state inherited",
        "reward": {
            "contract": "public-potential-v1",
            "terminal": "win/draw/loss = +1/0/-1",
            "gamma": 0.997,
            "coefficient": 0.15,
            "phi": "clip(0.50*powered_munkidori_diff + 0.25*all_munkidori_diff, -1, 1)",
            "terminal_phi": 0.0,
        },
    })
    payload["decision_rules"] = {
        "candidate": "only fixed terminal update 130 is selection-eligible",
        "intermediates": "recovery-only; never deployable or post-hoc selectable",
        "gameplay_gate": "fixed 10,240-game exact-mirror A/B versus frozen Dobi-v1; Wilson CI95 lower bound > 50%",
        "field_gate": "only after mirror pass; 5,120-game recent non-mirror field A/B with delta CI95 lower bound >= -1.5pp",
        "upload": "requires deployment audits and a separately user-approved name",
    }
    payload["frozen_parent_evidence"] = {
        "archive_sha256": payload["artifacts"]["dobi_archive"]["sha256"],
        "weights_sha256": payload["artifacts"]["parent_weights"]["sha256"],
        "checkpoint_sha256": payload["artifacts"]["parent_checkpoint"]["sha256"],
        "mirror_gate_sha256": payload["artifacts"]["dobi_mirror_gate"]["sha256"],
        "field_gate_sha256": payload["artifacts"]["dobi_field_gate"]["sha256"],
        "exact_tarball_audit_sha256": payload["artifacts"]["dobi_exact_tarball_audit"]["sha256"],
    }
    payload.pop("lock_sha256", None)
    payload["lock_sha256"] = BASE.canonical_sha256(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_LOCK))
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    payload = build_lock()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "path": BASE.display_path(output),
        "lock_sha256": payload["lock_sha256"],
        "updates": UPDATES,
        "games": TOTAL_GAMES,
    }, sort_keys=True))


_configure()


if __name__ == "__main__":
    main()
