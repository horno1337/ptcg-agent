"""Prospectively lock the 299,520-game dobi-v1 mirror PPO continuation."""

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
from tools.research import lock_md_v3_mirror_league_100k as BASE  # noqa: E402


SCHEMA = "ptcg.dobi-v1.exact-mirror-league-300k-lock.v1"
RUN_ROOT = ROOT / "tools/checkpoints/dobi-v1-mirror-league-300k"
DEFAULT_LOCK = RUN_ROOT / "training-lock.json"
DEFAULT_OUTPUT = RUN_ROOT / "training"
UPDATES = 390
GAMES_PER_UPDATE = 768
TOTAL_GAMES = UPDATES * GAMES_PER_UPDATE
PAIRS_PER_UPDATE = 384
SEATS_PER_UPDATE = {"0": 384, "1": 384}
BASE_SEED = 2_026_080_301
SEED_STRIDE = 1_000_003
ROLLOUT_SEEDS = [BASE_SEED + i * SEED_STRIDE for i in range(UPDATES)]
PPO_SEEDS = [BASE_SEED + (UPDATES + i) * SEED_STRIDE for i in range(UPDATES)]


ARTIFACT_PATHS = {
    **BASE.ARTIFACT_PATHS,
    "lock_builder": Path(__file__).resolve(),
    "wrapper": ROOT / "tools/research/run_dobi_v1_mirror_league_300k.py",
    "population": Path(POP.__file__).resolve(),
    "parent_checkpoint": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/training/terminal-update-16-candidate/candidate-qu-v2a-ppo-v2-checkpoint.pt",
    "parent_weights": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/training/terminal-update-16-candidate/candidate-qu-v2a-weights.npz",
    # The base runner passes this legacy argument to the population builder;
    # this lock binds it explicitly as the frozen MD-v3 league opponent.
    "md_v1_weights": ROOT / "tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz",
    "field_snapshot": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/field-gate/recent-field-20260729-31.json",
    "dobi_archive": ROOT / "submission-dobi-v1-unsigned.tar.gz",
    "dobi_package_manifest": ROOT / "tools/checkpoints/dobi-v1/package-manifest.json",
    "dobi_exact_tarball_audit": ROOT / "tools/checkpoints/dobi-v1/exact-tarball-audit.json",
    "dobi_random_smoke": ROOT / "tools/checkpoints/dobi-v1/random-smoke-result.json",
    "dobi_mirror_gate": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/gameplay-replication-10240/result.json",
    "dobi_field_gate": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/field-gate/result.json",
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
        "A fixed 299,520-game exact-mirror PPO continuation from frozen "
        "dobi-v1 can extend its validated mirror gain without changing deck, "
        "runtime routing, ST_CARD, or the Qu-v2B residual."
    )
    payload["scope"] = {
        "learner": "byte-frozen dobi-v1 Torch/NumPy-identical initialization",
        "trainable": "ST_MAIN actor and private critic only",
        "frozen": ["shared representation", "ST_CARD", "Qu-v2B residual", "deck"],
        "deployment_authorized": False,
    }
    payload["training"].update({
        "actor_learning_rate": 2.5e-6,
        "maximum_parent_kl_per_update": 0.04,
        "maximum_final_parent_kl": 0.04,
        "parent": "frozen dobi-v1; fresh optimizer; no optimizer state inherited",
    })
    payload["decision_rules"] = {
        "candidate": "only fixed terminal update 390 is selection-eligible",
        "intermediates": "recovery-only; never deployable and never eligible for post-hoc selection",
        "gameplay_gate": "fixed direct exact-mirror A/B versus frozen dobi-v1; require CI95 lower bound above 50%",
        "field_gate": "only after mirror pass; recent-frequency noninferiority margin fixed at -1.5pp",
        "upload": "requires exact-tarball safety audits plus a separately user-approved name and upload",
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
