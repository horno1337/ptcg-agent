"""Prospectively lock the 99,840-game exact-Grimmsnarl PPO league."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools.research import md_v3_mirror_league_100k_population as POP  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


SCHEMA = "ptcg.md-v3.exact-mirror-league-100k-lock.v1"
UPDATES = 130
GAMES_PER_UPDATE = 768
TOTAL_GAMES = 99_840
PAIRS_PER_UPDATE = 384
SEATS_PER_UPDATE = {"0": 384, "1": 384}
BASE_SEED = 2_026_080_101
SEED_STRIDE = 1_000_003
ROLLOUT_SEEDS = [BASE_SEED + i * SEED_STRIDE for i in range(UPDATES)]
PPO_SEEDS = [BASE_SEED + (UPDATES + i) * SEED_STRIDE for i in range(UPDATES)]
RUN_ROOT = ROOT / "tools/checkpoints/md-v3-mirror-league-100k"
DEFAULT_LOCK = RUN_ROOT / "training-lock.json"
DEFAULT_OUTPUT = RUN_ROOT / "training"

ARTIFACT_PATHS = {
    "lock_builder": Path(__file__).resolve(),
    "wrapper": ROOT / "tools/research/run_md_v3_mirror_league_100k.py",
    "runner": ROOT / "tools/research/run_md_v3_ppo_v2.py",
    "trainer": ROOT / "tools/research/train_md_v3_ppo_v2.py",
    "population": ROOT / "tools/research/md_v3_mirror_league_100k_population.py",
    "parent_checkpoint": ROOT / "tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-checkpoint.pt",
    "parent_weights": ROOT / "tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz",
    "card_weights": ROOT / "agent/md_v2_card_weights.npz",
    "qu_weights": ROOT / "agent/weights.npz",
    "ppo_v1_weights": ROOT / "tools/checkpoints/md-v3-ppo-v1/training/candidate-qu-v2a-weights.npz",
    "md_v1_weights": ROOT / "agent/md_v1_weights.npz",
    "bc51_weights": ROOT / "tools/checkpoints/md-v3-mirror-main-v1/model-51/candidate-qu-v2a-weights.npz",
    "md_v5_weights": RUN_ROOT / "inputs/agent/md_v1_weights.npz",
    "md_v5_archive": ROOT / "submission-md-v5-experimental-unsigned.tar.gz",
    "field_snapshot": ROOT / "tools/checkpoints/md-v3-mirror-main-v1/field-july27-28.json",
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
    "runtime_model": ROOT / "agent/model.py",
    "runtime_features": ROOT / "agent/qu_v2_features.py",
    "base_features": ROOT / "agent/features.py",
    "obsview": ROOT / "agent/obsview.py",
    "policy": ROOT / "agent/policy.py",
    "safety": ROOT / "agent/safety.py",
    "cards": ROOT / "agent/cards.py",
    "card_router": ROOT / "agent/md_v2_card.py",
    "training_features": ROOT / "tools/research/qu_v2a_features.py",
    "torch_model": ROOT / "tools/research/qu_v2a_model.py",
    "v1_ppo_helpers": ROOT / "tools/research/train_md_v3_ppo.py",
    "bc_helpers": ROOT / "tools/research/train_qu_v2a.py",
    "layered_controller": ROOT / "tools/research/eval_md_v2_card_v1_gameplay.py",
    "rl_env": ROOT / "tools/rl_env.py",
    "battle_engine": ROOT / "engine/libcg.so",
    "cards_data": ROOT / "data/cards.json",
    "attacks_data": ROOT / "data/attacks.json",
}


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"not a regular artifact: {resolved}")
    return {
        "path": display_path(resolved),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def verify_bound_artifacts(lock: Mapping[str, Any]) -> dict[str, Path]:
    records = lock.get("artifacts")
    if not isinstance(records, Mapping):
        raise RuntimeError("lock has no artifact map")
    paths: dict[str, Path] = {}
    for name, row in records.items():
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise RuntimeError("invalid artifact row")
        path = resolve_path(str(row.get("path", "")))
        if (not path.is_file() or file_sha256(path) != row.get("sha256")
                or path.stat().st_size != row.get("bytes")):
            raise RuntimeError(f"bound artifact drift: {name}")
        paths[name] = path
    if set(paths) != set(ARTIFACT_PATHS):
        raise RuntimeError("artifact set drifted")
    return paths


def load_lock(path: Path, *, verify_artifacts: bool = True) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("lock_sha256")
    body = {key: value for key, value in payload.items() if key != "lock_sha256"}
    if payload.get("schema") != SCHEMA or recorded != canonical_sha256(body):
        raise RuntimeError("league lock identity is invalid")
    if verify_artifacts:
        verify_bound_artifacts(payload)
    return payload


def _schedule_contract(population: POP.FrozenPopulation, update: int) -> dict[str, Any]:
    schedule = build_paired_schedule(
        population.opponents, GAMES_PER_UPDATE, seed=ROLLOUT_SEEDS[update - 1],
    )
    manifest = schedule_manifest(schedule, population.opponents)
    pairs = {row.pair_id: row for row in schedule}
    groups: Counter[str] = Counter()
    opponents: Counter[str] = Counter()
    for row in pairs.values():
        opponent = population.opponents[row.opponent_index]
        groups[str(opponent.schedule_group)] += 1
        opponents[str(opponent.key)] += 1
    return {
        "update": update,
        "rollout_seed": ROLLOUT_SEEDS[update - 1],
        "ppo_seed": PPO_SEEDS[update - 1],
        "games": len(schedule),
        "pairs": len(pairs),
        "seat_counts": dict(sorted(Counter(str(row.learner_seat) for row in schedule).items())),
        "family_pair_counts": {"mirror": len(pairs)},
        "schedule_group_pair_counts": dict(sorted(groups.items())),
        "opponent_pair_counts": dict(sorted(opponents.items())),
        "manifest_sha256": canonical_sha256(manifest),
    }


def build_lock() -> dict[str, Any]:
    artifacts = {name: artifact(path) for name, path in ARTIFACT_PATHS.items()}
    POP.BINDINGS = {
        "md_v5_weights": resolve_path(artifacts["md_v5_weights"]["path"]),
        "bc51_weights": resolve_path(artifacts["bc51_weights"]["path"]),
    }
    deck = tuple(int(line) for line in ARTIFACT_PATHS["deck"].read_text().splitlines() if line)
    population = POP.build_population(
        grim_deck=deck,
        field_snapshot=ARTIFACT_PATHS["field_snapshot"],
        md_v3_main_weights=ARTIFACT_PATHS["parent_weights"],
        md_v3_card_weights=ARTIFACT_PATHS["card_weights"],
        ppo_v1_main_weights=ARTIFACT_PATHS["ppo_v1_weights"],
        md_v1_main_weights=ARTIFACT_PATHS["md_v1_weights"],
        qu_v2b_weights=ARTIFACT_PATHS["qu_weights"],
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Outcome PPO over an all-exact-Grimmsnarl league can produce a material mirror policy improvement that conservative mixed-field PPO did not.",
        "scope": {
            "learner": "frozen MD-v3 initialization",
            "trainable": "ST_MAIN actor and private critic only",
            "frozen": ["shared representation", "ST_CARD", "Qu-v2B residual", "deck"],
            "deployment_authorized": False,
        },
        "training": {
            "updates": UPDATES,
            "games_per_update": GAMES_PER_UPDATE,
            "total_games": TOTAL_GAMES,
            "seat_balance_per_update": SEATS_PER_UPDATE,
            "rollout_seed_base": BASE_SEED,
            "seed_stride": SEED_STRIDE,
            "rollout_seeds": ROLLOUT_SEEDS,
            "ppo_seeds": PPO_SEEDS,
            "actor_learning_rate": 5e-6,
            "critic_learning_rate": 2e-5,
            "gamma": 0.997,
            "gae_lambda": 0.95,
            "ppo_epochs": 2,
            "minibatch_size": 512,
            "clip": 0.15,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.005,
            "parent_kl_coefficient": 0.10,
            "maximum_parent_kl_per_update": 0.08,
            "maximum_final_parent_kl": 0.08,
            "minimum_st_main_decisions_per_update": 20_000,
            "reward": "terminal win/draw/loss only; no potentially exploitable shaping",
        },
        "population": {
            "schema": POP.POPULATION_SCHEMA,
            "pilot_mass": dict(POP.PILOT_MASS),
            "manifest": population.manifest,
            "manifest_sha256": canonical_sha256(population.manifest),
            "family_pair_mass": {"mirror": 1.0},
        },
        "schedules": {"updates": [_schedule_contract(population, i) for i in range(1, UPDATES + 1)]},
        "decision_rules": {
            "candidate": "only fixed terminal update 130 is selection-eligible",
            "behavioral_floor": "must change at least 5% of locked exact-mirror ST_MAIN prompts and touch at least 50% of games",
            "gameplay_gate": "paired seat-swapped exact-mirror A/B versus frozen MD-v3 must have Wilson CI95 lower bound above 50%",
            "field_gate": "after mirror pass, recent-frequency field A/B must be non-inferior before packaging",
            "upload": "requires a separate user-named tag and explicit approval",
        },
        "artifacts": artifacts,
    }
    payload["lock_sha256"] = canonical_sha256(payload)
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
    print(json.dumps({"path": display_path(output), "lock_sha256": payload["lock_sha256"], "games": TOTAL_GAMES}, sort_keys=True))


if __name__ == "__main__":
    main()
