"""Run the prospectively locked 99,840-game exact-mirror PPO league."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools.research import lock_md_v3_mirror_league_100k as LOCK  # noqa: E402
from tools.research import md_v3_mirror_league_100k_population as POP  # noqa: E402
from tools.research import run_md_v3_ppo_v2 as BASE  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


class LeagueRunnerError(RuntimeError):
    pass


def _schedule_contract(
    opponents: Sequence[Any],
    *,
    update: int,
    games: int,
    rollout_seed: int,
    ppo_seed: int,
) -> tuple[list[Any], dict[str, Any]]:
    schedule = build_paired_schedule(opponents, games, seed=rollout_seed)
    manifest = schedule_manifest(schedule, opponents)
    pairs = {row.pair_id: row for row in schedule}
    groups: Counter[str] = Counter()
    opponent_counts: Counter[str] = Counter()
    for row in pairs.values():
        opponent = opponents[row.opponent_index]
        group = str(opponent.schedule_group)
        if not group.startswith("mirror_"):
            raise LeagueRunnerError(f"non-mirror league group {group!r}")
        groups[group] += 1
        opponent_counts[str(opponent.key)] += 1
    contract = {
        "update": int(update),
        "rollout_seed": int(rollout_seed),
        "ppo_seed": int(ppo_seed),
        "games": int(games),
        "pairs": len(pairs),
        "seat_counts": dict(sorted(Counter(str(row.learner_seat) for row in schedule).items())),
        "family_pair_counts": {"mirror": len(pairs)},
        "schedule_group_pair_counts": dict(sorted(groups.items())),
        "opponent_pair_counts": dict(sorted(opponent_counts.items())),
        "manifest_sha256": LOCK.canonical_sha256(manifest),
    }
    return schedule, contract


def _install_contract() -> None:
    """Parameterize the audited PPO-v2 engine without editing the retired run."""
    BASE.LOCK = LOCK
    BASE.POP = POP
    BASE.EXPECTED_UPDATES = LOCK.UPDATES
    BASE.EXPECTED_GAMES_PER_UPDATE = LOCK.GAMES_PER_UPDATE
    BASE.EXPECTED_TOTAL_GAMES = LOCK.TOTAL_GAMES
    BASE.EXPECTED_MINIMUM_ST_MAIN = 20_000
    BASE.EXPECTED_MAXIMUM_PARENT_KL = 0.08
    BASE.EXPECTED_ROLLOUT_SEED_BASE = LOCK.BASE_SEED
    BASE.EXPECTED_SEED_STRIDE = LOCK.SEED_STRIDE
    BASE.EXPECTED_SEAT_COUNTS = dict(LOCK.SEATS_PER_UPDATE)
    BASE.EXPECTED_FAMILY_PAIR_COUNTS = {"mirror": LOCK.PAIRS_PER_UPDATE}
    BASE.EXPECTED_HYPERPARAMETERS = {
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
    }
    BASE.build_locked_schedule_contract = _schedule_contract


def _restore_snapshot_slots(population: POP.FrozenPopulation, directory: Path) -> None:
    snapshot_dir = directory / "league-snapshots"
    if not snapshot_dir.is_dir():
        return
    for slot in range(POP.SNAPSHOT_SLOTS):
        candidates = sorted(snapshot_dir.glob(f"update-*-slot-{slot}.npz"))
        if not candidates:
            continue
        path = candidates[-1]
        with np.load(path, allow_pickle=False) as archive:
            population.controllers[f"mirror_snapshot_{slot}"].main_net = QM.NumpyQuV2A(archive)


def _save_snapshot(net: QM.TorchQuV2A, output: Path, event: Mapping[str, Any]) -> Path:
    directory = output / "league-snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"update-{int(event['completed_update']):03d}-slot-{int(event['slot'])}.npz"
    if path.exists():
        raise LeagueRunnerError(f"snapshot already exists: {path}")
    arrays = QM.export_numpy_weights(net)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(path)
    return path


def _patch_snapshot_rotation(output: Path, *, completed_updates: int) -> None:
    original_collect = BASE.collect_population_games
    state = {"completed": int(completed_updates), "restored": False}

    def collect(net, card_net, qu_net, deck, population, schedule, **kwargs):
        if not state["restored"]:
            _restore_snapshot_slots(population, output)
            state["restored"] = True
        update = state["completed"] + 1
        event = POP.refresh_snapshot(population, net, update=update)
        if event is not None:
            path = _save_snapshot(net, output, event)
            print(json.dumps({"league_snapshot": dict(event), "path": str(path)}, sort_keys=True), flush=True)
        result = original_collect(
            net, card_net, qu_net, deck, population, schedule, **kwargs,
        )
        state["completed"] = update
        return result

    BASE.collect_population_games = collect


def _resume_completed(path: Path | None) -> int:
    if path is None:
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=True)
    completed = payload.get("completed_updates")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise LeagueRunnerError("resume checkpoint has no completed update count")
    return completed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=str(LOCK.DEFAULT_LOCK))
    parser.add_argument("--output-dir", default=str(LOCK.DEFAULT_OUTPUT))
    parser.add_argument("--resume-from")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    lock_path = Path(args.lock).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    resume = Path(args.resume_from).expanduser().resolve() if args.resume_from else None
    if resume is None and output.exists():
        parser.error(f"fresh output already exists: {output}")
    if resume is not None and not output.is_dir():
        parser.error("resume output directory is absent")
    lock = LOCK.load_lock(lock_path, verify_artifacts=True)
    paths = LOCK.verify_bound_artifacts(lock)
    POP.BINDINGS = {
        "md_v5_weights": paths["md_v5_weights"],
        "bc51_weights": paths["bc51_weights"],
    }
    _install_contract()
    completed = _resume_completed(resume)
    _patch_snapshot_rotation(output, completed_updates=completed)

    # The retired runner rejected an existing directory before its own valid
    # recovery branch.  The wrapper performs the fresh/resume checks above.
    original_assert = BASE._assert_candidate_output
    if resume is not None:
        BASE._assert_candidate_output = lambda path, must_not_exist=True: original_assert(path, must_not_exist=False)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("the official league run requires an available CUDA device")
    result = BASE.execute_training(lock, paths, output, device=device, resume_from=resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
