"""Run the locked, actor-safe MD-v3 PPO-v2 training experiment.

This is a research-only driver.  It requires the prospective self-hashed
PPO-v2 preregistration before constructing an engine, uses the exact frozen
opponent population bound by that lock, and can select only the terminal
update-16 checkpoint.  Recovery files after updates 1--15 contain optimizer
state but deliberately contain no deployable NumPy weights and are marked
ineligible for selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model  # noqa: E402
from agent.obsview import ObsView, ST_MAIN  # noqa: E402
from tools.research import lock_md_v3_ppo_v2 as LOCK  # noqa: E402
from tools.research import md_v3_ppo_v2_population as POP  # noqa: E402
from tools.research import qu_v2a_features as TRAIN_QF  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import train_md_v3_ppo as V1  # noqa: E402
from tools.research import train_md_v3_ppo_v2 as PPO  # noqa: E402
from tools.research import train_qu_v2a as BC  # noqa: E402
from tools.rl_env import (  # noqa: E402
    EpisodeSpec,
    OpponentSpec,
    PTCGRLEnv,
    SelectionSpec,
    build_paired_schedule,
    schedule_manifest,
)


RESULT_SCHEMA = "ptcg.md-v3.st-main-ppo-v2-training-result.v1"
RECOVERY_SCHEMA = "ptcg.md-v3.st-main-ppo-v2-recovery.v1"
RECOVERY_MANIFEST_SCHEMA = "ptcg.md-v3.st-main-ppo-v2-recovery-manifest.v1"
ROLLOUT_POTENTIAL: Callable[[Mapping[str, Any]], float] | None = None
TRAJECTORY_FINALIZER: Callable[..., None] = PPO._finish_trajectory
REWARD_CONTRACT = "terminal"
TERMINAL_DIRECTORY = "terminal-update-16-candidate"
EXPECTED_UPDATES = 16
EXPECTED_GAMES_PER_UPDATE = 768
EXPECTED_TOTAL_GAMES = 12_288
EXPECTED_MINIMUM_ST_MAIN = 20_000
EXPECTED_MAXIMUM_PARENT_KL = 0.02
EXPECTED_ROLLOUT_SEED_BASE = 2_026_073_101
EXPECTED_SEED_STRIDE = 1_000_003
EXPECTED_SEAT_COUNTS = {"0": 384, "1": 384}
EXPECTED_FAMILY_PAIR_COUNTS = {"field": 192, "mirror": 192}
EXPECTED_HYPERPARAMETERS = {
    "actor_learning_rate": 1e-6,
    "critic_learning_rate": 1e-5,
    "gamma": 0.997,
    "gae_lambda": 0.95,
    "ppo_epochs": 2,
    "minibatch_size": 512,
    "clip": 0.10,
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.002,
    "parent_kl_coefficient": 1.0,
}
NUMPY_LOGIT_TOLERANCE = 2e-4
NUMPY_VALUE_TOLERANCE = 2e-5


class RunnerError(RuntimeError):
    """The prospective PPO-v2 execution contract was violated."""


def _read_deck(path: Path) -> tuple[int, ...]:
    try:
        deck = tuple(
            int(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValueError) as error:
        raise RunnerError(f"cannot read locked deck {path}") from error
    if len(deck) != 60:
        raise RunnerError("locked learner deck does not contain 60 cards")
    return deck


def _assert_candidate_output(path: Path, *, must_not_exist: bool = True) -> Path:
    output = path.expanduser().resolve()
    for protected in (ROOT / "agent", ROOT / "decks"):
        try:
            output.relative_to(protected.resolve())
        except ValueError:
            continue
        raise RunnerError(f"research output cannot be below {protected}")
    if must_not_exist and output.exists():
        raise RunnerError(f"refusing to overwrite research output {output}")
    return output


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _training_config(lock: Mapping[str, Any]) -> Mapping[str, Any]:
    config = lock.get("training")
    if not isinstance(config, Mapping):
        raise RunnerError("PPO-v2 lock has no training contract")
    required = {
        "updates",
        "games_per_update",
        "total_games",
        "seat_balance_per_update",
        "rollout_seed_base",
        "seed_stride",
        "rollout_seeds",
        "ppo_seeds",
        "actor_learning_rate",
        "critic_learning_rate",
        "gamma",
        "gae_lambda",
        "ppo_epochs",
        "minibatch_size",
        "clip",
        "value_coefficient",
        "entropy_coefficient",
        "parent_kl_coefficient",
        "maximum_parent_kl_per_update",
        "maximum_final_parent_kl",
        "minimum_st_main_decisions_per_update",
    }
    if required - config.keys():
        raise RunnerError("PPO-v2 training contract is incomplete")
    if (
        config["updates"] != EXPECTED_UPDATES
        or config["games_per_update"] != EXPECTED_GAMES_PER_UPDATE
        or config["total_games"] != EXPECTED_TOTAL_GAMES
        or config["minimum_st_main_decisions_per_update"]
            != EXPECTED_MINIMUM_ST_MAIN
        or config["maximum_parent_kl_per_update"]
            != EXPECTED_MAXIMUM_PARENT_KL
        or config["maximum_final_parent_kl"]
            != EXPECTED_MAXIMUM_PARENT_KL
        or dict(config["seat_balance_per_update"]) != EXPECTED_SEAT_COUNTS
        or config["rollout_seed_base"] != EXPECTED_ROLLOUT_SEED_BASE
        or config["seed_stride"] != EXPECTED_SEED_STRIDE
    ):
        raise RunnerError("PPO-v2 fixed training size/gates drifted")
    if any(
        config.get(name) != expected
        for name, expected in EXPECTED_HYPERPARAMETERS.items()
    ):
        raise RunnerError("PPO-v2 fixed hyperparameters drifted")
    rollout_seeds = config["rollout_seeds"]
    ppo_seeds = config["ppo_seeds"]
    if (
        not isinstance(rollout_seeds, list)
        or not isinstance(ppo_seeds, list)
        or len(rollout_seeds) != EXPECTED_UPDATES
        or len(ppo_seeds) != EXPECTED_UPDATES
        or any(isinstance(seed, bool) or not isinstance(seed, int)
               for seed in (*rollout_seeds, *ppo_seeds))
    ):
        raise RunnerError("PPO-v2 update seeds are malformed")
    base = config["rollout_seed_base"]
    stride = config["seed_stride"]
    expected_rollout = [
        int(base) + index * int(stride) for index in range(EXPECTED_UPDATES)
    ]
    expected_ppo = [
        int(base) + (EXPECTED_UPDATES + index) * int(stride)
        for index in range(EXPECTED_UPDATES)
    ]
    if rollout_seeds != expected_rollout or ppo_seeds != expected_ppo:
        raise RunnerError("PPO-v2 explicit seeds violate the locked derivation")
    numeric_positive = (
        "actor_learning_rate",
        "critic_learning_rate",
        "gamma",
    )
    if any(
        isinstance(config[name], bool)
        or not isinstance(config[name], (int, float))
        or not math.isfinite(float(config[name]))
        or float(config[name]) <= 0
        for name in numeric_positive
    ):
        raise RunnerError("PPO-v2 training hyperparameters are invalid")
    if any(
        isinstance(config[name], bool)
        or not isinstance(config[name], int)
        or config[name] <= 0
        for name in ("ppo_epochs", "minibatch_size")
    ):
        raise RunnerError("PPO-v2 epoch/minibatch contract is invalid")
    if (
        not 0.0 < float(config["gamma"]) <= 1.0
        or not 0.0 <= float(config["gae_lambda"]) <= 1.0
        or not 0.0 <= float(config["clip"]) < 1.0
        or any(
            not isinstance(config[name], (int, float))
            or isinstance(config[name], bool)
            or not math.isfinite(float(config[name]))
            or float(config[name]) < 0.0
            for name in (
                "value_coefficient",
                "entropy_coefficient",
                "parent_kl_coefficient",
            )
        )
    ):
        raise RunnerError("PPO-v2 loss/discount contract is invalid")
    return config


def build_locked_schedule_contract(
    opponents: Sequence[Any],
    *,
    update: int,
    games: int,
    rollout_seed: int,
    ppo_seed: int,
) -> tuple[list[EpisodeSpec], dict[str, Any]]:
    """Rebuild one exact schedule and its prospective identity."""
    # The population has six pilot-level schedule groups.  Allocating those
    # independently can round a nominal 50/50 family split to 191/193 pairs.
    # Collapse only the scheduler's first hierarchy to family; the builder's
    # member allocator still distributes the exact original pilot/deck weights.
    scheduling_opponents = []
    for opponent in opponents:
        group = str(opponent.schedule_group)
        family = (
            "mirror" if group.startswith("mirror_")
            else "field" if group.startswith("field_")
            else None
        )
        if family is None:
            raise RunnerError(f"unclassified population schedule group {group!r}")
        scheduling_opponents.append(OpponentSpec(
            key=opponent.key,
            deck=opponent.deck,
            move=opponent.move,
            weight=opponent.weight,
            policy_id=opponent.policy_id,
            schedule_group=family,
        ))
    schedule = build_paired_schedule(
        scheduling_opponents, games, seed=rollout_seed,
    )
    rows = schedule_manifest(schedule, opponents)
    pair_rows = {row.pair_id: row for row in schedule}
    seat_counts = Counter(str(row.learner_seat) for row in schedule)
    schedule_groups = Counter()
    opponent_counts = Counter()
    family_counts = Counter()
    for row in pair_rows.values():
        opponent = opponents[row.opponent_index]
        group = str(opponent.schedule_group)
        schedule_groups[group] += 1
        opponent_counts[str(opponent.key)] += 1
        if group.startswith("mirror_"):
            family_counts["mirror"] += 1
        elif group.startswith("field_"):
            family_counts["field"] += 1
        else:
            raise RunnerError(f"unclassified population schedule group {group!r}")
    contract = {
        "update": int(update),
        "rollout_seed": int(rollout_seed),
        "ppo_seed": int(ppo_seed),
        "games": int(games),
        "pairs": len(pair_rows),
        "seat_counts": dict(sorted(seat_counts.items())),
        "family_pair_counts": dict(sorted(family_counts.items())),
        "schedule_group_pair_counts": dict(sorted(schedule_groups.items())),
        "opponent_pair_counts": dict(sorted(opponent_counts.items())),
        "manifest_sha256": LOCK.canonical_sha256(rows),
    }
    return schedule, contract


def enforce_locked_schedule(
    lock: Mapping[str, Any],
    population: POP.FrozenPopulation,
    *,
    update: int,
    config: Mapping[str, Any],
) -> list[EpisodeSpec]:
    schedules = lock.get("schedules")
    if not isinstance(schedules, Mapping):
        raise RunnerError("PPO-v2 lock has no schedule contracts")
    contracts = schedules.get("updates")
    if not isinstance(contracts, list) or len(contracts) != EXPECTED_UPDATES:
        raise RunnerError("PPO-v2 lock does not bind all update schedules")
    schedule, actual = build_locked_schedule_contract(
        population.opponents,
        update=update,
        games=int(config["games_per_update"]),
        rollout_seed=int(config["rollout_seeds"][update - 1]),
        ppo_seed=int(config["ppo_seeds"][update - 1]),
    )
    expected = contracts[update - 1]
    if not isinstance(expected, Mapping) or dict(expected) != actual:
        raise RunnerError(f"update {update} schedule differs from preregistration")
    if (
        actual["seat_counts"] != EXPECTED_SEAT_COUNTS
        or actual["family_pair_counts"] != EXPECTED_FAMILY_PAIR_COUNTS
    ):
        raise RunnerError(f"update {update} population/seat mass drifted")
    return schedule


def _controller_snapshot(
    population: POP.FrozenPopulation,
) -> dict[str, dict[str, Any]]:
    return {
        key: controller.diagnostics()
        for key, controller in population.controllers.items()
    }


def _counter_delta(
    after: Mapping[str, Any], before: Mapping[str, Any],
) -> dict[str, int]:
    keys = set(after) | set(before)
    return {
        str(key): int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(keys)
        if int(after.get(key, 0)) - int(before.get(key, 0))
    }


def _controller_deltas(
    population: POP.FrozenPopulation,
    before: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for key, controller in population.controllers.items():
        after = controller.diagnostics()
        prior = before[key]
        result[key] = {
            name: int(after.get(name, 0)) - int(prior.get(name, 0))
            for name in (
                "calls",
                "main_routes",
                "card_routes",
                "qu_routes",
                "off_deck_main_routes",
                "off_deck_card_routes",
                "fallbacks",
                "repairs",
            )
        }
        result[key]["exceptions"] = _counter_delta(
            after.get("exceptions", {}), prior.get("exceptions", {}),
        )
        result[key]["fallback_reasons"] = _counter_delta(
            after.get("fallback_reasons", {}),
            prior.get("fallback_reasons", {}),
        )
    return result


def collect_population_games(
    net: QM.TorchQuV2A,
    card_net: model.Net,
    qu_net: model.Net,
    deck: Sequence[int],
    population: POP.FrozenPopulation,
    schedule: Sequence[EpisodeSpec],
    *,
    seed: int,
    device: torch.device,
    env_factory: Callable[..., PTCGRLEnv] = PTCGRLEnv,
) -> tuple[list[PPO.Decision], dict[str, Any]]:
    """Collect one locked update against the complete frozen population."""
    if not schedule:
        raise RunnerError("cannot collect an empty PPO-v2 schedule")
    expected_ids = list(range(len(schedule)))
    if [episode.episode_id for episode in schedule] != expected_ids:
        raise RunnerError("PPO-v2 schedule episode IDs are not consecutive")
    before = _controller_snapshot(population)
    env = env_factory(
        deck,
        population.opponents,
        observation_encoder=lambda obs: obs,
        fault_mode="truncate",
        max_selects=5000,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    decisions: list[PPO.Decision] = []
    outcomes: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    opponents_seen: Counter[str] = Counter()
    seats: Counter[str] = Counter()
    zero_main_games = 0
    net.eval()
    try:
        for episode in schedule:
            trajectory: list[PPO.Decision] = []
            obs, info = env.reset(options={
                "episode_id": episode.episode_id,
                "opponent_index": episode.opponent_index,
                "learner_seat": episode.learner_seat,
                "policy_seed": episode.policy_seed,
            })
            opponents_seen[population.opponents[episode.opponent_index].key] += 1
            seats[str(episode.learner_seat)] += 1
            reward = float(info.get("reward", 0.0))
            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
            while obs is not None:
                raw = env.raw_observation
                if raw is None:
                    raise RunnerError("environment lost its learner observation")
                view = ObsView(raw)
                learner_select_index = int(
                    info["seat_selects"][episode.learner_seat],
                )
                if view.select_type == ST_MAIN:
                    routes["st_main"] += 1
                    features = TRAIN_QF.encode_public_observation(raw, deck)
                    with torch.no_grad():
                        logits, values = net(
                            QM.collate([features], device=device),
                        )
                        picks, logp, _ = V1.sample_selection(
                            logits[0],
                            SelectionSpec.from_observation(raw),
                            generator,
                        )
                    trajectory.append(PPO.Decision(
                        features=features,
                        picks=tuple(picks),
                        n_options=len(view.options),
                        min_count=view.min_count,
                        max_count=view.max_count,
                        old_logp=float(logp.cpu()),
                        old_value=float(values[0].cpu()),
                        episode_id=episode.episode_id,
                        decision_index=len(trajectory),
                        learner_select_index=learner_select_index,
                        public_potential=(
                            float(ROLLOUT_POTENTIAL(raw))
                            if ROLLOUT_POTENTIAL is not None else 0.0
                        ),
                    ))
                    action = picks
                else:
                    routes["frozen_non_main"] += 1
                    action = PPO._frozen_action(raw, deck, card_net, qu_net)
                obs, reward, terminated, truncated, info = env.step(action)
            clean_terminal = (
                terminated
                and not truncated
                and info.get("reason") == "engine_terminal"
                and info.get("agent_error") is None
                and info.get("engine_error") is None
                and info.get("result") in ("win", "draw", "loss")
            )
            if not clean_terminal:
                outcomes["invalid"] += 1
                invalid_reasons[str(info.get("reason") or "unknown")] += 1
                continue
            outcomes[str(info["result"])] += 1
            if not trajectory:
                zero_main_games += 1
                continue
            TRAJECTORY_FINALIZER(
                trajectory,
                final_learner_selects=int(
                    info["seat_selects"][episode.learner_seat],
                ),
                terminal_reward=float(reward),
            )
            decisions.extend(trajectory)
    finally:
        env.close()
    diagnostics = _controller_deltas(population, before)
    controller_faults = sum(
        row["fallbacks"]
        + row["repairs"]
        + sum(abs(value) for value in row["exceptions"].values())
        for row in diagnostics.values()
    )
    return decisions, {
        "games": len(schedule),
        "outcomes": dict(outcomes),
        "invalid": int(outcomes["invalid"]),
        "invalid_reasons": dict(invalid_reasons),
        "zero_main_games": zero_main_games,
        "routes": dict(routes),
        "opponents": dict(opponents_seen),
        "learner_seats": dict(seats),
        "controller_faults": controller_faults,
        "controllers": diagnostics,
    }


def measure_parent_kl(
    net: QM.TorchQuV2A,
    parent: QM.TorchQuV2A,
    decisions: Sequence[PPO.Decision],
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> dict[str, float]:
    """Measure post-update parent||candidate KL on the locked rollout states."""
    if not decisions or batch_size <= 0:
        raise RunnerError("cannot measure parent KL without decisions/batches")
    values: list[float] = []
    net.eval()
    parent.eval()
    with torch.no_grad():
        for start in range(0, len(decisions), batch_size):
            rows = decisions[start:start + batch_size]
            batch = QM.collate([row.features for row in rows], device=device)
            logits, _ = net(batch)
            parent_logits, _ = parent(batch)
            for index, row in enumerate(rows):
                _, _, kl = V1.sequence_statistics(
                    logits[index, :row.n_options + 1],
                    row.picks,
                    row.spec,
                    parent_logits=parent_logits[index, :row.n_options + 1],
                )
                values.append(float(kl.cpu()))
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise RunnerError("post-update parent KL is non-finite")
    return {
        "mean": float(array.mean()),
        "maximum": float(array.max()),
        "decisions": int(array.size),
    }


def _state_dict_matches(
    net: QM.TorchQuV2A, state_dict: Mapping[str, Any],
) -> bool:
    current = net.state_dict()
    return (
        set(current) == set(state_dict)
        and all(
            isinstance(state_dict[name], torch.Tensor)
            and torch.equal(current[name].detach().cpu(), state_dict[name].cpu())
            for name in current
        )
    )


def _write_recovery_checkpoint(
    output: Path,
    net: QM.TorchQuV2A,
    optimizer: torch.optim.Optimizer,
    scopes: PPO.ParameterScopes,
    *,
    update: int,
    lock_sha256: str,
    parent_checkpoint_sha256: str,
    update_rows: Sequence[Mapping[str, Any]],
) -> Path:
    directory = output / "recovery" / f"update-{update:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint_path = directory / f"RECOVERY-ONLY-update-{update:02d}.pt"
    state_dict = {
        name: tensor.detach().cpu() for name, tensor in net.state_dict().items()
    }
    payload = {
        "schema": RECOVERY_SCHEMA,
        "candidate_only": True,
        "recovery_only": True,
        "selection_eligible": False,
        "fixed_terminal_selection_update": EXPECTED_UPDATES,
        "completed_updates": int(update),
        "lock_sha256": str(lock_sha256),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "architecture": list(net.architecture),
        "state_dict": state_dict,
        "state_dict_sha256": BC._state_dict_sha256(state_dict),
        "optimizer_state_dict": optimizer.state_dict(),
        "trainable_parameter_names": {
            "actor": list(scopes.actor_names),
            "critic": list(scopes.critic_names),
        },
        "frozen_parameter_names": list(scopes.frozen_names),
        "update_rows": list(update_rows),
    }
    V1.atomic_torch(checkpoint_path, payload)
    manifest = {
        "schema": RECOVERY_MANIFEST_SCHEMA,
        "candidate_only": True,
        "recovery_only": True,
        "selection_eligible": False,
        "completed_updates": int(update),
        "fixed_terminal_selection_update": EXPECTED_UPDATES,
        "checkpoint": {
            "path": _display_path(checkpoint_path),
            "sha256": V1.sha256_file(checkpoint_path),
        },
        "deployable_numpy_weights": None,
    }
    V1.atomic_json(directory / "RECOVERY-ONLY.json", manifest)
    return checkpoint_path


def _load_torch_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RunnerError(f"{path} is not a Torch checkpoint mapping")
    return payload


def _restore_recovery(
    path: Path,
    net: QM.TorchQuV2A,
    optimizer: torch.optim.Optimizer,
    scopes: PPO.ParameterScopes,
    parent: QM.TorchQuV2A,
    *,
    lock_sha256: str,
    parent_checkpoint_sha256: str,
) -> tuple[int, list[dict[str, Any]]]:
    payload = _load_torch_payload(path)
    completed = payload.get("completed_updates")
    if (
        payload.get("schema") != RECOVERY_SCHEMA
        or payload.get("candidate_only") is not True
        or payload.get("recovery_only") is not True
        or payload.get("selection_eligible") is not False
        or payload.get("fixed_terminal_selection_update") != EXPECTED_UPDATES
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 1 <= completed < EXPECTED_UPDATES
        or payload.get("lock_sha256") != lock_sha256
        or payload.get("parent_checkpoint_sha256")
            != parent_checkpoint_sha256
        or tuple(payload.get("architecture", ())) != net.architecture
        or payload.get("trainable_parameter_names") != {
            "actor": list(scopes.actor_names),
            "critic": list(scopes.critic_names),
        }
        or payload.get("frozen_parameter_names") != list(scopes.frozen_names)
    ):
        raise RunnerError("recovery checkpoint violates the PPO-v2 contract")
    state_dict = payload.get("state_dict")
    if (
        not isinstance(state_dict, Mapping)
        or payload.get("state_dict_sha256") != BC._state_dict_sha256(state_dict)
    ):
        raise RunnerError("recovery checkpoint state hash is invalid")
    net.load_state_dict(state_dict, strict=True)
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise RunnerError("recovery checkpoint has no persistent optimizer")
    optimizer.load_state_dict(optimizer_state)
    PPO._validate_optimizer(optimizer, scopes)
    parent_named = dict(parent.named_parameters())
    for name, parameter in net.named_parameters():
        if name in scopes.frozen_names and not torch.equal(
            parameter.detach().cpu(), parent_named[name].detach().cpu(),
        ):
            raise RunnerError("recovery checkpoint changed a frozen parameter")
    rows = payload.get("update_rows")
    if (
        not isinstance(rows, list)
        or len(rows) != completed
        or [row.get("update") for row in rows] != list(range(1, completed + 1))
    ):
        raise RunnerError("recovery checkpoint has invalid update history")
    return completed, [dict(row) for row in rows]


def _verify_checkpoint_parity(
    path: Path,
    net: QM.TorchQuV2A,
    *,
    completed_updates: int,
) -> None:
    payload = _load_torch_payload(path)
    state_dict = payload.get("state_dict")
    provenance = payload.get("provenance")
    if (
        payload.get("schema") != PPO.CHECKPOINT_SCHEMA
        or payload.get("candidate_only") is not True
        or payload.get("completed_updates") != completed_updates
        or not isinstance(provenance, Mapping)
        or provenance.get("selection_eligible") is not True
        or provenance.get("recovery_only") is not False
        or provenance.get("fixed_terminal_selection_update")
            != EXPECTED_UPDATES
        or not isinstance(state_dict, Mapping)
        or payload.get("state_dict_sha256") != BC._state_dict_sha256(state_dict)
        or not _state_dict_matches(net, state_dict)
    ):
        raise RunnerError("terminal candidate checkpoint/in-memory parity failed")


def _verify_numpy_parity(
    net: QM.TorchQuV2A,
    weights_path: Path,
    decisions: Sequence[PPO.Decision],
    *,
    device: torch.device,
) -> dict[str, float]:
    with np.load(weights_path, allow_pickle=False) as archive:
        numpy_net = QM.NumpyQuV2A(archive)
    maximum_logits = 0.0
    maximum_value = 0.0
    net.eval()
    with torch.no_grad():
        for start in range(0, len(decisions), 512):
            rows = decisions[start:start + 512]
            logits, values = net(
                QM.collate([row.features for row in rows], device=device),
            )
            logits_numpy = logits.cpu().numpy()
            values_numpy = values.cpu().numpy()
            for index, row in enumerate(rows):
                numpy_logits, numpy_value = numpy_net.forward(row.features)
                count = len(numpy_logits)
                maximum_logits = max(
                    maximum_logits,
                    float(np.max(np.abs(
                        logits_numpy[index, :count] - numpy_logits,
                    ))),
                )
                maximum_value = max(
                    maximum_value,
                    abs(float(values_numpy[index]) - float(numpy_value)),
                )
    if (
        maximum_logits > NUMPY_LOGIT_TOLERANCE
        or maximum_value > NUMPY_VALUE_TOLERANCE
    ):
        raise RunnerError("terminal Torch/NumPy runtime parity failed")
    return {
        "max_abs_logit_error": maximum_logits,
        "max_abs_value_error": maximum_value,
        "decisions": len(decisions),
    }


def _population_is_bound(
    lock: Mapping[str, Any], population: POP.FrozenPopulation,
) -> None:
    expected = lock.get("population")
    if not isinstance(expected, Mapping):
        raise RunnerError("PPO-v2 lock has no population contract")
    if (
        expected.get("schema") != POP.POPULATION_SCHEMA
        or dict(expected.get("pilot_mass", {})) != POP.PILOT_MASS
        or expected.get("manifest_sha256")
            != LOCK.canonical_sha256(population.manifest)
    ):
        raise RunnerError("constructed frozen population differs from the lock")


def _rollout_is_clean(
    rollout: Mapping[str, Any],
    *,
    minimum_st_main: int,
    decisions: int,
) -> bool:
    mirror_rows = [
        row
        for key, row in rollout.get("controllers", {}).items()
        if str(key).startswith("mirror_")
    ]
    mirror_routes_clean = bool(mirror_rows) and all(
        row.get("off_deck_main_routes") == 0
        and row.get("off_deck_card_routes") == 0
        for row in mirror_rows
    )
    return (
        rollout.get("games") == EXPECTED_GAMES_PER_UPDATE
        and rollout.get("invalid") == 0
        and rollout.get("controller_faults") == 0
        and sum(int(value) for value in rollout.get("outcomes", {}).values())
            == EXPECTED_GAMES_PER_UPDATE
        and decisions >= minimum_st_main
        and rollout.get("learner_seats") == EXPECTED_SEAT_COUNTS
        and mirror_routes_clean
    )


def _scope_parameter_delta(
    net: QM.TorchQuV2A,
    parent: QM.TorchQuV2A,
    names: Sequence[str],
) -> dict[str, float]:
    current = dict(net.named_parameters())
    frozen = dict(parent.named_parameters())
    squared = 0.0
    maximum = 0.0
    changed = 0
    count = 0
    for name in names:
        difference = (
            current[name].detach().cpu() - frozen[name].detach().cpu()
        )
        squared += float(difference.square().sum())
        maximum = max(maximum, float(difference.abs().max()))
        changed += int(torch.count_nonzero(difference))
        count += difference.numel()
    return {
        "l2": math.sqrt(squared),
        "max_abs": maximum,
        "changed_parameters": changed,
        "total_parameters": count,
    }


def execute_training(
    lock: Mapping[str, Any],
    paths: Mapping[str, Path],
    output_dir: Path,
    *,
    device: torch.device,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Execute the fully bound experiment, or raise without a candidate."""
    config = _training_config(lock)
    output = _assert_candidate_output(output_dir)
    output.mkdir(parents=True)
    base_seed = int(config["rollout_seed_base"])
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(base_seed)

    required_paths = {
        "runner",
        "trainer",
        "population",
        "parent_checkpoint",
        "parent_weights",
        "card_weights",
        "qu_weights",
        "ppo_v1_weights",
        "md_v1_weights",
        "field_snapshot",
        "deck",
    }
    if required_paths - paths.keys():
        raise RunnerError("verified lock omitted a required runner artifact")
    if (
        paths["runner"] != Path(__file__).resolve()
        or paths["trainer"] != Path(PPO.__file__).resolve()
        or paths["population"] != Path(POP.__file__).resolve()
    ):
        raise RunnerError("lock names different PPO-v2 implementation files")
    input_hashes_before = {
        name: V1.sha256_file(path) for name, path in paths.items()
    }
    deck = _read_deck(paths["deck"])
    if tuple(sorted(deck)) != PPO.LAYERED.CARD.TARGET_DECK:
        raise RunnerError("locked learner deck is not the exact MD-v3 target deck")
    net, _ = V1.load_torch_parent(paths["parent_checkpoint"], device)
    parent, _ = V1.load_torch_parent(paths["parent_checkpoint"], device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    V1.verify_numpy_parity(net, paths["parent_weights"], device)
    V1.verify_numpy_parity(parent, paths["parent_weights"], device)
    optimizer, scopes = PPO.make_optimizer(
        net,
        actor_learning_rate=float(config["actor_learning_rate"]),
        critic_learning_rate=float(config["critic_learning_rate"]),
    )
    frozen_before = PPO.parameter_snapshot(scopes.frozen)
    completed_updates = 0
    update_rows: list[dict[str, Any]] = []
    parent_checkpoint_sha256 = input_hashes_before["parent_checkpoint"]
    if resume_from is not None:
        completed_updates, update_rows = _restore_recovery(
            resume_from.expanduser().resolve(),
            net,
            optimizer,
            scopes,
            parent,
            lock_sha256=str(lock["lock_sha256"]),
            parent_checkpoint_sha256=parent_checkpoint_sha256,
        )
        frozen_before = PPO.parameter_snapshot(scopes.frozen)

    card_net = POP.load_net(paths["card_weights"])
    qu_net = POP.load_net(paths["qu_weights"])
    population = POP.build_population(
        grim_deck=deck,
        field_snapshot=paths["field_snapshot"],
        md_v3_main_weights=paths["parent_weights"],
        md_v3_card_weights=paths["card_weights"],
        ppo_v1_main_weights=paths["ppo_v1_weights"],
        md_v1_main_weights=paths["md_v1_weights"],
        qu_v2b_weights=paths["qu_weights"],
    )
    _population_is_bound(lock, population)
    started = time.time()
    last_decisions: list[PPO.Decision] = []
    for update in range(completed_updates + 1, EXPECTED_UPDATES + 1):
        schedule = enforce_locked_schedule(
            lock, population, update=update, config=config,
        )
        update_started = time.time()
        last_decisions, rollout = collect_population_games(
            net,
            card_net,
            qu_net,
            deck,
            population,
            schedule,
            seed=int(config["rollout_seeds"][update - 1]),
            device=device,
        )
        if not _rollout_is_clean(
            rollout,
            minimum_st_main=int(
                config["minimum_st_main_decisions_per_update"],
            ),
            decisions=len(last_decisions),
        ):
            raise RunnerError(f"update {update} rollout failed the cleanliness gate")
        optimizer_identity = id(optimizer)
        metrics = PPO.ppo_update(
            net,
            parent,
            optimizer,
            scopes,
            last_decisions,
            device=device,
            seed=int(config["ppo_seeds"][update - 1]),
            epochs=int(config["ppo_epochs"]),
            minibatch_size=int(config["minibatch_size"]),
            clip=float(config["clip"]),
            value_coefficient=float(config["value_coefficient"]),
            entropy_coefficient=float(config["entropy_coefficient"]),
            parent_kl_coefficient=float(config["parent_kl_coefficient"]),
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
            reward_contract=REWARD_CONTRACT,
        )
        if id(optimizer) != optimizer_identity:
            raise RunnerError("PPO-v2 replaced its persistent optimizer")
        PPO.assert_parameters_unchanged(
            scopes.frozen, frozen_before, label="shared representation",
        )
        post_update_kl = measure_parent_kl(
            net, parent, last_decisions, device=device,
        )
        maximum_kl = float(config["maximum_parent_kl_per_update"])
        if (
            float(metrics["parent_kl"]) > maximum_kl
            or post_update_kl["mean"] > maximum_kl
        ):
            raise RunnerError(f"update {update} exceeded parent KL ceiling")
        row = {
            "update": update,
            "rollout_seed": int(config["rollout_seeds"][update - 1]),
            "ppo_seed": int(config["ppo_seeds"][update - 1]),
            "st_main_decisions": len(last_decisions),
            "rollout": rollout,
            "ppo": metrics,
            "post_update_parent_kl": post_update_kl,
            "timing_seconds": time.time() - update_started,
        }
        update_rows.append(row)
        recovery = None
        if update < EXPECTED_UPDATES:
            recovery = _write_recovery_checkpoint(
                output,
                net,
                optimizer,
                scopes,
                update=update,
                lock_sha256=str(lock["lock_sha256"]),
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                update_rows=update_rows,
            )
        print(json.dumps({
            "update": update,
            "games": rollout["games"],
            "st_main_decisions": len(last_decisions),
            "post_update_parent_kl": post_update_kl["mean"],
            "recovery_only": str(recovery) if recovery is not None else None,
        }, sort_keys=True), flush=True)

    if len(update_rows) != EXPECTED_UPDATES or not last_decisions:
        raise RunnerError("PPO-v2 did not reach its fixed terminal update")
    PPO.assert_parameters_unchanged(
        scopes.frozen, frozen_before, label="final shared representation",
    )
    final_parent_kl = measure_parent_kl(
        net, parent, last_decisions, device=device,
    )
    if final_parent_kl["mean"] > float(config["maximum_final_parent_kl"]):
        raise RunnerError("terminal update exceeded final parent KL ceiling")
    parameter_delta = {
        "actor": _scope_parameter_delta(net, parent, scopes.actor_names),
        "critic": _scope_parameter_delta(net, parent, scopes.critic_names),
        "frozen": _scope_parameter_delta(net, parent, scopes.frozen_names),
        "all": V1.parameter_delta(net, parent),
    }
    if (
        parameter_delta["actor"]["changed_parameters"] <= 0
        or parameter_delta["frozen"]["changed_parameters"] != 0
        or not all(
            math.isfinite(value)
            for scope in parameter_delta.values()
            for value in scope.values()
        )
    ):
        raise RunnerError("terminal PPO-v2 actor/frozen delta gate failed")

    terminal_dir = output / TERMINAL_DIRECTORY
    weights_path, checkpoint_path = PPO.write_candidate_checkpoint(
        terminal_dir,
        net,
        optimizer,
        scopes,
        completed_updates=EXPECTED_UPDATES,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        provenance={
            "lock_sha256": lock["lock_sha256"],
            "candidate_only": True,
            "recovery_only": False,
            "selection_eligible": True,
            "fixed_terminal_selection_update": EXPECTED_UPDATES,
            "selection_rule": "only the fixed update-16 terminal checkpoint",
        },
    )
    _verify_checkpoint_parity(
        checkpoint_path, net, completed_updates=EXPECTED_UPDATES,
    )
    runtime_parity = _verify_numpy_parity(
        net, weights_path, last_decisions, device=device,
    )
    verified_after = LOCK.verify_bound_artifacts(lock)
    input_hashes_after = {
        name: V1.sha256_file(path) for name, path in verified_after.items()
    }
    if input_hashes_before != input_hashes_after:
        raise RunnerError("one or more frozen locked inputs changed")

    result = {
        "schema": RESULT_SCHEMA,
        "lock_sha256": lock["lock_sha256"],
        "candidate_only": True,
        "selection": {
            "eligible": True,
            "selected_update": EXPECTED_UPDATES,
            "fixed_terminal_update": EXPECTED_UPDATES,
            "intermediate_recovery_eligible": False,
            "checkpoint_cherry_picking": False,
        },
        "device": str(device),
        "population": population.manifest,
        "updates": update_rows,
        "total_games": sum(row["rollout"]["games"] for row in update_rows),
        "total_st_main_decisions": sum(
            row["st_main_decisions"] for row in update_rows
        ),
        "final_parent_kl": final_parent_kl,
        "parameter_delta": parameter_delta,
        "runtime_parity": runtime_parity,
        "timing_seconds": {"total": time.time() - started},
        "artifacts": {
            "weights": {
                "path": _display_path(weights_path),
                "sha256": V1.sha256_file(weights_path),
            },
            "checkpoint": {
                "path": _display_path(checkpoint_path),
                "sha256": V1.sha256_file(checkpoint_path),
            },
        },
        "training_gate": {
            "passed": True,
            "zero_invalid_or_controller_faults": True,
            "minimum_st_main_per_update": EXPECTED_MINIMUM_ST_MAIN,
            "frozen_shared_parameters_byte_unchanged": True,
            "actor_scope_changed": True,
            "maximum_parent_kl": EXPECTED_MAXIMUM_PARENT_KL,
            "terminal_selection_only": True,
        },
    }
    V1.atomic_json(output / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    lock_path = Path(args.lock).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        parser.error("--output-dir already exists")
    # This prospective lock verification deliberately precedes population
    # construction and therefore every possible engine outcome.
    lock = LOCK.load_lock(lock_path, verify_artifacts=True)
    paths = LOCK.verify_bound_artifacts(lock)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    result = execute_training(
        lock,
        paths,
        output,
        device=device,
        resume_from=(
            Path(args.resume_from).expanduser().resolve()
            if args.resume_from else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
