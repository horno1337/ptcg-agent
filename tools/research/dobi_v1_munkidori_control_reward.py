"""Locked public-state potential for Dobi-v1 Munkidori-control PPO v1."""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableSequence, Sequence


MUNKIDORI = 112
DARK_ENERGY = 7
GAMMA = 0.997
SHAPING_COEFFICIENT = 0.15


class MunkidoriRewardError(ValueError):
    """The observation or macro-transition reward contract is invalid."""


def _board(player: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for area in ("active", "bench"):
        entries = player.get(area) or ()
        if not isinstance(entries, (list, tuple)):
            raise MunkidoriRewardError(f"player {area} is not a sequence")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise MunkidoriRewardError(f"player {area} contains a non-object")
            result.append(entry)
    return result


def _munkidori_counts(player: Mapping[str, Any]) -> tuple[int, int]:
    total = 0
    powered = 0
    for pokemon in _board(player):
        if pokemon.get("id") != MUNKIDORI:
            continue
        total += 1
        cards = pokemon.get("energyCards") or ()
        if not isinstance(cards, (list, tuple)):
            raise MunkidoriRewardError("energyCards is not a sequence")
        if any(
            isinstance(energy, Mapping) and energy.get("id") == DARK_ENERGY
            for energy in cards
        ):
            powered += 1
    return total, powered


def potential(observation: Mapping[str, Any]) -> float:
    """Return the locked seat-relative public Munkidori-control potential."""
    current = observation.get("current")
    if not isinstance(current, Mapping):
        raise MunkidoriRewardError("observation has no public current state")
    players = current.get("players")
    seat = current.get("yourIndex")
    if (
        not isinstance(players, (list, tuple))
        or len(players) != 2
        or seat not in (0, 1)
        or not all(isinstance(player, Mapping) for player in players)
    ):
        raise MunkidoriRewardError("invalid player or seat contract")
    mine_total, mine_powered = _munkidori_counts(players[int(seat)])
    opp_total, opp_powered = _munkidori_counts(players[1 - int(seat)])
    raw = (
        0.50 * (mine_powered - opp_powered)
        + 0.25 * (mine_total - opp_total)
    )
    value = max(-1.0, min(1.0, float(raw)))
    if not math.isfinite(value):
        raise MunkidoriRewardError("non-finite potential")
    return value


def macro_rewards(
    potentials: Sequence[float],
    durations: Sequence[int],
    terminal_reward: float,
    *,
    gamma: float = GAMMA,
    coefficient: float = SHAPING_COEFFICIENT,
) -> list[float]:
    """Attach locked potential shaping to a complete ST_MAIN trajectory."""
    if not potentials or len(potentials) != len(durations):
        raise MunkidoriRewardError("potential/duration trajectory mismatch")
    if terminal_reward not in (-1.0, 0.0, 1.0):
        raise MunkidoriRewardError("terminal reward is not win/draw/loss")
    if not 0.0 < gamma <= 1.0 or coefficient < 0.0:
        raise MunkidoriRewardError("invalid shaping hyperparameters")
    values = [float(value) for value in potentials]
    if any(not math.isfinite(value) or abs(value) > 1.0 for value in values):
        raise MunkidoriRewardError("potential outside [-1, 1]")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in durations):
        raise MunkidoriRewardError("durations must be positive integers")

    rewards: list[float] = []
    for index, (value, duration) in enumerate(zip(values, durations)):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        shaped = coefficient * (gamma ** duration * next_value - value)
        if index + 1 == len(values):
            shaped += terminal_reward
        if not math.isfinite(shaped):
            raise MunkidoriRewardError("non-finite shaped reward")
        rewards.append(float(shaped))
    return rewards


def finish_trajectory(
    rows: MutableSequence[Any],
    *,
    final_learner_selects: int,
    terminal_reward: float,
) -> None:
    """Attach macro durations and the locked shaped rewards in place."""
    if not rows:
        raise MunkidoriRewardError("cannot finish an empty trajectory")
    durations: list[int] = []
    potentials: list[float] = []
    for index, row in enumerate(rows):
        next_select = (
            rows[index + 1].learner_select_index
            if index + 1 < len(rows) else int(final_learner_selects)
        )
        duration = next_select - row.learner_select_index
        if duration <= 0:
            raise MunkidoriRewardError("non-positive learner select span")
        durations.append(duration)
        potentials.append(float(row.public_potential))
    rewards = macro_rewards(potentials, durations, terminal_reward)
    for index, row in enumerate(rows):
        row.transition_steps = durations[index]
        row.reward = rewards[index]
        row.terminal = index + 1 == len(rows)
