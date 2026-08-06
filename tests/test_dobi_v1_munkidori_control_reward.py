from __future__ import annotations

import math

import pytest

from tools.research import dobi_v1_munkidori_control_reward as REWARD


def mon(card_id: int, energies: tuple[int, ...] = ()) -> dict:
    return {
        "id": card_id,
        "energyCards": [{"id": energy} for energy in energies],
    }


def obs(mine: list[dict], opponent: list[dict], seat: int = 0) -> dict:
    players = [
        {"active": mine[:1], "bench": mine[1:]},
        {"active": opponent[:1], "bench": opponent[1:]},
    ]
    if seat == 1:
        players.reverse()
    return {"current": {"players": players, "yourIndex": seat}}


def test_potential_is_seat_relative_and_counts_only_dark_powered_munkidori() -> None:
    mine = [mon(REWARD.MUNKIDORI, (REWARD.DARK_ENERGY,)), mon(646)]
    opponent = [mon(REWARD.MUNKIDORI, (6,)), mon(REWARD.MUNKIDORI)]
    assert REWARD.potential(obs(mine, opponent, 0)) == pytest.approx(0.25)
    assert REWARD.potential(obs(mine, opponent, 1)) == pytest.approx(0.25)


def test_potential_clips_to_locked_range() -> None:
    powered = [mon(REWARD.MUNKIDORI, (REWARD.DARK_ENERGY,)) for _ in range(4)]
    assert REWARD.potential(obs(powered, [])) == 1.0
    assert REWARD.potential(obs([], powered)) == -1.0


def test_macro_rewards_put_terminal_authority_only_on_last_transition() -> None:
    rewards = REWARD.macro_rewards([0.0, 0.75], [2, 3], 1.0)
    assert rewards[0] == pytest.approx(
        REWARD.SHAPING_COEFFICIENT * REWARD.GAMMA**2 * 0.75
    )
    assert rewards[1] == pytest.approx(1.0 - REWARD.SHAPING_COEFFICIENT * 0.75)


def test_discounted_shaping_telescopes() -> None:
    potentials = [0.25, -0.5, 0.75]
    durations = [2, 1, 4]
    rewards = REWARD.macro_rewards(potentials, durations, -1.0)
    discounts = [1.0]
    for duration in durations[:-1]:
        discounts.append(discounts[-1] * REWARD.GAMMA**duration)
    discounted = sum(weight * reward for weight, reward in zip(discounts, rewards))
    terminal_discount = math.prod(REWARD.GAMMA**duration for duration in durations[:-1])
    expected = (
        terminal_discount * -1.0
        - REWARD.SHAPING_COEFFICIENT * potentials[0]
    )
    assert discounted == pytest.approx(expected)


def test_finish_trajectory_binds_potentials_durations_and_terminal() -> None:
    rows = [
        type("Row", (), {
            "learner_select_index": 4,
            "public_potential": 0.0,
            "transition_steps": 0,
            "reward": 0.0,
            "terminal": False,
        })(),
        type("Row", (), {
            "learner_select_index": 7,
            "public_potential": 0.75,
            "transition_steps": 0,
            "reward": 0.0,
            "terminal": False,
        })(),
    ]
    REWARD.finish_trajectory(rows, final_learner_selects=9, terminal_reward=1.0)
    expected = REWARD.macro_rewards([0.0, 0.75], [3, 2], 1.0)
    assert [(row.transition_steps, row.reward, row.terminal) for row in rows] == [
        (3, expected[0], False),
        (2, expected[1], True),
    ]


@pytest.mark.parametrize(
    "potentials,durations,terminal",
    [([], [], 1.0), ([0.0], [0], 1.0), ([1.1], [1], 1.0), ([0.0], [1], 0.5)],
)
def test_invalid_reward_contract_fails_closed(potentials, durations, terminal) -> None:
    with pytest.raises(REWARD.MunkidoriRewardError):
        REWARD.macro_rewards(potentials, durations, terminal)
