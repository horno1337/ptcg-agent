"""Contract tests for the locked Dobi Munkidori-control PPO experiment."""

from __future__ import annotations

from tools.research import lock_dobi_v1_mirror_league_300k as OLD
from tools.research import lock_dobi_v1_munkidori_control_ppo_v1 as LOCK
from tools.research import eval_dobi_v1_munkidori_control_ppo_v1_field as FIELD


def test_size_seeds_and_terminal_selection_are_fixed_and_fresh() -> None:
    assert LOCK.UPDATES == 130
    assert LOCK.GAMES_PER_UPDATE == 768
    assert LOCK.TOTAL_GAMES == 99_840
    assert len(LOCK.ROLLOUT_SEEDS) == len(LOCK.PPO_SEEDS) == 130
    assert not set(LOCK.ROLLOUT_SEEDS + LOCK.PPO_SEEDS) & set(
        OLD.ROLLOUT_SEEDS + OLD.PPO_SEEDS
    )
    payload = LOCK.build_lock()
    assert payload["decision_rules"]["candidate"] == (
        "only fixed terminal update 130 is selection-eligible"
    )
    assert len(payload["schedules"]["updates"]) == 130


def test_lock_binds_exact_public_potential_and_implementation_files() -> None:
    payload = LOCK.build_lock()
    reward = payload["training"]["reward"]
    assert reward == {
        "contract": "public-potential-v1",
        "terminal": "win/draw/loss = +1/0/-1",
        "gamma": 0.997,
        "coefficient": 0.15,
        "phi": (
            "clip(0.50*powered_munkidori_diff + "
            "0.25*all_munkidori_diff, -1, 1)"
        ),
        "terminal_phi": 0.0,
    }
    for key in ("reward", "preregistration", "runner", "trainer", "population"):
        artifact = payload["artifacts"][key]
        assert artifact["bytes"] > 0
        assert len(artifact["sha256"]) == 64
    assert payload["population"]["family_pair_mass"] == {"mirror": 1.0}


def test_field_gate_binds_new_snapshot_and_fixed_threshold() -> None:
    FIELD._configure()
    payload = FIELD.build_lock()
    assert payload["written_before_outcomes"] is True
    assert payload["field"]["source_dates"] == [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
        "2026-08-04",
        "2026-08-05",
    ]
    assert len(payload["field"]["primary"]) == 12
    assert payload["protocol"]["games_per_arm"] == 5_120
    assert payload["protocol"]["noninferiority_margin"] == -0.015
    assert payload["prior_mirror_evidence"]["score"] == 0.510302734375
