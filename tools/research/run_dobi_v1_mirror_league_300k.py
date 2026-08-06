"""Run the prospectively locked dobi-v1 299,520-game PPO continuation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model  # noqa: E402
from tools.research import dobi_v1_mirror_league_300k_population as POP  # noqa: E402
from tools.research import lock_dobi_v1_mirror_league_300k as LOCK  # noqa: E402
from tools.research import run_md_v3_mirror_league_100k as LEAGUE  # noqa: E402


def _install_contract() -> None:
    base = LEAGUE.BASE
    base.LOCK = LOCK
    base.POP = POP
    base.RESULT_SCHEMA = "ptcg.dobi-v1.st-main-ppo-300k-training-result.v1"
    base.EXPECTED_UPDATES = LOCK.UPDATES
    base.EXPECTED_GAMES_PER_UPDATE = LOCK.GAMES_PER_UPDATE
    base.EXPECTED_TOTAL_GAMES = LOCK.TOTAL_GAMES
    base.EXPECTED_MINIMUM_ST_MAIN = 20_000
    base.EXPECTED_MAXIMUM_PARENT_KL = 0.04
    base.EXPECTED_ROLLOUT_SEED_BASE = LOCK.BASE_SEED
    base.EXPECTED_SEED_STRIDE = LOCK.SEED_STRIDE
    base.EXPECTED_SEAT_COUNTS = dict(LOCK.SEATS_PER_UPDATE)
    base.EXPECTED_FAMILY_PAIR_COUNTS = {"mirror": LOCK.PAIRS_PER_UPDATE}
    base.EXPECTED_HYPERPARAMETERS = {
        "actor_learning_rate": 2.5e-6,
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
    base.build_locked_schedule_contract = LEAGUE._schedule_contract


def main(argv: Sequence[str] | None = None) -> None:
    # Snapshot exports must be readable by the exact deployed NumPy runtime.
    LEAGUE.QM.NumpyQuV2A = model.QuV2Net
    LEAGUE.POP = POP
    LEAGUE.LOCK = LOCK
    LEAGUE._install_contract = _install_contract
    LEAGUE.main(argv)


if __name__ == "__main__":
    main()
