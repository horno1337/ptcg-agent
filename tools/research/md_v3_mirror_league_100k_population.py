"""Exact-Grimmsnarl opponent league for the 100k MD-v3 PPO experiment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from agent import model
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED
from tools.research import qu_v2a_model as QM
from tools.rl_env import OpponentSpec


POPULATION_SCHEMA = "ptcg.md-v3.exact-mirror-league-100k-population.v1"
PILOT_MASS = {
    "mirror_md_v3": 0.35,
    "mirror_md_v5": 0.15,
    "mirror_bc51": 0.10,
    "mirror_ppo_v1": 0.10,
    "mirror_snapshot_0": 0.075,
    "mirror_snapshot_1": 0.075,
    "mirror_snapshot_2": 0.075,
    "mirror_snapshot_3": 0.075,
}
SNAPSHOT_INTERVAL_UPDATES = 10
SNAPSHOT_SLOTS = 4


class PopulationError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_net(path: Path) -> model.Net:
    with np.load(path, allow_pickle=False) as archive:
        net = model.QuV2Net(archive)
    if not getattr(net, "is_qu_v2", False):
        raise PopulationError(f"{path} is not a Qu-v2-compatible network")
    return net


def _move(controller: LAYERED.LayeredMirrorCardController):
    def act(obs: dict, rng: Any) -> list[int]:
        del rng
        return controller.act(obs)
    return act


@dataclass
class FrozenPopulation:
    opponents: list[OpponentSpec]
    controllers: dict[str, LAYERED.LayeredMirrorCardController]
    manifest: dict[str, Any]
    snapshot_events: list[dict[str, Any]]


def build_population(
    *,
    grim_deck: Sequence[int],
    field_snapshot: Path,
    md_v3_main_weights: Path,
    md_v3_card_weights: Path,
    ppo_v1_main_weights: Path,
    md_v1_main_weights: Path,
    qu_v2b_weights: Path,
) -> FrozenPopulation:
    """Build the fixed identities; snapshot policies are refreshed in place."""
    del field_snapshot, md_v1_main_weights
    deck = tuple(int(card) for card in grim_deck)
    if len(deck) != 60:
        raise PopulationError("exact mirror deck must contain 60 cards")
    try:
        md_v5_path = Path(BINDINGS["md_v5_weights"]).resolve()
        bc51_path = Path(BINDINGS["bc51_weights"]).resolve()
    except (NameError, KeyError) as error:
        raise PopulationError("runtime population bindings are absent") from error
    paths = {
        "md_v3": md_v3_main_weights.resolve(),
        "md_v5": md_v5_path,
        "bc51": bc51_path,
        "ppo_v1": ppo_v1_main_weights.resolve(),
        "card": md_v3_card_weights.resolve(),
        "qu": qu_v2b_weights.resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise PopulationError("one or more exact-mirror league artifacts are missing")
    nets = {name: load_net(path) for name, path in paths.items()}
    controllers: dict[str, LAYERED.LayeredMirrorCardController] = {}
    main_sources = {
        "mirror_md_v3": nets["md_v3"],
        "mirror_md_v5": nets["md_v5"],
        "mirror_bc51": nets["bc51"],
        "mirror_ppo_v1": nets["ppo_v1"],
        **{f"mirror_snapshot_{index}": nets["md_v3"] for index in range(4)},
    }
    opponents: list[OpponentSpec] = []
    for key, main_net in main_sources.items():
        controller = LAYERED.LayeredMirrorCardController(
            main_net, nets["card"], nets["qu"], f"league/{key}", deck,
        )
        controllers[key] = controller
        opponents.append(OpponentSpec(
            key=f"grimmsnarl/{key}",
            deck=deck,
            move=_move(controller),
            weight=PILOT_MASS[key],
            policy_id=f"league/{key}",
            schedule_group=key,
        ))
    if abs(sum(row.weight for row in opponents) - 1.0) > 1e-12:
        raise PopulationError("league mass does not sum to one")
    manifest = {
        "schema": POPULATION_SCHEMA,
        "pilot_mass": dict(PILOT_MASS),
        "all_games_exact_grimmsnarl_mirror": True,
        "snapshot_rotation": {
            "interval_updates": SNAPSHOT_INTERVAL_UPDATES,
            "slots": SNAPSHOT_SLOTS,
            "first_refresh_before_update": 11,
            "rotation": "round_robin_oldest_slot",
        },
        "artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "opponents": [
            {
                "key": row.key,
                "schedule_group": row.schedule_group,
                "weight": row.weight,
                "deck_sha256": row.deck_sha256,
                "policy_id": row.policy_id,
            }
            for row in opponents
        ],
    }
    return FrozenPopulation(opponents, controllers, manifest, [])


def refresh_snapshot(
    population: FrozenPopulation,
    net: QM.TorchQuV2A,
    *,
    update: int,
) -> Mapping[str, Any] | None:
    """Install the policy after each ten completed updates into one league slot."""
    if update <= 1 or (update - 1) % SNAPSHOT_INTERVAL_UPDATES:
        return None
    completed_update = update - 1
    slot = ((completed_update // SNAPSHOT_INTERVAL_UPDATES) - 1) % SNAPSHOT_SLOTS
    arrays = QM.export_numpy_weights(net)
    controller = population.controllers[f"mirror_snapshot_{slot}"]
    controller.main_net = QM.NumpyQuV2A(arrays)
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(value.tobytes())
    event = {
        "before_update": update,
        "completed_update": completed_update,
        "slot": slot,
        "weights_sha256": digest.hexdigest(),
    }
    population.snapshot_events.append(event)
    return event

