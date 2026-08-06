"""Frozen exact-mirror opponent league for the dobi-v1 300k PPO run."""

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


POPULATION_SCHEMA = "ptcg.dobi-v1.exact-mirror-league-300k-population.v1"
PILOT_MASS = {
    "mirror_dobi_v1": 0.40,
    "mirror_md_v3": 0.20,
    "mirror_md_v5": 0.075,
    "mirror_bc51": 0.0625,
    "mirror_ppo_v1": 0.0625,
    "mirror_snapshot_0": 0.05,
    "mirror_snapshot_1": 0.05,
    "mirror_snapshot_2": 0.05,
    "mirror_snapshot_3": 0.05,
}
SNAPSHOT_INTERVAL_UPDATES = 10
SNAPSHOT_SLOTS = 4
BINDINGS: dict[str, Path] = {}


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
        raise PopulationError(f"{path} is not Qu-v2 compatible")
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
    *, grim_deck: Sequence[int], field_snapshot: Path,
    md_v3_main_weights: Path, md_v3_card_weights: Path,
    ppo_v1_main_weights: Path, md_v1_main_weights: Path,
    qu_v2b_weights: Path,
) -> FrozenPopulation:
    """The legacy argument names map to dobi-v1 and frozen MD-v3 explicitly."""
    del field_snapshot
    deck = tuple(int(card) for card in grim_deck)
    if len(deck) != 60:
        raise PopulationError("exact mirror deck must contain 60 cards")
    try:
        md_v5 = Path(BINDINGS["md_v5_weights"]).resolve()
        bc51 = Path(BINDINGS["bc51_weights"]).resolve()
    except KeyError as error:
        raise PopulationError("runtime population bindings are absent") from error
    paths = {
        "dobi_v1": md_v3_main_weights.resolve(),
        "md_v3": md_v1_main_weights.resolve(),
        "md_v5": md_v5,
        "bc51": bc51,
        "ppo_v1": ppo_v1_main_weights.resolve(),
        "card": md_v3_card_weights.resolve(),
        "qu": qu_v2b_weights.resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise PopulationError("one or more league artifacts are missing")
    nets = {name: load_net(path) for name, path in paths.items()}
    main_sources = {
        "mirror_dobi_v1": nets["dobi_v1"],
        "mirror_md_v3": nets["md_v3"],
        "mirror_md_v5": nets["md_v5"],
        "mirror_bc51": nets["bc51"],
        "mirror_ppo_v1": nets["ppo_v1"],
        **{f"mirror_snapshot_{index}": nets["dobi_v1"] for index in range(4)},
    }
    controllers = {}
    opponents = []
    for key, main_net in main_sources.items():
        controller = LAYERED.LayeredMirrorCardController(
            main_net, nets["card"], nets["qu"], f"league/{key}", deck,
        )
        controllers[key] = controller
        opponents.append(OpponentSpec(
            key=f"grimmsnarl/{key}", deck=deck, move=_move(controller),
            weight=PILOT_MASS[key], policy_id=f"league/{key}",
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
        "opponents": [{
            "key": row.key, "schedule_group": row.schedule_group,
            "weight": row.weight, "deck_sha256": row.deck_sha256,
            "policy_id": row.policy_id,
        } for row in opponents],
    }
    return FrozenPopulation(opponents, controllers, manifest, [])


def refresh_snapshot(
    population: FrozenPopulation, net: QM.TorchQuV2A, *, update: int,
) -> Mapping[str, Any] | None:
    if update <= 1 or (update - 1) % SNAPSHOT_INTERVAL_UPDATES:
        return None
    completed = update - 1
    slot = ((completed // SNAPSHOT_INTERVAL_UPDATES) - 1) % SNAPSHOT_SLOTS
    arrays = QM.export_numpy_weights(net)
    population.controllers[f"mirror_snapshot_{slot}"].main_net = QM.NumpyQuV2A(arrays)
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode() + b"\0" + value.dtype.str.encode() + b"\0" + value.tobytes())
    event = {
        "before_update": update, "completed_update": completed,
        "slot": slot, "weights_sha256": digest.hexdigest(),
    }
    population.snapshot_events.append(event)
    return event
