"""Frozen opponent population for the MD-v3 actor-safe PPO-v2 experiment.

Population mass is fixed by pilot family.  Recent non-mirror deck weights are
read from a content-locked field snapshot and renormalized only after removing
Grimmsnarl.  This module constructs controllers and schedules; it never trains
or writes a runtime artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from agent import model
from tools import eval_ab as EVAL
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED
from tools.rl_env import OpponentSpec


POPULATION_SCHEMA = "ptcg.md-v3.ppo-v2-population.v1"
PILOT_MASS = {
    "mirror_md_v3": 0.25,
    "mirror_ppo_v1": 0.10,
    "mirror_md_v1": 0.10,
    "mirror_qu_v2b": 0.05,
    "field_qu_v2b": 0.40,
    "field_rules": 0.10,
}


class PopulationError(ValueError):
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


def _field_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("field")
    if not isinstance(rows, list):
        raise PopulationError("field snapshot has no field rows")
    nonmirror = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise PopulationError("field snapshot contains an invalid row")
        deck = row.get("deck")
        weight = row.get("field_weight")
        if (
            not isinstance(deck, list)
            or len(deck) != 60
            or any(not isinstance(card, int) or isinstance(card, bool)
                   for card in deck)
            or not isinstance(weight, (int, float))
            or float(weight) <= 0.0
        ):
            raise PopulationError("field snapshot row has an invalid deck/weight")
        if row.get("archetype") != "Grimmsnarl":
            nonmirror.append(dict(row))
    if not nonmirror:
        raise PopulationError("field snapshot has no non-mirror rows")
    return payload, nonmirror


@dataclass
class FrozenPopulation:
    opponents: list[OpponentSpec]
    controllers: dict[str, LAYERED.LayeredMirrorCardController]
    manifest: dict[str, Any]


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
    deck = tuple(int(card) for card in grim_deck)
    if len(deck) != 60:
        raise PopulationError("Grimmsnarl deck must contain 60 cards")
    paths = {
        "md_v3_main": md_v3_main_weights.resolve(),
        "md_v3_card": md_v3_card_weights.resolve(),
        "ppo_v1_main": ppo_v1_main_weights.resolve(),
        "md_v1_main": md_v1_main_weights.resolve(),
        "qu_v2b": qu_v2b_weights.resolve(),
        "field_snapshot": field_snapshot.resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise PopulationError("one or more population artifacts are missing")
    md_v3 = load_net(paths["md_v3_main"])
    card = load_net(paths["md_v3_card"])
    ppo_v1 = load_net(paths["ppo_v1_main"])
    md_v1 = load_net(paths["md_v1_main"])
    qu = load_net(paths["qu_v2b"])
    controllers = {
        "mirror_md_v3": LAYERED.LayeredMirrorCardController(
            md_v3, card, qu, "population/md-v3", deck,
        ),
        "mirror_ppo_v1": LAYERED.LayeredMirrorCardController(
            ppo_v1, card, qu, "population/ppo-v1", deck,
        ),
        "mirror_md_v1": LAYERED.LayeredMirrorCardController(
            md_v1, None, qu, "population/md-v1", deck,
        ),
        "mirror_qu_v2b": LAYERED.LayeredMirrorCardController(
            qu, None, qu, "population/qu-v2b-grim", deck,
        ),
    }
    opponents = [
        OpponentSpec(
            key=f"grimmsnarl/{key}",
            deck=deck,
            move=_move(controller),
            weight=PILOT_MASS[key],
            policy_id=controller.name,
            schedule_group=key,
        )
        for key, controller in controllers.items()
    ]
    snapshot, rows = _field_rows(paths["field_snapshot"])
    field_total = sum(float(row["field_weight"]) for row in rows)
    for row in rows:
        archetype = str(row["archetype"])
        opponent_deck = tuple(int(card) for card in row["deck"])
        conditional = float(row["field_weight"]) / field_total
        field_controller = LAYERED.LayeredMirrorCardController(
            qu, None, qu, f"population/qu-v2b/{archetype}", opponent_deck,
        )
        key = f"field_qu_v2b/{archetype}"
        controllers[key] = field_controller
        opponents.append(OpponentSpec(
            key=f"{archetype}/qu-v2b",
            deck=opponent_deck,
            move=_move(field_controller),
            weight=PILOT_MASS["field_qu_v2b"] * conditional,
            policy_id=field_controller.name,
            schedule_group="field_qu_v2b",
        ))
        opponents.append(OpponentSpec(
            key=f"{archetype}/rules",
            deck=opponent_deck,
            move=EVAL.safe_rules_move,
            weight=PILOT_MASS["field_rules"] * conditional,
            policy_id="rules-v1",
            schedule_group="field_rules",
        ))
    total = sum(opponent.weight for opponent in opponents)
    if abs(total - 1.0) > 1e-12:
        raise PopulationError(f"population mass is {total}, expected 1")
    manifest = {
        "schema": POPULATION_SCHEMA,
        "pilot_mass": dict(PILOT_MASS),
        "field_snapshot": {
            "path": str(paths["field_snapshot"]),
            "sha256": file_sha256(paths["field_snapshot"]),
            "schema": snapshot.get("schema"),
            "dates": snapshot.get("source", {}).get("dates"),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
            if name != "field_snapshot"
        },
        "opponents": [
            {
                "key": opponent.key,
                "schedule_group": opponent.schedule_group,
                "weight": opponent.weight,
                "deck_sha256": opponent.deck_sha256,
                "policy_id": opponent.policy_id,
            }
            for opponent in opponents
        ],
    }
    return FrozenPopulation(opponents, controllers, manifest)
