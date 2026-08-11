"""Lock and run conservative ST_MAIN PPO pilots against the frozen BC league.

Each of the four behavior-passing specialists is updated independently against
the same frozen population of the other three specialists.  This avoids a
non-stationary co-training confound.  CARD heads, public representation, and
all residual routes stay frozen.  Only terminal update 4 is eligible for a
later paired control league; this runner grants no integration or upload.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import festival_lead as FEST_RULES, model, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import CTX_TO_HAND, ST_CARD, ObsView  # noqa: E402
from tools import rl_env  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as GAME  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as FEST  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import eval_day2_specialist_league as LEAGUE  # noqa: E402
from tools.research import run_day2_expanded_bc as TRAINING  # noqa: E402
from tools.research import run_md_v3_ppo_v2 as ROLLOUT  # noqa: E402
from tools.research import train_md_v3_ppo as PPO_V1  # noqa: E402
from tools.research import train_md_v3_ppo_v2 as PPO  # noqa: E402
from tools.rl_env import OpponentSpec, build_paired_schedule, schedule_manifest  # noqa: E402


RUN = TRAINING.RUN / "ppo-league-v1"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
UPDATES = 4
GAMES_PER_UPDATE = 256
BASE_SEED = 202608151
AGENT_SEED_STRIDE = 10_000_019
UPDATE_SEED_STRIDE = 1_000_037
MIN_DECISIONS = 6_000
MAX_KL = 0.02
ACTOR_LR = 0.000001
CRITIC_LR = 0.00001


class PilotError(RuntimeError):
    """The preregistered specialist PPO league failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != canonical(value):
        raise PilotError(f"self-hash or schema failed: {path}")
    value[key] = claimed
    return value


def load_net(path: Path) -> model.Net:
    with np.load(path, allow_pickle=False) as archive:
        return model.QuV2Net(archive)


def rows() -> dict[str, dict[str, Any]]:
    return LEAGUE.candidates()


def make_controller(name: str, row: Mapping[str, Any], qu, tag: str):
    main = load_net(Path(row["main"]))
    card = load_net(Path(row["card"]))
    deck = tuple(int(card_id) for card_id in row["deck"])
    if name == "festival":
        return FEST.FestivalHybridController(main, card, deck, tag)
    return GAME.DualHeadController(main, card, qu, deck, tag)


def build_population(
    learner: str, candidates: Mapping[str, Mapping[str, Any]], qu,
):
    opponents = []
    controllers = {}
    members = [name for name in candidates if name != learner]
    if len(members) != 3:
        raise PilotError("frozen league must contain exactly three opponents")
    for name in members:
        row = candidates[name]
        controller = make_controller(
            name, row, qu, f"ppo/{learner}/frozen-{name}"
        )
        controllers[name] = controller
        deck = tuple(int(card) for card in row["deck"])

        def move(obs, rng, active=controller, registration=deck):
            del rng
            return active.act(obs, registration)

        opponents.append(OpponentSpec(
            key=f"frozen/{name}",
            deck=deck,
            move=move,
            weight=1.0 / 3.0,
            policy_id=(
                f"{name}:main:{sha256(Path(row['main']))}+"
                f"card:{sha256(Path(row['card']))}"
            ),
            schedule_group=f"peer/{name}",
        ))
    if not math.isclose(sum(row.weight for row in opponents), 1.0):
        raise PilotError("opponent population weights do not sum to one")
    manifest = [{
        "key": row.key,
        "weight": row.weight,
        "deck_sha256": row.deck_sha256,
        "policy_id": row.policy_id,
        "schedule_group": row.schedule_group,
    } for row in opponents]
    return SimpleNamespace(
        opponents=opponents, controllers=controllers, manifest=manifest,
    )


def ordinary_frozen_action(
    obs: dict, deck: Sequence[int], card_net: model.Net, qu_net: model.Net,
) -> list[int]:
    view = ObsView(obs)
    active = card_net if view.select_type == ST_CARD else qu_net
    sample = FEATURES.encode_public_observation(obs, deck)
    logits, _ = active.forward(sample)
    return model.decode_qu_v2(
        logits, len(view.options), view.min_count, view.max_count,
    )


def festival_frozen_action(
    obs: dict, deck: Sequence[int], card_net: model.Net, qu_net: model.Net,
) -> list[int]:
    del qu_net
    view = ObsView(obs)
    ordinary_card = not (
        view.context == CTX_TO_HAND and view.effect_card_id == FEST_RULES.THWACKEY
    )
    if view.select_type == ST_CARD and ordinary_card:
        sample = FEATURES.encode_public_observation(obs, deck)
        logits, _ = card_net.forward(sample)
        return model.decode_qu_v2(
            logits, len(view.options), view.min_count, view.max_count,
        )
    action = FEST_RULES.decide(view, deck)
    if action is None:
        raise PilotError("Festival frozen route returned None")
    return safety._repair(action, obs)


def schedule_contract(population, agent_index: int) -> list[dict[str, Any]]:
    contracts = []
    for update in range(UPDATES):
        seed = (
            BASE_SEED + agent_index * AGENT_SEED_STRIDE
            + update * UPDATE_SEED_STRIDE
        )
        schedule = build_paired_schedule(
            population.opponents, GAMES_PER_UPDATE, seed=seed,
        )
        seats = Counter(row.learner_seat for row in schedule)
        if seats != {0: GAMES_PER_UPDATE // 2, 1: GAMES_PER_UPDATE // 2}:
            raise PilotError("PPO schedule is not seat balanced")
        manifest = schedule_manifest(schedule, population.opponents)
        contracts.append({
            "update": update + 1,
            "rollout_seed": seed,
            "ppo_seed": seed + UPDATES * UPDATE_SEED_STRIDE,
            "manifest_sha256": canonical(manifest),
            "seat_counts": {
                str(key): value for key, value in sorted(seats.items())
            },
        })
    return contracts


def build_lock() -> dict[str, Any]:
    if RUN.exists():
        raise PilotError("PPO league output already exists")
    training_lock = TRAINING.load_lock()
    league = load_self(
        LEAGUE.RESULT, "ptcg.day2-specialist-league.result.v1", "result_sha256"
    )
    if league.get("all_valid") is not True:
        raise PilotError("input specialist league was not valid")
    candidates = rows()
    qu = load_net(LEAGUE.PARENT_WEIGHTS)
    populations = {}
    schedules = {}
    for agent_index, name in enumerate(candidates):
        population = build_population(name, candidates, qu)
        populations[name] = population.manifest
        schedules[name] = schedule_contract(population, agent_index)

    paths = {
        "runner": Path(__file__).resolve(),
        "training_runner": Path(TRAINING.__file__).resolve(),
        "training_lock": TRAINING.LOCK,
        "league_evaluator": Path(LEAGUE.__file__).resolve(),
        "league_result": LEAGUE.RESULT,
        "rollout_runner": Path(ROLLOUT.__file__).resolve(),
        "ppo_trainer": Path(PPO.__file__).resolve(),
        "ppo_helpers": Path(PPO_V1.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
        "qu_weights": LEAGUE.PARENT_WEIGHTS,
        "festival_rules": Path(FEST_RULES.__file__).resolve(),
    }
    for name, row in candidates.items():
        paths[f"{name}_main_checkpoint"] = (
            TRAINING.RUN / f"candidates/{name}/main/model/candidate-qu-v2a-checkpoint.pt"
        )
        paths[f"{name}_main_weights"] = Path(row["main"])
        paths[f"{name}_card_weights"] = Path(row["card"])

    payload = {
        "schema": "ptcg.day2-specialist-ppo-league.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "Small independent KL-anchored ST_MAIN updates against the frozen "
            "peer league can improve cross-deck outcomes without erasing BC."
        ),
        "training_lock_sha256": training_lock["lock_sha256"],
        "league_result_sha256": league["result_sha256"],
        "training": {
            "agents": list(candidates),
            "updates_per_agent": UPDATES,
            "games_per_update": GAMES_PER_UPDATE,
            "games_per_agent": UPDATES * GAMES_PER_UPDATE,
            "total_games": len(candidates) * UPDATES * GAMES_PER_UPDATE,
            "actor_learning_rate": ACTOR_LR,
            "critic_learning_rate": CRITIC_LR,
            "gamma": 0.997,
            "gae_lambda": 0.95,
            "ppo_epochs": 2,
            "minibatch_size": 512,
            "clip": 0.10,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.002,
            "parent_kl_coefficient": 1.0,
            "maximum_parent_kl": MAX_KL,
            "minimum_st_main_decisions_per_update": MIN_DECISIONS,
            "trainable": "ST_MAIN actor plus private critic",
            "frozen": "shared representation, CARD head, residual controller",
            "reward": "terminal win/draw/loss with SMDP GAE",
        },
        "population": {
            "design": "each agent independently faces the other three frozen BC stacks",
            "opponent_mass": "equal one-third per peer",
            "agents": populations,
        },
        "schedules": schedules,
        "selection": {
            "eligible": "terminal update 4 only for each agent",
            "early_stopping": False,
            "next_gate": "separate paired BC-control versus PPO league evaluation",
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "authorization": {
            "research_only": True,
            "integration": False,
            "package": False,
            "upload": False,
        },
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if claimed != canonical(value):
        raise PilotError("PPO league lock self-hash failed")
    value["lock_sha256"] = claimed
    for name, row in value["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise PilotError(f"locked artifact drifted: {name}")
    return value


def train_agent(
    lock: Mapping[str, Any], name: str, agent_index: int,
    candidates: Mapping[str, Mapping[str, Any]], device: torch.device,
) -> dict[str, Any]:
    row = candidates[name]
    parent_checkpoint = Path(
        lock["artifacts"][f"{name}_main_checkpoint"]["path"]
    )
    parent_weights = Path(lock["artifacts"][f"{name}_main_weights"]["path"])
    card_weights = Path(lock["artifacts"][f"{name}_card_weights"]["path"])
    output = RUN / f"{name}/terminal-update-4-candidate"
    net, _ = PPO_V1.load_torch_parent(parent_checkpoint, device)
    parent, _ = PPO_V1.load_torch_parent(parent_checkpoint, device)
    PPO_V1.verify_numpy_parity(net, parent_weights, device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    optimizer, scopes = PPO.make_optimizer(
        net, actor_learning_rate=ACTOR_LR, critic_learning_rate=CRITIC_LR,
    )
    frozen_before = PPO.parameter_snapshot(scopes.frozen)
    card = load_net(card_weights)
    qu = load_net(Path(lock["artifacts"]["qu_weights"]["path"]))
    population = build_population(name, candidates, qu)
    original = PPO._frozen_action
    PPO._frozen_action = (
        festival_frozen_action if name == "festival" else ordinary_frozen_action
    )
    updates = []
    try:
        for update, contract in enumerate(lock["schedules"][name], start=1):
            schedule = build_paired_schedule(
                population.opponents,
                GAMES_PER_UPDATE,
                seed=int(contract["rollout_seed"]),
            )
            manifest = schedule_manifest(schedule, population.opponents)
            if canonical(manifest) != contract["manifest_sha256"]:
                raise PilotError(f"runtime PPO schedule drifted: {name}/{update}")
            decisions, rollout = ROLLOUT.collect_population_games(
                net, card, qu, row["deck"], population, schedule,
                seed=int(contract["ppo_seed"]), device=device,
            )
            if (
                rollout["games"] != GAMES_PER_UPDATE
                or rollout["invalid"] != 0
                or rollout["controller_faults"] != 0
                or len(decisions) < MIN_DECISIONS
            ):
                raise PilotError(f"unclean update {name}/{update}: {rollout}")
            metrics = PPO.ppo_update(
                net, parent, optimizer, scopes, decisions, device=device,
                seed=int(contract["ppo_seed"]), epochs=2,
                minibatch_size=512, clip=0.10,
                value_coefficient=0.5, entropy_coefficient=0.002,
                parent_kl_coefficient=1.0, gamma=0.997, gae_lambda=0.95,
            )
            kl = ROLLOUT.measure_parent_kl(
                net, parent, decisions, device=device,
            )
            if kl["mean"] > MAX_KL:
                raise PilotError(f"{name}/{update} exceeded parent KL ceiling")
            updates.append({
                "update": update,
                "rollout": rollout,
                "decisions": len(decisions),
                "ppo": metrics,
                "post_update_parent_kl": kl,
            })
            print(json.dumps({
                "agent": name, "update": update,
                "outcomes": rollout["outcomes"],
                "decisions": len(decisions), "parent_kl": kl["mean"],
            }, sort_keys=True), flush=True)
    finally:
        PPO._frozen_action = original
    PPO.assert_parameters_unchanged(
        scopes.frozen, frozen_before, label=f"{name} shared representation",
    )
    weights, checkpoint = PPO.write_candidate_checkpoint(
        output, net, optimizer, scopes, completed_updates=UPDATES,
        parent_checkpoint_sha256=sha256(parent_checkpoint),
        provenance={
            "lock_sha256": lock["lock_sha256"],
            "agent": name,
            "pilot": True,
        },
    )
    return {
        "updates": updates,
        "candidate": {
            "weights": str(weights.resolve()),
            "weights_sha256": sha256(weights),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
        },
    }


def run(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise PilotError("PPO league attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.day2-specialist-ppo-league.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "written_before_first_training_outcome": True,
    })
    device = torch.device("cpu")
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)
    candidates = rows()
    agents = {}
    for agent_index, name in enumerate(candidates):
        agents[name] = train_agent(lock, name, agent_index, candidates, device)
    payload = {
        "schema": "ptcg.day2-specialist-ppo-league.result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "valid": True,
        "agents": agents,
        "next_gate": "paired BC-control versus PPO league evaluation",
        "authorization": {
            "integration": False, "package": False, "upload": False,
        },
    }
    payload["result_sha256"] = canonical(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock()
            write_new(LOCK, value)
            print(json.dumps({
                "lock_sha256": value["lock_sha256"],
                "training": value["training"],
            }, indent=2, sort_keys=True))
            return 0
        value = run(load_lock())
    except (PilotError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "result_sha256": value["result_sha256"],
        "agents": {
            name: row["candidate"] for name, row in value["agents"].items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
