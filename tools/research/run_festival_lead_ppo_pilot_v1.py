"""Lock and run a small ST_MAIN PPO feasibility pilot for Festival Lead."""

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
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import festival_lead as RULES, model, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import CTX_TO_HAND, ST_CARD, ObsView  # noqa: E402
from tools import eval_ab as EVAL, index_corpus, rl_env  # noqa: E402
from tools.research import eval_dobi_v1_elite_teacher_card_v1_field as FIELD  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as FEST  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import run_md_v3_ppo_v2 as RUNNER  # noqa: E402
from tools.research import train_md_v3_ppo as V1  # noqa: E402
from tools.research import train_md_v3_ppo_v2 as PPO  # noqa: E402
from tools.rl_env import OpponentSpec, build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/festival-lead-ppo-pilot-v1"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
OUTPUT = RUN / "terminal-update-4-candidate"
UPDATES = 4
GAMES_PER_UPDATE = 384
BASE_SEED = 2_026_080_85
SEED_STRIDE = 1_000_037
MIN_DECISIONS = 8_000
MAX_KL = 0.02

PATHS = {
    "runner": Path(__file__).resolve(),
    "parent_checkpoint": FEST.MAIN_CHECKPOINT,
    "parent_weights": FEST.MAIN_WEIGHTS,
    "card_weights": FEST.CARD_WEIGHTS,
    "qu_weights": FEST.PARENT_WEIGHTS,
    "deck": FEST.DECK,
    "rules": FEST.RULE_MODULE,
    "field_snapshot": FIELD.FIELD_SNAPSHOT,
    "grim_variants": FIELD.GRIM_VARIANT_SNAPSHOT,
    "ppo_runner": Path(RUNNER.__file__).resolve(),
    "ppo_trainer": Path(PPO.__file__).resolve(),
    "ppo_v1_helpers": Path(V1.__file__).resolve(),
    "rl_env": Path(rl_env.__file__).resolve(),
}


class PilotError(RuntimeError):
    pass


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
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")


def load_net(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return model.QuV2Net(archive)


def build_population(main_net, card_net, qu_net, deck):
    _snapshot, rows = FIELD.load_snapshot()
    expanded = FIELD.expand_field_rows(rows)
    field_controller = EVAL.DeployableReflex(qu_net, "pilot/field-qu-v2b")
    opponents: list[OpponentSpec] = []
    for row in expanded:
        registration = tuple(int(card) for card in row["deck"])

        def move(obs, rng, opponent_deck=registration):
            del rng
            return field_controller.act(obs, opponent_deck)

        opponents.append(OpponentSpec(
            key=f"field/{row['opponent_key']}", deck=registration, move=move,
            weight=0.90 * float(row["field_weight"]),
            policy_id="pilot/field-qu-v2b", schedule_group="field",
        ))
    mirror = FEST.FestivalHybridController(main_net, card_net, deck, "pilot/frozen-festival-v1")

    def mirror_move(obs, rng):
        del rng
        return mirror.act(obs)

    opponents.append(OpponentSpec(
        key="mirror/frozen-festival-v1", deck=tuple(deck), move=mirror_move,
        weight=0.10, policy_id="pilot/frozen-festival-v1", schedule_group="mirror",
    ))
    if not math.isclose(sum(row.weight for row in opponents), 1.0, abs_tol=1e-12):
        raise PilotError("population weights do not sum to one")
    manifest = [{
        "key": row.key, "weight": row.weight, "deck_sha256": row.deck_sha256,
        "policy_id": row.policy_id, "schedule_group": row.schedule_group,
    } for row in opponents]
    return SimpleNamespace(
        opponents=opponents,
        controllers={"field": field_controller, "mirror": mirror},
        manifest=manifest,
    )


def frozen_action(obs: dict, deck: Sequence[int], card_net, qu_net):
    del qu_net
    view = ObsView(obs)
    ordinary_card = not (
        view.context == CTX_TO_HAND and view.effect_card_id == RULES.THWACKEY
    )
    if view.select_type == ST_CARD and ordinary_card:
        sample = FEATURES.encode_public_observation(obs, deck)
        logits, _ = card_net.forward(sample)
        return model.decode_qu_v2(logits, len(view.options), view.min_count, view.max_count)
    action = RULES.decide(view, deck)
    if action is None:
        raise PilotError("frozen Festival route returned None")
    return safety._repair(action, obs)


def schedule_contract(population) -> list[dict[str, Any]]:
    contracts = []
    for update in range(UPDATES):
        seed = BASE_SEED + update * SEED_STRIDE
        schedule = build_paired_schedule(population.opponents, GAMES_PER_UPDATE, seed=seed)
        contracts.append({
            "update": update + 1, "seed": seed,
            "manifest": schedule_manifest(schedule, population.opponents),
            "manifest_sha256": canonical(schedule_manifest(schedule, population.opponents)),
        })
    return contracts


def build_lock() -> dict[str, Any]:
    if RUN.exists():
        raise PilotError("pilot output already exists")
    artifacts = {}
    for name, path in PATHS.items():
        if not path.is_file():
            raise PilotError(f"missing artifact: {name}")
        artifacts[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    if artifacts["parent_weights"]["sha256"] != "d6cfd897e93ef80048f4134aa03faddcf358b5bd3674f3f25cf4f83ab411a71d":
        raise PilotError("main parent drifted")
    deck = FEST.read_festival_deck(FEST.DECK)
    main_net, card_net, qu_net = map(load_net, (FEST.MAIN_WEIGHTS, FEST.CARD_WEIGHTS, FEST.PARENT_WEIGHTS))
    population = build_population(main_net, card_net, qu_net, deck)
    payload: dict[str, Any] = {
        "schema": "ptcg.festival-lead.st-main-ppo-pilot-v1.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": "A small KL-anchored ST_MAIN PPO pilot can improve recent-field gameplay while preserving Festival BC/rule routes.",
        "training": {
            "updates": UPDATES, "games_per_update": GAMES_PER_UPDATE,
            "total_games": UPDATES * GAMES_PER_UPDATE,
            "actor_learning_rate": 1e-6, "critic_learning_rate": 1e-5,
            "gamma": 0.997, "gae_lambda": 0.95, "ppo_epochs": 2,
            "minibatch_size": 512, "clip": 0.10,
            "value_coefficient": 0.5, "entropy_coefficient": 0.002,
            "parent_kl_coefficient": 1.0, "maximum_parent_kl": MAX_KL,
            "minimum_st_main_decisions_per_update": MIN_DECISIONS,
            "trainable": "ST_MAIN actor and private critic",
            "frozen": "representation, ordinary ST_CARD BC, Thwackey and residual rules",
            "reward": "terminal win/draw/loss; SMDP GAE",
        },
        "population": {"mass": {"recent_field_qu_v2b": 0.90, "exact_festival_mirror": 0.10},
                       "opponents": population.manifest},
        "schedules": schedule_contract(population),
        "selection": {"eligible": "terminal update 4 only", "early_stopping": False,
                      "development_gate": "separate fixed 1024-game v1-vs-pilot recent-field screen"},
        "artifacts": artifacts,
        "authorization": {"research_only": True, "package": False, "upload": False},
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8")); claimed = value.pop("lock_sha256", None)
    if claimed != canonical(value):
        raise PilotError("lock self-hash failed")
    value["lock_sha256"] = claimed
    for name, row in value["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise PilotError(f"artifact drifted: {name}")
    return value


def run(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists() or OUTPUT.exists():
        raise PilotError("pilot attempt already consumed")
    attempt = {"schema": "ptcg.festival-lead.st-main-ppo-pilot-v1.attempt.v1",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "lock_sha256": lock["lock_sha256"], "before_first_outcome": True}
    attempt["attempt_sha256"] = canonical(attempt); write_new(ATTEMPT, attempt)
    device = torch.device("cpu")
    random.seed(BASE_SEED); np.random.seed(BASE_SEED); torch.manual_seed(BASE_SEED)
    net, _ = V1.load_torch_parent(FEST.MAIN_CHECKPOINT, device)
    parent, _ = V1.load_torch_parent(FEST.MAIN_CHECKPOINT, device)
    V1.verify_numpy_parity(net, FEST.MAIN_WEIGHTS, device)
    parent.eval()
    for parameter in parent.parameters(): parameter.requires_grad_(False)
    optimizer, scopes = PPO.make_optimizer(net, actor_learning_rate=1e-6, critic_learning_rate=1e-5)
    frozen_before = PPO.parameter_snapshot(scopes.frozen)
    card_net, qu_net = load_net(FEST.CARD_WEIGHTS), load_net(FEST.PARENT_WEIGHTS)
    main_runtime = load_net(FEST.MAIN_WEIGHTS)
    deck = FEST.read_festival_deck(FEST.DECK)
    population = build_population(main_runtime, card_net, qu_net, deck)
    original_frozen = PPO._frozen_action
    PPO._frozen_action = frozen_action
    rows = []
    try:
        for update, contract in enumerate(lock["schedules"], start=1):
            schedule = build_paired_schedule(population.opponents, GAMES_PER_UPDATE, seed=contract["seed"])
            manifest = schedule_manifest(schedule, population.opponents)
            if canonical(manifest) != contract["manifest_sha256"]:
                raise PilotError("runtime schedule drifted")
            decisions, rollout = RUNNER.collect_population_games(
                net, card_net, qu_net, deck, population, schedule,
                seed=BASE_SEED + (UPDATES + update - 1) * SEED_STRIDE, device=device,
            )
            if (rollout["games"] != GAMES_PER_UPDATE or rollout["invalid"] != 0
                    or rollout["controller_faults"] != 0 or len(decisions) < MIN_DECISIONS):
                raise PilotError(f"unclean update {update}: {rollout}")
            metrics = PPO.ppo_update(
                net, parent, optimizer, scopes, decisions, device=device,
                seed=BASE_SEED + (UPDATES + update - 1) * SEED_STRIDE,
                epochs=2, minibatch_size=512, clip=0.10,
                value_coefficient=0.5, entropy_coefficient=0.002,
                parent_kl_coefficient=1.0, gamma=0.997, gae_lambda=0.95,
            )
            kl = RUNNER.measure_parent_kl(net, parent, decisions, device=device)
            if kl["mean"] > MAX_KL:
                raise PilotError(f"update {update} exceeded KL ceiling")
            rows.append({"update": update, "rollout": rollout, "decisions": len(decisions),
                         "ppo": metrics, "post_update_parent_kl": kl})
            print(json.dumps({"update": update, "outcomes": rollout["outcomes"],
                              "decisions": len(decisions), "kl": kl["mean"]}), flush=True)
    finally:
        PPO._frozen_action = original_frozen
    PPO.assert_parameters_unchanged(scopes.frozen, frozen_before, label="shared representation")
    weights, checkpoint = PPO.write_candidate_checkpoint(
        OUTPUT, net, optimizer, scopes, completed_updates=UPDATES,
        parent_checkpoint_sha256=sha256(FEST.MAIN_CHECKPOINT),
        provenance={"lock_sha256": lock["lock_sha256"], "pilot": True},
    )
    payload: dict[str, Any] = {
        "schema": "ptcg.festival-lead.st-main-ppo-pilot-v1.result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "lock_sha256": lock["lock_sha256"],
        "valid": True, "updates": rows,
        "candidate": {"weights": str(weights.resolve()), "weights_sha256": sha256(weights),
                      "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint)},
        "authorization": {"package": False, "upload": False},
    }
    payload["result_sha256"] = canonical(payload); write_new(RESULT, payload); return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True); args = parser.parse_args()
    if args.stage == "lock":
        value = build_lock(); write_new(LOCK, value); print(json.dumps({"lock_sha256": value["lock_sha256"]}))
    else:
        value = run(load_lock()); print(json.dumps({"result_sha256": value["result_sha256"], "candidate": value["candidate"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
