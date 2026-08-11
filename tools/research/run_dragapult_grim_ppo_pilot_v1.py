"""Lock and run a bounded Dragapult ST_MAIN PPO pilot for the Grim matchup."""

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

from agent import model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import ST_CARD, ObsView  # noqa: E402
from tools import eval_ab as EVAL, rl_env  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as FIELD  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as DOBI  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import qu_v2a_model as QM  # noqa: E402
from tools.research import run_dragapult_bc as DRAG  # noqa: E402
from tools.research import run_md_v3_ppo_v2 as ROLLOUT  # noqa: E402
from tools.research import train_md_v3_ppo as PPO_V1  # noqa: E402
from tools.research import train_md_v3_ppo_v2 as PPO  # noqa: E402
from tools.rl_env import OpponentSpec, build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-grim-ppo-pilot-v1"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
OUTPUT = RUN / "terminal-update-4-candidate"
PARENT_CHECKPOINT = DRAG.RUN / "candidates/main/model/candidate-qu-v2a-checkpoint.pt"
PARENT_WEIGHTS = DRAG.RUN / "candidates/main/model/candidate-qu-v2a-weights.npz"
CARD_WEIGHTS = DRAG.RUN / "candidates/card/model/candidate-qu-v2a-weights.npz"
QU_WEIGHTS = ROOT / "agent/weights.npz"
DOBI_MAIN = ROOT / (
    "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
    "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
)
DOBI_CARD = ROOT / "agent/md_v2_card_weights.npz"
DOBI_DECK = ROOT / "decks/md_v1_grimmsnarl.csv"
BASELINE = ROOT / "tools/checkpoints/bc-specialists-vs-grim-champions-20260811/result.json"
FIELD_RESULT = DRAG.RUN / "gameplay/result.json"
CORPUS = ROOT / "tools/checkpoints/dragapult-exact-v2/combined-corpus.json"
UPDATES = 4
GAMES_PER_UPDATE = 256
BASE_SEED = 202608122
SEED_STRIDE = 1_000_037
MIN_DECISIONS = 8_000
MAX_KL = 0.02
GRIM_MASS = 0.75
FIELD_MASS = 0.25


PATHS = {
    "runner": Path(__file__).resolve(),
    "parent_checkpoint": PARENT_CHECKPOINT,
    "parent_weights": PARENT_WEIGHTS,
    "card_weights": CARD_WEIGHTS,
    "qu_weights": QU_WEIGHTS,
    "dobi_main": DOBI_MAIN,
    "dobi_card": DOBI_CARD,
    "dobi_deck": DOBI_DECK,
    "training_lock": DRAG.LOCK,
    "sealed_behavior": DRAG.RUN / "sealed-test-result.json",
    "direct_baseline": BASELINE,
    "field_result": FIELD_RESULT,
    "corpus_inventory": CORPUS,
    "field_snapshot": FIELD.FIELD_SNAPSHOT,
    "rollout_runner": Path(ROLLOUT.__file__).resolve(),
    "ppo_trainer": Path(PPO.__file__).resolve(),
    "ppo_helpers": Path(PPO_V1.__file__).resolve(),
    "rl_env": Path(rl_env.__file__).resolve(),
}


class PilotError(RuntimeError):
    """The preregistered Dragapult PPO pilot failed closed."""


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


def load_net(path: Path) -> model.Net:
    with np.load(path, allow_pickle=False) as archive:
        return model.QuV2Net(archive)


def build_population(parent: model.Net, card: model.Net, qu: model.Net):
    deck = tuple(int(value) for value in DRAG.TARGET_DECK)
    dobi_deck = tuple(COMMON.read_deck(DOBI_DECK))
    dobi = DOBI.LayeredMirrorCardController(
        load_net(DOBI_MAIN), load_net(DOBI_CARD), qu,
        "drag-ppo/frozen-dobi", dobi_deck,
    )
    opponents = [OpponentSpec(
        key="grimmsnarl/frozen-dobi-v1",
        deck=dobi_deck,
        move=dobi.opponent_move,
        weight=GRIM_MASS,
        policy_id=(
            f"main:{sha256(DOBI_MAIN)}+card:{sha256(DOBI_CARD)}+qu:{sha256(QU_WEIGHTS)}"
        ),
        schedule_group="grim",
    )]
    _snapshot, rows = FIELD.load_field()
    field_controller = EVAL.DeployableReflex(qu, "drag-ppo/top20-qu-v2b")
    for row in rows:
        registration = tuple(int(value) for value in row["deck"])

        def move(obs, rng, registered_deck=registration):
            del rng
            return field_controller.act(obs, registered_deck)

        opponents.append(OpponentSpec(
            key=f"field/{row['opponent_key']}",
            deck=registration,
            move=move,
            weight=FIELD_MASS * float(row["field_weight"]),
            policy_id=f"qu-v2b:{sha256(QU_WEIGHTS)}",
            schedule_group="field",
        ))
    if not math.isclose(sum(row.weight for row in opponents), 1.0, abs_tol=1e-12):
        raise PilotError("population weights do not sum to one")
    manifest = [{
        "key": row.key,
        "weight": row.weight,
        "deck_sha256": row.deck_sha256,
        "policy_id": row.policy_id,
        "schedule_group": row.schedule_group,
    } for row in opponents]
    del parent, card, deck
    return SimpleNamespace(
        opponents=opponents,
        controllers={"dobi": dobi, "field": field_controller},
        manifest=manifest,
    )


def frozen_action(
    obs: dict, deck: Sequence[int], card_net: model.Net, qu_net: model.Net,
) -> list[int]:
    view = ObsView(obs)
    active = card_net if view.select_type == ST_CARD else qu_net
    sample = FEATURES.encode_public_observation(obs, deck)
    logits, _ = active.forward(sample)
    return model.decode_qu_v2(
        logits, len(view.options), view.min_count, view.max_count,
    )


def schedule_contract(population) -> list[dict[str, Any]]:
    rows = []
    for update in range(UPDATES):
        seed = BASE_SEED + update * SEED_STRIDE
        schedule = build_paired_schedule(
            population.opponents, GAMES_PER_UPDATE, seed=seed,
        )
        manifest = schedule_manifest(schedule, population.opponents)
        seats = Counter(row.learner_seat for row in schedule)
        if seats != {0: GAMES_PER_UPDATE // 2, 1: GAMES_PER_UPDATE // 2}:
            raise PilotError("PPO schedule is not seat balanced")
        rows.append({
            "update": update + 1,
            "seed": seed,
            "manifest": manifest,
            "manifest_sha256": canonical(manifest),
        })
    return rows


def validate_prior() -> dict[str, Any]:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    dobi = baseline.get("cells", {}).get("dragapult", {}).get("dobi-v1", {})
    field = json.loads(FIELD_RESULT.read_text(encoding="utf-8"))
    if dobi.get("valid") is not True or field.get("decision", {}).get(
        "supported_superiority"
    ) is not True:
        raise PilotError("Dragapult baseline contracts failed")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    target = [
        game for game in corpus["games"]
        if any(seat["registered_deck_sha256"] == DRAG.TARGET_SHA256
               for seat in game["seats"])
    ]
    exact = [
        game for game in target
        if any(seat["registered_deck_sha256"]
               == "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
               for seat in game["seats"])
    ]
    if len(target) != 139 or len(exact) != 24:
        raise PilotError("Dragapult corpus inventory drifted")
    return {
        "dobi": {"games": dobi["summary"]["scheduled_games"], "score": dobi["score"]},
        "field": {
            "games": field["summaries"]["candidate"]["scheduled_games"],
            "score": field["summaries"]["candidate"]["score"],
            "paired_gain": field["decision"]["candidate_minus_control"],
        },
        "corpus": {"target_games": len(target), "exact_dobi_games": len(exact)},
    }


def build_lock() -> dict[str, Any]:
    if RUN.exists():
        raise PilotError("pilot output already exists")
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
        raise PilotError(f"required artifacts missing: {missing}")
    prior = validate_prior()
    parent, card, qu = map(load_net, (PARENT_WEIGHTS, CARD_WEIGHTS, QU_WEIGHTS))
    population = build_population(parent, card, qu)
    payload = {
        "schema": "ptcg.dragapult-grim.st-main-ppo-pilot-v1.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "hypothesis": (
            "A small KL-anchored outcome-optimized ST_MAIN update can repair "
            "Dragapult's Dobi deficit without erasing its strong general-field policy."
        ),
        "prior": prior,
        "training": {
            "updates": UPDATES,
            "games_per_update": GAMES_PER_UPDATE,
            "total_games": UPDATES * GAMES_PER_UPDATE,
            "actor_learning_rate": 0.000003,
            "critic_learning_rate": 0.00001,
            "gamma": 0.997,
            "gae_lambda": 0.95,
            "ppo_epochs": 2,
            "minibatch_size": 512,
            "clip": 0.10,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.002,
            "parent_kl_coefficient": 0.5,
            "maximum_parent_kl": MAX_KL,
            "minimum_st_main_decisions_per_update": MIN_DECISIONS,
            "trainable": "ST_MAIN actor plus private critic",
            "frozen": "shared representation, Dragapult CARD, Qu-v2B residual",
            "reward": "terminal win/draw/loss with SMDP GAE",
        },
        "population": {
            "mass": {"frozen_dobi": GRIM_MASS, "top20_qu_v2b": FIELD_MASS},
            "opponents": population.manifest,
        },
        "schedules": schedule_contract(population),
        "selection": {
            "eligible": "terminal update 4 only",
            "early_stopping": False,
            "next_gate": (
                "fixed paired direct Dobi screen versus current Dragapult; "
                "only a positive point estimate may enter a separate field guard"
            ),
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in PATHS.items()
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
        raise PilotError("pilot lock self-hash failed")
    value["lock_sha256"] = claimed
    for name, row in value["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise PilotError(f"locked artifact drifted: {name}")
    return value


def run(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists() or OUTPUT.exists():
        raise PilotError("pilot attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.dragapult-grim.st-main-ppo-pilot-v1.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "written_before_first_outcome": True,
    })
    device = torch.device("cpu")
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)
    net, _ = PPO_V1.load_torch_parent(PARENT_CHECKPOINT, device)
    parent, _ = PPO_V1.load_torch_parent(PARENT_CHECKPOINT, device)
    PPO_V1.verify_numpy_parity(net, PARENT_WEIGHTS, device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    optimizer, scopes = PPO.make_optimizer(
        net, actor_learning_rate=0.000003, critic_learning_rate=0.00001,
    )
    frozen_before = PPO.parameter_snapshot(scopes.frozen)
    card, qu = load_net(CARD_WEIGHTS), load_net(QU_WEIGHTS)
    population = build_population(load_net(PARENT_WEIGHTS), card, qu)
    original = PPO._frozen_action
    PPO._frozen_action = frozen_action
    rows = []
    try:
        for update, contract in enumerate(lock["schedules"], start=1):
            schedule = build_paired_schedule(
                population.opponents, GAMES_PER_UPDATE, seed=contract["seed"],
            )
            manifest = schedule_manifest(schedule, population.opponents)
            if canonical(manifest) != contract["manifest_sha256"]:
                raise PilotError("runtime PPO schedule drifted")
            decisions, rollout = ROLLOUT.collect_population_games(
                net, card, qu, DRAG.TARGET_DECK, population, schedule,
                seed=BASE_SEED + (UPDATES + update - 1) * SEED_STRIDE,
                device=device,
            )
            if (
                rollout["games"] != GAMES_PER_UPDATE
                or rollout["invalid"] != 0
                or rollout["controller_faults"] != 0
                or len(decisions) < MIN_DECISIONS
            ):
                raise PilotError(f"unclean update {update}: {rollout}")
            metrics = PPO.ppo_update(
                net, parent, optimizer, scopes, decisions, device=device,
                seed=BASE_SEED + (UPDATES + update - 1) * SEED_STRIDE,
                epochs=2, minibatch_size=512, clip=0.10,
                value_coefficient=0.5, entropy_coefficient=0.002,
                parent_kl_coefficient=0.5, gamma=0.997, gae_lambda=0.95,
            )
            kl = ROLLOUT.measure_parent_kl(net, parent, decisions, device=device)
            if kl["mean"] > MAX_KL:
                raise PilotError(f"update {update} exceeded parent KL ceiling")
            rows.append({
                "update": update,
                "rollout": rollout,
                "decisions": len(decisions),
                "ppo": metrics,
                "post_update_parent_kl": kl,
            })
            print(json.dumps({
                "update": update,
                "outcomes": rollout["outcomes"],
                "decisions": len(decisions),
                "parent_kl": kl["mean"],
            }, sort_keys=True), flush=True)
    finally:
        PPO._frozen_action = original
    PPO.assert_parameters_unchanged(
        scopes.frozen, frozen_before, label="shared representation",
    )
    weights, checkpoint = PPO.write_candidate_checkpoint(
        OUTPUT, net, optimizer, scopes, completed_updates=UPDATES,
        parent_checkpoint_sha256=sha256(PARENT_CHECKPOINT),
        provenance={"lock_sha256": lock["lock_sha256"], "pilot": True},
    )
    payload = {
        "schema": "ptcg.dragapult-grim.st-main-ppo-pilot-v1.result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "valid": True,
        "updates": rows,
        "candidate": {
            "weights": str(weights.resolve()),
            "weights_sha256": sha256(weights),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
        },
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
            print(json.dumps({"lock_sha256": value["lock_sha256"]}, sort_keys=True))
            return 0
        value = run(load_lock())
    except (PilotError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "result_sha256": value["result_sha256"],
        "candidate": value["candidate"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
