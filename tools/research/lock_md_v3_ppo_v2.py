"""Prospectively seal the actor-safe MD-v3 ST_MAIN PPO-v2 experiment.

This module contains no training or game execution.  It binds the immutable
inputs, all deterministic rollout schedules, the fixed terminal checkpoint,
and the gameplay decision rules before any PPO-v2 outcome is generated.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA = "ptcg.md-v3.st-main-ppo-v2-training-lock.v1"
POPULATION_SCHEMA = "ptcg.md-v3.ppo-v2-population.v1"

UPDATES = 16
GAMES_PER_UPDATE = 768
TOTAL_GAMES = 12_288
PAIRS_PER_UPDATE = 384
SEATS_PER_UPDATE = {"0": 384, "1": 384}
BASE_SEED = 2_026_073_101
SEED_STRIDE = 1_000_003
ROLLOUT_SEEDS = tuple(BASE_SEED + index * SEED_STRIDE for index in range(UPDATES))
PPO_SEEDS = tuple(
    BASE_SEED + (UPDATES + index) * SEED_STRIDE
    for index in range(UPDATES)
)
DIRECT_GAMEPLAY_SEED = 2_026_073_102
DIRECT_GAMEPLAY_GAMES = 2_560

PILOT_MASS = {
    "mirror_md_v3": 0.25,
    "mirror_ppo_v1": 0.10,
    "mirror_md_v1": 0.10,
    "mirror_qu_v2b": 0.05,
    "field_qu_v2b": 0.40,
    "field_rules": 0.10,
}

TARGET_DECK = (
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    104, 104,
    112, 112, 112, 112,
    646, 646, 646, 646,
    647, 647, 647,
    648, 648, 648,
    860, 860,
    1079, 1079, 1079,
    1080,
    1086, 1086, 1086, 1086,
    1097, 1097, 1097,
    1122,
    1137,
    1152, 1152, 1152, 1152,
    1182, 1182,
    1219, 1219, 1219, 1219,
    1227, 1227, 1227, 1227,
    1231,
    1259, 1259, 1259, 1259,
)
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)

DEFAULT_ARTIFACT_PATHS: dict[str, Path] = {
    "lock_builder": ROOT / "tools/research/lock_md_v3_ppo_v2.py",
    "runner": ROOT / "tools/research/run_md_v3_ppo_v2.py",
    "trainer": ROOT / "tools/research/train_md_v3_ppo_v2.py",
    "population": ROOT / "tools/research/md_v3_ppo_v2_population.py",
    "direct_evaluator": (
        ROOT / "tools/research/eval_md_v3_ppo_v2_gameplay.py"
    ),
    "direct_lock_builder": (
        ROOT / "tools/research/lock_md_v3_ppo_v2_gameplay.py"
    ),
    "parent_checkpoint": (
        ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
        "candidate-qu-v2a-checkpoint.pt"
    ),
    "parent_weights": (
        ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
        "candidate-qu-v2a-weights.npz"
    ),
    "card_weights": ROOT / "agent/md_v2_card_weights.npz",
    "qu_weights": ROOT / "agent/weights.npz",
    "ppo_v1_weights": (
        ROOT / "tools/checkpoints/md-v3-ppo-v1/training/"
        "candidate-qu-v2a-weights.npz"
    ),
    "md_v1_weights": ROOT / "agent/md_v1_weights.npz",
    "field_snapshot": (
        ROOT / "tools/checkpoints/md-v3-mirror-main-v1/"
        "field-july27-28.json"
    ),
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
    "runtime_model": ROOT / "agent/model.py",
    "runtime_features": ROOT / "agent/qu_v2_features.py",
    "base_features": ROOT / "agent/features.py",
    "obsview": ROOT / "agent/obsview.py",
    "policy": ROOT / "agent/policy.py",
    "safety": ROOT / "agent/safety.py",
    "cards": ROOT / "agent/cards.py",
    "card_router": ROOT / "agent/md_v2_card.py",
    "training_features": ROOT / "tools/research/qu_v2a_features.py",
    "torch_model": ROOT / "tools/research/qu_v2a_model.py",
    "v1_ppo_helpers": ROOT / "tools/research/train_md_v3_ppo.py",
    "bc_helpers": ROOT / "tools/research/train_qu_v2a.py",
    "layered_controller": (
        ROOT / "tools/research/eval_md_v2_card_v1_gameplay.py"
    ),
    "rl_env": ROOT / "tools/rl_env.py",
    "eval_ab": ROOT / "tools/eval_ab.py",
    "index_corpus": ROOT / "tools/index_corpus.py",
    "cabt_wrapper": ROOT / "tools/cabt.py",
    "battle_engine": ROOT / "engine/libcg.so",
    "cards_data": ROOT / "data/cards.json",
    "attacks_data": ROOT / "data/attacks.json",
}
REQUIRED_ARTIFACTS = frozenset(DEFAULT_ARTIFACT_PATHS)


class PPOV2LockError(RuntimeError):
    """The PPO-v2 prospective contract is absent, invalid, or has drifted."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve_recorded_path(raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PPOV2LockError("artifact path is missing")
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _record_artifact(path: Path) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise PPOV2LockError(f"missing artifact {path}: {error}") from error
    if not resolved.is_file():
        raise PPOV2LockError(f"artifact is not a regular file: {resolved}")
    return {
        "path": _display_path(resolved),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _read_deck(path: Path) -> tuple[int, ...]:
    try:
        deck = tuple(
            int(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValueError) as error:
        raise PPOV2LockError(f"cannot read target deck {path}: {error}") from error
    if deck != TARGET_DECK:
        raise PPOV2LockError("target deck is not the exact frozen Grimmsnarl list")
    return deck


def _read_field_snapshot(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PPOV2LockError(f"cannot read field snapshot {path}: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "ptcg.recent-frequency-weighted-field.v1"
        or payload.get("source", {}).get("dates")
        != ["2026-07-27", "2026-07-28"]
    ):
        raise PPOV2LockError("field snapshot is not the locked Jul27-28 field")
    rows = payload.get("field")
    if not isinstance(rows, list) or not rows:
        raise PPOV2LockError("field snapshot is empty")
    total = 0.0
    for row in rows:
        if not isinstance(row, Mapping):
            raise PPOV2LockError("field snapshot has a non-object row")
        deck = row.get("deck")
        weight = row.get("field_weight")
        if (
            not isinstance(row.get("archetype"), str)
            or not isinstance(deck, list)
            or len(deck) != 60
            or any(
                not isinstance(card, int) or isinstance(card, bool)
                for card in deck
            )
            or not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise PPOV2LockError("field snapshot has an invalid row")
        total += float(weight)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise PPOV2LockError("field snapshot weights do not sum to one")
    return payload


def _schedule_row(
    schedule: Sequence[Any],
    opponents: Sequence[Any],
    *,
    update: int,
    rollout_seed: int,
    ppo_seed: int,
    manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seat_counts = Counter(str(int(episode.learner_seat)) for episode in schedule)
    representatives = {}
    for episode in schedule:
        representatives.setdefault(int(episode.pair_id), episode)
    group_counts: Counter[str] = Counter()
    opponent_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for episode in representatives.values():
        opponent = opponents[int(episode.opponent_index)]
        group = str(opponent.schedule_group)
        group_counts[group] += 1
        opponent_counts[str(opponent.key)] += 1
        if group.startswith("mirror_"):
            family_counts["mirror"] += 1
        elif group.startswith("field_"):
            family_counts["field"] += 1
        else:
            raise PPOV2LockError(f"unclassified schedule group {group!r}")
    return {
        "update": int(update),
        "rollout_seed": int(rollout_seed),
        "ppo_seed": int(ppo_seed),
        "games": len(schedule),
        "pairs": len(representatives),
        "seat_counts": dict(sorted(seat_counts.items())),
        "family_pair_counts": dict(sorted(family_counts.items())),
        "schedule_group_pair_counts": dict(sorted(group_counts.items())),
        "opponent_pair_counts": dict(sorted(opponent_counts.items())),
        "manifest_sha256": canonical_sha256(list(manifest)),
    }


def _compute_schedule_contracts(
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    # Dynamic imports keep inspection/tests independent of the native engine.
    from tools.research import md_v3_ppo_v2_population as POP
    from tools.rl_env import (
        OpponentSpec,
        build_paired_schedule,
        schedule_manifest,
    )

    deck = _read_deck(paths["deck"])
    population = POP.build_population(
        grim_deck=deck,
        field_snapshot=paths["field_snapshot"],
        md_v3_main_weights=paths["parent_weights"],
        md_v3_card_weights=paths["card_weights"],
        ppo_v1_main_weights=paths["ppo_v1_weights"],
        md_v1_main_weights=paths["md_v1_weights"],
        qu_v2b_weights=paths["qu_weights"],
    )
    population_contract = {
        "schema": POP.POPULATION_SCHEMA,
        "pilot_mass": dict(POP.PILOT_MASS),
        "manifest": population.manifest,
        "manifest_sha256": canonical_sha256(population.manifest),
        "family_pair_mass": {"mirror": 0.5, "field": 0.5},
    }
    # Allocate the 50/50 mirror/field curriculum at the top level.  Opponent
    # indices and weights stay unchanged, so the scheduler still allocates the
    # individual frozen pilots/decks within each family by their exact masses.
    scheduling_opponents = [
        OpponentSpec(
            key=opponent.key,
            deck=opponent.deck,
            move=opponent.move,
            weight=opponent.weight,
            policy_id=opponent.policy_id,
            schedule_group=(
                "mirror"
                if str(opponent.schedule_group).startswith("mirror_")
                else "field"
            ),
        )
        for opponent in population.opponents
    ]
    updates = []
    for update, (rollout_seed, ppo_seed) in enumerate(
        zip(ROLLOUT_SEEDS, PPO_SEEDS), start=1,
    ):
        schedule = build_paired_schedule(
            scheduling_opponents, GAMES_PER_UPDATE, seed=rollout_seed,
        )
        manifest = schedule_manifest(schedule, population.opponents)
        updates.append(_schedule_row(
            schedule,
            population.opponents,
            update=update,
            rollout_seed=rollout_seed,
            ppo_seed=ppo_seed,
            manifest=manifest,
        ))

    def _noop(_obs: dict, _rng: Any) -> list[int]:
        return [0]

    direct_opponents = [OpponentSpec(
        key="grimmsnarl/frozen-md-v3",
        deck=deck,
        move=_noop,
        weight=1.0,
        # Frozen weight identities are bound separately in ``artifacts``.
        # Keep the schedule label stable so the post-training binder can copy
        # this exact prospective schedule without rewriting it.
        policy_id="complete-frozen-md-v3",
        schedule_group="frozen-md-v3",
    )]
    direct_schedule = build_paired_schedule(
        direct_opponents,
        DIRECT_GAMEPLAY_GAMES,
        seed=DIRECT_GAMEPLAY_SEED,
    )
    direct_manifest = schedule_manifest(direct_schedule, direct_opponents)
    direct_record = {
        "generator": "tools.rl_env.build_paired_schedule",
        "manifest": "tools.rl_env.schedule_manifest",
        "manifest_sha256": canonical_sha256(direct_manifest),
        "games": len(direct_schedule),
        "pairs": len({episode.pair_id for episode in direct_schedule}),
        "candidate_seat_counts": dict(sorted(Counter(
            str(int(episode.learner_seat)) for episode in direct_schedule
        ).items())),
        "opponent": {
            "key": direct_opponents[0].key,
            "policy_id": direct_opponents[0].policy_id,
            "schedule_group": direct_opponents[0].schedule_group,
            "deck_sha256": direct_opponents[0].deck_sha256,
        },
    }
    return population_contract, updates, direct_record


def _validate_schedule_rows(rows: Any) -> None:
    if not isinstance(rows, list) or len(rows) != UPDATES:
        raise PPOV2LockError("lock must contain exactly 16 update schedules")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PPOV2LockError("schedule row is not an object")
        expected_identity = {
            "update": index + 1,
            "rollout_seed": ROLLOUT_SEEDS[index],
            "ppo_seed": PPO_SEEDS[index],
            "games": GAMES_PER_UPDATE,
            "pairs": PAIRS_PER_UPDATE,
            "seat_counts": SEATS_PER_UPDATE,
            "family_pair_counts": {"field": 192, "mirror": 192},
        }
        for key, expected in expected_identity.items():
            if row.get(key) != expected:
                raise PPOV2LockError(
                    f"update {index + 1} schedule has invalid {key}"
                )
        groups = row.get("schedule_group_pair_counts")
        opponents = row.get("opponent_pair_counts")
        if (
            not isinstance(groups, Mapping)
            or set(groups) != set(PILOT_MASS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in groups.values()
            )
            or sum(groups.values()) != PAIRS_PER_UPDATE
        ):
            raise PPOV2LockError("schedule-group pair quotas are invalid")
        for name, mass in PILOT_MASS.items():
            if abs(int(groups[name]) - PAIRS_PER_UPDATE * mass) > 1.0:
                raise PPOV2LockError(f"schedule-group quota drifted for {name}")
        if (
            not isinstance(opponents, Mapping)
            or not opponents
            or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for key, value in opponents.items()
            )
            or sum(opponents.values()) != PAIRS_PER_UPDATE
        ):
            raise PPOV2LockError("opponent pair quotas are invalid")
        if not _is_sha256(row.get("manifest_sha256")):
            raise PPOV2LockError("update schedule manifest hash is invalid")


def _validate_population(value: Any) -> None:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != POPULATION_SCHEMA
        or value.get("pilot_mass") != PILOT_MASS
        or value.get("family_pair_mass") != {"mirror": 0.5, "field": 0.5}
        or not isinstance(value.get("manifest"), Mapping)
        or value.get("manifest_sha256")
        != canonical_sha256(value.get("manifest"))
    ):
        raise PPOV2LockError("frozen population contract is invalid")


def _validate_direct_gameplay(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise PPOV2LockError("direct gameplay contract is missing")
    protocol = value.get("protocol")
    schedule = value.get("schedule")
    expected_protocol = {
        "games": DIRECT_GAMEPLAY_GAMES,
        "seed": DIRECT_GAMEPLAY_SEED,
        "matchup": "complete frozen MD-v3 exact Grimmsnarl mirror",
        "candidate_seat_balance": {"0": 1280, "1": 1280},
        "fixed_terminal_update": 16,
        "one_schedule_one_attempt": True,
        "score": "(wins + 0.5 * draws) / 2560",
        "confidence_interval": "two-sided Wilson score interval at 95%",
        "pass": "Wilson CI95 lower bound strictly greater than 0.50",
        "zero_faults": True,
        "field_gate_only_after_pass": True,
    }
    if protocol != expected_protocol:
        raise PPOV2LockError("direct gameplay protocol drifted")
    if (
        not isinstance(schedule, Mapping)
        or schedule.get("generator") != "tools.rl_env.build_paired_schedule"
        or schedule.get("manifest") != "tools.rl_env.schedule_manifest"
        or schedule.get("games") != DIRECT_GAMEPLAY_GAMES
        or schedule.get("pairs") != DIRECT_GAMEPLAY_GAMES // 2
        or schedule.get("candidate_seat_counts") != {"0": 1280, "1": 1280}
        or not _is_sha256(schedule.get("manifest_sha256"))
        or schedule.get("opponent") != {
            "key": "grimmsnarl/frozen-md-v3",
            "policy_id": "complete-frozen-md-v3",
            "schedule_group": "frozen-md-v3",
            "deck_sha256": canonical_sha256(list(TARGET_DECK)),
        }
    ):
        raise PPOV2LockError("direct gameplay schedule drifted")


def validate_contract(lock: Mapping[str, Any]) -> None:
    if lock.get("schema") != SCHEMA:
        raise PPOV2LockError("PPO-v2 lock schema mismatch")
    if (
        lock.get("prospective") is not True
        or lock.get("changed_runtime_component")
        != "exact-deck ST_MAIN weights only"
        or lock.get("field_snapshot") != {
            "schema": "ptcg.recent-frequency-weighted-field.v1",
            "dates": ["2026-07-27", "2026-07-28"],
            "use": (
                "training population and a separately locked field gate only "
                "after the direct gate passes"
            ),
        }
        or lock.get("target_deck") != {
            "name": "exact MD-v1 Grimmsnarl registration",
            "canonical_sha256": TARGET_DECK_SHA256,
            "cards": 60,
        }
    ):
        raise PPOV2LockError("PPO-v2 prospective scope drifted")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != REQUIRED_ARTIFACTS:
        raise PPOV2LockError("PPO-v2 artifact set is incomplete or unexpected")
    for name, record in artifacts.items():
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("path"), str)
            or not _is_sha256(record.get("sha256"))
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record.get("bytes") < 0
        ):
            raise PPOV2LockError(f"invalid artifact record {name}")

    training = lock.get("training")
    expected_training = {
        "updates": UPDATES,
        "games_per_update": GAMES_PER_UPDATE,
        "total_games": TOTAL_GAMES,
        "seat_balance_per_update": SEATS_PER_UPDATE,
        "rollout_seed_base": BASE_SEED,
        "seed_stride": SEED_STRIDE,
        "rollout_seeds": list(ROLLOUT_SEEDS),
        "ppo_seeds": list(PPO_SEEDS),
        "actor_learning_rate": 1e-6,
        "critic_learning_rate": 1e-5,
        "gamma": 0.997,
        "gae_lambda": 0.95,
        "ppo_epochs": 2,
        "minibatch_size": 512,
        "clip": 0.10,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.002,
        "parent_kl_coefficient": 1.0,
        "maximum_parent_kl_per_update": 0.02,
        "maximum_final_parent_kl": 0.02,
        "minimum_st_main_decisions_per_update": 20_000,
        "invalid_games_per_update": 0,
        "faults_repairs_exceptions_per_update": 0,
        "optimizer": "one persistent Adam instance across all 16 updates",
        "reward": "terminal win/draw/loss only",
        "return_estimator": "episode-local variable-duration semi-MDP GAE",
        "actor_trainable_modules": ["option1", "context1", "policy"],
        "critic_trainable_modules": ["value1", "value2"],
        "frozen_modules": [
            "embedding", "board1", "board_relation", "state1", "state2",
        ],
        "critic_shared_representation": "detached",
    }
    if training != expected_training:
        raise PPOV2LockError("fixed PPO-v2 training configuration drifted")

    _validate_population(lock.get("population"))
    schedules = lock.get("schedules")
    if (
        not isinstance(schedules, Mapping)
        or schedules.get("generator")
        != "tools.rl_env.build_paired_schedule"
        or schedules.get("manifest") != "tools.rl_env.schedule_manifest"
        or schedules.get("native_engine_rng")
        != "unseedable; deterministic schedule only"
        or schedules.get("invalid_game_policy")
        != "fail the experiment; never refill or replace"
    ):
        raise PPOV2LockError("schedule protocol drifted")
    _validate_schedule_rows(schedules.get("updates"))

    selection = lock.get("candidate_selection")
    if selection != {
        "eligible_update": 16,
        "eligible_checkpoint": "fixed terminal checkpoint after update 16 only",
        "intermediate_checkpoints": "crash recovery only; never eligible",
        "early_stopping": False,
        "posthoc_checkpoint_selection": False,
        "failure": (
            "any invalid game, nonzero fault, insufficient ST_MAIN decisions, "
            "non-finite update, frozen-parameter drift, artifact drift, or "
            "parent KL above 0.02 rejects the experiment"
        ),
    }:
        raise PPOV2LockError("candidate-selection rule drifted")
    _validate_direct_gameplay(lock.get("direct_gameplay"))
    if lock.get("authorization") != {
        "research_only": True,
        "production_mutation": False,
        "package_build": False,
        "ladder_upload": False,
        "field_gate": "forbidden unless the direct gameplay gate passes",
        "upload_requires_user_named_tag_and_explicit_approval": True,
    }:
        raise PPOV2LockError("authorization scope drifted")


def verify_bound_artifacts(lock: Mapping[str, Any]) -> dict[str, Path]:
    validate_contract(lock)
    resolved: dict[str, Path] = {}
    for name, record in lock["artifacts"].items():
        path = resolve_recorded_path(record["path"])
        if not path.is_file():
            raise PPOV2LockError(f"bound artifact is missing: {name}")
        if path.stat().st_size != record["bytes"]:
            raise PPOV2LockError(f"bound artifact size drifted: {name}")
        if file_sha256(path) != record["sha256"]:
            raise PPOV2LockError(f"bound artifact hash drifted: {name}")
        resolved[name] = path
    return resolved


def build_lock(
    *,
    artifact_paths: Mapping[str, Path] | None = None,
    created_at: str | None = None,
    population_contract: Mapping[str, Any] | None = None,
    update_schedules: Sequence[Mapping[str, Any]] | None = None,
    direct_schedule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = {
        name: Path(path)
        for name, path in (
            DEFAULT_ARTIFACT_PATHS if artifact_paths is None else artifact_paths
        ).items()
    }
    if set(paths) != REQUIRED_ARTIFACTS:
        missing = sorted(REQUIRED_ARTIFACTS - set(paths))
        extra = sorted(set(paths) - REQUIRED_ARTIFACTS)
        raise PPOV2LockError(
            f"artifact path set mismatch; missing={missing}, extra={extra}"
        )
    _read_deck(paths["deck"])
    field = _read_field_snapshot(paths["field_snapshot"])
    if (
        population_contract is None
        or update_schedules is None
        or direct_schedule is None
    ):
        if not (
            population_contract is None
            and update_schedules is None
            and direct_schedule is None
        ):
            raise PPOV2LockError(
                "schedule injection must provide population, updates, and direct"
            )
        population_contract, computed_updates, direct_schedule = (
            _compute_schedule_contracts(paths)
        )
        update_schedules = computed_updates
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    if not isinstance(timestamp, str) or not timestamp:
        raise PPOV2LockError("created_at must be a non-empty string")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": timestamp,
        "prospective": True,
        "hypothesis": (
            "Actor-safe terminal-outcome PPO with semi-MDP credit assignment "
            "and a frozen mirror/recent-field opponent population improves the "
            "exact-deck MD-v3 ST_MAIN policy."
        ),
        "changed_runtime_component": "exact-deck ST_MAIN weights only",
        "training": {
            "updates": UPDATES,
            "games_per_update": GAMES_PER_UPDATE,
            "total_games": TOTAL_GAMES,
            "seat_balance_per_update": dict(SEATS_PER_UPDATE),
            "rollout_seed_base": BASE_SEED,
            "seed_stride": SEED_STRIDE,
            "rollout_seeds": list(ROLLOUT_SEEDS),
            "ppo_seeds": list(PPO_SEEDS),
            "actor_learning_rate": 1e-6,
            "critic_learning_rate": 1e-5,
            "gamma": 0.997,
            "gae_lambda": 0.95,
            "ppo_epochs": 2,
            "minibatch_size": 512,
            "clip": 0.10,
            "value_coefficient": 0.5,
            "entropy_coefficient": 0.002,
            "parent_kl_coefficient": 1.0,
            "maximum_parent_kl_per_update": 0.02,
            "maximum_final_parent_kl": 0.02,
            "minimum_st_main_decisions_per_update": 20_000,
            "invalid_games_per_update": 0,
            "faults_repairs_exceptions_per_update": 0,
            "optimizer": "one persistent Adam instance across all 16 updates",
            "reward": "terminal win/draw/loss only",
            "return_estimator": (
                "episode-local variable-duration semi-MDP GAE"
            ),
            "actor_trainable_modules": ["option1", "context1", "policy"],
            "critic_trainable_modules": ["value1", "value2"],
            "frozen_modules": [
                "embedding", "board1", "board_relation", "state1", "state2",
            ],
            "critic_shared_representation": "detached",
        },
        "population": dict(population_contract),
        "schedules": {
            "generator": "tools.rl_env.build_paired_schedule",
            "manifest": "tools.rl_env.schedule_manifest",
            "native_engine_rng": "unseedable; deterministic schedule only",
            "invalid_game_policy": "fail the experiment; never refill or replace",
            "updates": [dict(row) for row in update_schedules],
        },
        "candidate_selection": {
            "eligible_update": 16,
            "eligible_checkpoint": (
                "fixed terminal checkpoint after update 16 only"
            ),
            "intermediate_checkpoints": "crash recovery only; never eligible",
            "early_stopping": False,
            "posthoc_checkpoint_selection": False,
            "failure": (
                "any invalid game, nonzero fault, insufficient ST_MAIN decisions, "
                "non-finite update, frozen-parameter drift, artifact drift, or "
                "parent KL above 0.02 rejects the experiment"
            ),
        },
        "direct_gameplay": {
            "protocol": {
                "games": DIRECT_GAMEPLAY_GAMES,
                "seed": DIRECT_GAMEPLAY_SEED,
                "matchup": (
                    "complete frozen MD-v3 exact Grimmsnarl mirror"
                ),
                "candidate_seat_balance": {"0": 1280, "1": 1280},
                "fixed_terminal_update": 16,
                "one_schedule_one_attempt": True,
                "score": "(wins + 0.5 * draws) / 2560",
                "confidence_interval": (
                    "two-sided Wilson score interval at 95%"
                ),
                "pass": (
                    "Wilson CI95 lower bound strictly greater than 0.50"
                ),
                "zero_faults": True,
                "field_gate_only_after_pass": True,
            },
            "schedule": dict(direct_schedule),
        },
        "field_snapshot": {
            "schema": field["schema"],
            "dates": ["2026-07-27", "2026-07-28"],
            "use": (
                "training population and a separately locked field gate only "
                "after the direct gate passes"
            ),
        },
        "target_deck": {
            "name": "exact MD-v1 Grimmsnarl registration",
            "canonical_sha256": TARGET_DECK_SHA256,
            "cards": 60,
        },
        "authorization": {
            "research_only": True,
            "production_mutation": False,
            "package_build": False,
            "ladder_upload": False,
            "field_gate": "forbidden unless the direct gameplay gate passes",
            "upload_requires_user_named_tag_and_explicit_approval": True,
        },
        "artifacts": {
            name: _record_artifact(path)
            for name, path in sorted(paths.items())
        },
    }
    validate_contract(payload)
    payload["lock_sha256"] = canonical_sha256(payload)
    return payload


def load_lock(
    path: Path,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PPOV2LockError(f"cannot read PPO-v2 lock {path}: {error}") from error
    if not isinstance(value, dict):
        raise PPOV2LockError("PPO-v2 lock root is not an object")
    recorded = value.pop("lock_sha256", None)
    calculated = canonical_sha256(value)
    value["lock_sha256"] = recorded
    if not _is_sha256(recorded) or recorded != calculated:
        raise PPOV2LockError("PPO-v2 lock self-hash mismatch")
    validate_contract(value)
    if verify_artifacts:
        verify_bound_artifacts(value)
    return value


def write_lock(path: Path, payload: Mapping[str, Any]) -> None:
    value = dict(payload)
    recorded = value.pop("lock_sha256", None)
    if recorded != canonical_sha256(value):
        raise PPOV2LockError("refusing to write a lock with an invalid self-hash")
    value["lock_sha256"] = recorded
    validate_contract(value)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".partial",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise PPOV2LockError(f"refusing to overwrite {output}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = build_lock()
        write_lock(args.output, payload)
    except (
        PPOV2LockError,
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({
        "lock": str(args.output.resolve()),
        "lock_sha256": payload["lock_sha256"],
        "updates": UPDATES,
        "games": TOTAL_GAMES,
        "direct_gameplay_games": DIRECT_GAMEPLAY_GAMES,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
