"""Prospectively seal the MD-v4 resource-aware PPO v1 experiment.

This module creates no engine, reads no replay action/reward/outcome, and does
not train a model.  It binds the fixed warm start, deployed NumPy parent,
frozen population, all 24 rollout schedules, optimization constants, terminal
selection rule, offline rejection screens, and the still-sealed July 29
archive before an official run can begin.
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

from tools.rl_env import OpponentSpec, build_paired_schedule, schedule_manifest


GENERATION = os.environ.get("PTCG_RESOURCE_PPO_GENERATION", "v1")
if GENERATION not in {"v1", "v2", "v3", "v4"}:
    raise RuntimeError(
        "PTCG_RESOURCE_PPO_GENERATION must be v1, v2, v3, or v4"
    )
SCHEMA = f"ptcg.md-v4.resource-ppo-training-lock.{GENERATION}"
POPULATION_SCHEMA = "ptcg.md-v4.resource-ppo-population.v1"
CANDIDATE = f"md-v4-resource-ppo-{GENERATION}"

UPDATES = 24
GAMES_PER_UPDATE = 768
TOTAL_GAMES = 18_432
PAIRS_PER_UPDATE = 384
SEATS_PER_UPDATE = {"0": 384, "1": 384}
FAMILY_PAIRS_PER_UPDATE = {"field": 192, "mirror": 192}
ROLLOUT_SEED_BASE = {
    "v1": 2_026_073_201,
    "v2": 2_026_073_107,
    "v3": 2_026_073_108,
    "v4": 2_026_073_109,
}[GENERATION]
SEED_STRIDE = 1_000_003
ROLLOUT_SEEDS = tuple(
    ROLLOUT_SEED_BASE + index * SEED_STRIDE
    for index in range(UPDATES)
)
PPO_SEEDS = tuple(
    ROLLOUT_SEED_BASE + (UPDATES + index) * SEED_STRIDE
    for index in range(UPDATES)
)

ACTOR_LEARNING_RATE = 2e-6
CRITIC_LEARNING_RATE = 1e-5
GAMMA = 0.997
GAE_LAMBDA = 0.95
PPO_EPOCHS = 2
MINIBATCH_SIZE = 512
PPO_CLIP = 0.10
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.002
PARENT_KL_COEFFICIENT = 1.0
GRADIENT_NORM = 1.0
MIN_ST_MAIN_PER_UPDATE = 20_000
MAX_PARENT_KL = 0.02
CRITIC_HIDDEN = 64

EXPECTED_VALIDATION_CALLBACKS = 99_946
EXPECTED_VALIDATION_GAMES = 2_090
MIN_PARENT_DISAGREEMENTS = 2_999
MIN_PARENT_GAMES_TOUCHED = 1_045
MIN_INITIAL_DISAGREEMENTS = 1_000
MIN_INITIAL_GAMES_TOUCHED = 418

PILOT_MASS = {
    "mirror_md_v3": 0.25,
    "mirror_md_v4_epoch4": 0.10,
    "mirror_ppo_v2": 0.05,
    "mirror_md_v1": 0.05,
    "mirror_qu_v2b": 0.05,
    "field_qu_v2b": 0.40,
    "field_rules": 0.10,
}

ACTOR_MODULES = (
    "resource1",
    "event_embedding",
    "role_embedding",
    "log_card_projections",
    "attack_embedding",
    "area_embedding",
    "log_gru",
    "fusion",
    "residual1",
    "residual2",
)

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

WARM_CHECKPOINT_SHA256 = (
    "ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad"
)
WARM_STATE_SHA256 = (
    "6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908"
)
WARM_NUMPY_MAPPING_SHA256 = (
    "e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01"
)
FROZEN_PARENT_STATE_SHA256 = (
    "5976a58846bacb4e06e86365f2ad9603c7dd333fff369a6ebbb1539a1175bb36"
)
DEPLOYED_PARENT_WEIGHTS_SHA256 = (
    "76420fc2e031127143f5816e64973ec49e150da22b84604308b649c9a4dc8df8"
)
CORRECTION_LOCK_SHA256 = (
    "482920b6162ac5037218ac5a324144137dcc04592101f8713ffd8313ce8b1d42"
)
CORRECTION_RESULT_SHA256 = (
    "4c3ddc4abfe5fc087f2d563b7660272465be07db870bde20c966edbe48c25fb9"
)
JULY29_SHA256 = (
    "dbe39b021aebe75c46805105604b56c6263ebb0b4f49d7bc41025517a1f4168e"
)
JULY29_BYTES = 744_279_881
JULY29_ENTRIES = 4_387
JULY29_JSON_ENTRIES = 4_386
JULY29_UNCOMPRESSED_BYTES = 21_474_480_425
JULY29_INVENTORY_SHA256 = (
    "32620d4df3e6a7a88a3416464e67f510914fca95755b39f3416dfbd9f094b40d"
)

DIRECT_PAIR_SEED = 2_026_073_202
DIRECT_PAIRS = 1_280
DIRECT_GAMES = 2_560

RUN_ROOT = ROOT / f"tools/checkpoints/md-v4-resource-ppo-{GENERATION}"
DEFAULT_OUTPUT = RUN_ROOT / "training-lock.json"
OFFICIAL_TRAINING_OUTPUT = RUN_ROOT / "training"

DEFAULT_ARTIFACT_PATHS: dict[str, Path] = {
    "preregistration": (
        ROOT / (
            "tools/research/md-v4-resource-ppo-v1-preregistration.md"
            if GENERATION == "v1"
            else f"tools/research/md-v4-resource-ppo-{GENERATION}-preregistration.md"
        )
    ),
    "lock_builder": ROOT / "tools/research/lock_md_v4_resource_ppo_v1.py",
    "runner": ROOT / "tools/research/run_md_v4_resource_ppo_v1.py",
    "lock_tests": ROOT / "tests/test_lock_md_v4_resource_ppo_v1.py",
    "runner_tests": ROOT / "tests/test_run_md_v4_resource_ppo_v1.py",
    "warm_checkpoint": (
        ROOT / "tools/checkpoints/md-v4-public-window-v1/model/"
        "candidate-md-v4-recovery.pt"
    ),
    "correction_lock": (
        ROOT / "tools/checkpoints/md-v4-public-window-v1/"
        "deployed-parent-correction-lock.json"
    ),
    "correction_result": (
        ROOT / "tools/checkpoints/md-v4-public-window-v1/model/"
        "deployed-parent-correction-evaluation-result.json"
    ),
    "correction_record": (
        ROOT / "tools/research/md-v4-deployed-parent-correction-result.md"
    ),
    "correction_evaluator": (
        ROOT / "tools/research/eval_md_v4_deployed_parent_correction.py"
    ),
    "correction_lock_builder": (
        ROOT / "tools/research/lock_md_v4_deployed_parent_correction.py"
    ),
    "md_v4_features": ROOT / "tools/research/md_v4_features.py",
    "md_v4_model": ROOT / "tools/research/md_v4_model.py",
    "md_v4_explicit_reference": (
        ROOT / "tools/research/md_v4_explicit_reference.py"
    ),
    "md_v4_runtime": ROOT / "tools/research/md_v4_runtime.py",
    "md_v4_trainer": ROOT / "tools/research/train_md_v4.py",
    "md_v4_numpy_evaluator": (
        ROOT / "tools/research/eval_md_v4_numpy_deployable.py"
    ),
    "deployed_parent_checkpoint": (
        ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
        "candidate-qu-v2a-checkpoint.pt"
    ),
    "deployed_parent_weights": (
        ROOT / "tools/checkpoints/md-v2-allthrough26/model/"
        "candidate-qu-v2a-weights.npz"
    ),
    "frozen_md_v3_archive": (
        ROOT / "submission-md-v2-card-v1-experimental-unsigned.tar.gz"
    ),
    "card_weights": ROOT / "agent/md_v2_card_weights.npz",
    "qu_v2b_weights": ROOT / "agent/weights.npz",
    "md_v1_weights": ROOT / "agent/md_v1_weights.npz",
    "ppo_v2_weights": (
        ROOT / "tools/checkpoints/md-v3-ppo-v2/training/"
        "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
    ),
    "ppo_v2_result": (
        ROOT / "tools/checkpoints/md-v3-ppo-v2/direct-gameplay-result.json"
    ),
    "field_snapshot": (
        ROOT / "tools/checkpoints/md-v3-mirror-main-v1/"
        "field-july27-28.json"
    ),
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
    "agent_model": ROOT / "agent/model.py",
    "agent_features": ROOT / "agent/qu_v2_features.py",
    "agent_legacy_features": ROOT / "agent/features.py",
    "agent_obsview": ROOT / "agent/obsview.py",
    "agent_cards": ROOT / "agent/cards.py",
    "agent_policy": ROOT / "agent/policy.py",
    "agent_safety": ROOT / "agent/safety.py",
    "agent_card_router": ROOT / "agent/md_v2_card.py",
    "base_torch_model": ROOT / "tools/research/qu_v2a_model.py",
    "base_public_features": ROOT / "tools/research/qu_v2a_features.py",
    "ppo_v1_helpers": ROOT / "tools/research/train_md_v3_ppo.py",
    "ppo_v2_helpers": ROOT / "tools/research/train_md_v3_ppo_v2.py",
    "population_helpers": ROOT / "tools/research/md_v3_ppo_v2_population.py",
    "layered_controller": (
        ROOT / "tools/research/eval_md_v2_card_v1_gameplay.py"
    ),
    "scaled_gameplay_helpers": (
        ROOT / "tools/research/eval_md_v2_scaled_gameplay.py"
    ),
    "eval_ab": ROOT / "tools/eval_ab.py",
    "corpus_indexer": ROOT / "tools/index_corpus.py",
    "imitation_loader": ROOT / "tools/il_dataset.py",
    "rl_env": ROOT / "tools/rl_env.py",
    "battle_bindings": ROOT / "tools/cabt.py",
    "battle_engine": ROOT / "engine/libcg.so",
    "cards_data": ROOT / "data/cards.json",
    "attacks_data": ROOT / "data/attacks.json",
    "july29_archive": Path("/home/horn/Desktop/ptcg_official_2026-07-29.zip"),
}
REQUIRED_ARTIFACTS = frozenset(DEFAULT_ARTIFACT_PATHS)


class ResourcePPOLockError(RuntimeError):
    """The prospective resource-PPO contract is absent or has drifted."""


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
        raise ResourcePPOLockError("artifact path is missing")
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _record_artifact(path: Path) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ResourcePPOLockError(f"missing artifact {path}: {error}") from error
    if not resolved.is_file():
        raise ResourcePPOLockError(f"artifact is not a regular file: {resolved}")
    return {
        "path": _display_path(resolved),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ResourcePPOLockError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ResourcePPOLockError(f"{path} is not a JSON object")
    return payload


def _read_deck(path: Path) -> tuple[int, ...]:
    try:
        deck = tuple(
            int(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValueError) as error:
        raise ResourcePPOLockError(f"cannot read deck: {error}") from error
    if deck != TARGET_DECK:
        raise ResourcePPOLockError("deck is not the exact Grimmsnarl registration")
    return deck


def _read_field(path: Path) -> dict[str, Any]:
    payload = _strict_json(path)
    if (
        payload.get("schema") != "ptcg.recent-frequency-weighted-field.v1"
        or payload.get("source", {}).get("dates")
        != ["2026-07-27", "2026-07-28"]
    ):
        raise ResourcePPOLockError("field is not the fixed July 27--28 snapshot")
    rows = payload.get("field")
    if not isinstance(rows, list) or not rows:
        raise ResourcePPOLockError("field snapshot is empty")
    total = 0.0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ResourcePPOLockError("field row is not an object")
        deck = row.get("deck")
        weight = row.get("field_weight")
        if (
            not isinstance(row.get("archetype"), str)
            or not isinstance(deck, list)
            or len(deck) != 60
            or any(
                isinstance(card, bool) or not isinstance(card, int)
                for card in deck
            )
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise ResourcePPOLockError("field row is malformed")
        total += float(weight)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ResourcePPOLockError("field weights do not sum to one")
    return payload


def _deck_sha256(deck: Sequence[int]) -> str:
    return canonical_sha256([int(card) for card in deck])


def population_contract(field_path: Path) -> dict[str, Any]:
    """Build the frozen population manifest without creating controllers."""
    payload = _read_field(field_path)
    nonmirror = [
        dict(row)
        for row in payload["field"]
        if row["archetype"] != "Grimmsnarl"
    ]
    if not nonmirror:
        raise ResourcePPOLockError("field has no non-Grimmsnarl decks")
    denominator = sum(float(row["field_weight"]) for row in nonmirror)
    opponents: list[dict[str, Any]] = []
    mirror_specs = (
        ("mirror_md_v3", "grimmsnarl/md-v3", "complete-frozen-md-v3"),
        (
            "mirror_md_v4_epoch4",
            "grimmsnarl/md-v4-epoch4",
            "frozen-md-v4-epoch4",
        ),
        ("mirror_ppo_v2", "grimmsnarl/ppo-v2", "retired-ppo-v2-terminal"),
        ("mirror_md_v1", "grimmsnarl/md-v1", "frozen-md-v1"),
        ("mirror_qu_v2b", "grimmsnarl/qu-v2b", "frozen-qu-v2b"),
    )
    for group, key, policy_id in mirror_specs:
        opponents.append({
            "key": key,
            "policy_id": policy_id,
            "schedule_group": group,
            "weight": PILOT_MASS[group],
            "deck": list(TARGET_DECK),
            # This is the runtime manifest hash of the 60-card value, not the
            # byte hash of decks/md_v1_grimmsnarl.csv bound above.
            "deck_sha256": _deck_sha256(TARGET_DECK),
        })
    for row in nonmirror:
        conditional = float(row["field_weight"]) / denominator
        deck = [int(card) for card in row["deck"]]
        archetype = str(row["archetype"])
        for group, suffix, policy_id in (
            ("field_qu_v2b", "qu-v2b", f"frozen-qu-v2b/{archetype}"),
            ("field_rules", "rules", "rules-v1"),
        ):
            opponents.append({
                "key": f"{archetype}/{suffix}",
                "policy_id": policy_id,
                "schedule_group": group,
                "weight": PILOT_MASS[group] * conditional,
                "deck": deck,
                "deck_sha256": _deck_sha256(deck),
            })
    if not math.isclose(
        sum(float(row["weight"]) for row in opponents),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ResourcePPOLockError("population mass does not sum to one")
    manifest = {
        "schema": POPULATION_SCHEMA,
        "pilot_mass": dict(PILOT_MASS),
        "field_snapshot": {
            "schema": payload["schema"],
            "dates": payload["source"]["dates"],
            "sha256": file_sha256(field_path),
        },
        "opponents": opponents,
    }
    return {
        "schema": POPULATION_SCHEMA,
        "pilot_mass": dict(PILOT_MASS),
        "family_pair_mass": {"mirror": 0.5, "field": 0.5},
        "manifest": manifest,
        "manifest_sha256": canonical_sha256(manifest),
    }


def _noop_move(_observation: dict, _rng: Any) -> list[int]:
    return [0]


def opponent_specs(population: Mapping[str, Any]) -> list[OpponentSpec]:
    manifest = population.get("manifest")
    rows = manifest.get("opponents") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ResourcePPOLockError("population has no opponent rows")
    return [
        OpponentSpec(
            key=str(row["key"]),
            deck=tuple(int(card) for card in row["deck"]),
            move=_noop_move,
            weight=float(row["weight"]),
            policy_id=str(row["policy_id"]),
            schedule_group=str(row["schedule_group"]),
        )
        for row in rows
    ]


def schedule_contract(
    opponents: Sequence[OpponentSpec],
    *,
    update: int,
    rollout_seed: int,
    ppo_seed: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Build one exact 50/50-family paired schedule and its digest."""
    collapsed = []
    for opponent in opponents:
        group = str(opponent.schedule_group)
        family = (
            "mirror" if group.startswith("mirror_")
            else "field" if group.startswith("field_")
            else None
        )
        if family is None:
            raise ResourcePPOLockError(f"unclassified group {group!r}")
        collapsed.append(OpponentSpec(
            key=opponent.key,
            deck=opponent.deck,
            move=opponent.move,
            weight=opponent.weight,
            policy_id=opponent.policy_id,
            schedule_group=family,
        ))
    schedule = build_paired_schedule(
        collapsed,
        GAMES_PER_UPDATE,
        seed=int(rollout_seed),
    )
    representatives: dict[int, Any] = {}
    for episode in schedule:
        representatives.setdefault(int(episode.pair_id), episode)
    seats = Counter(str(int(row.learner_seat)) for row in schedule)
    families: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    members: Counter[str] = Counter()
    for episode in representatives.values():
        opponent = opponents[int(episode.opponent_index)]
        group = str(opponent.schedule_group)
        family = "mirror" if group.startswith("mirror_") else "field"
        families[family] += 1
        groups[group] += 1
        members[str(opponent.key)] += 1
    row = {
        "update": int(update),
        "rollout_seed": int(rollout_seed),
        "ppo_seed": int(ppo_seed),
        "games": len(schedule),
        "pairs": len(representatives),
        "seat_counts": dict(sorted(seats.items())),
        "family_pair_counts": dict(sorted(families.items())),
        "schedule_group_pair_counts": dict(sorted(groups.items())),
        "opponent_pair_counts": dict(sorted(members.items())),
        "manifest_sha256": canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
    }
    return schedule, row


def all_schedule_contracts(
    population: Mapping[str, Any],
) -> list[dict[str, Any]]:
    opponents = opponent_specs(population)
    return [
        schedule_contract(
            opponents,
            update=index + 1,
            rollout_seed=ROLLOUT_SEEDS[index],
            ppo_seed=PPO_SEEDS[index],
        )[1]
        for index in range(UPDATES)
    ]


def direct_pair_seeds(
    seed: int = DIRECT_PAIR_SEED,
    pairs: int = DIRECT_PAIRS,
) -> list[int]:
    values = []
    for index in range(pairs):
        raw = hashlib.sha256(
            b"ptcg.md-v4.resource-ppo.direct.v1\0"
            + int(seed).to_bytes(8, "big", signed=False)
            + int(index).to_bytes(8, "big", signed=False)
        ).digest()
        values.append(int.from_bytes(raw[:8], "big", signed=False))
    if len(set(values)) != pairs:
        raise ResourcePPOLockError("direct pair seeds collided")
    return values


def correction_provenance(
    correction_lock_path: Path,
    correction_result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _strict_json(correction_lock_path)
    result = _strict_json(correction_result_path)
    final = result.get("final_validation")
    if (
        lock.get("schema") != "ptcg.md-v4.deployed-parent-correction-lock.v1"
        or lock.get("lock_sha256") != CORRECTION_LOCK_SHA256
        or result.get("schema")
        != "ptcg.md-v4.deployed-parent-correction-result.v1"
        or result.get("result_sha256") != CORRECTION_RESULT_SHA256
        or result.get("passed") is not False
        or not isinstance(final, Mapping)
        or final.get("samples") != EXPECTED_VALIDATION_CALLBACKS
        or final.get("games") != EXPECTED_VALIDATION_GAMES
        or final.get("greedy_disagreements") != 2_376
        or final.get("games_touched") != 1_330
        or result.get("cached_parent_logits_used") is not False
        or result.get("temporal_archive_opened") is not False
    ):
        raise ResourcePPOLockError("correction failure provenance drifted")
    provenance = {
        "schema": result["schema"],
        "lock_sha256": lock["lock_sha256"],
        "result_sha256": result["result_sha256"],
        "passed": False,
        "callbacks": final["samples"],
        "games": final["games"],
        "deployed_parent_kl": final["parent_kl"],
        "deployed_parent_disagreements": final["greedy_disagreements"],
        "deployed_parent_disagreement_rate":
            final["greedy_disagreement_rate"],
        "games_touched": final["games_touched"],
        "games_touched_rate": final["games_touched_rate"],
        "cached_parent_logits_used": False,
        "role": "immutable_failed_warm_start_provenance_only",
    }
    temporal = dict(lock.get("temporal_seal", {}))
    return provenance, temporal


def _validate_correction(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema")
        != "ptcg.md-v4.deployed-parent-correction-result.v1"
        or value.get("lock_sha256") != CORRECTION_LOCK_SHA256
        or value.get("result_sha256") != CORRECTION_RESULT_SHA256
        or value.get("passed") is not False
        or value.get("callbacks") != EXPECTED_VALIDATION_CALLBACKS
        or value.get("games") != EXPECTED_VALIDATION_GAMES
        or value.get("deployed_parent_disagreements") != 2_376
        or value.get("games_touched") != 1_330
        or value.get("cached_parent_logits_used") is not False
    ):
        raise ResourcePPOLockError("correction provenance is not exact")


def _validate_temporal(value: Mapping[str, Any]) -> None:
    archive = value.get("archive")
    if (
        not isinstance(archive, Mapping)
        or archive.get("sha256") != JULY29_SHA256
        or archive.get("bytes") != JULY29_BYTES
        or value.get("entries") != JULY29_ENTRIES
        or value.get("json_entries") != JULY29_JSON_ENTRIES
        or value.get("uncompressed_bytes") != JULY29_UNCOMPRESSED_BYTES
        or value.get("central_directory_inventory_sha256")
        != JULY29_INVENTORY_SHA256
        or value.get("replay_content_opened") is not False
    ):
        raise ResourcePPOLockError("July 29 temporal seal drifted")


def _validate_population(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema") != POPULATION_SCHEMA
        or value.get("pilot_mass") != PILOT_MASS
        or value.get("family_pair_mass") != {"mirror": 0.5, "field": 0.5}
        or not _is_sha256(value.get("manifest_sha256"))
        or canonical_sha256(value.get("manifest"))
        != value.get("manifest_sha256")
    ):
        raise ResourcePPOLockError("population contract drifted")
    opponents = opponent_specs(value)
    if not math.isclose(
        sum(row.weight for row in opponents),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ResourcePPOLockError("population weights drifted")


def _validate_schedules(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != UPDATES:
        raise ResourcePPOLockError("schedule count drifted")
    for index, row in enumerate(rows):
        if (
            row.get("update") != index + 1
            or row.get("rollout_seed") != ROLLOUT_SEEDS[index]
            or row.get("ppo_seed") != PPO_SEEDS[index]
            or row.get("games") != GAMES_PER_UPDATE
            or row.get("pairs") != PAIRS_PER_UPDATE
            or row.get("seat_counts") != SEATS_PER_UPDATE
            or row.get("family_pair_counts") != FAMILY_PAIRS_PER_UPDATE
            or sum(row.get("schedule_group_pair_counts", {}).values())
            != PAIRS_PER_UPDATE
            or sum(row.get("opponent_pair_counts", {}).values())
            != PAIRS_PER_UPDATE
            or not _is_sha256(row.get("manifest_sha256"))
        ):
            raise ResourcePPOLockError(
                f"schedule contract drifted at update {index + 1}"
            )


def _training_contract() -> dict[str, Any]:
    return {
        "device": "cuda-explicit-fp32-tf32-disabled",
        "updates": UPDATES,
        "games_per_update": GAMES_PER_UPDATE,
        "total_games": TOTAL_GAMES,
        "pairs_per_update": PAIRS_PER_UPDATE,
        "seat_balance_per_update": dict(SEATS_PER_UPDATE),
        "family_pair_counts_per_update": dict(FAMILY_PAIRS_PER_UPDATE),
        "rollout_seed_base": ROLLOUT_SEED_BASE,
        "seed_stride": SEED_STRIDE,
        "rollout_seeds": list(ROLLOUT_SEEDS),
        "ppo_seeds": list(PPO_SEEDS),
        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "ppo_epochs": PPO_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "clip": PPO_CLIP,
        "value_coefficient": VALUE_COEFFICIENT,
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "deployed_parent_kl_coefficient": PARENT_KL_COEFFICIENT,
        "gradient_norm": GRADIENT_NORM,
        "minimum_st_main_decisions_per_update": MIN_ST_MAIN_PER_UPDATE,
        "maximum_deployed_parent_kl_per_update": MAX_PARENT_KL,
        "maximum_final_deployed_parent_kl": MAX_PARENT_KL,
        "actor_modules": list(ACTOR_MODULES),
        "critic": {
            "hidden": CRITIC_HIDDEN,
            "input": "detached stored float32[192]: MD-v3 state[160] plus MD-v4 fusion[32]",
            "architecture": [
                "Linear(192,64,bias=True)",
                "ReLU",
                "Linear(64,1,bias=True)",
                "tanh",
            ],
            "initialization": {
                "hidden_weight": "orthogonal gain sqrt(2)",
                "hidden_bias": 0.0,
                "output_weight": "orthogonal gain 1",
                "output_bias": 0.0,
            },
            "recomputed_after_actor_update": False,
            "deployment_artifact_member": False,
            "gradient_into_actor": False,
        },
        "reward": "terminal win=+1 draw=0 loss=-1",
        "return_estimator": "episode-local variable-duration semi-MDP GAE",
        "persistent_optimizer_across_updates_and_resume": True,
        "cached_parent_logits_authority": False,
        "parent_anchor":
            "exact deployed NumPy agent.model.QuV2Net/76420fc2",
        "rollout_actor":
            "tools.research.md_v4_explicit_reference.TorchMDV4ExplicitFP32",
        "cuda_matmul_tf32": False,
        "single_attempt": {
            "count": 1,
            "official_output": _display_path(OFFICIAL_TRAINING_OUTPUT),
            "consume_before_first_rollout": True,
            "arbitrary_output_directories": False,
            "controlled_failure_retires_attempt": True,
            "resume_only_in_same_output_tree": True,
            "each_recovery_checkpoint_consumed_at_most_once": True,
        },
    }


def build_lock(
    *,
    artifact_paths: Mapping[str, Path],
    created_at: str,
    population: Mapping[str, Any],
    schedules: Sequence[Mapping[str, Any]],
    correction: Mapping[str, Any],
    temporal_seal: Mapping[str, Any],
) -> dict[str, Any]:
    if set(artifact_paths) != REQUIRED_ARTIFACTS:
        raise ResourcePPOLockError("artifact path set mismatch")
    _validate_population(population)
    _validate_schedules(schedules)
    _validate_correction(correction)
    _validate_temporal(temporal_seal)
    records = {
        name: _record_artifact(path)
        for name, path in sorted(artifact_paths.items())
    }
    if records["warm_checkpoint"]["sha256"] != WARM_CHECKPOINT_SHA256:
        raise ResourcePPOLockError("warm checkpoint bytes drifted")
    if (
        records["deployed_parent_weights"]["sha256"]
        != DEPLOYED_PARENT_WEIGHTS_SHA256
    ):
        raise ResourcePPOLockError("deployed NumPy parent bytes drifted")
    if records["july29_archive"]["sha256"] != JULY29_SHA256:
        raise ResourcePPOLockError("July 29 archive bytes drifted")
    deck = _read_deck(Path(artifact_paths["deck"]))
    direct_seeds = direct_pair_seeds()
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "prospective": True,
        "created_at_utc": str(created_at),
        "candidate": {
            "name": CANDIDATE,
            "internal_identifier_only": True,
            "warm_checkpoint_sha256": WARM_CHECKPOINT_SHA256,
            "warm_state_dict_sha256": WARM_STATE_SHA256,
            "warm_numpy_mapping_sha256": WARM_NUMPY_MAPPING_SHA256,
            "frozen_embedded_parent_state_sha256":
                FROZEN_PARENT_STATE_SHA256,
            "eligible_update": UPDATES,
            "intermediate_updates_recovery_only": True,
            "checkpoint_or_seed_selection": False,
            "changed_runtime_component":
                "exact-deck ST_MAIN MD-v4 resource/log actor only",
            "training_only_critic_exported": False,
        },
        "target_deck": {
            "cards": len(deck),
            "sha256": TARGET_DECK_SHA256,
            "exact": True,
        },
        "deployed_parent": {
            "runtime_class": "agent.model.QuV2Net",
            "decoder": "agent.model.decode_qu_v2",
            "weights_sha256": DEPLOYED_PARENT_WEIGHTS_SHA256,
            "cached_torch_parent_logits_authority": False,
        },
        "prior_correction_failure": dict(correction),
        "training": _training_contract(),
        "population": dict(population),
        "schedules": {"updates": [dict(row) for row in schedules]},
        "terminal_offline_rejection": {
            "population": {
                "callbacks": EXPECTED_VALIDATION_CALLBACKS,
                "games": EXPECTED_VALIDATION_GAMES,
                "role": "reused development/integrity/behavior-sizing only",
                "promotion_evidence": False,
            },
            "full_parent_runtime_conformance": {
                "candidate_embedded_parent_vs_deployed_parent_logit_bits": 0,
                "candidate_embedded_parent_vs_deployed_parent_value_bits": 0,
                "candidate_direct_vs_staged_logit_bit_mismatches": 0,
                "candidate_direct_vs_staged_value_bit_mismatches": 0,
                "research_vs_production_decoder_mismatches_each_arm": 0,
                "feature_conversion_inference_nonfinite_decode_metric_faults": 0,
                "cached_parent_logits_reads": 0,
            },
            "maximum_deployed_parent_kl_inclusive": MAX_PARENT_KL,
            "minimum_deployed_parent_disagreements_inclusive":
                MIN_PARENT_DISAGREEMENTS,
            "minimum_deployed_parent_games_touched_inclusive":
                MIN_PARENT_GAMES_TOUCHED,
            "minimum_warm_start_disagreements_inclusive":
                MIN_INITIAL_DISAGREEMENTS,
            "minimum_warm_start_games_touched_inclusive":
                MIN_INITIAL_GAMES_TOUCHED,
            "all_required": True,
        },
        "evaluation": {
            "sanity_games": 20,
            "direct_exact_mirror": {
                "games": DIRECT_GAMES,
                "pairs": DIRECT_PAIRS,
                "candidate_each_physical_seat": DIRECT_PAIRS,
                "pair_seed": DIRECT_PAIR_SEED,
                "pair_seeds": direct_seeds,
                "pair_seed_sha256": canonical_sha256(direct_seeds),
                "score": "(wins + 0.5 * draws) / 2560",
                "pass":
                    "zero faults and Wilson CI95 lower bound strictly above 0.50",
            },
            "recent_frequency_field": {
                "requires_direct_pass": True,
                "games_per_arm": 1_280,
                "point_delta_minimum_inclusive": 0.0,
                "conservative_wilson_difference_lower_strictly_above": -0.05,
                "separate_fresh_lock_required": True,
            },
            "untouched_temporal": {
                "requires_direct_and_field_pass": True,
                "date": "2026-07-29",
                "candidate_fixed_before_open": True,
                "rejection_only": True,
                "maximum_parent_kl_inclusive": 0.02,
                "candidate_nll_minus_parent_maximum_inclusive": 0.010,
                "mirror_winning_seat_candidate_nll_strictly_below_parent": True,
            },
        },
        "temporal_seal": dict(temporal_seal),
        "artifacts": records,
        "authorization": {
            "research_only": True,
            "production_mutation": False,
            "package_build": False,
            "tag": False,
            "upload": False,
            "ship_name_user_controlled": True,
        },
    }
    body["lock_sha256"] = canonical_sha256(body)
    _validate_payload(body)
    return body


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("prospective") is not True:
        raise ResourcePPOLockError("lock schema/prospective marker drifted")
    candidate = payload.get("candidate")
    training = payload.get("training")
    offline = payload.get("terminal_offline_rejection")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("warm_state_dict_sha256") != WARM_STATE_SHA256
        or candidate.get("eligible_update") != UPDATES
        or candidate.get("checkpoint_or_seed_selection") is not False
        or not isinstance(training, Mapping)
        or dict(training) != _training_contract()
        or not isinstance(offline, Mapping)
        or offline.get("minimum_deployed_parent_disagreements_inclusive")
        != MIN_PARENT_DISAGREEMENTS
        or offline.get("minimum_deployed_parent_games_touched_inclusive")
        != MIN_PARENT_GAMES_TOUCHED
        or offline.get("minimum_warm_start_disagreements_inclusive")
        != MIN_INITIAL_DISAGREEMENTS
        or offline.get("minimum_warm_start_games_touched_inclusive")
        != MIN_INITIAL_GAMES_TOUCHED
    ):
        raise ResourcePPOLockError("candidate/training/offline contract drifted")
    _validate_population(payload.get("population", {}))
    _validate_schedules(payload.get("schedules", {}).get("updates", []))
    _validate_correction(payload.get("prior_correction_failure", {}))
    _validate_temporal(payload.get("temporal_seal", {}))
    if set(payload.get("artifacts", {})) != REQUIRED_ARTIFACTS:
        raise ResourcePPOLockError("bound artifact set drifted")
    direct = payload.get("evaluation", {}).get("direct_exact_mirror", {})
    if (
        direct.get("games") != DIRECT_GAMES
        or direct.get("pairs") != DIRECT_PAIRS
        or direct.get("pair_seeds") != direct_pair_seeds()
        or direct.get("pair_seed_sha256")
        != canonical_sha256(direct_pair_seeds())
    ):
        raise ResourcePPOLockError("direct gameplay preregistration drifted")


def write_lock(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise ResourcePPOLockError(f"refusing to overwrite {destination}")
    _validate_payload(payload)
    without_hash = {
        key: value for key, value in payload.items() if key != "lock_sha256"
    }
    if payload.get("lock_sha256") != canonical_sha256(without_hash):
        raise ResourcePPOLockError("lock self-hash is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ).encode("utf-8") + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ResourcePPOLockError(
                f"refusing to overwrite {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def verify_bound_artifacts(payload: Mapping[str, Any]) -> dict[str, Path]:
    records = payload.get("artifacts")
    if not isinstance(records, Mapping) or set(records) != REQUIRED_ARTIFACTS:
        raise ResourcePPOLockError("lock artifact map is incomplete")
    result = {}
    for name, record in records.items():
        if not isinstance(record, Mapping):
            raise ResourcePPOLockError(f"artifact {name} record is malformed")
        path = resolve_recorded_path(record.get("path"))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or file_sha256(path) != record.get("sha256")
        ):
            raise ResourcePPOLockError(f"artifact {name} drifted")
        result[str(name)] = path
    return result


def load_lock(
    path: Path,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    payload = _strict_json(path.expanduser().resolve())
    recorded = payload.get("lock_sha256")
    body = {key: value for key, value in payload.items() if key != "lock_sha256"}
    if not _is_sha256(recorded) or recorded != canonical_sha256(body):
        raise ResourcePPOLockError("lock self-hash is invalid")
    _validate_payload(payload)
    if verify_artifacts:
        verify_bound_artifacts(payload)
    return payload


def build_default_lock(created_at: str | None = None) -> dict[str, Any]:
    population = population_contract(DEFAULT_ARTIFACT_PATHS["field_snapshot"])
    schedules = all_schedule_contracts(population)
    correction, temporal = correction_provenance(
        DEFAULT_ARTIFACT_PATHS["correction_lock"],
        DEFAULT_ARTIFACT_PATHS["correction_result"],
    )
    return build_lock(
        artifact_paths=DEFAULT_ARTIFACT_PATHS,
        created_at=(
            datetime.now(timezone.utc).isoformat()
            if created_at is None else created_at
        ),
        population=population,
        schedules=schedules,
        correction=correction,
        temporal_seal=temporal,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output != DEFAULT_OUTPUT.resolve():
        parser.error(f"official lock path is fixed to {DEFAULT_OUTPUT}")
    payload = build_default_lock()
    write_lock(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
