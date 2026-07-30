"""Focused tests for the prospective MD-v3 PPO-v2 experiment lock."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_md_v3_ppo_v2 as LOCK  # noqa: E402


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for index, name in enumerate(sorted(LOCK.REQUIRED_ARTIFACTS)):
        path = tmp_path / "artifacts" / f"{index:02d}-{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{name}:bound\n".encode())
        paths[name] = path
    paths["deck"].write_text(
        "".join(f"{card}\n" for card in LOCK.TARGET_DECK),
        encoding="utf-8",
    )
    paths["field_snapshot"].write_text(
        json.dumps({
            "schema": "ptcg.recent-frequency-weighted-field.v1",
            "source": {"dates": ["2026-07-27", "2026-07-28"]},
            "field": [{
                "archetype": "Grimmsnarl",
                "deck": list(LOCK.TARGET_DECK),
                "field_weight": 1.0,
            }],
        }),
        encoding="utf-8",
    )
    return paths


def _population() -> dict:
    manifest = {
        "schema": LOCK.POPULATION_SCHEMA,
        "pilot_mass": dict(LOCK.PILOT_MASS),
        "opponents": [{"key": "synthetic/test"}],
    }
    return {
        "schema": LOCK.POPULATION_SCHEMA,
        "pilot_mass": dict(LOCK.PILOT_MASS),
        "manifest": manifest,
        "manifest_sha256": LOCK.canonical_sha256(manifest),
        "family_pair_mass": {"mirror": 0.5, "field": 0.5},
    }


def _update_schedules() -> list[dict]:
    groups = {
        "mirror_md_v3": 96,
        "mirror_ppo_v1": 38,
        "mirror_md_v1": 39,
        "mirror_qu_v2b": 19,
        "field_qu_v2b": 154,
        "field_rules": 38,
    }
    return [
        {
            "update": index + 1,
            "rollout_seed": LOCK.ROLLOUT_SEEDS[index],
            "ppo_seed": LOCK.PPO_SEEDS[index],
            "games": LOCK.GAMES_PER_UPDATE,
            "pairs": LOCK.PAIRS_PER_UPDATE,
            "seat_counts": dict(LOCK.SEATS_PER_UPDATE),
            "family_pair_counts": {"field": 192, "mirror": 192},
            "schedule_group_pair_counts": dict(groups),
            "opponent_pair_counts": {"synthetic/test": 384},
            "manifest_sha256": f"{index + 1:064x}",
        }
        for index in range(LOCK.UPDATES)
    ]


def _direct_schedule() -> dict:
    return {
        "generator": "tools.rl_env.build_paired_schedule",
        "manifest": "tools.rl_env.schedule_manifest",
        "manifest_sha256": "f" * 64,
        "games": 2560,
        "pairs": 1280,
        "candidate_seat_counts": {"0": 1280, "1": 1280},
        "opponent": {
            "key": "grimmsnarl/frozen-md-v3",
            "policy_id": "complete-frozen-md-v3",
            "schedule_group": "frozen-md-v3",
            "deck_sha256": LOCK.canonical_sha256(list(LOCK.TARGET_DECK)),
        },
    }


def _build(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    artifacts = _artifacts(tmp_path)
    payload = LOCK.build_lock(
        artifact_paths=artifacts,
        created_at="2026-07-30T12:00:00+00:00",
        population_contract=_population(),
        update_schedules=_update_schedules(),
        direct_schedule=_direct_schedule(),
    )
    return payload, artifacts


def test_contract_is_deterministic_self_hashed_and_fully_preregistered(tmp_path):
    first, _ = _build(tmp_path / "one")
    second, _ = _build(tmp_path / "two")

    first_without_hash = dict(first)
    recorded = first_without_hash.pop("lock_sha256")
    assert recorded == LOCK.canonical_sha256(first_without_hash)
    assert first["training"] == second["training"]
    assert first["population"] == second["population"]
    assert first["schedules"] == second["schedules"]
    assert first["training"]["updates"] == 16
    assert first["training"]["games_per_update"] == 768
    assert first["training"]["total_games"] == 12_288
    assert first["training"]["rollout_seeds"] == list(LOCK.ROLLOUT_SEEDS)
    assert first["training"]["ppo_seeds"] == list(LOCK.PPO_SEEDS)
    assert first["candidate_selection"]["eligible_update"] == 16
    assert first["candidate_selection"]["posthoc_checkpoint_selection"] is False
    assert first["direct_gameplay"]["protocol"]["games"] == 2560
    assert first["direct_gameplay"]["protocol"]["seed"] == 2026073102
    assert first["direct_gameplay"]["protocol"]["pass"] == (
        "Wilson CI95 lower bound strictly greater than 0.50"
    )
    assert first["authorization"]["ladder_upload"] is False
    assert set(first["artifacts"]) == LOCK.REQUIRED_ARTIFACTS


def test_load_rejects_self_hash_contract_and_artifact_tampering(tmp_path):
    payload, artifacts = _build(tmp_path)
    lock_path = tmp_path / "lock.json"
    LOCK.write_lock(lock_path, payload)
    assert LOCK.load_lock(lock_path)["lock_sha256"] == payload["lock_sha256"]

    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    raw["training"]["gamma"] = 0.5
    lock_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(LOCK.PPOV2LockError, match="self-hash"):
        LOCK.load_lock(lock_path)

    raw["lock_sha256"] = LOCK.canonical_sha256({
        key: value for key, value in raw.items() if key != "lock_sha256"
    })
    lock_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(LOCK.PPOV2LockError, match="configuration drifted"):
        LOCK.load_lock(lock_path, verify_artifacts=False)

    LOCK.write_lock(tmp_path / "clean.json", payload)
    artifacts["qu_weights"].write_bytes(b"changed-but-still-a-file\n")
    with pytest.raises(LOCK.PPOV2LockError, match="artifact .* drifted"):
        LOCK.load_lock(tmp_path / "clean.json")


def test_schedule_population_and_path_drift_fail_closed(tmp_path):
    artifacts = _artifacts(tmp_path)
    schedules = _update_schedules()
    schedules[3]["family_pair_counts"] = {"field": 191, "mirror": 193}
    with pytest.raises(LOCK.PPOV2LockError, match="family_pair_counts"):
        LOCK.build_lock(
            artifact_paths=artifacts,
            created_at="2026-07-30T12:00:00+00:00",
            population_contract=_population(),
            update_schedules=schedules,
            direct_schedule=_direct_schedule(),
        )

    bad_population = _population()
    bad_population["pilot_mass"]["mirror_md_v3"] = 0.24
    with pytest.raises(LOCK.PPOV2LockError, match="population"):
        LOCK.build_lock(
            artifact_paths=artifacts,
            created_at="2026-07-30T12:00:00+00:00",
            population_contract=bad_population,
            update_schedules=_update_schedules(),
            direct_schedule=_direct_schedule(),
        )

    missing = dict(artifacts)
    missing.pop("runner")
    with pytest.raises(LOCK.PPOV2LockError, match="artifact path set mismatch"):
        LOCK.build_lock(
            artifact_paths=missing,
            created_at="2026-07-30T12:00:00+00:00",
            population_contract=_population(),
            update_schedules=_update_schedules(),
            direct_schedule=_direct_schedule(),
        )


def test_write_is_immutable_and_direct_protocol_is_exact(tmp_path):
    payload, _ = _build(tmp_path)
    destination = tmp_path / "sealed.json"
    LOCK.write_lock(destination, payload)
    with pytest.raises(LOCK.PPOV2LockError, match="overwrite"):
        LOCK.write_lock(destination, payload)

    changed = copy.deepcopy(payload)
    changed["direct_gameplay"]["protocol"]["games"] = 2558
    changed_without_hash = dict(changed)
    changed_without_hash.pop("lock_sha256")
    changed["lock_sha256"] = LOCK.canonical_sha256(changed_without_hash)
    with pytest.raises(LOCK.PPOV2LockError, match="direct gameplay protocol"):
        LOCK.write_lock(tmp_path / "bad.json", changed)
