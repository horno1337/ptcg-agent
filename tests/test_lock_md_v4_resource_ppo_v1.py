"""Focused tests for the prospective MD-v4 resource-PPO v1 lock."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.research import lock_md_v4_resource_ppo_v1 as LOCK


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for index, name in enumerate(sorted(LOCK.REQUIRED_ARTIFACTS)):
        path = tmp_path / "artifacts" / f"{index:02d}-{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{name}:prospectively-bound\n".encode())
        paths[name] = path
    paths["deck"].write_text(
        "".join(f"{card}\n" for card in LOCK.TARGET_DECK),
        encoding="utf-8",
    )
    paths["field_snapshot"].write_text(
        json.dumps({
            "schema": "ptcg.recent-frequency-weighted-field.v1",
            "source": {"dates": ["2026-07-27", "2026-07-28"]},
            "field": [
                {
                    "archetype": "Grimmsnarl",
                    "deck": list(LOCK.TARGET_DECK),
                    "field_weight": 0.51,
                },
                {
                    "archetype": "Alakazam",
                    "deck": [7] * 60,
                    "field_weight": 0.49,
                },
            ],
        }),
        encoding="utf-8",
    )
    return paths


def _correction() -> dict:
    return {
        "schema": "ptcg.md-v4.deployed-parent-correction-result.v1",
        "lock_sha256": LOCK.CORRECTION_LOCK_SHA256,
        "result_sha256": LOCK.CORRECTION_RESULT_SHA256,
        "passed": False,
        "callbacks": LOCK.EXPECTED_VALIDATION_CALLBACKS,
        "games": LOCK.EXPECTED_VALIDATION_GAMES,
        "deployed_parent_kl": 0.002864179944542191,
        "deployed_parent_disagreements": 2_376,
        "deployed_parent_disagreement_rate": 0.0237728,
        "games_touched": 1_330,
        "games_touched_rate": 1_330 / 2_090,
        "cached_parent_logits_used": False,
        "role": "immutable_failed_warm_start_provenance_only",
    }


def _temporal() -> dict:
    return {
        "archive": {
            "sha256": LOCK.JULY29_SHA256,
            "bytes": LOCK.JULY29_BYTES,
        },
        "entries": LOCK.JULY29_ENTRIES,
        "json_entries": LOCK.JULY29_JSON_ENTRIES,
        "uncompressed_bytes": LOCK.JULY29_UNCOMPRESSED_BYTES,
        "central_directory_inventory_sha256":
            LOCK.JULY29_INVENTORY_SHA256,
        "replay_content_opened": False,
    }


def _build(tmp_path: Path, monkeypatch) -> tuple[dict, dict[str, Path]]:
    paths = _artifacts(tmp_path)
    monkeypatch.setattr(
        LOCK,
        "WARM_CHECKPOINT_SHA256",
        LOCK.file_sha256(paths["warm_checkpoint"]),
    )
    monkeypatch.setattr(
        LOCK,
        "DEPLOYED_PARENT_WEIGHTS_SHA256",
        LOCK.file_sha256(paths["deployed_parent_weights"]),
    )
    monkeypatch.setattr(
        LOCK,
        "JULY29_SHA256",
        LOCK.file_sha256(paths["july29_archive"]),
    )
    population = LOCK.population_contract(paths["field_snapshot"])
    payload = LOCK.build_lock(
        artifact_paths=paths,
        created_at="2026-07-30T12:00:00+00:00",
        population=population,
        schedules=LOCK.all_schedule_contracts(population),
        correction=_correction(),
        temporal_seal=_temporal(),
    )
    return payload, paths


def test_contract_fixes_training_terminal_and_authority(tmp_path, monkeypatch):
    payload, _ = _build(tmp_path, monkeypatch)
    body = dict(payload)
    recorded = body.pop("lock_sha256")
    assert recorded == LOCK.canonical_sha256(body)
    assert payload["training"]["updates"] == 24
    assert payload["training"]["games_per_update"] == 768
    assert payload["training"]["total_games"] == 18_432
    assert payload["training"]["actor_learning_rate"] == 2e-6
    assert payload["training"]["critic"]["gradient_into_actor"] is False
    assert payload["training"]["critic"]["deployment_artifact_member"] is False
    assert payload["candidate"]["eligible_update"] == 24
    assert payload["candidate"]["checkpoint_or_seed_selection"] is False
    assert payload["deployed_parent"]["runtime_class"] == "agent.model.QuV2Net"
    assert payload["deployed_parent"][
        "cached_torch_parent_logits_authority"
    ] is False
    assert payload["prior_correction_failure"][
        "deployed_parent_disagreements"
    ] == 2_376
    assert payload["authorization"]["upload"] is False
    assert payload["authorization"]["ship_name_user_controlled"] is True
    assert len(payload["schedules"]["updates"]) == 24
    assert set(payload["artifacts"]) == LOCK.REQUIRED_ARTIFACTS


def test_population_uses_runtime_deck_hash_and_exact_half_families(
    tmp_path,
    monkeypatch,
):
    payload, _ = _build(tmp_path, monkeypatch)
    rows = payload["population"]["manifest"]["opponents"]
    mirror = [row for row in rows if row["schedule_group"].startswith("mirror_")]
    assert len(mirror) == 5
    assert all(
        row["deck_sha256"] == LOCK._deck_sha256(LOCK.TARGET_DECK)
        for row in mirror
    )
    # The registration identity and runtime JSON-value identity are different
    # namespaces and must not be silently interchanged.
    assert LOCK._deck_sha256(LOCK.TARGET_DECK) != LOCK.TARGET_DECK_SHA256
    assert sum(row["weight"] for row in mirror) == pytest.approx(0.5)
    for schedule in payload["schedules"]["updates"]:
        assert schedule["seat_counts"] == {"0": 384, "1": 384}
        assert schedule["family_pair_counts"] == {
            "field": 192,
            "mirror": 192,
        }


def test_load_rejects_contract_and_artifact_tampering(tmp_path, monkeypatch):
    payload, paths = _build(tmp_path, monkeypatch)
    lock_path = tmp_path / "lock.json"
    LOCK.write_lock(lock_path, payload)
    assert LOCK.load_lock(lock_path)["lock_sha256"] == payload["lock_sha256"]

    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    raw["training"]["updates"] = 23
    lock_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(LOCK.ResourcePPOLockError, match="self-hash"):
        LOCK.load_lock(lock_path)

    clean_path = tmp_path / "clean.json"
    LOCK.write_lock(clean_path, payload)
    paths["runner"].write_bytes(b"runner drift\n")
    with pytest.raises(LOCK.ResourcePPOLockError, match="artifact runner drifted"):
        LOCK.load_lock(clean_path)


def test_lock_rejects_schedule_correction_and_path_drift(
    tmp_path,
    monkeypatch,
):
    payload, paths = _build(tmp_path, monkeypatch)
    population = payload["population"]
    schedules = copy.deepcopy(payload["schedules"]["updates"])
    schedules[3]["family_pair_counts"] = {"field": 191, "mirror": 193}
    with pytest.raises(LOCK.ResourcePPOLockError, match="schedule contract"):
        LOCK.build_lock(
            artifact_paths=paths,
            created_at="2026-07-30T12:00:00+00:00",
            population=population,
            schedules=schedules,
            correction=_correction(),
            temporal_seal=_temporal(),
        )

    correction = _correction()
    correction["deployed_parent_disagreements"] = 2_951
    with pytest.raises(LOCK.ResourcePPOLockError, match="correction provenance"):
        LOCK.build_lock(
            artifact_paths=paths,
            created_at="2026-07-30T12:00:00+00:00",
            population=population,
            schedules=payload["schedules"]["updates"],
            correction=correction,
            temporal_seal=_temporal(),
        )

    missing = dict(paths)
    missing.pop("runner")
    with pytest.raises(LOCK.ResourcePPOLockError, match="path set mismatch"):
        LOCK.build_lock(
            artifact_paths=missing,
            created_at="2026-07-30T12:00:00+00:00",
            population=population,
            schedules=payload["schedules"]["updates"],
            correction=_correction(),
            temporal_seal=_temporal(),
        )


def test_write_is_immutable(tmp_path, monkeypatch):
    payload, _ = _build(tmp_path, monkeypatch)
    destination = tmp_path / "sealed.json"
    LOCK.write_lock(destination, payload)
    with pytest.raises(LOCK.ResourcePPOLockError, match="overwrite"):
        LOCK.write_lock(destination, payload)
