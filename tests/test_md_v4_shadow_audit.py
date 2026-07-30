"""Synthetic tests for the locked MD-v4 shadow-audit population and gates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import index_corpus as INDEX
from tools.research import lock_md_v4_shadow_audit as LOCK
from tools.research import md_v4_features as FEATURES
from tools.research import run_md_v4_shadow_audit as AUDIT


def _player(*, actor: bool, hand_card: int | None = None) -> dict:
    hand = (
        [{"id": hand_card, "serial": 1, "playerIndex": 0}]
        if actor and hand_card is not None else ([] if actor else None)
    )
    visible = len(hand or ())
    return {
        "active": [],
        "bench": [],
        "benchMax": 5,
        "deckCount": 54 - visible,
        "discard": [],
        "hand": hand,
        "handCount": visible if actor else 0,
        "prize": [None] * 6,
        "asleep": False,
        "burned": False,
        "confused": False,
        "paralyzed": False,
        "poisoned": False,
    }


def _observation(seat: int, *, active: bool) -> dict:
    me = _player(actor=True, hand_card=7)
    opponent = _player(actor=False)
    players = [me, opponent] if seat == 0 else [opponent, me]
    # Physical labels are part of transport only; actor-relative features must
    # remain unchanged when the complete fixture is relabelled.
    for entry in me.get("hand") or ():
        entry["playerIndex"] = seat
    return {
        "current": {
            "yourIndex": seat,
            "players": players,
            "turn": 1,
            "turnActionCount": 0,
            "firstPlayer": 0,
            "looking": [],
            "stadium": [],
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
        },
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}],
            "deck": None,
            "contextCard": None,
            "effect": None,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
        },
        "logs": (
            [{"type": 4, "playerIndex": seat, "cardId": 7, "serial": 1}]
            if active else []
        ),
        "remainingOverageTime": 599.0,
        "search_begin_input": "transport-only",
    }


def _document(episode_id: int = 101) -> dict:
    deck = list(FEATURES.TARGET_DECK)
    return {
        "id": episode_id,
        "module_version": "1.32.2",
        "info": {
            "EpisodeId": episode_id,
            "TeamNames": ["alpha", "beta"],
            "Agents": [{"Name": "a"}, {"Name": "b"}],
        },
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [
                {
                    "action": deck,
                    "status": "ACTIVE",
                    "observation": _observation(0, active=True),
                },
                {
                    "action": deck,
                    "status": "INACTIVE",
                    # This copied/non-callback select must not enter the lock.
                    "observation": _observation(1, active=False),
                },
            ],
            [
                {
                    "action": [0],
                    "status": "INACTIVE",
                    # A second inactive copied prompt: still not a callback.
                    "observation": _observation(0, active=False),
                },
                {
                    "action": [],
                    "status": "ACTIVE",
                    "observation": _observation(1, active=True),
                },
            ],
            [
                {"action": [], "status": "DONE", "observation": {}},
                {"action": [0], "status": "DONE", "observation": {}},
            ],
        ],
    }


def _validation_manifest(root: Path) -> Path:
    replay_dir = root / "replays"
    replay_dir.mkdir()
    (replay_dir / "101.json").write_text(
        json.dumps(_document(), sort_keys=True), encoding="utf-8")
    manifest = INDEX.build_index(
        (INDEX.SourceSpec("synthetic", replay_dir.resolve()),),
        split_seed=19,
    )
    assert manifest["summary"]["valid_bc_games"] == 1
    manifest["games"][0]["split"] = "validation"
    for name in ("split_games", "split_valid_bc_games"):
        manifest["summary"][name] = {
            "train": 0, "validation": 1, "test": 0,
        }
    manifest["corpus_content_sha256"] = INDEX._corpus_content_hash(
        manifest["games"])
    manifest = INDEX.add_manifest_sha256(manifest)
    return INDEX.write_index(manifest, root / "corpus.json")


def _build_small_lock(root: Path) -> tuple[Path, dict]:
    manifest = _validation_manifest(root)
    payload = LOCK.build_lock(
        corpus_path=manifest,
        expected_games=1,
        expected_callbacks=2,
        metamorphic_samples=1,
        runtime_samples=1,
        enforce_production_corpus=False,
    )
    path = root / "shadow-audit-lock.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, payload


def test_lock_uses_only_strict_paired_callbacks_and_self_hashes():
    with tempfile.TemporaryDirectory() as temporary:
        lock_path, payload = _build_small_lock(Path(temporary))
        assert payload["cohort"]["games"] == 1
        assert payload["cohort"]["callbacks"] == 2
        assert payload["cohort"]["rows"][0]["decision_count"] == 2
        assert len(payload["golden_samples"]["metamorphic_rows"]) == 1
        assert len(payload["golden_samples"]["runtime_rows"]) == 1
        assert LOCK.load_lock(lock_path)["lock_sha256"] == payload["lock_sha256"]

        tampered = copy.deepcopy(payload)
        tampered["cohort"]["callbacks"] = 3
        lock_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(LOCK.LockError, match="self-hash"):
            LOCK.load_lock(lock_path)


def test_resource_and_tensor_invariant_helpers_are_strict():
    sample = FEATURES.encode_public_observation(
        _observation(0, active=True), FEATURES.TARGET_DECK)
    arrays = sample.arrays()
    assert AUDIT._base_tensor_parity(
        sample, _observation(0, active=True), FEATURES.TARGET_DECK)
    assert AUDIT._tensor_contract_failures(arrays) == []
    assert AUDIT._resource_invariant_failures(arrays) == []

    bad = {name: np.array(value, copy=True) for name, value in arrays.items()}
    bad["resource_features"][0, 14] = 9.0
    bad["resource_features"][0, 15] = 1.0
    assert any(
        "deck bounds" in reason
        for reason in AUDIT._resource_invariant_failures(bad)
    )
    bad["log_mask"] = bad["log_mask"].astype(np.int32)
    assert any(
        failure.startswith("log_mask:dtype")
        for failure in AUDIT._tensor_contract_failures(bad)
    )


def test_redaction_audit_matches_standalone_sanitizer():
    logs = [
        {
            "type": 5, "playerIndex": 1, "cardId": 648,
            "cardIdTarget": 112, "cardIdBefore": 647,
            "cardIdAfter": 646, "attackId": 937,
        },
        {"type": 4, "playerIndex": 1, "cardId": 7},
    ]
    arrays, stats = FEATURES.sanitize_log_window(logs, actor_index=0)
    failures, removed = AUDIT._redaction_failures(logs, 0, arrays)
    assert failures == 0
    assert removed == 6
    assert stats.removed_identity_count >= removed
    assert not np.any(arrays["log_card_ids"])


def test_golden_mutation_seat_swap_and_cross_call_independence():
    observation = _observation(0, active=True)
    obs_hash = LOCK._observation_sha256(observation)
    identity = ("synthetic", 0, 0, obs_hash)
    result = AUDIT._golden_checks(
        [(observation, tuple(FEATURES.TARGET_DECK), 0)],
        {identity},
        [identity],
        (copy.deepcopy(observation), tuple(FEATURES.TARGET_DECK)),
    )
    assert result == {
        "golden_mutation_mismatches": 0,
        "golden_seat_swap_mismatches": 0,
        "golden_serial_bijection_mismatches": 0,
        "golden_repeat_state_mismatches": 0,
        "golden_transport_mismatches": 0,
        "golden_runtime_adapter_mismatches": 0,
        "hidden_input_acceptances": 0,
        "malformed_full_deck_reveal_acceptances": 0,
        "off_deck_acceptances": 0,
        "forbidden_key_consumption": 0,
    }


def test_one_game_shadow_audit_passes_without_training_or_promotion():
    with tempfile.TemporaryDirectory() as temporary:
        lock_path, _payload = _build_small_lock(Path(temporary))
        result = AUDIT.run_audit(lock_path)
        assert result["cohort"]["games"] == 1
        assert result["cohort"]["callbacks"] == 2
        assert result["cohort"]["st_main_prompts"] == 2
        assert result["gates"]["passed"]
        assert result["promotion_authority"] is False
        assert result["privacy_audit"][
            "future_observations_actions_rewards_never_passed_to_encoder"
        ] is True
