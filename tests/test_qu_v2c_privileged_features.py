"""Isolation contracts for Qu-v2C tooling-only privileged critic features."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import counterfactual_oracle as CFO  # noqa: E402
from agent import policy  # noqa: E402
from agent.obsview import AREA_ACTIVE, OT_ATTACH, OT_END, OT_PLAY, ST_MAIN  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402


def _card(card_id, owner=0, serial=1):
    return {"id": card_id, "playerIndex": owner, "serial": serial}


def _pokemon(card_id, owner, serial):
    return {
        "id": card_id,
        "playerIndex": owner,
        "serial": serial,
        "hp": 120,
        "maxHp": 140,
        "energies": [],
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
        "appearThisTurn": False,
    }


def _observation():
    hand = [_card(13, 0, 30), _card(305, 0, 31)]
    return {
        "current": {
            "yourIndex": 0,
            "turn": 4,
            "turnActionCount": 7,
            "result": 0,
            "firstPlayer": 1,
            "supporterPlayed": False,
            "energyAttached": False,
            "stadiumPlayed": False,
            "retreated": False,
            "stadium": [],
            "looking": [],
            "players": [
                {
                    "active": [_pokemon(743, 0, 10)],
                    "bench": [],
                    "hand": hand,
                    "handCount": len(hand),
                    "discard": [],
                    "deckCount": 3,
                    "prize": [None] * 2,
                },
                {
                    "active": [_pokemon(723, 1, 20)],
                    "bench": [],
                    "hand": None,
                    "handCount": 2,
                    "discard": [],
                    "deckCount": 2,
                    "prize": [None],
                },
            ],
        },
        "select": {
            "type": ST_MAIN,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": OT_ATTACH, "index": 0,
                 "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
                {"type": OT_PLAY, "index": 1},
                {"type": OT_END},
            ],
        },
        "logs": ["transport log must not enter features"],
        "search_begin_input": "opaque-native-search-state",
    }


def _payload(obs):
    return {
        "schema": PF.EXACT_PAYLOAD_SCHEMA,
        "selecting_player": 0,
        "turn": 4,
        "public_root_fingerprint": PF.public_root_fingerprint(obs),
        "search_begin_sha256": hashlib.sha256(
            obs["search_begin_input"].encode("utf-8")).hexdigest(),
        "my_deck": [101, 102, 103],
        "my_prize": [104, 105],
        "opponent_deck": [201, 202],
        "opponent_prize": [203],
        "opponent_hand": [204, 205],
        "opponent_active": [],
    }


def _record(obs=None, payload=None):
    obs = _observation() if obs is None else obs
    payload = _payload(obs) if payload is None else payload
    return PF.encode_privileged_observation(
        obs, payload, policy.load_deck())


def _assert_public_equal(left, right):
    for name in PF.PUBLIC_ARRAY_NAMES:
        np.testing.assert_array_equal(
            getattr(left, name), getattr(right, name), err_msg=name)


def _expect_error(callback, text):
    try:
        callback()
    except PF.PrivilegedFeatureError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError("expected PrivilegedFeatureError")


def test_encode_uses_fixed_hidden_arrays_masks_and_public_sample():
    obs = _observation()
    assert PF.public_root_fingerprint(obs) == CFO.public_root_fingerprint(obs)
    record = _record(obs)
    QF.validate_public_features(record.public)
    PF.validate_privileged_features(record)

    assert record.my_deck_ids.shape == (PF.DECK_SLOTS,)
    assert record.my_deck_ids[:4].tolist() == [101, 102, 103, 0]
    assert record.my_deck_mask[:4].tolist() == [True, True, True, False]
    assert record.my_prize_ids[:3].tolist() == [104, 105, 0]
    assert record.opponent_deck_ids[:3].tolist() == [201, 202, 0]
    assert record.opponent_prize_ids[:2].tolist() == [203, 0]
    assert record.opponent_hand_ids[:3].tolist() == [204, 205, 0]
    assert not record.opponent_active_mask.any()
    assert not record.my_deck_ids.flags.writeable
    assert not record.public.board_ids.flags.writeable

    expected_public = QF.encode_public_observation(obs, policy.load_deck())
    _assert_public_equal(record.public, expected_public)

    arrays = record.arrays()
    assert tuple(arrays) == PF.ARRAY_NAMES
    assert all(isinstance(value, np.ndarray) for value in arrays.values())
    assert all(value.dtype != np.dtype(object) for value in arrays.values())
    assert not any(
        token in name for name in arrays
        for token in ("visual", "log", "serialized", "search_begin"))

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as loaded:
        assert tuple(loaded.files) == PF.ARRAY_NAMES
        for name, value in arrays.items():
            np.testing.assert_array_equal(loaded[name], value)


def test_strip_to_public_is_whitelisted_and_defensive():
    record = _record()
    public = PF.strip_to_public(record)
    assert isinstance(public, QF.PublicFeatures)
    assert set(public.arrays()) == set(PF.PUBLIC_ARRAY_NAMES)
    assert not any(hasattr(public, name) for name in PF.HIDDEN_ARRAY_NAMES)
    _assert_public_equal(public, record.public)

    original = int(record.public.board_ids[0])
    public.board_ids[0] = 0
    assert int(record.public.board_ids[0]) == original
    public.registered_deck_ids[0] = 1
    assert int(record.public.registered_deck_ids[0]) != 1


def test_payload_binding_and_no_raw_material_fail_closed():
    obs = _observation()
    base = _payload(obs)

    extra = dict(base)
    extra["visual"] = {"raw": b"not-a-feature"}
    _expect_error(
        lambda: _record(obs, extra), "keys differ from the strict schema")

    bytes_zone = dict(base)
    bytes_zone["opponent_hand"] = [b"204", 205]
    _expect_error(
        lambda: _record(obs, bytes_zone), "invalid card ID")

    short = dict(base)
    short["opponent_hand"] = [204]
    _expect_error(
        lambda: _record(obs, short), "opponent_hand has length")

    bad_id = dict(base)
    bad_id["my_deck"] = [101, 0, 103]
    _expect_error(
        lambda: _record(obs, bad_id), "invalid card ID")

    stale_root = dict(base)
    stale_root["public_root_fingerprint"] = "0" * 64
    _expect_error(
        lambda: _record(obs, stale_root), "public-root fingerprint mismatch")

    stale_search = dict(base)
    stale_search["search_begin_sha256"] = "0" * 64
    _expect_error(
        lambda: _record(obs, stale_search),
        "serialized-search fingerprint mismatch")

    bytes_search = _observation()
    bytes_search["search_begin_input"] = b"native bytes forbidden"
    bytes_payload = _payload(_observation())
    bytes_payload["public_root_fingerprint"] = PF.public_root_fingerprint(
        bytes_search)
    _expect_error(
        lambda: _record(bytes_search, bytes_payload), "never bytes")


def test_validator_rejects_mutation_bad_padding_and_memory_aliases():
    record = _record()

    writeable = record.my_prize_ids.copy()
    broken = replace(record, my_prize_ids=writeable)
    _expect_error(
        lambda: PF.validate_privileged_features(broken), "must be immutable")

    bad_ids = record.my_prize_ids.copy()
    bad_ids[5] = 123
    bad_ids.setflags(write=False)
    broken = replace(record, my_prize_ids=bad_ids)
    _expect_error(
        lambda: PF.validate_privileged_features(broken),
        "mask/padding is not a canonical dense prefix")

    alias = record.public.registered_deck_ids
    alias_mask = np.ones(PF.DECK_SLOTS, dtype=np.bool_)
    alias_mask.setflags(write=False)
    broken = replace(
        record, my_deck_ids=alias, my_deck_mask=alias_mask)
    _expect_error(
        lambda: PF.validate_privileged_features(broken),
        "public and privileged arrays must not alias memory")


def test_canonical_hash_is_stable_sensitive_and_transport_free():
    obs = _observation()
    first = _record(obs)
    second = _record(obs)
    assert first.canonical_hash() == second.canonical_hash()
    assert len(first.canonical_hash()) == 64
    assert first.canonical_hash() == PF.canonical_hash(first)

    changed_payload = _payload(obs)
    changed_payload["opponent_hand"] = [204, 206]
    changed = _record(obs, changed_payload)
    assert changed.canonical_hash() != first.canonical_hash()

    # Logs and native serialization are inputs used only for provenance/binding.
    # With identical public and hidden numeric features they do not enter the
    # critic record or its canonical feature identity.
    transport = _observation()
    transport["logs"] = ["different", "arbitrary", "debug data"]
    transport["search_begin_input"] = "different serialized root"
    transport_payload = _payload(transport)
    transported = _record(transport, transport_payload)
    assert transported.canonical_hash() == first.canonical_hash()


if __name__ == "__main__":
    test_encode_uses_fixed_hidden_arrays_masks_and_public_sample()
    test_strip_to_public_is_whitelisted_and_defensive()
    test_payload_binding_and_no_raw_material_fail_closed()
    test_validator_rejects_mutation_bad_padding_and_memory_aliases()
    test_canonical_hash_is_stable_sensitive_and_transport_free()
    print("ok")
