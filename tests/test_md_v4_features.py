from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from agent.obsview import (
    AREA_ACTIVE,
    AREA_BENCH,
    AREA_DECK,
    AREA_DISCARD,
    AREA_HAND,
    OT_ABILITY,
    OT_PLAY,
    ST_MAIN,
)
from tools.research import md_v4_features as F
from tools.research import qu_v2a_features as QF


def _card(card_id: int, player: int, serial: int) -> dict:
    return {"id": card_id, "playerIndex": player, "serial": serial}


def _pokemon(
    card_id: int,
    player: int,
    serial: int,
    *,
    hp: int,
    max_hp: int,
    energy: list[dict] | None = None,
    tools: list[dict] | None = None,
    evolution: list[dict] | None = None,
) -> dict:
    energy = [] if energy is None else energy
    return {
        **_card(card_id, player, serial),
        "appearThisTurn": False,
        "hp": hp,
        "maxHp": max_hp,
        "energies": [entry["id"] for entry in energy],
        "energyCards": energy,
        "tools": [] if tools is None else tools,
        "preEvolution": [] if evolution is None else evolution,
    }


def _player(
    *,
    active: list,
    bench: list,
    hand,
    discard: list,
    deck_count: int,
    prizes: int,
) -> dict:
    return {
        "active": active,
        "bench": bench,
        "benchMax": 5,
        "hand": hand,
        "handCount": 0 if hand is None else len(hand),
        "discard": discard,
        "deckCount": deck_count,
        "prize": [None] * prizes,
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def observation() -> dict:
    me = _player(
        active=[_pokemon(
            648, 0, 5, hp=250, max_hp=320,
            energy=[_card(7, 0, 6)],
            evolution=[_card(647, 0, 7)],
        )],
        bench=[_pokemon(
            112, 0, 8, hp=80, max_hp=110,
            energy=[_card(7, 0, 9)],
            tools=[_card(1137, 0, 10)],
        )],
        hand=[_card(112, 0, 1), _card(1219, 0, 2)],
        discard=[_card(7, 0, 3), _card(1079, 0, 4)],
        deck_count=42,
        prizes=6,
    )
    opponent = _player(
        active=[_pokemon(
            648, 1, 105, hp=210, max_hp=320,
            energy=[_card(7, 1, 106)],
            evolution=[_card(647, 1, 107)],
        )],
        bench=[_pokemon(646, 1, 108, hp=70, max_hp=70)],
        hand=None,
        discard=[_card(112, 1, 103)],
        deck_count=45,
        prizes=6,
    )
    return {
        "current": {
            "yourIndex": 0,
            "turn": 6,
            "turnActionCount": 3,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "energyAttached": True,
            "stadiumPlayed": False,
            "retreated": False,
            "looking": [_card(646, 0, 12)],
            "stadium": [_card(1259, 0, 11)],
            "players": [me, opponent],
        },
        "select": {
            "type": ST_MAIN,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "effect": _card(1219, 0, 2),
            "contextCard": _card(648, 0, 5),
            "option": [
                {"type": OT_PLAY, "index": 0},
                {
                    "type": OT_ABILITY,
                    "area": AREA_BENCH,
                    "index": 0,
                    "inPlayArea": AREA_BENCH,
                    "inPlayIndex": 0,
                    "playerIndex": 1,
                },
                {
                    "type": OT_ABILITY,
                    "area": AREA_ACTIVE,
                    "index": 0,
                    "inPlayArea": AREA_ACTIVE,
                    "inPlayIndex": 0,
                    "playerIndex": 1,
                },
            ],
        },
        "logs": [
            {
                "type": 6,
                "playerIndex": 0,
                "cardId": 1259,
                "serial": 11,
                "fromArea": AREA_HAND,
                "toArea": 7,
            },
            {
                "type": 4,
                "playerIndex": 1,
                "cardId": 112,
                "serial": 103,
            },
        ],
        "remainingOverageTime": 599.0,
        "search_begin_input": "opaque runtime transport",
        "step": 17,
    }


def _row(sample: F.PublicResourceWindowFeatures, card_id: int) -> np.ndarray:
    index = F.RESOURCE_CARD_IDS.index(card_id)
    return sample.resource_features[index]


def _assert_same(
    left: F.PublicResourceWindowFeatures,
    right: F.PublicResourceWindowFeatures,
) -> None:
    assert left.arrays().keys() == right.arrays().keys()
    for name, expected in left.arrays().items():
        np.testing.assert_array_equal(expected, right.arrays()[name], err_msg=name)


def _visible_self_cards(obs: dict) -> list[int]:
    actor = obs["current"]["yourIndex"]
    me = obs["current"]["players"][actor]
    result: list[int] = []
    result.extend(entry["id"] for entry in me["hand"])
    result.extend(entry["id"] for entry in me["discard"])
    for pokemon in me["active"] + me["bench"]:
        result.append(pokemon["id"])
        result.extend(entry["id"] for entry in pokemon["energyCards"])
        result.extend(entry["id"] for entry in pokemon["tools"])
        result.extend(entry["id"] for entry in pokemon["preEvolution"])
    result.extend(
        entry["id"]
        for entry in obs["current"]["stadium"]
        if entry["playerIndex"] == actor
    )
    result.extend(entry["id"] for entry in obs["current"]["looking"])
    return result


def _current_deck_reveal(obs: dict) -> list[dict]:
    remaining = Counter(F.TARGET_DECK)
    remaining.subtract(_visible_self_cards(obs))
    assert all(value >= 0 for value in remaining.values())
    cards = [
        card_id
        for card_id in sorted(remaining)
        for _ in range(remaining[card_id])
    ]
    assert len(cards) == 48
    # Six unobserved copies are the prizes; the other 42 are the current deck.
    return [
        _card(card_id, obs["current"]["yourIndex"], 1000 + index)
        for index, card_id in enumerate(cards[:42])
    ]


def _swap_player_indices(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "playerIndex" and isinstance(item, int) and item in (0, 1):
                result[key] = 1 - item
            else:
                result[key] = _swap_player_indices(item)
        return result
    if isinstance(value, list):
        return [_swap_player_indices(item) for item in value]
    return value


def seat_swapped(obs: dict) -> dict:
    result = _swap_player_indices(copy.deepcopy(obs))
    current = result["current"]
    current["players"] = list(reversed(current["players"]))
    current["yourIndex"] = 1 - obs["current"]["yourIndex"]
    current["firstPlayer"] = 1 - obs["current"]["firstPlayer"]
    return result


def serials_renamed(value):
    if isinstance(value, dict):
        return {
            key: (
                10_000 + int(item) * 17
                if key == "serial" and isinstance(item, int)
                else serials_renamed(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [serials_renamed(item) for item in value]
    return value


def test_schema_shapes_base_bytes_and_dependency_lock() -> None:
    sample = F.encode_public_observation(observation(), list(F.TARGET_DECK))
    F.validate_public_features(sample)
    assert F.SCHEMA == "ptcg.md-v4.public-resource-window.v1"
    assert F.KNOWN_NUMERIC_EVENT_TYPES == (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 15, 16,
    )
    assert set(F.EVENT_TYPE_TO_ENUM.values()) == set(range(2, 16))
    assert sample.resource_ids.shape == (19,)
    assert sample.resource_features.shape == (19, 22)
    assert sample.resource_prompt_features.shape == (2,)
    assert sample.log_event_type.shape == (64,)
    assert sample.log_actor_role.shape == (64,)
    assert sample.log_card_ids.shape == (64, 4)
    assert sample.log_attack_ids.shape == (64,)
    assert sample.log_areas.shape == (64, 2)
    assert sample.log_features.shape == (64, 8)
    assert sample.log_mask.shape == (64,)
    assert sample.log_prompt_features.shape == (3,)

    expected_base = QF.encode_public_observation(
        {
            "current": observation()["current"],
            "select": observation()["select"],
            "logs": observation()["logs"],
        },
        list(F.TARGET_DECK),
    )
    actual_base = sample.base_features()
    for name, expected in expected_base.arrays().items():
        np.testing.assert_array_equal(expected, actual_base.arrays()[name])

    assert F.FEATURE_DEPENDENCY_PATHS[0] == (
        "tools/research/md_v4_features.py"
    )
    assert F.compute_feature_dependency_hashes() == F.FEATURE_DEPENDENCY_HASHES
    assert F.feature_dependency_fingerprint(F.FEATURE_DEPENDENCY_HASHES) == (
        F.FEATURE_DEPENDENCY_FINGERPRINT
    )
    assert F.assert_feature_dependency_lock() == F.FEATURE_DEPENDENCY_FINGERPRINT
    assert len(F.feature_fingerprint(sample)) > sum(
        array.nbytes for array in sample.arrays().values()
    )


def test_resource_counts_bounds_options_and_unaccounted_transient() -> None:
    sample = F.encode_public_observation(observation(), list(F.TARGET_DECK))
    assert sample.resource_prompt_features.tolist() == [0.0, 1.0]

    energy = _row(sample, 7)
    assert energy[0] == 10
    assert energy[2] == 1  # self discard
    assert energy[6] == 2  # self attachments
    assert energy[7] == 1  # opponent attachment
    assert energy[13] == 7  # hidden after three visible self copies
    assert energy[14:18].tolist() == [1.0, 7.0, 0.0, 6.0]

    munkidori = _row(sample, 112)
    assert munkidori[1] == 1
    assert munkidori[4] == 1
    assert munkidori[3] == 1
    assert munkidori[13] == 2
    assert munkidori[20] == 1  # first legal option's semantic subject

    impidimp = _row(sample, 646)
    assert impidimp[12] == 1
    assert impidimp[5] == 1
    assert impidimp[20] == 1
    assert impidimp[21] == 1  # public opponent-board target
    grimmsnarl = _row(sample, 648)
    assert grimmsnarl[4] == 1
    assert grimmsnarl[5] == 1
    assert grimmsnarl[21] == 1

    transient = observation()
    transient["current"]["players"][0]["deckCount"] = 41
    transient_sample = F.encode_public_observation(
        transient, list(F.TARGET_DECK)
    )
    assert transient_sample.resource_prompt_features.tolist() == [1.0, 1.0]
    F.validate_public_features(transient_sample)


def test_serial_deduplication_and_invalid_accounting_flag() -> None:
    duplicated = observation()
    duplicated["current"]["players"][0]["hand"].append(
        _card(7, 0, 3)
    )
    duplicated["current"]["players"][0]["handCount"] += 1
    sample = F.encode_public_observation(duplicated, list(F.TARGET_DECK))
    energy = _row(sample, 7)
    assert energy[1] == 1 and energy[2] == 1
    assert energy[13] == 7  # same serial contributes once to hidden bounds
    assert sample.resource_prompt_features.tolist() == [0.0, 1.0]

    missing_serial = observation()
    del missing_serial["current"]["players"][0]["hand"][0]["serial"]
    invalid = F.encode_public_observation(
        missing_serial, list(F.TARGET_DECK)
    )
    assert invalid.resource_prompt_features[1] == 0.0
    F.validate_public_features(invalid)


def test_full_select_deck_tightens_only_deck_and_prize_bounds() -> None:
    exact = observation()
    exact["select"]["deck"] = _current_deck_reveal(exact)
    sample = F.encode_public_observation(exact, list(F.TARGET_DECK))
    for row in sample.resource_features:
        assert row[19] == 1.0
        assert row[14] == row[15] == row[18]
        remaining = row[13] - row[18]
        assert row[16] == row[17] == remaining

    partial = copy.deepcopy(exact)
    partial["select"]["deck"].pop()
    partial_sample = F.encode_public_observation(
        partial, list(F.TARGET_DECK)
    )
    assert not partial_sample.resource_features[:, 19].any()
    assert not partial_sample.resource_features[:, 18].any()


def test_malformed_full_select_deck_reveal_fails_closed() -> None:
    malformed = observation()
    reveal = _current_deck_reveal(malformed)
    reveal[0]["id"] = 1
    malformed["select"]["deck"] = reveal
    with pytest.raises(F.PublicFeatureError, match="registration"):
        F.encode_public_observation(malformed, list(F.TARGET_DECK))

    over_count = observation()
    reveal = _current_deck_reveal(over_count)
    reveal[0]["id"] = 648
    reveal[1]["id"] = 648
    reveal[2]["id"] = 648
    over_count["select"]["deck"] = reveal
    with pytest.raises(F.PublicFeatureError, match="registration"):
        F.encode_public_observation(over_count, list(F.TARGET_DECK))


def test_log_sanitizer_redacts_reverse_opponent_draw_and_unknown_payloads() -> None:
    logs = [
        {
            "type": 4, "playerIndex": 1, "cardId": 112, "serial": 1,
            "attackId": 937,
        },
        {
            "type": 5, "playerIndex": 0, "cardId": 646,
            "cardIdTarget": 648, "attackId": 937,
        },
        {
            "type": 7, "playerIndex": 1, "cardIdBefore": 647,
            "cardIdAfter": 648, "attackId": 937,
        },
        {
            "type": 6, "playerIndex": 0, "cardId": 1259,
            "fromArea": AREA_HAND, "toArea": 7,
        },
        {
            "type": 999, "playerIndex": 1, "cardId": 112,
            "cardIdTarget": 648, "attackId": 937,
            "fromArea": AREA_DECK, "toArea": AREA_DISCARD,
            "value": 340, "putDamageCounter": True,
        },
        "legacy-string-event",
    ]
    arrays, stats = F.sanitize_log_window(logs, 0)
    assert arrays["log_mask"][:6].all()
    assert not arrays["log_mask"][6:].any()
    assert np.all(arrays["log_card_ids"][:3] == 0)
    assert np.all(arrays["log_attack_ids"][:3] == 0)
    assert arrays["log_features"][:3, 5].tolist() == [1.0, 1.0, 1.0]
    assert arrays["log_card_ids"][3, 0] == 1259
    assert arrays["log_areas"][3].tolist() == [AREA_HAND, 7]
    assert arrays["log_event_type"][4] == F.EVENT_UNKNOWN
    assert arrays["log_actor_role"][4] == F.ROLE_OPPONENT
    assert np.all(arrays["log_card_ids"][4] == 0)
    assert arrays["log_attack_ids"][4] == 0
    assert np.all(arrays["log_areas"][4] == 0)
    assert np.all(arrays["log_features"][4, :5] == 0.0)
    assert arrays["log_features"][4, 5] == 1.0
    assert arrays["log_actor_role"][5] == F.ROLE_NONE_OR_UNKNOWN
    assert stats.unknown_event_count == 2
    assert stats.removed_identity_count == 11
    assert stats.as_dict()["numeric_type_counts"] == {
        "4": 1, "5": 1, "6": 1, "7": 1,
    }


def test_log_sanitizer_redacts_draw_when_actor_is_unknown() -> None:
    arrays, stats = F.sanitize_log_window(
        [{"type": 4, "cardId": 112, "attackId": 937}],
        actor_index=0,
    )
    assert arrays["log_actor_role"][0] == F.ROLE_NONE_OR_UNKNOWN
    assert arrays["log_features"][0, 5] == 1.0
    assert not np.any(arrays["log_card_ids"][0])
    assert arrays["log_attack_ids"][0] == 0
    assert stats.removed_identity_count == 2


def test_all_known_event_types_have_fixed_distinct_normalized_values() -> None:
    logs = [
        {"type": raw_type, "playerIndex": 0}
        for raw_type in F.KNOWN_NUMERIC_EVENT_TYPES
    ]
    arrays, stats = F.sanitize_log_window(logs, 0)
    actual = arrays["log_event_type"][:len(logs)].tolist()
    assert actual == [
        F.EVENT_TYPE_TO_ENUM[raw_type]
        for raw_type in F.KNOWN_NUMERIC_EVENT_TYPES
    ]
    assert len(set(actual)) == 14
    assert stats.unknown_event_count == 0


def test_attack_ids_use_zero_padding_and_cover_the_full_engine_id_range() -> None:
    assert F.ATTACK_VOCAB_SIZE == 1557
    arrays, _ = F.sanitize_log_window([
        {"type": 15, "playerIndex": 0, "attackId": 1},
        {"type": 15, "playerIndex": 0, "attackId": 1556},
        {"type": 15, "playerIndex": 0, "attackId": 1557},
    ], 0)
    assert arrays["log_attack_ids"][:3].tolist() == [1, 1556, 0]


def test_log_window_keeps_latest_64_left_aligned_with_absolute_position() -> None:
    logs = [
        {
            "type": 6,
            "playerIndex": 0,
            "cardId": 7 if index % 2 else 112,
            "fromArea": AREA_DECK,
            "toArea": AREA_HAND,
        }
        for index in range(70)
    ]
    arrays, stats = F.sanitize_log_window(logs, 0)
    assert arrays["log_mask"].all()
    assert arrays["log_card_ids"][0, 0] == 112  # original event index six
    assert arrays["log_card_ids"][-1, 0] == 7
    assert arrays["log_features"][0, 6] == np.float32(7 / 255)
    assert arrays["log_features"][-1, 6] == np.float32(70 / 255)
    assert arrays["log_prompt_features"].tolist() == [70.0, 6.0, 1.0]
    assert stats.raw_event_count == 70
    assert stats.retained_event_count == 64
    assert stats.dropped_event_count == 6

    long_arrays, _ = F.sanitize_log_window(logs * 5, 0)
    assert long_arrays["log_prompt_features"].tolist() == [255.0, 255.0, 1.0]
    assert long_arrays["log_features"][-1, 6] == 1.0


def test_transport_private_metadata_is_ignored_but_actor_private_state_rejected() -> None:
    base_obs = observation()
    expected = F.encode_public_observation(base_obs, list(F.TARGET_DECK))
    transport = copy.deepcopy(base_obs)
    transport.update({
        "visualize": {
            "players": [
                {"deck": [_card(7, 0, index) for index in range(60)]},
                {"hand": [_card(648, 1, 999)]},
            ]
        },
        "search_begin_input": "different native serialization",
        "remainingOverageTime": 1.0,
        "step": 9999,
        "reward": -1,
        "future_steps": [{"opponentHand": [648]}],
        "episode_id": "private-transport-id",
        "agent_name": "must-not-enter-features",
        "wall_clock_date": "2099-01-01",
    })
    _assert_same(
        expected,
        F.encode_public_observation(transport, list(F.TARGET_DECK)),
    )

    exact_hidden = observation()
    exact_hidden["_counterfactual_exact_hidden_v1"] = {"opponent_hand": [648]}
    with pytest.raises(F.PublicFeatureError, match="exact-hidden"):
        F.encode_public_observation(exact_hidden, list(F.TARGET_DECK))

    opponent_hand = observation()
    opponent_hand["current"]["players"][1]["hand"] = [_card(648, 1, 999)]
    with pytest.raises(F.PublicFeatureError, match="opponent hand"):
        F.encode_public_observation(opponent_hand, list(F.TARGET_DECK))

    hidden_deck = observation()
    hidden_deck["current"]["players"][0]["deck"] = [_card(7, 0, 999)]
    with pytest.raises(F.PublicFeatureError, match="hidden deck"):
        F.encode_public_observation(hidden_deck, list(F.TARGET_DECK))


def test_exact_deck_scope_and_actor_relative_seat_swap() -> None:
    obs = observation()
    original = F.encode_public_observation(obs, list(F.TARGET_DECK))
    swapped = F.encode_public_observation(
        seat_swapped(obs), list(reversed(F.TARGET_DECK))
    )
    _assert_same(original, swapped)

    off_deck = list(F.TARGET_DECK)
    off_deck[0] = 1
    assert not F.supports_deck(off_deck)
    with pytest.raises(F.PublicFeatureError, match="exact registered deck"):
        F.encode_public_observation(obs, off_deck)
    for malformed in (
        list(F.TARGET_DECK[:-1]),
        list(F.TARGET_DECK) + [7],
        [True] * 60,
        [0] * 60,
    ):
        with pytest.raises(F.PublicFeatureError):
            F.encode_public_observation(obs, malformed)


def test_offline_runtime_callback_parity_and_serial_renaming_invariance() -> None:
    obs = observation()
    deck = list(F.TARGET_DECK)
    offline = F.encode_public_observation(obs, deck)
    runtime = F.encode_runtime_observation(copy.deepcopy(obs), tuple(deck))
    _assert_same(offline, runtime)
    assert F.feature_fingerprint(offline) == F.feature_fingerprint(runtime)

    renamed = F.encode_runtime_observation(serials_renamed(obs), deck)
    _assert_same(offline, renamed)
    assert F.feature_fingerprint(offline) == F.feature_fingerprint(renamed)


def test_stateless_idempotence_across_unrelated_calls() -> None:
    obs = observation()
    first = F.encode_public_observation(obs, list(F.TARGET_DECK))
    second = F.encode_public_observation(obs, list(F.TARGET_DECK))
    _assert_same(first, second)
    assert F.feature_fingerprint(first) == F.feature_fingerprint(second)

    unrelated = observation()
    unrelated["current"]["turn"] = 30
    unrelated["current"]["players"][0]["discard"].append(
        _card(1086, 0, 50)
    )
    unrelated["current"]["players"][0]["deckCount"] -= 1
    unrelated["logs"] = [
        {"type": 16, "playerIndex": 1, "cardId": 648, "value": -340}
    ] * 90
    F.encode_public_observation(unrelated, list(F.TARGET_DECK))
    after = F.encode_public_observation(obs, list(F.TARGET_DECK))
    _assert_same(first, after)


def test_strict_validation_rejects_corrupted_extension_tensors() -> None:
    sample = F.encode_public_observation(observation(), list(F.TARGET_DECK))

    wrong_ids = sample.resource_ids.copy()
    wrong_ids[[0, 1]] = wrong_ids[[1, 0]]
    with pytest.raises(F.PublicFeatureError, match="canonical"):
        F.validate_public_features(replace(sample, resource_ids=wrong_ids))

    broken_bound = sample.resource_features.copy()
    broken_bound[0, 14] = broken_bound[0, 15] + 1
    with pytest.raises(F.PublicFeatureError, match="invariant"):
        F.validate_public_features(
            replace(sample, resource_features=broken_bound)
        )

    bad_padding = sample.log_card_ids.copy()
    bad_padding[-1, 0] = 112
    with pytest.raises(F.PublicFeatureError, match="padding"):
        F.validate_public_features(replace(sample, log_card_ids=bad_padding))

    bad_unknown = sample.log_event_type.copy()
    bad_unknown[0] = F.EVENT_UNKNOWN
    with pytest.raises(F.PublicFeatureError, match="redacted|unknown"):
        F.validate_public_features(replace(sample, log_event_type=bad_unknown))

    bad_mask = sample.log_mask.copy()
    bad_mask[0] = False
    bad_mask[2] = True
    with pytest.raises(F.PublicFeatureError, match="left-aligned"):
        F.validate_public_features(replace(sample, log_mask=bad_mask))
