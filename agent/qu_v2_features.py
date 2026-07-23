"""Vendored public-only relational features for the Qu-v2 production runtime.

This is the reviewed, Torch-free encoder used by the promoted Qu-v2 model.
Production deliberately keeps it under :mod:`agent`: importing the submission
must not depend on a top-level ``tools`` package or read repository source files
to recompute research provenance.  The hard-coded feature fingerprint is the
content lock embedded in the evaluated weights; parity tests bind this runtime
copy to the research encoder that produced those weights.

Qu-v2A is a stateless representation experiment.  It adds information that is
already public at the current prompt:

* one token per active/bench object, including public energy, tool and
  pre-evolution identities;
* prompt effect/context identities and ``turnActionCount``;
* the learner's explicitly registered 60-card multiset;
* the semantic subject and public board target of every legal option.

It intentionally excludes observation logs, ``search_begin_input``, exact
counterfactual payloads, opponent-hand identities and opponent deck lists.
History/recurrent state is outside the v2A scope.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np

from agent import cards
from agent import features as BASE
from agent.obsview import AREA_ACTIVE, AREA_BENCH, ObsView


SCHEMA = "ptcg.qu-v2a.public-relational.v4"

# Qu-v2A delegates semantic option binding and static card metadata to the
# shipped agent modules. Their public constants remain explicit guards here;
# research-time source/data hashes are represented by the pinned fingerprint.
EXPECTED_BASE_FEATURE_VERSION = 3
EXPECTED_CARD_VOCAB = 1300
EXPECTED_BASE_OPTION_FEATURES = 91
if BASE.FEAT_VERSION != EXPECTED_BASE_FEATURE_VERSION:
    raise RuntimeError(
        "Qu-v2A requires agent.features.FEAT_VERSION "
        f"{EXPECTED_BASE_FEATURE_VERSION}, got {BASE.FEAT_VERSION}"
    )
if BASE.N_CARD_IDS != EXPECTED_CARD_VOCAB:
    raise RuntimeError(
        "Qu-v2A requires agent.features.N_CARD_IDS "
        f"{EXPECTED_CARD_VOCAB}, got {BASE.N_CARD_IDS}"
    )
if BASE.OPT_FEATS != EXPECTED_BASE_OPTION_FEATURES:
    raise RuntimeError(
        "Qu-v2A requires agent.features.OPT_FEATS "
        f"{EXPECTED_BASE_OPTION_FEATURES}, got {BASE.OPT_FEATS}"
    )
BASE_FEATURE_VERSION = EXPECTED_BASE_FEATURE_VERSION

FEATURE_DEPENDENCY_FINGERPRINT = (
    "c6747b6c2848baf0c70dca80e0423240f9024dcf207797e927cb46e596060123"
)


def assert_feature_dependency_lock() -> str:
    """Return the evaluated feature identity without production filesystem I/O."""
    return FEATURE_DEPENDENCY_FINGERPRINT

BOARD_SLOTS = 12             # my active/bench x5, opponent active/bench x5
ENERGY_SLOTS = 6
TOOL_SLOTS = 2
EVOLUTION_SLOTS = 2
HAND_SLOTS = 48
DISCARD_SLOTS = 60
LOOKING_SLOTS = 60
STADIUM_SLOTS = 2
PROMPT_ID_SLOTS = 2          # effect, contextCard
REGISTERED_DECK_SLOTS = 60

BOARD_FEATURES = 20
PROMPT_FEATURES = 74
OPTION_EXTRA_FEATURES = 16
OPTION_FEATURES = BASE.OPT_FEATS + OPTION_EXTRA_FEATURES
BASE_OPTION_HP_FRACTION_INDEX = 79

_EXACT_HIDDEN_KEY = "_counterfactual_exact_hidden_v1"
_HP_NORM = 340.0
_CONTEXTS = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19,
    21, 22, 25, 35, 38, 41, 42, 43, 46,
)


class PublicFeatureError(ValueError):
    """The candidate encoder received malformed or privileged information."""


@dataclass(frozen=True)
class PublicFeatures:
    board_ids: np.ndarray
    board_energy_ids: np.ndarray
    board_tool_ids: np.ndarray
    board_evolution_ids: np.ndarray
    board_features: np.ndarray
    hand_ids: np.ndarray
    my_discard_ids: np.ndarray
    opponent_discard_ids: np.ndarray
    looking_ids: np.ndarray
    stadium_ids: np.ndarray
    prompt_ids: np.ndarray
    prompt_features: np.ndarray
    registered_deck_ids: np.ndarray
    option_ids: np.ndarray
    option_target_ids: np.ndarray
    option_features: np.ndarray
    option_mask: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        """Return defensive copies for tests, batching and artifact tooling."""
        return {
            item.name: np.array(getattr(self, item.name), copy=True)
            for item in fields(self)
        }


def _require_array(
        features: PublicFeatures, name: str, shape: tuple[int, ...],
        dtype: np.dtype,
) -> np.ndarray:
    value = getattr(features, name, None)
    if not isinstance(value, np.ndarray):
        raise PublicFeatureError(f"{name} is not a NumPy array")
    if value.shape != shape:
        raise PublicFeatureError(
            f"{name} has shape {value.shape}, expected {shape}")
    if value.dtype != dtype:
        raise PublicFeatureError(
            f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.flags.c_contiguous:
        raise PublicFeatureError(f"{name} is not C-contiguous")
    return value


def validate_public_features(features: PublicFeatures) -> None:
    """Validate one unpadded, single-observation feature record.

    The NumPy evaluator indexes embeddings directly, so accepting a subtly
    malformed record would turn schema drift into either undefined decisions
    or an indexing exception.  Keep this contract exact and fail closed.
    """
    if not isinstance(features, PublicFeatures):
        raise PublicFeatureError("expected a Qu-v2A PublicFeatures record")
    option_ids = getattr(features, "option_ids", None)
    if (not isinstance(option_ids, np.ndarray) or option_ids.ndim != 1
            or not 1 <= len(option_ids) <= 4097):
        raise PublicFeatureError("option_ids has an invalid row count")
    n_options = len(option_ids)

    integer_shapes = {
        "board_ids": (BOARD_SLOTS,),
        "board_energy_ids": (BOARD_SLOTS, ENERGY_SLOTS),
        "board_tool_ids": (BOARD_SLOTS, TOOL_SLOTS),
        "board_evolution_ids": (BOARD_SLOTS, EVOLUTION_SLOTS),
        "hand_ids": (HAND_SLOTS,),
        "my_discard_ids": (DISCARD_SLOTS,),
        "opponent_discard_ids": (DISCARD_SLOTS,),
        "looking_ids": (LOOKING_SLOTS,),
        "stadium_ids": (STADIUM_SLOTS,),
        "prompt_ids": (PROMPT_ID_SLOTS,),
        "registered_deck_ids": (REGISTERED_DECK_SLOTS,),
        "option_ids": (n_options,),
        "option_target_ids": (n_options,),
    }
    for name, shape in integer_shapes.items():
        value = _require_array(features, name, shape, np.dtype(np.int32))
        if np.any(value < 0) or np.any(value >= EXPECTED_CARD_VOCAB):
            raise PublicFeatureError(f"{name} contains an out-of-range card ID")

    float_shapes = {
        "board_features": (BOARD_SLOTS, BOARD_FEATURES),
        "prompt_features": (PROMPT_FEATURES,),
        "option_features": (n_options, OPTION_FEATURES),
    }
    for name, shape in float_shapes.items():
        value = _require_array(features, name, shape, np.dtype(np.float32))
        if not np.isfinite(value).all():
            raise PublicFeatureError(f"{name} contains a non-finite value")
        if np.any(value < 0.0) or np.any(value > 64.0):
            raise PublicFeatureError(f"{name} contains an implausible value")

    option_mask = _require_array(
        features, "option_mask", (n_options,), np.dtype(np.bool_))
    if not option_mask.all():
        raise PublicFeatureError(
            "single-observation option_mask must mark every real/STOP row")

    deck = features.registered_deck_ids
    if np.any(deck == 0) or np.any(deck[1:] < deck[:-1]):
        raise PublicFeatureError(
            "registered_deck_ids is not a canonical 60-card multiset")
    expected_my_discard = np.float32(
        np.count_nonzero(features.my_discard_ids) / DISCARD_SLOTS)
    expected_opponent_discard = np.float32(
        np.count_nonzero(features.opponent_discard_ids) / DISCARD_SLOTS)
    if (features.prompt_features[71] != 1.0
            or features.prompt_features[72] != expected_my_discard
            or features.prompt_features[73] != expected_opponent_discard):
        raise PublicFeatureError(
            "prompt schema sentinel/discard cardinalities are inconsistent")
    presence = features.board_features[:, 0]
    if not np.isin(presence, np.asarray([0.0, 1.0], dtype=np.float32)).all():
        raise PublicFeatureError("board presence mask is not binary")

    # BASE.encode_options appends exactly one virtual STOP row.  Keeping that
    # invariant here protects sequential decoding and evaluator alignment.
    stop_flags = features.option_features[:, 88]
    if (features.option_ids[-1] != 0 or features.option_target_ids[-1] != 0
            or stop_flags[-1] != 1.0 or np.any(stop_flags[:-1] != 0.0)
            or np.any(features.option_features[-1, BASE.OPT_FEATS:] != 0.0)):
        raise PublicFeatureError("option rows do not end in one canonical STOP")


def _valid_card_id(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value < BASE.N_CARD_IDS
    )


def _entry_id(entry: Any) -> int:
    if isinstance(entry, Mapping):
        value = entry.get("id")
    else:
        value = entry
    return int(value) if _valid_card_id(value) else 0


def _finite_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if np.isfinite(number) else default
    return default


def _nonnegative_number(value: Any, default: float = 0.0) -> float:
    """Normalize public counts while treating negative engine sentinels as zero."""
    return max(_finite_number(value, default), 0.0)


def _copy_ids(entries: Any, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.int32)
    if not isinstance(entries, (list, tuple)):
        return result
    for index, entry in enumerate(entries[:size]):
        result[index] = _entry_id(entry)
    return result


def _registered_deck(deck: Sequence[int]) -> np.ndarray:
    if isinstance(deck, (str, bytes)) or not isinstance(deck, Sequence):
        raise PublicFeatureError("registered learner deck must be a sequence")
    values = list(deck)
    if len(values) != REGISTERED_DECK_SLOTS or not all(
            _valid_card_id(value) for value in values):
        raise PublicFeatureError(
            "registered learner deck must contain exactly 60 valid integer IDs"
        )
    # Registration is a multiset.  Canonical ordering makes deck-file order a
    # mathematical no-op and gives artifacts a stable representation.
    return np.asarray(sorted(int(value) for value in values), dtype=np.int32)


def _validate_public_root(obs: Mapping[str, Any]) -> tuple[Mapping[str, Any], int]:
    if not isinstance(obs, Mapping):
        raise PublicFeatureError("observation must be a mapping")
    if _EXACT_HIDDEN_KEY in obs:
        raise PublicFeatureError("exact-hidden metadata reached Qu-v2A")
    current = obs.get("current")
    if not isinstance(current, Mapping):
        raise PublicFeatureError("observation has no current public state")
    players = current.get("players")
    selecting = current.get("yourIndex")
    if (not isinstance(selecting, int) or isinstance(selecting, bool)
            or selecting not in (0, 1) or not isinstance(players, list)
            or len(players) != 2
            or not all(isinstance(player, Mapping) for player in players)):
        raise PublicFeatureError("observation has invalid public player state")
    if any("deck" in player for player in players):
        raise PublicFeatureError("hidden deck identities reached Qu-v2A")
    # In the normal actor observation, only the selecting player's hand is
    # visible.  Fail closed instead of accidentally training on a visualization
    # or the other seat's replay row.  A future explicitly modelled reveal
    # mechanic should receive its own versioned public feature.
    if players[1 - selecting].get("hand") is not None:
        raise PublicFeatureError("opponent hand identities reached Qu-v2A")
    return current, int(selecting)


def _board_entries(current: Mapping[str, Any], selecting: int):
    players = current.get("players") or []
    for relative_owner, player_index in enumerate((selecting, 1 - selecting)):
        player = players[player_index]
        active = player.get("active") or ()
        if not isinstance(active, (list, tuple)):
            active = ()
        yield relative_owner, True, 0, active[0] if active else None, player
        bench = player.get("bench") or ()
        if not isinstance(bench, (list, tuple)):
            bench = ()
        for bench_index in range(5):
            entry = bench[bench_index] if bench_index < len(bench) else None
            yield relative_owner, False, bench_index, entry, player


def _attack_readiness(entry: Mapping[str, Any], energy_count: int) -> tuple[float, float]:
    card = cards.card(_entry_id(entry))
    costs: list[int] = []
    if card:
        for attack_id in card.get("attacks") or ():
            attack = cards.attack(attack_id)
            if attack:
                costs.append(len(attack.get("energies") or ()))
    if not costs:
        return 0.0, 0.0
    minimum = min(costs)
    return minimum / 5.0, 1.0 if energy_count >= minimum else 0.0


def _encode_board(current: Mapping[str, Any], selecting: int):
    board_ids = np.zeros(BOARD_SLOTS, dtype=np.int32)
    energy_ids = np.zeros((BOARD_SLOTS, ENERGY_SLOTS), dtype=np.int32)
    tool_ids = np.zeros((BOARD_SLOTS, TOOL_SLOTS), dtype=np.int32)
    evolution_ids = np.zeros((BOARD_SLOTS, EVOLUTION_SLOTS), dtype=np.int32)
    numeric = np.zeros((BOARD_SLOTS, BOARD_FEATURES), dtype=np.float32)

    for slot, (owner, active, position, raw_entry, player) in enumerate(
            _board_entries(current, selecting)):
        if not isinstance(raw_entry, Mapping):
            continue
        card_id = _entry_id(raw_entry)
        board_ids[slot] = card_id
        energy_cards = raw_entry.get("energyCards") or ()
        tools = raw_entry.get("tools") or ()
        evolution = raw_entry.get("preEvolution") or ()
        energy_ids[slot] = _copy_ids(energy_cards, ENERGY_SLOTS)
        tool_ids[slot] = _copy_ids(tools, TOOL_SLOTS)
        evolution_ids[slot] = _copy_ids(evolution, EVOLUTION_SLOTS)

        max_hp = max(_finite_number(raw_entry.get("maxHp")), 0.0)
        hp = max(_finite_number(raw_entry.get("hp")), 0.0)
        hp_fraction = hp / max_hp if max_hp else 0.0
        energy_count = max(
            len(raw_entry.get("energies") or ()), len(energy_cards))
        card = cards.card(card_id) or {}
        min_cost, ready = _attack_readiness(raw_entry, energy_count)
        row = numeric[slot]
        row[0] = 1.0
        row[1] = 1.0 if owner == 0 else 0.0
        row[2] = 1.0 if active else 0.0
        row[3] = 0.0 if active else position / 4.0
        row[4] = hp_fraction
        row[5] = max_hp / _HP_NORM
        row[6] = max(0.0, 1.0 - hp_fraction)
        row[7] = energy_count / float(ENERGY_SLOTS)
        row[8] = len(energy_cards) / float(ENERGY_SLOTS)
        row[9] = len(tools) / float(TOOL_SLOTS)
        row[10] = len(evolution) / float(EVOLUTION_SLOTS)
        row[11] = 1.0 if raw_entry.get("appearThisTurn") else 0.0
        row[12] = 1.0 if card.get("ex") else 0.0
        row[13] = 1.0 if card.get("megaEx") else 0.0
        row[14] = 1.0 if card.get("basic") else 0.0
        row[15] = 1.0 if card.get("stage1") else 0.0
        row[16] = 1.0 if card.get("stage2") else 0.0
        row[17] = min_cost
        row[18] = ready
        status = active and any(player.get(name) for name in (
            "poisoned", "burned", "asleep", "paralyzed", "confused"))
        row[19] = 1.0 if status else 0.0
    return board_ids, energy_ids, tool_ids, evolution_ids, numeric


def _prompt_ids(select: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([
        _entry_id(select.get("effect")),
        _entry_id(select.get("contextCard")),
    ], dtype=np.int32)


def _prompt_features(view: ObsView, registered_deck: np.ndarray) -> np.ndarray:
    current = view.current or {}
    select = view.select or {}
    me, opponent = view.me or {}, view.opp or {}
    result = np.zeros(PROMPT_FEATURES, dtype=np.float32)

    select_type = select.get("type")
    if isinstance(select_type, int) and not isinstance(select_type, bool) \
            and 0 <= select_type < 11:
        result[select_type] = 1.0
    context = select.get("context")
    known_context = (isinstance(context, int) and not isinstance(context, bool)
                     and context in _CONTEXTS)
    context_index = _CONTEXTS.index(context) if known_context else len(_CONTEXTS)
    result[11 + context_index] = 1.0

    result[37] = _nonnegative_number(select.get("minCount"), 1.0) / 5.0
    result[38] = _nonnegative_number(select.get("maxCount"), 1.0) / 5.0
    result[39] = len(select.get("option") or ()) / 24.0
    result[40] = _nonnegative_number(current.get("turn")) / 40.0
    result[41] = _nonnegative_number(current.get("turnActionCount")) / 32.0
    result[42] = _nonnegative_number(select.get("remainDamageCounter")) / 34.0
    energy_cost = select.get("remainEnergyCost")
    result[43] = (len(energy_cost) if isinstance(energy_cost, (list, tuple))
                  else _nonnegative_number(energy_cost)) / 5.0
    for offset, name in enumerate((
            "supporterPlayed", "energyAttached", "stadiumPlayed", "retreated")):
        result[44 + offset] = 1.0 if current.get(name) else 0.0
    result[48] = 1.0 if current.get("firstPlayer") == view.my_index else 0.0
    result[49] = _nonnegative_number(me.get("deckCount")) / 60.0
    result[50] = _nonnegative_number(opponent.get("deckCount")) / 60.0
    result[51] = _nonnegative_number(
        me.get("handCount"), float(len(me.get("hand") or ()))) / 30.0
    result[52] = _nonnegative_number(opponent.get("handCount")) / 30.0
    result[53] = len(me.get("prize") or ()) / 6.0
    result[54] = len(opponent.get("prize") or ()) / 6.0
    for owner, player in enumerate((me, opponent)):
        for status_index, name in enumerate((
                "poisoned", "burned", "asleep", "paralyzed", "confused")):
            result[55 + owner * 5 + status_index] = 1.0 if player.get(name) else 0.0
    result[65] = len(me.get("active") or ()) + len(me.get("bench") or ())
    result[65] /= 6.0
    result[66] = len(opponent.get("active") or ()) + len(opponent.get("bench") or ())
    result[66] /= 6.0
    result[67] = 1.0 if len(registered_deck) == REGISTERED_DECK_SLOTS else 0.0
    result[68] = 1.0 if _entry_id(select.get("effect")) else 0.0
    result[69] = 1.0 if _entry_id(select.get("contextCard")) else 0.0
    result[70] = len(current.get("looking") or ()) / 60.0
    result[71] = 1.0  # schema-presence sentinel for strict round-trip checks
    # Mean-pooled discard embeddings cannot distinguish one copy from repeated
    # copies of the same card.  Cardinalities resolve that representation
    # collision without leaking any hidden identity.
    result[72] = len(me.get("discard") or ()) / 60.0
    result[73] = len(opponent.get("discard") or ()) / 60.0
    return result


def _target_entry(view: ObsView, option: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, bool, int, bool]:
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    valid_index = isinstance(index, int) and not isinstance(index, bool)
    if area not in (AREA_ACTIVE, AREA_BENCH) or not valid_index:
        area = option.get("area")
        index = option.get("index")
        valid_index = isinstance(index, int) and not isinstance(index, bool)
    if area not in (AREA_ACTIVE, AREA_BENCH) or not valid_index:
        return None, False, 0, False
    player_index = option.get("playerIndex", view.my_index)
    if (not isinstance(player_index, int) or isinstance(player_index, bool)
            or player_index not in (0, 1)):
        return None, False, 0, False
    entry = view.board_entry(area, index, int(player_index))
    return entry, int(player_index) == view.my_index, int(index), area == AREA_ACTIVE


def _encode_option_extras(view: ObsView, option_ids: np.ndarray,
                          registered_deck: np.ndarray):
    count = len(view.options)
    target_ids = np.zeros(count + 1, dtype=np.int32)
    extra = np.zeros((count + 1, OPTION_EXTRA_FEATURES), dtype=np.float32)
    deck_counts = Counter(int(value) for value in registered_deck)
    prompt = _prompt_ids(view.select or {})
    for index, option in enumerate(view.options):
        target, mine, target_index, active = _target_entry(view, option)
        subject_id = int(option_ids[index])
        if isinstance(target, Mapping):
            target_id = _entry_id(target)
            target_ids[index] = target_id
            max_hp = max(_finite_number(target.get("maxHp")), 0.0)
            hp = max(_finite_number(target.get("hp")), 0.0)
            hp_fraction = hp / max_hp if max_hp else 0.0
            energies = max(len(target.get("energies") or ()),
                           len(target.get("energyCards") or ()))
            extra[index, 0] = 1.0
            extra[index, 1] = 1.0 if mine else 0.0
            extra[index, 2] = 1.0 if active else 0.0
            extra[index, 3] = target_index / 4.0
            extra[index, 4] = hp_fraction
            extra[index, 5] = max(0.0, 1.0 - hp_fraction)
            extra[index, 6] = energies / float(ENERGY_SLOTS)
            extra[index, 7] = len(target.get("tools") or ()) / float(TOOL_SLOTS)
            extra[index, 8] = len(target.get("preEvolution") or ()) / float(
                EVOLUTION_SLOTS)
            extra[index, 10] = 1.0 if target.get("appearThisTurn") else 0.0
            target_card = cards.card(target_id) or {}
            extra[index, 11] = 1.0 if (
                target_card.get("ex") or target_card.get("megaEx")) else 0.0
        extra[index, 9] = deck_counts.get(subject_id, 0) / 4.0
        extra[index, 12] = 1.0 if subject_id else 0.0
        extra[index, 13] = 1.0 if subject_id and subject_id == int(prompt[0]) else 0.0
        extra[index, 14] = 1.0 if subject_id and subject_id == int(prompt[1]) else 0.0
        extra[index, 15] = 1.0 if target_ids[index] == subject_id and subject_id else 0.0
    return target_ids, extra


def encode_public_observation(
        obs: Mapping[str, Any], registered_learner_deck: Sequence[int],
) -> PublicFeatures:
    """Encode one actor observation without consuming privileged state."""
    current, selecting = _validate_public_root(obs)
    deck = _registered_deck(registered_learner_deck)
    view = ObsView(dict(obs))
    if view.select is None or not isinstance(view.select, Mapping):
        raise PublicFeatureError("Qu-v2A expects an action-selection prompt")
    board = _encode_board(current, selecting)
    me, opponent = view.me or {}, view.opp or {}

    option_ids, base_option_features = BASE.encode_options(
        view, BASE_FEATURE_VERSION)
    # The engine temporarily retains knocked-out Pokemon while resolving
    # multi-target damage and KO prompts.  Their public ``hp`` can therefore
    # be negative (overkill damage), while BASE feature 79 is documented and
    # consumed as an HP fraction.  Qu-v2A's board and relational encoders
    # already map that state to zero HP; normalize the inherited copy to the
    # same [0, 1] contract without changing Qu-v1's frozen feature semantics.
    hp_fraction = base_option_features[:, BASE_OPTION_HP_FRACTION_INDEX]
    np.clip(hp_fraction, 0.0, 1.0, out=hp_fraction)
    target_ids, option_extra = _encode_option_extras(view, option_ids, deck)
    option_features = np.concatenate(
        [base_option_features, option_extra], axis=1).astype(np.float32, copy=False)
    if option_features.shape[1] != OPTION_FEATURES:
        raise AssertionError("Qu-v2A option feature width drifted")

    stadium = current.get("stadium")
    stadium_entries = stadium if isinstance(stadium, (list, tuple)) else [stadium]
    result = PublicFeatures(
        board_ids=board[0],
        board_energy_ids=board[1],
        board_tool_ids=board[2],
        board_evolution_ids=board[3],
        board_features=board[4],
        hand_ids=_copy_ids(me.get("hand"), HAND_SLOTS),
        my_discard_ids=_copy_ids(me.get("discard"), DISCARD_SLOTS),
        opponent_discard_ids=_copy_ids(opponent.get("discard"), DISCARD_SLOTS),
        looking_ids=_copy_ids(current.get("looking"), LOOKING_SLOTS),
        stadium_ids=_copy_ids(stadium_entries, STADIUM_SLOTS),
        prompt_ids=_prompt_ids(view.select),
        prompt_features=_prompt_features(view, deck),
        registered_deck_ids=deck,
        option_ids=option_ids.astype(np.int32, copy=False),
        option_target_ids=target_ids,
        option_features=option_features,
        option_mask=np.ones(len(option_ids), dtype=np.bool_),
    )
    validate_public_features(result)
    return result


def feature_fingerprint(features: PublicFeatures) -> bytes:
    """Canonical byte material useful for deterministic research manifests."""
    chunks = [SCHEMA.encode("ascii")]
    for item in fields(features):
        array = np.ascontiguousarray(getattr(features, item.name))
        chunks.extend((item.name.encode("ascii"), array.dtype.str.encode("ascii"),
                       repr(array.shape).encode("ascii"), array.tobytes()))
    return b"\0".join(chunks)
