"""Research-only privileged features for a Qu-v2C asymmetric critic.

The deployable Qu-v2 actor remains public-only.  This module combines one
validated :class:`qu_v2a_features.PublicFeatures` record with fixed-width,
masked card-ID arrays from the exact-hidden terminal-oracle payload.  It is
tooling-only by design and must never be imported from :mod:`agent`.

Raw visualizations, logs, and serialized native search state are deliberately
not retained.  The serialized search input and public-root fingerprint are
used only to prove that the exact payload belongs to the supplied public root.
``strip_to_public`` then rebuilds a public record through an explicit whitelist
so adding a future privileged field cannot silently expose it to the actor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from agent import turn_search as TS
from tools.research import qu_v2a_features as QF


SCHEMA = "ptcg.qu-v2c.privileged-features.v1"
EXACT_PAYLOAD_SCHEMA = "ptcg.counterfactual.oracle.v1"

DECK_SLOTS = 60
PRIZE_SLOTS = 6
OPPONENT_HAND_SLOTS = 60
OPPONENT_ACTIVE_SLOTS = 1

PUBLIC_ARRAY_NAMES = (
    "board_ids",
    "board_energy_ids",
    "board_tool_ids",
    "board_evolution_ids",
    "board_features",
    "hand_ids",
    "my_discard_ids",
    "opponent_discard_ids",
    "looking_ids",
    "stadium_ids",
    "prompt_ids",
    "prompt_features",
    "registered_deck_ids",
    "option_ids",
    "option_target_ids",
    "option_features",
    "option_mask",
)
HIDDEN_ARRAY_NAMES = (
    "my_deck_ids",
    "my_deck_mask",
    "my_prize_ids",
    "my_prize_mask",
    "opponent_deck_ids",
    "opponent_deck_mask",
    "opponent_prize_ids",
    "opponent_prize_mask",
    "opponent_hand_ids",
    "opponent_hand_mask",
    "opponent_active_ids",
    "opponent_active_mask",
)
ARRAY_NAMES = tuple(
    f"public_{name}" for name in PUBLIC_ARRAY_NAMES
) + HIDDEN_ARRAY_NAMES

_EXACT_PAYLOAD_KEYS = frozenset({
    "schema",
    "selecting_player",
    "turn",
    "public_root_fingerprint",
    "search_begin_sha256",
    "my_deck",
    "my_prize",
    "opponent_deck",
    "opponent_prize",
    "opponent_hand",
    "opponent_active",
})
_HEX_DIGITS = frozenset("0123456789abcdef")


class PrivilegedFeatureError(ValueError):
    """A privileged feature or exact-root binding is malformed."""


@dataclass(frozen=True, slots=True)
class PrivilegedFeatures:
    """One immutable public actor sample plus fixed-width critic-only zones."""

    public: QF.PublicFeatures
    my_deck_ids: np.ndarray
    my_deck_mask: np.ndarray
    my_prize_ids: np.ndarray
    my_prize_mask: np.ndarray
    opponent_deck_ids: np.ndarray
    opponent_deck_mask: np.ndarray
    opponent_prize_ids: np.ndarray
    opponent_prize_mask: np.ndarray
    opponent_hand_ids: np.ndarray
    opponent_hand_mask: np.ndarray
    opponent_active_ids: np.ndarray
    opponent_active_mask: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        """Return validated defensive copies suitable for ``numpy.savez``."""
        validate_privileged_features(self)
        result = {
            f"public_{name}": np.array(
                getattr(self.public, name), copy=True, order="C")
            for name in PUBLIC_ARRAY_NAMES
        }
        result.update({
            name: np.array(getattr(self, name), copy=True, order="C")
            for name in HIDDEN_ARRAY_NAMES
        })
        return result

    def canonical_hash(self) -> str:
        """Return the canonical SHA-256 identity of this feature record."""
        return canonical_hash(self)


def _fingerprint_value(value: Any) -> Any:
    """Mirror the exact oracle's public JSON canonicalization."""
    if isinstance(value, Mapping):
        return tuple(sorted(
            (key, _fingerprint_value(item))
            for key, item in value.items() if key != "name"
        ))
    if isinstance(value, list):
        return tuple(_fingerprint_value(item) for item in value)
    return value if isinstance(value, (str, int, float, bool)) else None


def public_root_fingerprint(obs: Mapping[str, Any]) -> str:
    """Recompute the exact-oracle public-root binding without hidden state."""
    if not isinstance(obs, Mapping):
        raise PrivilegedFeatureError("public observation must be a mapping")
    current = obs.get("current")
    selecting = current.get("yourIndex") if isinstance(current, Mapping) else None
    if (not isinstance(selecting, int) or isinstance(selecting, bool)
            or selecting not in (0, 1)):
        raise PrivilegedFeatureError("public observation selecting seat is invalid")
    try:
        key = (
            current.get("turn"),
            current.get("turnActionCount"),
            current.get("result"),
            selecting,
            _fingerprint_value(current.get("looking")),
            TS.canonical_info_key(dict(obs), selecting),
        )
    except Exception as exc:
        raise PrivilegedFeatureError(
            "cannot fingerprint public root selection") from exc
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _copy_public(
        public: QF.PublicFeatures, *, readonly: bool,
) -> QF.PublicFeatures:
    """Copy only the deployable public whitelist, never dataclass fields."""
    QF.validate_public_features(public)
    copier = _readonly_copy if readonly else (
        lambda value: np.array(value, copy=True, order="C"))
    result = QF.PublicFeatures(
        board_ids=copier(public.board_ids),
        board_energy_ids=copier(public.board_energy_ids),
        board_tool_ids=copier(public.board_tool_ids),
        board_evolution_ids=copier(public.board_evolution_ids),
        board_features=copier(public.board_features),
        hand_ids=copier(public.hand_ids),
        my_discard_ids=copier(public.my_discard_ids),
        opponent_discard_ids=copier(public.opponent_discard_ids),
        looking_ids=copier(public.looking_ids),
        stadium_ids=copier(public.stadium_ids),
        prompt_ids=copier(public.prompt_ids),
        prompt_features=copier(public.prompt_features),
        registered_deck_ids=copier(public.registered_deck_ids),
        option_ids=copier(public.option_ids),
        option_target_ids=copier(public.option_target_ids),
        option_features=copier(public.option_features),
        option_mask=copier(public.option_mask),
    )
    QF.validate_public_features(result)
    return result


def _strict_count(player: Mapping[str, Any], name: str, maximum: int) -> int:
    value = player.get(name)
    if (not isinstance(value, int) or isinstance(value, bool)
            or not 0 <= value <= maximum):
        raise PrivilegedFeatureError(
            f"public {name} must be an integer in [0, {maximum}]")
    return int(value)


def _prize_count(player: Mapping[str, Any], label: str) -> int:
    prize = player.get("prize")
    if not isinstance(prize, list) or len(prize) > PRIZE_SLOTS:
        raise PrivilegedFeatureError(
            f"public {label} prize must be a list of at most {PRIZE_SLOTS} cards")
    return len(prize)


def _zone(
        payload: Mapping[str, Any], name: str, capacity: int, expected: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = payload.get(name)
    if not isinstance(raw, list):
        raise PrivilegedFeatureError(f"exact payload {name} must be a list")
    if len(raw) != expected:
        raise PrivilegedFeatureError(
            f"exact payload {name} has length {len(raw)}, expected {expected}")
    if len(raw) > capacity:
        raise PrivilegedFeatureError(
            f"exact payload {name} exceeds fixed capacity {capacity}")
    for index, value in enumerate(raw):
        if (not isinstance(value, int) or isinstance(value, bool)
                or not 0 < value < QF.EXPECTED_CARD_VOCAB):
            raise PrivilegedFeatureError(
                f"exact payload {name}[{index}] has an invalid card ID")
    ids = np.zeros(capacity, dtype=np.int32)
    mask = np.zeros(capacity, dtype=np.bool_)
    if raw:
        ids[:len(raw)] = np.asarray(raw, dtype=np.int32)
        mask[:len(raw)] = True
    return _readonly_copy(ids), _readonly_copy(mask)


def _validate_payload_identity(
        public_obs: Mapping[str, Any], exact_payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(exact_payload, Mapping):
        raise PrivilegedFeatureError("exact payload must be a mapping")
    keys = set(exact_payload)
    if keys != _EXACT_PAYLOAD_KEYS:
        missing = sorted(_EXACT_PAYLOAD_KEYS - keys, key=repr)
        extra = sorted(keys - _EXACT_PAYLOAD_KEYS, key=repr)
        raise PrivilegedFeatureError(
            f"exact payload keys differ from the strict schema; "
            f"missing={missing}, extra={extra}")
    if exact_payload.get("schema") != EXACT_PAYLOAD_SCHEMA:
        raise PrivilegedFeatureError("exact payload schema is not supported")

    current = public_obs.get("current")
    if not isinstance(current, Mapping):
        raise PrivilegedFeatureError("public observation current state is missing")
    players = current.get("players")
    selecting = current.get("yourIndex")
    if (not isinstance(selecting, int) or isinstance(selecting, bool)
            or selecting not in (0, 1)
            or not isinstance(players, list) or len(players) != 2
            or not all(isinstance(player, Mapping) for player in players)):
        raise PrivilegedFeatureError("public observation players are malformed")
    payload_selecting = exact_payload.get("selecting_player")
    public_turn = current.get("turn")
    payload_turn = exact_payload.get("turn")
    if (not isinstance(payload_selecting, int)
            or isinstance(payload_selecting, bool)
            or not isinstance(public_turn, int) or isinstance(public_turn, bool)
            or not isinstance(payload_turn, int) or isinstance(payload_turn, bool)
            or payload_selecting != selecting or payload_turn != public_turn):
        raise PrivilegedFeatureError(
            "exact payload seat/turn does not match the public observation")

    root_hash = exact_payload.get("public_root_fingerprint")
    if (not isinstance(root_hash, str) or len(root_hash) != 64
            or any(char not in _HEX_DIGITS for char in root_hash)
            or root_hash != public_root_fingerprint(public_obs)):
        raise PrivilegedFeatureError(
            "exact payload public-root fingerprint mismatch")

    serialized = public_obs.get("search_begin_input")
    if not isinstance(serialized, str):
        raise PrivilegedFeatureError(
            "public observation search_begin_input must be text, never bytes")
    search_hash = exact_payload.get("search_begin_sha256")
    expected_search_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if (not isinstance(search_hash, str) or len(search_hash) != 64
            or any(char not in _HEX_DIGITS for char in search_hash)
            or search_hash != expected_search_hash):
        raise PrivilegedFeatureError(
            "exact payload serialized-search fingerprint mismatch")
    return players[selecting], players[1 - selecting]


def encode_privileged_observation(
        public_obs: Mapping[str, Any],
        exact_payload: Mapping[str, Any],
        registered_learner_deck: Sequence[int],
) -> PrivilegedFeatures:
    """Encode a public actor sample and its bound exact-hidden critic state.

    ``logs`` and ``search_begin_input`` may exist on the source observation,
    as they do in the local engine.  They are not features: only the latter's
    SHA-256 is checked against the oracle payload, and neither is retained.
    """
    if not isinstance(public_obs, Mapping):
        raise PrivilegedFeatureError("public observation must be a mapping")
    me, opponent = _validate_payload_identity(public_obs, exact_payload)

    public = QF.encode_public_observation(
        public_obs, registered_learner_deck)
    public = _copy_public(public, readonly=True)

    my_deck = _zone(
        exact_payload, "my_deck", DECK_SLOTS,
        _strict_count(me, "deckCount", DECK_SLOTS))
    my_prize = _zone(
        exact_payload, "my_prize", PRIZE_SLOTS,
        _prize_count(me, "learner"))
    opponent_deck = _zone(
        exact_payload, "opponent_deck", DECK_SLOTS,
        _strict_count(opponent, "deckCount", DECK_SLOTS))
    opponent_prize = _zone(
        exact_payload, "opponent_prize", PRIZE_SLOTS,
        _prize_count(opponent, "opponent"))
    opponent_hand = _zone(
        exact_payload, "opponent_hand", OPPONENT_HAND_SLOTS,
        _strict_count(opponent, "handCount", OPPONENT_HAND_SLOTS))
    # The v1 exact oracle supports ST_MAIN only, where the opponent active is
    # already public.  Its hidden-active vector must therefore remain empty.
    opponent_active = _zone(
        exact_payload, "opponent_active", OPPONENT_ACTIVE_SLOTS, 0)

    result = PrivilegedFeatures(
        public=public,
        my_deck_ids=my_deck[0],
        my_deck_mask=my_deck[1],
        my_prize_ids=my_prize[0],
        my_prize_mask=my_prize[1],
        opponent_deck_ids=opponent_deck[0],
        opponent_deck_mask=opponent_deck[1],
        opponent_prize_ids=opponent_prize[0],
        opponent_prize_mask=opponent_prize[1],
        opponent_hand_ids=opponent_hand[0],
        opponent_hand_mask=opponent_hand[1],
        opponent_active_ids=opponent_active[0],
        opponent_active_mask=opponent_active[1],
    )
    validate_privileged_features(result)
    return result


def _validate_hidden_pair(
        features: PrivilegedFeatures, prefix: str, capacity: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids = getattr(features, f"{prefix}_ids", None)
    mask = getattr(features, f"{prefix}_mask", None)
    if (not isinstance(ids, np.ndarray) or ids.shape != (capacity,)
            or ids.dtype != np.dtype(np.int32) or not ids.flags.c_contiguous):
        raise PrivilegedFeatureError(
            f"{prefix}_ids must be C-contiguous int32[{capacity}]")
    if (not isinstance(mask, np.ndarray) or mask.shape != (capacity,)
            or mask.dtype != np.dtype(np.bool_) or not mask.flags.c_contiguous):
        raise PrivilegedFeatureError(
            f"{prefix}_mask must be C-contiguous bool[{capacity}]")
    if ids.flags.writeable or mask.flags.writeable:
        raise PrivilegedFeatureError(
            f"{prefix} arrays must be immutable")
    count = int(np.count_nonzero(mask))
    if (np.any(~mask[:count]) or np.any(mask[count:])
            or np.any(ids[~mask] != 0)):
        raise PrivilegedFeatureError(
            f"{prefix} mask/padding is not a canonical dense prefix")
    if np.any(ids[mask] <= 0) or np.any(
            ids[mask] >= QF.EXPECTED_CARD_VOCAB):
        raise PrivilegedFeatureError(
            f"{prefix} contains an out-of-range card ID")
    return ids, mask


def validate_privileged_features(features: PrivilegedFeatures) -> None:
    """Fail closed on schema, mutability, aliasing, shape, or padding drift."""
    if not isinstance(features, PrivilegedFeatures):
        raise PrivilegedFeatureError(
            "expected a Qu-v2C PrivilegedFeatures record")
    QF.validate_public_features(features.public)
    public_arrays = []
    for name in PUBLIC_ARRAY_NAMES:
        value = getattr(features.public, name)
        if value.flags.writeable:
            raise PrivilegedFeatureError(
                f"public {name} must be immutable inside a privileged record")
        public_arrays.append(value)

    hidden_arrays: list[np.ndarray] = []
    for prefix, capacity in (
            ("my_deck", DECK_SLOTS),
            ("my_prize", PRIZE_SLOTS),
            ("opponent_deck", DECK_SLOTS),
            ("opponent_prize", PRIZE_SLOTS),
            ("opponent_hand", OPPONENT_HAND_SLOTS),
            ("opponent_active", OPPONENT_ACTIVE_SLOTS)):
        hidden_arrays.extend(_validate_hidden_pair(features, prefix, capacity))

    for public in public_arrays:
        if any(np.shares_memory(public, hidden) for hidden in hidden_arrays):
            raise PrivilegedFeatureError(
                "public and privileged arrays must not alias memory")


def strip_to_public(features: PrivilegedFeatures) -> QF.PublicFeatures:
    """Return a defensive public-only copy through a fixed field whitelist."""
    validate_privileged_features(features)
    return _copy_public(features.public, readonly=False)


def _frame(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _canonical_array(array: np.ndarray) -> tuple[bytes, bytes]:
    if array.dtype == np.dtype(np.int32):
        canonical = np.ascontiguousarray(array.astype("<i4", copy=False))
        dtype = b"int32-le"
    elif array.dtype == np.dtype(np.float32):
        canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
        dtype = b"float32-le"
    elif array.dtype == np.dtype(np.bool_):
        canonical = np.ascontiguousarray(array.astype(np.uint8, copy=False))
        dtype = b"bool-u8"
    else:  # pragma: no cover - validation makes this unreachable.
        raise PrivilegedFeatureError(
            f"unsupported canonical dtype {array.dtype}")
    shape = struct.pack("<Q", canonical.ndim) + b"".join(
        struct.pack("<Q", int(size)) for size in canonical.shape)
    return dtype + b"\0" + shape, canonical.tobytes(order="C")


def canonical_hash(features: PrivilegedFeatures) -> str:
    """Hash only canonical numeric features, never transport/raw metadata."""
    validate_privileged_features(features)
    digest = hashlib.sha256()
    _frame(digest, SCHEMA.encode("ascii"))
    _frame(digest, QF.SCHEMA.encode("ascii"))
    for name in PUBLIC_ARRAY_NAMES:
        array = getattr(features.public, name)
        header, payload = _canonical_array(array)
        _frame(digest, f"public_{name}".encode("ascii"))
        _frame(digest, header)
        _frame(digest, payload)
    for name in HIDDEN_ARRAY_NAMES:
        array = getattr(features, name)
        header, payload = _canonical_array(array)
        _frame(digest, name.encode("ascii"))
        _frame(digest, header)
        _frame(digest, payload)
    return digest.hexdigest()
