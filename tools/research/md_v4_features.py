"""Research-only MD-v4 public resource and current-log features.

This module extends the frozen Qu-v2A public observation without changing its
schema or implementation.  Every output is a pure function of the actor's
current observation and the exact registered Grimmsnarl deck.  In particular,
there is no module-global game state and no cross-callback history.

Nothing under :mod:`agent` imports this module.  A separately reviewed runtime
copy may be vendored only after the MD-v4 gameplay gates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from agent.obsview import (
    AREA_ACTIVE,
    AREA_BENCH,
    AREA_DECK,
    AREA_DISCARD,
    AREA_HAND,
    AREA_LOOKING,
)
from tools.research import qu_v2a_features as QF


SCHEMA = "ptcg.md-v4.public-resource-window.v1"
EXPECTED_BASE_SCHEMA = "ptcg.qu-v2a.public-relational.v4"
if QF.SCHEMA != EXPECTED_BASE_SCHEMA:
    raise RuntimeError(
        f"MD-v4 requires Qu-v2A schema {EXPECTED_BASE_SCHEMA}, got {QF.SCHEMA}"
    )

TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
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
RESOURCE_CARD_IDS = (
    7, 104, 112, 646, 647, 648, 860, 1079, 1080, 1086,
    1097, 1122, 1137, 1152, 1182, 1219, 1227, 1231, 1259,
)
_TARGET_COUNTS = Counter(TARGET_DECK)

RESOURCE_ROWS = 19
RESOURCE_FEATURES = 22
RESOURCE_PROMPT_FEATURES = 2

LOG_SLOTS = 64
LOG_CARD_SLOTS = 4
LOG_AREA_SLOTS = 2
LOG_FEATURES = 8
LOG_PROMPT_FEATURES = 3

# These are the fourteen numeric event types observed in the frozen 350-game
# exact-mirror validation cohort.  They are normalized to a compact, versioned
# enum rather than used as direct embedding indices.
KNOWN_NUMERIC_EVENT_TYPES = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 15, 16,
)
EVENT_PAD = 0
EVENT_UNKNOWN = 1
EVENT_TYPE_TO_ENUM = {
    raw_type: index + 2
    for index, raw_type in enumerate(KNOWN_NUMERIC_EVENT_TYPES)
}
EVENT_ENUM_TO_TYPE = {
    normalized: raw_type
    for raw_type, normalized in EVENT_TYPE_TO_ENUM.items()
}
EVENT_VOCAB_SIZE = len(EVENT_TYPE_TO_ENUM) + 2

ROLE_PAD = 0
ROLE_SELF = 1
ROLE_OPPONENT = 2
ROLE_NONE_OR_UNKNOWN = 3

REDACTED_RAW_EVENT_TYPES = frozenset((5, 7))
DRAW_RAW_EVENT_TYPE = 4
MAX_ATTACK_ID = 1556
ATTACK_VOCAB_SIZE = MAX_ATTACK_ID + 1  # zero is canonical PAD/absent
MAX_AREA_ID = AREA_LOOKING

_EXACT_HIDDEN_KEY = "_counterfactual_exact_hidden_v1"
_BASE_FIELD_NAMES = tuple(item.name for item in fields(QF.PublicFeatures))
_ROOT_INPUT_KEYS = ("current", "select", "logs")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DEPENDENCY_PATHS = (
    "tools/research/md_v4_features.py",
    "tools/research/qu_v2a_features.py",
    "agent/features.py",
    "agent/obsview.py",
    "agent/cards.py",
    "data/cards.json",
    "data/attacks.json",
)


class PublicFeatureError(ValueError):
    """The MD-v4 encoder received malformed, incompatible, or private input."""


@dataclass(frozen=True)
class LogSanitizationStats:
    raw_event_count: int
    retained_event_count: int
    dropped_event_count: int
    unknown_event_count: int
    removed_identity_count: int
    numeric_type_counts: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_event_count": self.raw_event_count,
            "retained_event_count": self.retained_event_count,
            "dropped_event_count": self.dropped_event_count,
            "unknown_event_count": self.unknown_event_count,
            "removed_identity_count": self.removed_identity_count,
            "numeric_type_counts": {
                str(raw_type): count
                for raw_type, count in self.numeric_type_counts
            },
        }


@dataclass(frozen=True)
class PublicResourceWindowFeatures:
    # Frozen Qu-v2A tensors.
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

    # MD-v4 tensors.
    resource_ids: np.ndarray
    resource_features: np.ndarray
    resource_prompt_features: np.ndarray
    log_event_type: np.ndarray
    log_actor_role: np.ndarray
    log_card_ids: np.ndarray
    log_attack_ids: np.ndarray
    log_areas: np.ndarray
    log_features: np.ndarray
    log_mask: np.ndarray
    log_prompt_features: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        """Return defensive copies for audits, batching, and fingerprints."""
        return {
            item.name: np.array(getattr(self, item.name), copy=True)
            for item in fields(self)
        }

    def base_features(self) -> QF.PublicFeatures:
        """Reconstruct the nominal Qu-v2A record without changing any bytes."""
        return QF.PublicFeatures(**{
            name: getattr(self, name)
            for name in _BASE_FIELD_NAMES
        })


def compute_feature_dependency_hashes() -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for relative in FEATURE_DEPENDENCY_PATHS:
        path = _REPOSITORY_ROOT / relative
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"MD-v4 feature dependency is unavailable: {relative}"
            ) from error
        result.append((relative, hashlib.sha256(payload).hexdigest()))
    return tuple(result)


def feature_dependency_fingerprint(
    hashes: Sequence[tuple[str, str]] | None = None,
) -> str:
    manifest = (
        compute_feature_dependency_hashes()
        if hashes is None
        else tuple(hashes)
    )
    payload = b"ptcg.md-v4.feature-dependencies.v1\0" + b"\0".join(
        relative.encode("utf-8") + b"\0" + digest.encode("ascii")
        for relative, digest in manifest
    )
    return hashlib.sha256(payload).hexdigest()


FEATURE_DEPENDENCY_HASHES = compute_feature_dependency_hashes()
FEATURE_DEPENDENCY_FINGERPRINT = feature_dependency_fingerprint(
    FEATURE_DEPENDENCY_HASHES
)


def assert_feature_dependency_lock() -> str:
    current = compute_feature_dependency_hashes()
    if current != FEATURE_DEPENDENCY_HASHES:
        raise RuntimeError(
            "MD-v4 feature dependencies changed after import; restart and "
            "bump the feature schema before creating or loading artifacts"
        )
    if QF.assert_feature_dependency_lock() != QF.FEATURE_DEPENDENCY_FINGERPRINT:
        raise RuntimeError("frozen Qu-v2A feature dependency lock drifted")
    return FEATURE_DEPENDENCY_FINGERPRINT


def _valid_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _valid_card_id(value: Any) -> bool:
    return _valid_int(value) and 0 < int(value) < QF.EXPECTED_CARD_VOCAB


def _entry_card_id(entry: Any) -> int:
    value = entry.get("id") if isinstance(entry, Mapping) else entry
    return int(value) if _valid_card_id(value) else 0


def _entry_serial(entry: Any) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("serial")
    return int(value) if _valid_int(value) and int(value) >= 0 else None


def _canonical_deck(deck: Sequence[int]) -> tuple[int, ...]:
    if isinstance(deck, (str, bytes)) or not isinstance(deck, Sequence):
        raise PublicFeatureError("registered deck must be a sequence")
    values = list(deck)
    if len(values) != 60 or not all(_valid_card_id(value) for value in values):
        raise PublicFeatureError(
            "registered deck must contain exactly 60 valid integer card IDs"
        )
    return tuple(sorted(int(value) for value in values))


def supports_deck(deck: Sequence[int]) -> bool:
    try:
        return _canonical_deck(deck) == TARGET_DECK
    except (PublicFeatureError, TypeError, ValueError):
        return False


def _contains_exact_hidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            key == _EXACT_HIDDEN_KEY or _contains_exact_hidden(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_exact_hidden(item) for item in value)
    return False


def _public_root(obs: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(obs, Mapping):
        raise PublicFeatureError("observation must be a mapping")
    if _contains_exact_hidden(obs):
        raise PublicFeatureError("exact-hidden research metadata reached MD-v4")
    # Deliberately do not forward transport, visualization, timing, replay, or
    # outcome metadata into the frozen base encoder.
    return {
        "current": obs.get("current"),
        "select": obs.get("select"),
        "logs": obs.get("logs", []),
    }


def _role(player_index: Any, actor_index: int) -> int:
    if not _valid_int(player_index) or int(player_index) not in (0, 1):
        return ROLE_NONE_OR_UNKNOWN
    return ROLE_SELF if int(player_index) == actor_index else ROLE_OPPONENT


def _bounded_card_identity(value: Any) -> int:
    return int(value) if _valid_card_id(value) else 0


def _bounded_attack_identity(value: Any) -> int:
    return (
        int(value)
        if _valid_int(value) and 0 < int(value) <= MAX_ATTACK_ID
        else 0
    )


def _bounded_area(value: Any) -> int:
    return (
        int(value)
        if _valid_int(value) and AREA_DECK <= int(value) <= MAX_AREA_ID
        else 0
    )


def _finite_number(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)) \
            and not isinstance(value, (bool, np.bool_)):
        result = float(value)
        return result if np.isfinite(result) else 0.0
    return 0.0


def _identity_payload_count(event: Mapping[str, Any]) -> int:
    result = 0
    for key in (
        "cardId", "cardIdTarget", "cardIdBefore", "cardIdActive",
        "cardIdAfter", "cardIdBench",
    ):
        result += int(_bounded_card_identity(event.get(key)) != 0)
    result += int(_bounded_attack_identity(event.get("attackId")) != 0)
    return result


def sanitize_log_window(
    logs: Any,
    actor_index: int,
) -> tuple[dict[str, np.ndarray], LogSanitizationStats]:
    """Sanitize and left-align the most recent 64 events in engine order."""
    if actor_index not in (0, 1):
        raise PublicFeatureError("actor index must be 0 or 1")
    if logs is None:
        raw_events: list[Any] = []
    elif isinstance(logs, (list, tuple)):
        raw_events = list(logs)
    else:
        raise PublicFeatureError("observation logs must be an array")

    event_type = np.zeros(LOG_SLOTS, dtype=np.int32)
    actor_role = np.zeros(LOG_SLOTS, dtype=np.int32)
    card_ids = np.zeros((LOG_SLOTS, LOG_CARD_SLOTS), dtype=np.int32)
    attack_ids = np.zeros(LOG_SLOTS, dtype=np.int32)
    areas = np.zeros((LOG_SLOTS, LOG_AREA_SLOTS), dtype=np.int32)
    numeric = np.zeros((LOG_SLOTS, LOG_FEATURES), dtype=np.float32)
    mask = np.zeros(LOG_SLOTS, dtype=np.bool_)

    raw_count = len(raw_events)
    dropped = max(raw_count - LOG_SLOTS, 0)
    retained = raw_events[-LOG_SLOTS:]
    raw_type_counts: Counter[int] = Counter()
    unknown_count = 0
    removed_identity_count = 0

    for slot, raw_event in enumerate(retained):
        original_index = dropped + slot
        mask[slot] = True
        numeric[slot, 6] = np.float32(
            min((original_index + 1) / 255.0, 1.0)
        )
        numeric[slot, 7] = 1.0

        event = raw_event if isinstance(raw_event, Mapping) else {}
        raw_type = event.get("type")
        known_numeric = (
            _valid_int(raw_type)
            and int(raw_type) in EVENT_TYPE_TO_ENUM
        )
        if known_numeric:
            normalized = EVENT_TYPE_TO_ENUM[int(raw_type)]
            raw_type_counts[int(raw_type)] += 1
        else:
            normalized = EVENT_UNKNOWN
            unknown_count += 1
        event_type[slot] = normalized
        actor_role[slot] = _role(event.get("playerIndex"), actor_index)

        redact = (
            not known_numeric
            or int(raw_type) in REDACTED_RAW_EVENT_TYPES
            or (
                int(raw_type) == DRAW_RAW_EVENT_TYPE
                and actor_role[slot] != ROLE_SELF
            )
        )
        if redact:
            removed_identity_count += _identity_payload_count(event)
            numeric[slot, 5] = 1.0
        else:
            card_ids[slot, 0] = _bounded_card_identity(event.get("cardId"))
            card_ids[slot, 1] = _bounded_card_identity(
                event.get("cardIdTarget")
            )
            before = _bounded_card_identity(event.get("cardIdBefore"))
            card_ids[slot, 2] = (
                before
                if before
                else _bounded_card_identity(event.get("cardIdActive"))
            )
            after = _bounded_card_identity(event.get("cardIdAfter"))
            card_ids[slot, 3] = (
                after
                if after
                else _bounded_card_identity(event.get("cardIdBench"))
            )
            attack_ids[slot] = _bounded_attack_identity(event.get("attackId"))

        if known_numeric:
            areas[slot, 0] = _bounded_area(event.get("fromArea"))
            areas[slot, 1] = _bounded_area(event.get("toArea"))
            numeric[slot, 0] = np.float32(
                np.clip(_finite_number(event.get("value")), -340.0, 340.0)
                / 340.0
            )
            numeric[slot, 1] = 1.0 if event.get("putDamageCounter") is True else 0.0
            numeric[slot, 2] = 1.0 if event.get("hasBasicPokemon") is True else 0.0
            numeric[slot, 3] = 1.0 if event.get("head") is True else 0.0
            numeric[slot, 4] = 1.0 if event.get("isRecover") is True else 0.0

        # Unknown events lose all numeric payload except their actor role and
        # fixed position/sentinel fields. Reverse events lose identities but
        # may retain non-identity schema fields if malformed input supplies
        # them; this is conservative and does not reveal hidden cards.

    prompt = np.asarray([
        min(raw_count, 255),
        min(dropped, 255),
        1 if dropped else 0,
    ], dtype=np.float32)
    arrays = {
        "log_event_type": event_type,
        "log_actor_role": actor_role,
        "log_card_ids": card_ids,
        "log_attack_ids": attack_ids,
        "log_areas": areas,
        "log_features": numeric,
        "log_mask": mask,
        "log_prompt_features": prompt,
    }
    stats = LogSanitizationStats(
        raw_event_count=raw_count,
        retained_event_count=len(retained),
        dropped_event_count=dropped,
        unknown_event_count=unknown_count,
        removed_identity_count=removed_identity_count,
        numeric_type_counts=tuple(sorted(raw_type_counts.items())),
    )
    return arrays, stats


@dataclass
class _ResourceAccumulator:
    features: np.ndarray
    self_visible_by_serial: dict[int, int]
    missing_self_serial: bool = False
    contradictory_self_serial: bool = False


def _new_accumulator() -> _ResourceAccumulator:
    result = np.zeros((RESOURCE_ROWS, RESOURCE_FEATURES), dtype=np.float32)
    for row, card_id in enumerate(RESOURCE_CARD_IDS):
        result[row, 0] = float(_TARGET_COUNTS[card_id])
    return _ResourceAccumulator(result, {})


_RESOURCE_ROW_BY_ID = {
    card_id: row for row, card_id in enumerate(RESOURCE_CARD_IDS)
}


def _add_zone_entry(
    accumulator: _ResourceAccumulator,
    entry: Any,
    *,
    feature_column: int,
    self_owned: bool,
    visible_for_bounds: bool,
) -> None:
    card_id = _entry_card_id(entry)
    row = _RESOURCE_ROW_BY_ID.get(card_id)
    if row is not None:
        accumulator.features[row, feature_column] += 1.0
    if not self_owned or not visible_for_bounds or card_id == 0:
        return
    serial = _entry_serial(entry)
    if serial is None:
        accumulator.missing_self_serial = True
        return
    previous = accumulator.self_visible_by_serial.get(serial)
    if previous is None:
        accumulator.self_visible_by_serial[serial] = card_id
    elif previous != card_id:
        accumulator.contradictory_self_serial = True


def _entries(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()


def _board_entries(player: Mapping[str, Any]) -> tuple[Any, ...]:
    return _entries(player.get("active")) + _entries(player.get("bench"))


def _collect_player_zones(
    accumulator: _ResourceAccumulator,
    player: Mapping[str, Any],
    *,
    self_owned: bool,
) -> None:
    if self_owned:
        for entry in _entries(player.get("hand")):
            _add_zone_entry(
                accumulator, entry, feature_column=1, self_owned=True,
                visible_for_bounds=True,
            )
    for entry in _entries(player.get("discard")):
        _add_zone_entry(
            accumulator, entry, feature_column=2 if self_owned else 3,
            self_owned=self_owned, visible_for_bounds=self_owned,
        )
    for pokemon in _board_entries(player):
        if not isinstance(pokemon, Mapping):
            continue
        _add_zone_entry(
            accumulator, pokemon, feature_column=4 if self_owned else 5,
            self_owned=self_owned, visible_for_bounds=self_owned,
        )
        for key in ("energyCards", "tools"):
            for entry in _entries(pokemon.get(key)):
                _add_zone_entry(
                    accumulator, entry,
                    feature_column=6 if self_owned else 7,
                    self_owned=self_owned, visible_for_bounds=self_owned,
                )
        for entry in _entries(pokemon.get("preEvolution")):
            _add_zone_entry(
                accumulator, entry, feature_column=8 if self_owned else 9,
                self_owned=self_owned, visible_for_bounds=self_owned,
            )


def _nonnegative_count(value: Any) -> tuple[int, bool]:
    if _valid_int(value):
        number = int(value)
        return min(max(number, 0), 60), 0 <= number <= 60
    return 0, False


def _prize_count(player: Mapping[str, Any]) -> tuple[int, bool]:
    prize = player.get("prize")
    if isinstance(prize, (list, tuple)):
        return min(len(prize), 60), len(prize) <= 60
    return 0, False


def _exact_deck_counts(
    select: Mapping[str, Any],
    deck_count: int,
    visible_counts: Counter[int],
) -> Counter[int] | None:
    reveal = select.get("deck")
    if not isinstance(reveal, (list, tuple)) or len(reveal) != deck_count:
        return None
    card_ids = [_entry_card_id(entry) for entry in reveal]
    if any(card_id == 0 for card_id in card_ids):
        raise PublicFeatureError("full select.deck reveal has an invalid card ID")
    counts = Counter(card_ids)
    if (
        sum(counts.values()) != deck_count
        or any(card_id not in _TARGET_COUNTS for card_id in counts)
        or any(
            count + int(visible_counts.get(card_id, 0))
            > int(_TARGET_COUNTS[card_id])
            for card_id, count in counts.items()
        )
    ):
        raise PublicFeatureError(
            "full select.deck reveal is incompatible with registration"
        )
    return counts


def _encode_resources(
    public_obs: Mapping[str, Any],
    base: QF.PublicFeatures,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = public_obs.get("current")
    select = public_obs.get("select")
    if not isinstance(current, Mapping) or not isinstance(select, Mapping):
        raise PublicFeatureError("resource encoder requires current/select mappings")
    players = current.get("players")
    actor = current.get("yourIndex")
    if (
        not isinstance(players, list)
        or len(players) != 2
        or actor not in (0, 1)
        or not all(isinstance(player, Mapping) for player in players)
    ):
        raise PublicFeatureError("resource encoder received invalid players")
    me = players[int(actor)]
    opponent = players[1 - int(actor)]
    accumulator = _new_accumulator()
    _collect_player_zones(accumulator, me, self_owned=True)
    _collect_player_zones(accumulator, opponent, self_owned=False)

    stadium = current.get("stadium")
    stadium_entries = _entries(stadium)
    if isinstance(stadium, Mapping):
        stadium_entries = (stadium,)
    for entry in stadium_entries:
        owner = (
            int(entry.get("playerIndex"))
            if isinstance(entry, Mapping)
            and _valid_int(entry.get("playerIndex"))
            and int(entry.get("playerIndex")) in (0, 1)
            else None
        )
        if owner is None:
            continue
        self_owned = owner == int(actor)
        _add_zone_entry(
            accumulator, entry, feature_column=10 if self_owned else 11,
            self_owned=self_owned, visible_for_bounds=self_owned,
        )
    for entry in _entries(current.get("looking")):
        _add_zone_entry(
            accumulator, entry, feature_column=12, self_owned=True,
            visible_for_bounds=True,
        )

    deck_count, deck_count_valid = _nonnegative_count(me.get("deckCount"))
    prize_count, prize_count_valid = _prize_count(me)
    visible_total = len(accumulator.self_visible_by_serial)
    raw_unaccounted = 60 - visible_total - deck_count - prize_count
    unaccounted = min(max(raw_unaccounted, 0), 60)
    accounting_valid = (
        deck_count_valid
        and prize_count_valid
        and not accumulator.missing_self_serial
        and not accumulator.contradictory_self_serial
        and 0 <= raw_unaccounted <= 60
    )

    visible_counts = Counter(accumulator.self_visible_by_serial.values())
    exact_deck = _exact_deck_counts(select, deck_count, visible_counts)
    for row, card_id in enumerate(RESOURCE_CARD_IDS):
        registered = int(_TARGET_COUNTS[card_id])
        visible = int(visible_counts.get(card_id, 0))
        if visible > registered:
            accounting_valid = False
        hidden = max(registered - visible, 0)
        accumulator.features[row, 13] = float(hidden)

        deck_lower = max(hidden - prize_count - unaccounted, 0)
        deck_upper = min(hidden, deck_count)
        prize_lower = max(hidden - deck_count - unaccounted, 0)
        prize_upper = min(hidden, prize_count)

        if exact_deck is not None:
            exact = int(exact_deck.get(card_id, 0))
            if exact > hidden:
                accounting_valid = False
            exact = min(exact, registered)
            accumulator.features[row, 18] = float(exact)
            accumulator.features[row, 19] = 1.0
            deck_lower = deck_upper = exact
            remaining = max(hidden - exact, 0)
            prize_lower = max(remaining - unaccounted, 0)
            prize_upper = min(remaining, prize_count)

        accumulator.features[row, 14] = float(deck_lower)
        accumulator.features[row, 15] = float(deck_upper)
        accumulator.features[row, 16] = float(prize_lower)
        accumulator.features[row, 17] = float(prize_upper)

    subject_counts = Counter(
        int(card_id) for card_id in base.option_ids[:-1] if int(card_id) > 0
    )
    target_counts = Counter(
        int(card_id)
        for card_id in base.option_target_ids[:-1]
        if int(card_id) > 0
    )
    for row, card_id in enumerate(RESOURCE_CARD_IDS):
        accumulator.features[row, 20] = float(subject_counts.get(card_id, 0))
        accumulator.features[row, 21] = float(target_counts.get(card_id, 0))

    prompt = np.asarray([
        unaccounted,
        1 if accounting_valid else 0,
    ], dtype=np.float32)
    return (
        np.asarray(RESOURCE_CARD_IDS, dtype=np.int32),
        accumulator.features,
        prompt,
    )


def _require_array(
    sample: PublicResourceWindowFeatures,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    value = getattr(sample, name, None)
    if not isinstance(value, np.ndarray):
        raise PublicFeatureError(f"{name} is not a NumPy array")
    if value.shape != shape:
        raise PublicFeatureError(
            f"{name} has shape {value.shape}, expected {shape}"
        )
    if value.dtype != dtype:
        raise PublicFeatureError(
            f"{name} has dtype {value.dtype}, expected {dtype}"
        )
    if not value.flags.c_contiguous:
        raise PublicFeatureError(f"{name} is not C-contiguous")
    return value


def _validate_resource_features(sample: PublicResourceWindowFeatures) -> None:
    ids = _require_array(
        sample, "resource_ids", (RESOURCE_ROWS,), np.dtype(np.int32)
    )
    if not np.array_equal(ids, np.asarray(RESOURCE_CARD_IDS, dtype=np.int32)):
        raise PublicFeatureError("resource IDs are not in canonical deck order")
    values = _require_array(
        sample, "resource_features",
        (RESOURCE_ROWS, RESOURCE_FEATURES), np.dtype(np.float32),
    )
    if not np.isfinite(values).all() or np.any(values < 0.0) \
            or np.any(values > 4096.0):
        raise PublicFeatureError("resource features are non-finite/out of range")
    prompt = _require_array(
        sample, "resource_prompt_features",
        (RESOURCE_PROMPT_FEATURES,), np.dtype(np.float32),
    )
    if (
        not np.isfinite(prompt).all()
        or not 0.0 <= float(prompt[0]) <= 60.0
        or float(prompt[1]) not in (0.0, 1.0)
    ):
        raise PublicFeatureError("resource prompt features are invalid")

    expected_copies = np.asarray(
        [_TARGET_COUNTS[card_id] for card_id in RESOURCE_CARD_IDS],
        dtype=np.float32,
    )
    if not np.array_equal(values[:, 0], expected_copies):
        raise PublicFeatureError("registered resource-copy counts drifted")
    for row in values:
        registered = float(row[0])
        hidden = float(row[13])
        deck_lower, deck_upper = float(row[14]), float(row[15])
        prize_lower, prize_upper = float(row[16]), float(row[17])
        exact, exact_known = float(row[18]), float(row[19])
        unaccounted = float(prompt[0])
        if (
            hidden > registered
            or not 0.0 <= deck_lower <= deck_upper <= registered
            or not 0.0 <= prize_lower <= prize_upper <= registered
            or deck_lower + prize_lower > hidden
            or deck_upper + prize_upper + unaccounted < hidden
            or exact_known not in (0.0, 1.0)
            or exact > registered
            or (
                exact_known == 1.0
                and (deck_lower != exact or deck_upper != exact)
            )
            or (exact_known == 0.0 and exact != 0.0)
        ):
            raise PublicFeatureError("resource-bound invariant failed")


def _validate_log_features(sample: PublicResourceWindowFeatures) -> None:
    event_type = _require_array(
        sample, "log_event_type", (LOG_SLOTS,), np.dtype(np.int32)
    )
    actor_role = _require_array(
        sample, "log_actor_role", (LOG_SLOTS,), np.dtype(np.int32)
    )
    card_ids = _require_array(
        sample, "log_card_ids",
        (LOG_SLOTS, LOG_CARD_SLOTS), np.dtype(np.int32),
    )
    attack_ids = _require_array(
        sample, "log_attack_ids", (LOG_SLOTS,), np.dtype(np.int32)
    )
    areas = _require_array(
        sample, "log_areas",
        (LOG_SLOTS, LOG_AREA_SLOTS), np.dtype(np.int32),
    )
    numeric = _require_array(
        sample, "log_features",
        (LOG_SLOTS, LOG_FEATURES), np.dtype(np.float32),
    )
    mask = _require_array(
        sample, "log_mask", (LOG_SLOTS,), np.dtype(np.bool_)
    )
    prompt = _require_array(
        sample, "log_prompt_features",
        (LOG_PROMPT_FEATURES,), np.dtype(np.float32),
    )

    if np.any(event_type < EVENT_PAD) or np.any(event_type >= EVENT_VOCAB_SIZE):
        raise PublicFeatureError("log event enum is out of range")
    if np.any(actor_role < ROLE_PAD) or np.any(actor_role > ROLE_NONE_OR_UNKNOWN):
        raise PublicFeatureError("log actor role is out of range")
    if np.any(card_ids < 0) or np.any(card_ids >= QF.EXPECTED_CARD_VOCAB):
        raise PublicFeatureError("log card identity is out of range")
    if np.any(attack_ids < 0) or np.any(attack_ids > MAX_ATTACK_ID):
        raise PublicFeatureError("log attack identity is out of range")
    if np.any(areas < 0) or np.any(areas > MAX_AREA_ID):
        raise PublicFeatureError("log area is out of range")
    if not np.isfinite(numeric).all():
        raise PublicFeatureError("log numeric features are non-finite")
    if np.any(numeric[:, 0] < -1.0) or np.any(numeric[:, 0] > 1.0):
        raise PublicFeatureError("log signed values are out of range")
    if not np.isin(
        numeric[:, [1, 2, 3, 4, 5, 7]],
        np.asarray([0.0, 1.0], dtype=np.float32),
    ).all():
        raise PublicFeatureError("log boolean features are not binary")
    if np.any(numeric[:, 6] < 0.0) or np.any(numeric[:, 6] > 1.0):
        raise PublicFeatureError("log positions are out of range")
    if np.any(mask[1:] & ~mask[:-1]):
        raise PublicFeatureError("log mask is not left-aligned")

    padded = ~mask
    if (
        np.any(event_type[padded] != 0)
        or np.any(actor_role[padded] != 0)
        or np.any(card_ids[padded] != 0)
        or np.any(attack_ids[padded] != 0)
        or np.any(areas[padded] != 0)
        or np.any(numeric[padded] != 0.0)
    ):
        raise PublicFeatureError("log padding is not canonical zero padding")
    if (
        np.any(event_type[mask] == EVENT_PAD)
        or np.any(actor_role[mask] == ROLE_PAD)
        or np.any(numeric[mask, 7] != 1.0)
    ):
        raise PublicFeatureError("retained log rows lack schema presence")

    for slot in np.flatnonzero(mask):
        normalized = int(event_type[slot])
        raw_type = EVENT_ENUM_TO_TYPE.get(normalized)
        redacted = (
            normalized == EVENT_UNKNOWN
            or raw_type in REDACTED_RAW_EVENT_TYPES
            or (
                raw_type == DRAW_RAW_EVENT_TYPE
                and int(actor_role[slot]) != ROLE_SELF
            )
        )
        if redacted and (
            np.any(card_ids[slot] != 0) or attack_ids[slot] != 0
        ):
            raise PublicFeatureError("redacted log row retains an identity")
        if normalized == EVENT_UNKNOWN and (
            np.any(areas[slot] != 0)
            or np.any(numeric[slot, :5] != 0.0)
            or numeric[slot, 5] != 1.0
        ):
            raise PublicFeatureError("unknown log row retained a payload")

    if (
        not np.isfinite(prompt).all()
        or np.any(prompt < 0.0)
        or prompt[0] > 255.0
        or prompt[1] > 255.0
        or prompt[2] not in (0.0, 1.0)
        or not float(prompt[0]).is_integer()
        or not float(prompt[1]).is_integer()
    ):
        raise PublicFeatureError("log prompt features are invalid")
    raw_count = int(prompt[0])
    dropped = int(prompt[1])
    retained_count = int(mask.sum())
    if raw_count < 255:
        expected_dropped = max(raw_count - LOG_SLOTS, 0)
        if dropped != expected_dropped or retained_count != min(raw_count, LOG_SLOTS):
            raise PublicFeatureError("log count/truncation metadata is inconsistent")
    else:
        if not 191 <= dropped <= 255 or retained_count != LOG_SLOTS:
            raise PublicFeatureError("clipped log count metadata is inconsistent")
    if bool(prompt[2]) != bool(dropped):
        raise PublicFeatureError("log truncation flag is inconsistent")


def validate_public_features(sample: PublicResourceWindowFeatures) -> None:
    if not isinstance(sample, PublicResourceWindowFeatures):
        raise PublicFeatureError("expected an MD-v4 feature record")
    try:
        QF.validate_public_features(sample.base_features())
    except (QF.PublicFeatureError, TypeError, ValueError) as error:
        raise PublicFeatureError(f"frozen Qu-v2A feature failure: {error}") from error
    if tuple(int(value) for value in sample.registered_deck_ids) != TARGET_DECK:
        raise PublicFeatureError("MD-v4 base registration is not the exact deck")
    _validate_resource_features(sample)
    _validate_log_features(sample)


def _encode_public_root(
    public_obs: Mapping[str, Any],
    canonical_registered_deck: tuple[int, ...],
) -> PublicResourceWindowFeatures:
    """Pure shared core for offline materialization and runtime callbacks."""
    if canonical_registered_deck != TARGET_DECK:
        raise PublicFeatureError("MD-v4 is scoped to the exact registered deck")
    try:
        base = QF.encode_public_observation(
            public_obs, canonical_registered_deck
        )
    except (QF.PublicFeatureError, TypeError, ValueError) as error:
        raise PublicFeatureError(f"frozen Qu-v2A encoder rejected input: {error}") from error
    current = public_obs.get("current")
    if not isinstance(current, Mapping) or current.get("yourIndex") not in (0, 1):
        raise PublicFeatureError("MD-v4 observation has no actor index")
    resource_ids, resource_features, resource_prompt = _encode_resources(
        public_obs, base
    )
    log_arrays, _ = sanitize_log_window(
        public_obs.get("logs"), int(current["yourIndex"])
    )
    result = PublicResourceWindowFeatures(
        **base.arrays(),
        resource_ids=resource_ids,
        resource_features=resource_features,
        resource_prompt_features=resource_prompt,
        **log_arrays,
    )
    validate_public_features(result)
    return result


def encode_public_observation(
    obs: Mapping[str, Any],
    registered_learner_deck: Sequence[int],
) -> PublicResourceWindowFeatures:
    """Offline/materialization entrypoint for one exact-deck actor prompt."""
    canonical_deck = _canonical_deck(registered_learner_deck)
    return _encode_public_root(_public_root(obs), canonical_deck)


def encode_runtime_observation(
    obs: Mapping[str, Any],
    registered_learner_deck: Sequence[int],
) -> PublicResourceWindowFeatures:
    """Research runtime-callback entrypoint using the identical pure core.

    This distinct boundary lets the shadow audit exercise callback wiring
    independently from replay materialization while guaranteeing that neither
    path can accumulate state or reinterpret the feature schema.
    """
    canonical_deck = _canonical_deck(registered_learner_deck)
    runtime_public_root = _public_root(obs)
    return _encode_public_root(runtime_public_root, canonical_deck)


def feature_fingerprint(sample: PublicResourceWindowFeatures) -> bytes:
    validate_public_features(sample)
    chunks = [SCHEMA.encode("ascii")]
    for item in fields(sample):
        array = np.ascontiguousarray(getattr(sample, item.name))
        chunks.extend((
            item.name.encode("ascii"),
            array.dtype.str.encode("ascii"),
            repr(array.shape).encode("ascii"),
            array.tobytes(),
        ))
    return b"\0".join(chunks)


__all__ = [
    "ATTACK_VOCAB_SIZE",
    "EVENT_ENUM_TO_TYPE",
    "EVENT_PAD",
    "EVENT_TYPE_TO_ENUM",
    "EVENT_UNKNOWN",
    "EVENT_VOCAB_SIZE",
    "FEATURE_DEPENDENCY_FINGERPRINT",
    "FEATURE_DEPENDENCY_HASHES",
    "FEATURE_DEPENDENCY_PATHS",
    "KNOWN_NUMERIC_EVENT_TYPES",
    "LOG_AREA_SLOTS",
    "LOG_CARD_SLOTS",
    "LOG_FEATURES",
    "LOG_PROMPT_FEATURES",
    "LOG_SLOTS",
    "LogSanitizationStats",
    "PublicFeatureError",
    "PublicResourceWindowFeatures",
    "RESOURCE_CARD_IDS",
    "RESOURCE_FEATURES",
    "RESOURCE_PROMPT_FEATURES",
    "RESOURCE_ROWS",
    "ROLE_NONE_OR_UNKNOWN",
    "ROLE_OPPONENT",
    "ROLE_PAD",
    "ROLE_SELF",
    "SCHEMA",
    "TARGET_DECK",
    "TARGET_DECK_SHA256",
    "assert_feature_dependency_lock",
    "compute_feature_dependency_hashes",
    "encode_public_observation",
    "encode_runtime_observation",
    "feature_dependency_fingerprint",
    "feature_fingerprint",
    "sanitize_log_window",
    "supports_deck",
    "validate_public_features",
]
