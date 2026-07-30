"""Run the locked MD-v4 public resource/log feature shadow audit.

The audit is research-only.  It encodes every deployable callback in the
locked 350-game validation mirror cohort, then runs expensive mutation and
state-independence checks on the preselected golden callbacks.  It neither
trains a model nor authorizes packaging or upload.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from agent.obsview import ST_MAIN  # noqa: E402
from tools import il_dataset  # noqa: E402
from tools.research import lock_md_v4_shadow_audit as LOCK  # noqa: E402
from tools.research import md_v4_features as FEATURES  # noqa: E402
from tools.research import qu_v2a_features as BASE_QF  # noqa: E402
from tools.research import train_qu_v2a as TRAIN  # noqa: E402


DEFAULT_LOCK = LOCK.OUTPUT
DEFAULT_OUTPUT = LOCK.RUN / "shadow-audit-result.json"
_EXACT_HIDDEN_KEY = "_counterfactual_exact_hidden_v1"
_IDENTITY_FIELDS = (
    "cardId", "cardIdTarget", "cardIdBefore", "cardIdActive",
    "cardIdAfter", "cardIdBench",
)
_SANITIZER_ARRAYS = (
    "log_event_type",
    "log_actor_role",
    "log_card_ids",
    "log_attack_ids",
    "log_areas",
    "log_features",
    "log_mask",
    "log_prompt_features",
)
_MD_V4_FIXED_SHAPES = {
    "resource_ids": ("int32", (19,)),
    "resource_features": ("float32", (19, 22)),
    "resource_prompt_features": ("float32", (2,)),
    "log_event_type": ("int32", (64,)),
    "log_actor_role": ("int32", (64,)),
    "log_card_ids": ("int32", (64, 4)),
    "log_attack_ids": ("int32", (64,)),
    "log_areas": ("int32", (64, 2)),
    "log_features": ("float32", (64, 8)),
    "log_mask": ("bool", (64,)),
    "log_prompt_features": ("float32", (3,)),
}


class AuditError(RuntimeError):
    """The locked population or audit implementation drifted."""


def _artifact_paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    raw = lock.get("artifacts")
    if not isinstance(raw, Mapping):
        raise AuditError("shadow-audit lock has no artifact map")
    result: dict[str, Path] = {}
    for label, record in raw.items():
        if not isinstance(label, str) or not isinstance(record, Mapping):
            raise AuditError("shadow-audit lock has a malformed artifact row")
        value = record.get("path")
        expected = record.get("sha256")
        if not isinstance(value, str) or not isinstance(expected, str):
            raise AuditError(f"shadow-audit artifact {label} is malformed")
        path = Path(value).expanduser().resolve()
        if not path.is_file() or LOCK.file_sha256(path) != expected:
            raise AuditError(f"shadow-audit artifact drift: {label}")
        result[label] = path
    required = {
        "corpus", "contract", "features", "runner", "feature_tests",
        "shadow_tests", "lock_builder", "imitation_loader", "corpus_indexer",
        "corpus_loader",
    }
    if set(result) != required:
        raise AuditError(
            f"shadow-audit artifact keys drifted: {sorted(result)}")
    if result["runner"] != Path(__file__).resolve():
        raise AuditError("shadow-audit lock names another runner")
    if result["features"] != Path(FEATURES.__file__).resolve():
        raise AuditError("shadow-audit lock names another feature module")
    return result


def _feature_identity(lock: Mapping[str, Any]) -> None:
    expected = lock.get("feature_contract")
    if not isinstance(expected, Mapping):
        raise AuditError("shadow-audit lock has no feature contract")
    try:
        fingerprint = FEATURES.assert_feature_dependency_lock()
    except Exception as error:
        raise AuditError(f"MD-v4 feature dependency lock failed: {error}") from error
    if (
        expected.get("schema") != FEATURES.SCHEMA
        or expected.get("dependency_fingerprint") != fingerprint
    ):
        raise AuditError("MD-v4 feature identity differs from the lock")


def _selected_games(
    lock: Mapping[str, Any],
    corpus_path: Path,
) -> tuple[TRAIN.CorpusPlan, tuple[TRAIN.LockedGame, ...]]:
    plan = TRAIN.load_corpus_plan(
        corpus_path, required_splits=("validation",))
    cohort = lock.get("cohort")
    if not isinstance(cohort, Mapping):
        raise AuditError("shadow-audit lock has no cohort")
    if (
        plan.manifest_sha256 != cohort.get("source_manifest_sha256")
        or plan.corpus_content_sha256
        != cohort.get("source_corpus_content_sha256")
    ):
        raise AuditError("shadow-audit corpus identity drifted")
    games = LOCK._selected_games(
        plan, target_deck_sha256=LOCK.TARGET_DECK_SHA256)
    locked_rows = cohort.get("rows")
    if not isinstance(locked_rows, list) or len(locked_rows) != len(games):
        raise AuditError("shadow-audit locked rows are malformed")
    for index, (game, row) in enumerate(zip(games, locked_rows)):
        expected = {
            "index": index,
            "game_uid": game.game_uid,
            "episode_id": game.episode_id,
            "split_rank": game.split_rank,
            "content_sha256": game.content_sha256,
            "decision_count": game.decision_count,
            "registered_deck_sha256s": list(game.registered_deck_sha256s),
        }
        if not isinstance(row, Mapping) or any(
                row.get(name) != value for name, value in expected.items()):
            raise AuditError(f"shadow-audit locked game row {index} drifted")
    return plan, games


def _arrays(record: Any) -> dict[str, np.ndarray]:
    if is_dataclass(record):
        values = {
            item.name: getattr(record, item.name)
            for item in fields(record)
        }
    else:
        exporter = getattr(record, "arrays", None)
        if not callable(exporter):
            raise ValueError("MD-v4 feature record has no arrays() method")
        values = exporter()
    if not isinstance(values, Mapping):
        raise ValueError("MD-v4 arrays() did not return a mapping")
    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not isinstance(value, np.ndarray):
            raise ValueError("MD-v4 arrays() contains a malformed entry")
        result[name] = value
    return result


def _fingerprint(record: Any) -> bytes:
    value = FEATURES.feature_fingerprint(record)
    if not isinstance(value, bytes):
        raise ValueError("MD-v4 feature_fingerprint must return bytes")
    return value


def _same_features(left: Any, right: Any) -> bool:
    left_arrays = _arrays(left)
    right_arrays = _arrays(right)
    return (
        set(left_arrays) == set(right_arrays)
        and all(
            left_arrays[name].dtype == right_arrays[name].dtype
            and left_arrays[name].shape == right_arrays[name].shape
            and np.array_equal(left_arrays[name], right_arrays[name])
            for name in left_arrays
        )
        and _fingerprint(left) == _fingerprint(right)
    )


def _base_tensor_parity(
    md_v4_record: Any,
    observation: Mapping[str, Any],
    deck: Sequence[int],
) -> bool:
    """Compare every frozen Qu-v2 tensor with an independent base encode."""
    base_method = getattr(md_v4_record, "base_features", None)
    if not callable(base_method):
        return False
    try:
        projected = base_method()
        direct = BASE_QF.encode_public_observation(observation, deck)
        left = projected.arrays()
        right = direct.arrays()
    except Exception:
        return False
    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and set(left) == set(right)
        and all(
            isinstance(left[name], np.ndarray)
            and isinstance(right[name], np.ndarray)
            and left[name].dtype == right[name].dtype
            and left[name].shape == right[name].shape
            and np.array_equal(left[name], right[name])
            for name in left
        )
    )


def _tensor_contract_failures(arrays: Mapping[str, np.ndarray]) -> list[str]:
    failures: list[str] = []
    for name, (dtype_name, shape) in _MD_V4_FIXED_SHAPES.items():
        value = arrays.get(name)
        expected_dtype = (
            np.dtype(np.bool_) if dtype_name == "bool" else np.dtype(dtype_name)
        )
        if not isinstance(value, np.ndarray):
            failures.append(f"{name}:missing")
            continue
        if value.dtype != expected_dtype:
            failures.append(
                f"{name}:dtype:{value.dtype.str}!={expected_dtype.str}")
        if value.shape != shape:
            failures.append(f"{name}:shape:{value.shape}!={shape}")
        if not value.flags.c_contiguous:
            failures.append(f"{name}:not-c-contiguous")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            failures.append(f"{name}:nonfinite")
    return failures


def _resource_invariant_failures(arrays: Mapping[str, np.ndarray]) -> list[str]:
    table = arrays.get("resource_features")
    prompt = arrays.get("resource_prompt_features")
    if (
        not isinstance(table, np.ndarray)
        or table.shape != (19, 22)
        or not isinstance(prompt, np.ndarray)
        or prompt.shape != (2,)
    ):
        return ["resource tensors unavailable"]
    failures: list[str] = []
    registered = table[:, 0]
    hidden = table[:, 13]
    deck_lower, deck_upper = table[:, 14], table[:, 15]
    prize_lower, prize_upper = table[:, 16], table[:, 17]
    exact_deck, exact_known = table[:, 18], table[:, 19]
    count_columns = np.concatenate((table[:, :19], table[:, 20:22]), axis=1)
    if not np.isfinite(table).all() or np.any(count_columns < 0.0):
        failures.append("resource table has nonfinite or negative counts")
    if np.any(np.abs(count_columns - np.rint(count_columns)) > 1e-6):
        failures.append("resource table count columns are not integral")
    if np.any((exact_known != 0.0) & (exact_known != 1.0)):
        failures.append("exact-deck-known flag is not binary")
    if np.any(hidden > registered) or np.any(exact_deck > registered):
        failures.append("hidden or exact-deck count exceeds registration")
    if np.any(deck_lower > deck_upper) or np.any(deck_upper > registered):
        failures.append("deck bounds violate 0<=lower<=upper<=registered")
    if np.any(prize_lower > prize_upper) or np.any(prize_upper > registered):
        failures.append("prize bounds violate 0<=lower<=upper<=registered")
    if np.any(deck_lower + prize_lower > hidden):
        failures.append("lower-bound sum exceeds hidden count")
    unaccounted = float(prompt[0])
    valid = float(prompt[1])
    if (
        not math.isfinite(unaccounted)
        or unaccounted < 0.0
        or unaccounted > 60.0
        or abs(unaccounted - round(unaccounted)) > 1e-6
    ):
        failures.append("unaccounted_count is invalid")
    if valid != 1.0:
        failures.append("resource-accounting-valid flag is not one")
    if np.any(deck_upper + prize_upper + unaccounted < hidden - 1e-6):
        failures.append("upper-bound sum plus unaccounted misses hidden count")
    known = exact_known == 1.0
    if np.any(deck_lower[known] != exact_deck[known]) \
            or np.any(deck_upper[known] != exact_deck[known]):
        failures.append("exact deck reveal did not collapse deck bounds")
    return failures


def _update_array_statistics(
    statistics: MutableMapping[str, dict[str, Any]],
    arrays: Mapping[str, np.ndarray],
) -> None:
    for name, value in arrays.items():
        row = statistics.setdefault(name, {
            "dtypes": Counter(),
            "shapes": Counter(),
            "records": 0,
            "finite_failures": 0,
            "c_contiguous_failures": 0,
            "minimum": None,
            "maximum": None,
        })
        row["dtypes"][value.dtype.str] += 1
        row["shapes"][repr(value.shape)] += 1
        row["records"] += 1
        row["c_contiguous_failures"] += int(not value.flags.c_contiguous)
        if value.size and value.dtype.kind in "fiub":
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                row["finite_failures"] += 1
            else:
                minimum = float(np.min(value))
                maximum = float(np.max(value))
                row["minimum"] = (
                    minimum if row["minimum"] is None
                    else min(float(row["minimum"]), minimum)
                )
                row["maximum"] = (
                    maximum if row["maximum"] is None
                    else max(float(row["maximum"]), maximum)
                )


def _serialize_array_statistics(
    statistics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        name: {
            **{
                key: value
                for key, value in row.items()
                if key not in ("dtypes", "shapes")
            },
            "dtypes": dict(sorted(row["dtypes"].items())),
            "shapes": dict(sorted(row["shapes"].items())),
        }
        for name, row in sorted(statistics.items())
    }


def _stats_mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        result = value.as_dict()
    elif is_dataclass(value):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = value
    else:
        result = vars(value) if hasattr(value, "__dict__") else {}
    return result if isinstance(result, Mapping) else {}


def _merge_stats(
    totals: MutableMapping[str, Any],
    prefix: str,
    value: Any,
) -> None:
    if isinstance(value, bool):
        totals[prefix] = int(totals.get(prefix, 0)) + int(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        totals[prefix] = totals.get(prefix, 0) + value
    elif isinstance(value, Mapping):
        for name, nested in value.items():
            _merge_stats(totals, f"{prefix}.{name}" if prefix else str(name), nested)


def _select_deck_audit(
    observation: Mapping[str, Any],
    seat: int,
) -> tuple[bool, bool]:
    select = observation.get("select")
    current = observation.get("current")
    if not isinstance(select, Mapping) or not isinstance(current, Mapping):
        return False, True
    players = current.get("players")
    if not isinstance(players, list) or len(players) != 2 \
            or not isinstance(players[seat], Mapping):
        return False, True
    revealed = select.get("deck")
    if not isinstance(revealed, list):
        return False, False
    deck_count = players[seat].get("deckCount")
    return True, (
        isinstance(deck_count, bool)
        or not isinstance(deck_count, int)
        or len(revealed) != deck_count
    )


def _input_privacy_violations(
    observation: Mapping[str, Any],
    seat: int,
) -> list[str]:
    failures: list[str] = []
    current = observation.get("current")
    if not isinstance(current, Mapping):
        return ["current_not_mapping"]
    players = current.get("players")
    if not isinstance(players, list) or len(players) != 2 \
            or not all(isinstance(player, Mapping) for player in players):
        return ["players_not_two_mappings"]
    if current.get("yourIndex") != seat:
        failures.append("actor_seat_mismatch")
    if players[1 - seat].get("hand") is not None:
        failures.append("opponent_hand_visible")
    if any("deck" in player for player in players):
        failures.append("hidden_deck_present")
    if _EXACT_HIDDEN_KEY in observation:
        failures.append("exact_hidden_key_present")
    return failures


def _sanitizer_arrays(
    logs: Any,
    seat: int,
) -> tuple[dict[str, np.ndarray], Any]:
    result = FEATURES.sanitize_log_window(logs, seat)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("sanitize_log_window returned an invalid result")
    arrays, statistics = result
    if not isinstance(arrays, Mapping):
        raise ValueError("sanitize_log_window arrays are not a mapping")
    checked = {
        name: value
        for name, value in arrays.items()
        if isinstance(name, str) and isinstance(value, np.ndarray)
    }
    if set(checked) != set(_SANITIZER_ARRAYS):
        raise ValueError(
            f"sanitizer array keys drifted: {sorted(checked)}")
    return checked, statistics


def _redaction_failures(
    logs: Sequence[Any],
    seat: int,
    sanitized: Mapping[str, np.ndarray],
) -> tuple[int, int]:
    mask = sanitized["log_mask"]
    slots = np.flatnonzero(mask)
    raw_window = list(logs[-64:])
    if len(slots) != len(raw_window):
        return 1, 0
    failures = 0
    removed = 0
    card_ids = sanitized["log_card_ids"]
    attack_ids = sanitized["log_attack_ids"]
    for slot, raw in zip(slots.tolist(), raw_window):
        if not isinstance(raw, Mapping):
            continue
        event_type = raw.get("type")
        actor = raw.get("playerIndex")
        must_redact = (
            event_type in (5, 7)
            or (event_type == 4 and actor != seat)
            or event_type in ("DrawReverse", "MoveCardReverse")
        )
        if not must_redact:
            continue
        removed += sum(
            int(isinstance(raw.get(name), int) and not isinstance(raw.get(name), bool)
                and raw.get(name) != 0)
            for name in _IDENTITY_FIELDS
        )
        removed += int(
            isinstance(raw.get("attackId"), int)
            and not isinstance(raw.get("attackId"), bool)
            and raw.get("attackId") != 0
        )
        if np.any(card_ids[int(slot)] != 0) or attack_ids[int(slot)] != 0:
            failures += 1
    return failures, removed


def _swap_absolute_seats(observation: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(observation))
    current = result.get("current")
    if not isinstance(current, dict):
        raise ValueError("cannot swap an observation without current state")
    old = current.get("yourIndex")
    players = current.get("players")
    if old not in (0, 1) or not isinstance(players, list) or len(players) != 2:
        raise ValueError("cannot swap malformed absolute seats")
    current["players"] = [players[1], players[0]]
    current["yourIndex"] = 1 - int(old)
    if current.get("firstPlayer") in (0, 1):
        current["firstPlayer"] = 1 - int(current["firstPlayer"])
    if current.get("result") in (0, 1):
        current["result"] = 1 - int(current["result"])

    def rewrite(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("playerIndex") in (0, 1):
                value["playerIndex"] = 1 - int(value["playerIndex"])
            for child in value.values():
                rewrite(child)
        elif isinstance(value, list):
            for child in value:
                rewrite(child)

    rewrite(current["players"])
    rewrite(current.get("stadium"))
    rewrite(current.get("looking"))
    rewrite(result.get("select"))
    rewrite(result.get("logs"))
    return result


def _ignored_root_mutation(observation: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(observation))
    result.update({
        "visualize": {
            "audit_canary": "must-not-enter-features",
            "players": [{"hand": [1], "deck": [2]}, {"hand": [3], "deck": [4]}],
        },
        "search_begin_input": "audit-mutated-private-engine-state",
        "remainingOverageTime": -123456.5,
        "step": 2_147_483_647,
        "final_reward": 1,
        "future_action": [999],
        "episode_id": "audit-episode-canary",
        "agent_name": "audit-agent-canary",
        "team_name": "audit-team-canary",
        "rank": 1,
        "rating": 9999,
        "data_source": "audit-source-canary",
        "wall_clock_date": "2099-01-01",
        "_audit_opponent_hidden_fixture": {"card_ids": [1, 2, 3]},
    })
    return result


def _serial_bijection(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Rename every public serial consistently; serial identity is not a feature."""
    result = copy.deepcopy(dict(observation))

    def rewrite(value: Any) -> None:
        if isinstance(value, dict):
            serial = value.get("serial")
            if isinstance(serial, int) and not isinstance(serial, bool):
                value["serial"] = int(serial) + 1_000_003
            for child in value.values():
                rewrite(child)
        elif isinstance(value, list):
            for child in value:
                rewrite(child)

    rewrite(result)
    return result


def _hidden_mutations(
    observation: Mapping[str, Any],
    seat: int,
) -> Iterable[tuple[str, dict[str, Any]]]:
    opponent_hand = copy.deepcopy(dict(observation))
    opponent_hand["current"]["players"][1 - seat]["hand"] = [{"id": 1}]
    yield "opponent_hand", opponent_hand

    hidden_deck = copy.deepcopy(dict(observation))
    hidden_deck["current"]["players"][1 - seat]["deck"] = [{"id": 1}]
    yield "opponent_hidden_deck", hidden_deck

    self_hidden_deck = copy.deepcopy(dict(observation))
    self_hidden_deck["current"]["players"][seat]["deck"] = [{"id": 1}]
    yield "self_hidden_deck", self_hidden_deck

    exact_hidden = copy.deepcopy(dict(observation))
    exact_hidden[_EXACT_HIDDEN_KEY] = {"opponent_hand": [1]}
    yield "exact_hidden_key", exact_hidden


def _off_deck(registration: Sequence[int]) -> tuple[int, ...]:
    result = list(int(card) for card in registration)
    replacement = 1 if result[0] != 1 else 2
    result[0] = replacement
    return tuple(result)


def _malformed_full_deck_reveal(
    observation: Mapping[str, Any],
    seat: int,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(observation))
    current = result.get("current")
    select = result.get("select")
    if not isinstance(current, dict) or not isinstance(select, dict):
        raise ValueError("cannot inject a full reveal into a malformed callback")
    players = current.get("players")
    if (
        not isinstance(players, list)
        or len(players) != 2
        or not isinstance(players[seat], dict)
    ):
        raise ValueError("cannot inject a full reveal without actor state")
    deck_count = players[seat].get("deckCount")
    if (
        not isinstance(deck_count, int)
        or isinstance(deck_count, bool)
        or deck_count < 0
        or deck_count > 60
    ):
        raise ValueError("cannot inject a full reveal without a deck count")
    if deck_count == 0:
        deck_count = 1
        players[seat]["deckCount"] = deck_count
    # Card ID 1 is valid for the engine vocabulary but absent from the locked
    # exact Grimmsnarl registration.
    select["deck"] = [{"id": 1} for _ in range(deck_count)]
    return result


def _golden_checks(
    metamorphic: Sequence[tuple[Mapping[str, Any], tuple[int, ...], int]],
    runtime_keys: set[tuple[str, int, int, str]],
    metamorphic_identities: Sequence[tuple[str, int, int, str]],
    unrelated: tuple[Mapping[str, Any], tuple[int, ...]] | None,
) -> dict[str, int]:
    result = {
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
    if (
        not metamorphic
        or len(metamorphic) != len(metamorphic_identities)
        or unrelated is None
    ):
        raise AuditError("golden audit did not capture required callbacks")
    unrelated_obs, unrelated_deck = unrelated
    for (observation, deck, seat), identity in zip(
            metamorphic, metamorphic_identities):
        baseline = FEATURES.encode_public_observation(observation, deck)
        if identity in runtime_keys:
            transport = json.loads(LOCK.canonical_json(observation))
            if not _same_features(
                    baseline, FEATURES.encode_public_observation(transport, deck)):
                result["golden_transport_mismatches"] += 1
            try:
                runtime = FEATURES.encode_runtime_observation(observation, deck)
            except Exception:
                result["golden_runtime_adapter_mismatches"] += 1
            else:
                if not _same_features(baseline, runtime):
                    result["golden_runtime_adapter_mismatches"] += 1

        mutated = _ignored_root_mutation(observation)
        if not _same_features(
                baseline, FEATURES.encode_public_observation(mutated, deck)):
            result["golden_mutation_mismatches"] += 1
            result["forbidden_key_consumption"] += 1

        swapped = _swap_absolute_seats(observation)
        if not _same_features(
                baseline, FEATURES.encode_public_observation(swapped, deck)):
            result["golden_seat_swap_mismatches"] += 1
        renamed = _serial_bijection(observation)
        if not _same_features(
                baseline, FEATURES.encode_public_observation(renamed, deck)):
            result["golden_serial_bijection_mismatches"] += 1

        FEATURES.encode_public_observation(unrelated_obs, unrelated_deck)
        repeated = FEATURES.encode_public_observation(observation, deck)
        if not _same_features(baseline, repeated):
            result["golden_repeat_state_mismatches"] += 1

        for _label, hidden in _hidden_mutations(observation, seat):
            try:
                FEATURES.encode_public_observation(hidden, deck)
            except Exception:
                pass
            else:
                result["hidden_input_acceptances"] += 1
        malformed_reveal = _malformed_full_deck_reveal(observation, seat)
        try:
            FEATURES.encode_public_observation(malformed_reveal, deck)
        except Exception:
            pass
        else:
            result["malformed_full_deck_reveal_acceptances"] += 1
        try:
            FEATURES.encode_public_observation(observation, _off_deck(deck))
        except Exception:
            pass
        else:
            result["off_deck_acceptances"] += 1
    return result


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with destination.open("xb") as handle:
            handle.write(raw)
    except FileExistsError as error:
        raise AuditError(f"refusing to overwrite {destination}") from error


def run_audit(
    lock_path: Path,
) -> dict[str, Any]:
    lock_file = lock_path.expanduser().resolve()
    lock = LOCK.load_lock(lock_file)
    artifacts = _artifact_paths(lock)
    _feature_identity(lock)
    _plan, games = _selected_games(lock, artifacts["corpus"])

    locked_rows = lock["cohort"]["rows"]
    golden_contract = lock.get("golden_samples")
    if not isinstance(golden_contract, Mapping):
        raise AuditError("shadow-audit golden contract is malformed")
    metamorphic_rows = golden_contract.get("metamorphic_rows")
    runtime_rows = golden_contract.get("runtime_rows")
    if not isinstance(metamorphic_rows, list) or not isinstance(runtime_rows, list):
        raise AuditError("shadow-audit golden rows are malformed")

    def identity(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
        return (
            str(row.get("game_uid")),
            int(row.get("callback_ordinal")),
            int(row.get("seat")),
            str(row.get("observation_sha256")),
        )

    metamorphic_identities = [
        identity(row) for row in metamorphic_rows if isinstance(row, Mapping)
    ]
    runtime_identities = [
        identity(row) for row in runtime_rows if isinstance(row, Mapping)
    ]
    metamorphic_keys = set(metamorphic_identities)
    runtime_keys = set(runtime_identities)
    if (
        len(metamorphic_identities)
        != int(golden_contract["metamorphic_count"])
        or len(metamorphic_keys) != len(metamorphic_identities)
        or len(runtime_identities) != int(golden_contract["runtime_count"])
        or len(runtime_keys) != len(runtime_identities)
        or not runtime_keys.issubset(metamorphic_keys)
    ):
        raise AuditError("shadow-audit golden callback identities drifted")
    if list(runtime_identities) != metamorphic_identities[:len(runtime_identities)]:
        raise AuditError("runtime golden rows are not the locked prefix")

    gate_counts = {
        name: 0
        for name, expected in lock["pass_rule"].items()
        if expected == 0
    }
    gate_counts.update({
        "cohort_games": len(games),
        "cohort_callbacks": 0,
        "discovery_seat_observations": int(
            lock["cohort"]["discovery_census"]["seat_relative_observations"]),
        "discovery_log_events": int(
            lock["cohort"]["discovery_census"]["log_events"]),
        "callback_log_events": 0,
    })
    exception_types: Counter[str] = Counter()
    exception_examples: list[dict[str, Any]] = []
    privacy_violation_types: Counter[str] = Counter()
    raw_event_types: Counter[str] = Counter()
    normalized_event_types: Counter[str] = Counter()
    module_versions: Counter[str] = Counter()
    array_statistics: dict[str, dict[str, Any]] = {}
    sanitizer_totals: dict[str, Any] = {}
    encoded_digest = hashlib.sha256(
        b"ptcg.md-v4.shadow-audit-encoded-features.v1\0")
    callback_rows: list[dict[str, Any]] = []
    golden_captured: dict[
        tuple[str, int, int, str],
        tuple[Mapping[str, Any], tuple[int, ...], int],
    ] = {}
    unrelated: tuple[Mapping[str, Any], tuple[int, ...]] | None = None

    information = Counter()
    raw_counts = Counter()
    redacted_identities_removed = 0
    for game, locked_row in zip(games, locked_rows):
        raw = TRAIN._stable_locked_read(
            game.path, game.content_sha256,
            f"shadow-audit replay {game.game_uid}",
        )
        document = TRAIN._verify_replay_metadata(game, raw)
        module_version = document.get("module_version")
        module_versions[str(module_version)] += 1
        game_digest = hashlib.sha256(LOCK.GAME_DIGEST_DOMAIN)
        callback_count = 0
        for ordinal, (observation, _picks, _reward) in enumerate(
                il_dataset.iter_document(document)):
            callback_count += 1
            gate_counts["cohort_callbacks"] += 1
            current = (
                observation.get("current")
                if isinstance(observation, Mapping) else None
            )
            seat = (
                current.get("yourIndex")
                if isinstance(current, Mapping) else None
            )
            if seat not in (0, 1):
                gate_counts["malformed_feature_records"] += 1
                continue
            seat = int(seat)
            obs_hash = hashlib.sha256(
                LOCK.canonical_json(observation)).hexdigest()
            LOCK._framed(
                game_digest, game.game_uid, ordinal, seat, obs_hash)
            callback_key = (game.game_uid, ordinal, seat, obs_hash)
            deck = game.registered_decks[seat]
            if game.registered_deck_sha256s[seat] != LOCK.TARGET_DECK_SHA256:
                gate_counts["exact_deck_scope_failures"] += 1

            privacy = _input_privacy_violations(observation, seat)
            for name in privacy:
                privacy_violation_types[name] += 1
            if privacy:
                gate_counts["malformed_feature_records"] += 1
            select = observation.get("select")
            select_type = (
                select.get("type") if isinstance(select, Mapping) else None
            )
            is_main = select_type == ST_MAIN
            logs = observation.get("logs")
            if not isinstance(logs, list):
                logs = []
                gate_counts["malformed_feature_records"] += 1
            raw_counts["prompts"] += 1
            raw_counts["st_main_prompts"] += int(is_main)
            raw_counts["log_events"] += len(logs)
            gate_counts["callback_log_events"] += len(logs)
            raw_counts["nonempty_log_prompts"] += int(bool(logs))
            raw_counts["truncated_log_prompts"] += int(len(logs) > 64)
            raw_counts["dropped_log_events"] += max(len(logs) - 64, 0)
            for event in logs:
                raw_type = (
                    event.get("type") if isinstance(event, Mapping)
                    else f"<{type(event).__name__}>"
                )
                raw_event_types[str(raw_type)] += 1
            reveal, reveal_mismatch = _select_deck_audit(
                observation, seat)
            raw_counts["exact_deck_reveal_prompts"] += int(reveal)
            raw_counts["select_deck_length_mismatches"] += int(reveal_mismatch)
            gate_counts["select_deck_length_mismatches"] += int(reveal_mismatch)

            if unrelated is None and callback_key not in metamorphic_keys:
                unrelated = (copy.deepcopy(observation), tuple(deck))
            if callback_key in metamorphic_keys:
                golden_captured[callback_key] = (
                    copy.deepcopy(observation), tuple(deck), seat)

            try:
                encoded = FEATURES.encode_public_observation(observation, deck)
            except Exception as error:
                gate_counts["encoder_exceptions"] += 1
                exception_types[f"encode:{type(error).__name__}"] += 1
                if len(exception_examples) < 20:
                    exception_examples.append({
                        "game_uid": game.game_uid,
                        "callback_ordinal": ordinal,
                        "stage": "encode",
                        "error": repr(error),
                    })
                continue
            try:
                FEATURES.validate_public_features(encoded)
                arrays = _arrays(encoded)
            except Exception as error:
                gate_counts["malformed_feature_records"] += 1
                exception_types[f"validate:{type(error).__name__}"] += 1
                if len(exception_examples) < 20:
                    exception_examples.append({
                        "game_uid": game.game_uid,
                        "callback_ordinal": ordinal,
                        "stage": "validate",
                        "error": repr(error),
                    })
                continue

            if not _base_tensor_parity(encoded, observation, deck):
                gate_counts["base_qu_v2_tensor_mismatches"] += 1
            _update_array_statistics(array_statistics, arrays)
            tensor_failures = _tensor_contract_failures(arrays)
            if tensor_failures:
                gate_counts["nonfinite_or_shape_dtype_failures"] += 1
                exception_types["tensor_contract"] += 1
            resource_failures = _resource_invariant_failures(arrays)
            if resource_failures:
                gate_counts["resource_bound_failures"] += 1
                exception_types["resource_invariants"] += 1

            try:
                sanitized, sanitizer_stats = _sanitizer_arrays(logs, seat)
            except Exception as error:
                gate_counts["sanitizer_encoder_mismatches"] += 1
                exception_types[f"sanitizer:{type(error).__name__}"] += 1
            else:
                _merge_stats(
                    sanitizer_totals, "", _stats_mapping(sanitizer_stats))
                if any(
                    name not in arrays
                    or arrays[name].dtype != sanitized[name].dtype
                    or arrays[name].shape != sanitized[name].shape
                    or not np.array_equal(arrays[name], sanitized[name])
                    for name in _SANITIZER_ARRAYS
                ):
                    gate_counts["sanitizer_encoder_mismatches"] += 1
                redaction_failures, removed = _redaction_failures(
                    logs, seat, sanitized)
                gate_counts[
                    "opponent_draw_or_reverse_identity_failures"
                ] += redaction_failures
                redacted_identities_removed += removed
                for value in sanitized["log_event_type"][
                        sanitized["log_mask"]].tolist():
                    normalized_event_types[str(int(value))] += 1

            if is_main:
                information["st_main_prompts"] += 1
                resource = arrays["resource_features"]
                dynamic = (
                    np.any(resource[:, 1:13] != 0.0)
                    or np.any(resource[:, 20:22] != 0.0)
                )
                information["dynamic_public_resource"] += int(dynamic)
                information["nonempty_logs"] += int(bool(logs))
                information["truncated_logs"] += int(len(logs) > 64)
                information["exact_current_deck_reveal"] += int(reveal)
            fingerprint = _fingerprint(encoded)
            LOCK._framed(
                encoded_digest, game.game_uid, ordinal, seat, fingerprint)

        if callback_count != game.decision_count:
            raise AuditError(
                f"game {game.game_uid} callback count drifted during audit")
        game_callback_hash = game_digest.hexdigest()
        if game_callback_hash != locked_row["callback_observations_sha256"]:
            raise AuditError(
                f"game {game.game_uid} callback observation digest drifted")
        callback_rows.append({
            "game_uid": game.game_uid,
            "episode_id": game.episode_id,
            "split_rank": game.split_rank,
            "content_sha256": game.content_sha256,
            "decision_count": game.decision_count,
            "callback_observations_sha256": game_callback_hash,
        })

    if LOCK._ordered_cohort_digest(callback_rows) \
            != lock["cohort"]["ordered_callback_cohort_sha256"]:
        raise AuditError("ordered callback cohort digest drifted")
    if dict(sorted(module_versions.items())) \
            != lock["cohort"]["engine_module_version_games"]:
        raise AuditError("engine module-version population drifted")
    observed_callback_census = {
        "callbacks": int(raw_counts["prompts"]),
        "log_events": int(raw_counts["log_events"]),
        "empty_log_prompts": (
            int(raw_counts["prompts"]) - int(raw_counts["nonempty_log_prompts"])
        ),
        "truncated_log_prompts": int(raw_counts["truncated_log_prompts"]),
        "st_main_prompts": int(raw_counts["st_main_prompts"]),
        "select_deck_reveals": int(raw_counts["exact_deck_reveal_prompts"]),
    }
    if observed_callback_census != lock["cohort"]["callback_census"]:
        raise AuditError(
            f"callback census drifted: {observed_callback_census}")
    if len(golden_captured) != len(metamorphic_keys):
        raise AuditError(
            f"captured {len(golden_captured)} of "
            f"{len(metamorphic_keys)} metamorphic callbacks")
    metamorphic = [
        golden_captured[key] for key in metamorphic_identities
    ]
    golden_counts = _golden_checks(
        metamorphic, runtime_keys, metamorphic_identities, unrelated)
    for name, value in golden_counts.items():
        gate_counts[name] += int(value)

    # Presence in the actor view is distinct from consumption.  Any malformed
    # hidden state fails the encoder; ignored transport/private-root canaries
    # are tested byte-for-byte on the golden callbacks.
    gate_counts["forbidden_key_consumption"] += 0
    denominator = max(int(information["st_main_prompts"]), 1)
    availability = {
        "st_main_prompts": int(information["st_main_prompts"]),
        "dynamic_public_resource": {
            "prompts": int(information["dynamic_public_resource"]),
            "fraction": information["dynamic_public_resource"] / denominator,
        },
        "nonempty_logs": {
            "prompts": int(information["nonempty_logs"]),
            "fraction": information["nonempty_logs"] / denominator,
        },
        "truncated_logs": {
            "prompts": int(information["truncated_logs"]),
            "fraction": information["truncated_logs"] / denominator,
        },
        "exact_current_deck_reveal": {
            "prompts": int(information["exact_current_deck_reveal"]),
            "fraction": information["exact_current_deck_reveal"] / denominator,
        },
    }
    expected_rule = lock["pass_rule"]
    passed_by_gate = {
        name: gate_counts.get(name) == expected
        for name, expected in expected_rule.items()
    }
    passed = all(passed_by_gate.values())
    payload: dict[str, Any] = {
        "schema": LOCK.RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock": {
            "path": str(lock_file),
            "file_sha256": LOCK.file_sha256(lock_file),
            "lock_sha256": lock["lock_sha256"],
        },
        "feature_contract": dict(lock["feature_contract"]),
        "cohort": {
            "games": len(games),
            "discovery_census": {
                **dict(lock["cohort"]["discovery_census"]),
                "role": (
                    "locked seat-observation discovery census; not feature "
                    "examples or training callbacks"
                ),
            },
            "callback_census": observed_callback_census,
            "callbacks": int(raw_counts["prompts"]),
            "st_main_prompts": int(raw_counts["st_main_prompts"]),
            "log_events": int(raw_counts["log_events"]),
            "nonempty_log_prompts": int(raw_counts["nonempty_log_prompts"]),
            "truncated_log_prompts": int(raw_counts["truncated_log_prompts"]),
            "dropped_log_events": int(raw_counts["dropped_log_events"]),
            "exact_deck_reveal_prompts": int(
                raw_counts["exact_deck_reveal_prompts"]),
            "select_deck_length_mismatches": int(
                raw_counts["select_deck_length_mismatches"]),
            "ordered_callback_cohort_sha256": (
                lock["cohort"]["ordered_callback_cohort_sha256"]
            ),
            "encoded_features_sha256": encoded_digest.hexdigest(),
            "engine_module_version_games": dict(sorted(module_versions.items())),
        },
        "tensor_audit": {
            "records_expected": int(raw_counts["prompts"]),
            "array_statistics": _serialize_array_statistics(array_statistics),
        },
        "resource_audit": {
            "bound_failure_records": gate_counts["resource_bound_failures"],
            "information_availability": availability,
        },
        "log_audit": {
            "raw_event_types": dict(sorted(raw_event_types.items())),
            "normalized_event_types": dict(
                sorted(normalized_event_types.items(), key=lambda row: int(row[0]))
            ),
            "unknown_normalized_events": int(
                normalized_event_types.get(
                    str(int(getattr(FEATURES, "EVENT_UNKNOWN", 1))), 0)
            ),
            "sanitizer_statistics": dict(sorted(sanitizer_totals.items())),
            "redacted_input_identities_removed": redacted_identities_removed,
        },
        "privacy_audit": {
            "raw_input_violation_types": dict(
                sorted(privacy_violation_types.items())),
            "metamorphic_golden_callbacks": len(metamorphic),
            "runtime_golden_callbacks": len(runtime_keys),
            "ignored_root_mutation_fields": sorted(
                _ignored_root_mutation(metamorphic[0][0]).keys()
                - metamorphic[0][0].keys()
            ),
            "future_observations_actions_rewards_never_passed_to_encoder": True,
        },
        "exceptions": {
            "by_stage_and_type": dict(sorted(exception_types.items())),
            "examples": exception_examples,
            "examples_truncated": max(
                sum(exception_types.values()) - len(exception_examples), 0),
        },
        "gates": {
            "counts": gate_counts,
            "expected": expected_rule,
            "passed_by_gate": passed_by_gate,
            "passed": passed,
        },
        "promotion_authority": False,
    }
    payload["result_sha256"] = LOCK.value_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        payload = run_audit(args.lock)
        _write_new(args.out, payload)
    except (
        OSError, ValueError, TypeError, KeyError,
        TRAIN.TrainingError, LOCK.LockError, AuditError,
    ) as error:
        parser.error(str(error))
    print(json.dumps({
        "out": str(args.out.expanduser().resolve()),
        "file_sha256": LOCK.file_sha256(args.out.expanduser().resolve()),
        "result_sha256": payload["result_sha256"],
        "passed": payload["gates"]["passed"],
        "gate_counts": payload["gates"]["counts"],
    }, indent=2, sort_keys=True))
    return 0 if payload["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
