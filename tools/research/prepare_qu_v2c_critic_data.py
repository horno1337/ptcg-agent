"""Prepare private, game-grouped factual data for the Qu-v2C critic.

The input must be a ``factual-critic`` root-mining artifact plus a complete
native-root validation report for that exact artifact.  Each logged real
action receives only its factual terminal game return.  This tool does not
invent counterfactual labels and its outputs are not actor-training data.

One pickle-free compressed NPZ is written per complete episode beneath the
deterministic ``train``, ``validation``, or ``test`` directory.  Every NPZ is
mode 0600 and contains both the public Qu-v2 representation and the
tooling-only privileged critic arrays.  The JSON manifest contains hashes,
counts, and contract metadata only; it never contains card or root IDs.

The strict :func:`load_game_npz` API opens one named game file at a time.  A
trainer can therefore select a checkpoint using train/validation files without
ever opening the physically separate test files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import policy  # noqa: E402
from agent import qu_v2_features as PRODUCTION_QF  # noqa: E402
from agent import turn_search as TS  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import qu_v2a_features as QF  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import qu_v2c_splits as SPLITS  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.critic-data.v1"
GAME_SCHEMA = "ptcg.qu-v2c.critic-game.v1"
DEFAULT_ROOT_DIR = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-roots-v1"
    / "factual-critic-ladder"
)
DEFAULT_OUT = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-critic-data-v1"
    / "factual-ladder"
)
DEFAULT_NATIVE_VALIDATION = "native-validation.json"
SPLIT_SEED = SPLITS.DEFAULT_SEED

PROTECTED_OUTPUT_TREES = tuple(
    (ROOT / name).resolve() for name in ("agent", "data", "decks")
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_VARIABLE_PUBLIC_ARRAYS = frozenset({
    "public_option_ids",
    "public_option_target_ids",
    "public_option_features",
    "public_option_mask",
})
_META_ARRAYS = (
    "schema",
    "feature_schema",
    "privileged_feature_schema",
    "split_schema",
    "split",
    "split_seed",
    "episode_id_sha256",
    "root_manifest_sha256",
    "native_validation_sha256",
    "record_count",
    "root_ids",
    "option_count",
    "chosen_index",
    "terminal_return",
    "game_weight",
)
GAME_ARRAY_NAMES = _META_ARRAYS + PF.ARRAY_NAMES
_MANIFEST_FORBIDDEN_KEYS = frozenset({
    *PF.HIDDEN_ARRAY_NAMES,
    "root_id",
    "root_ids",
    "exact_hidden_payload",
    "registered_learner_deck_cards",
})


class CriticDataError(RuntimeError):
    """A critic-data input, output, or serialized game failed closed."""


@dataclass(frozen=True, slots=True)
class CriticLabels:
    """Immutable labels and provenance returned beside one game's records."""

    split: str
    episode_id_sha256: str
    root_manifest_sha256: str
    native_validation_sha256: str
    root_ids: tuple[str, ...]
    option_counts: np.ndarray
    chosen_indices: np.ndarray
    terminal_returns: np.ndarray
    game_weights: np.ndarray


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CriticDataError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CriticDataError(f"{label} is not a JSON object")
    return value, _sha256_bytes(raw)


def _validate_embedded_checksum(
    value: Mapping[str, Any], checksum_key: str, label: str,
) -> str:
    recorded = value.get(checksum_key)
    without = dict(value)
    without.pop(checksum_key, None)
    if not _is_sha256(recorded) or recorded != _value_sha256(without):
        raise CriticDataError(f"{label} embedded checksum mismatch")
    return str(recorded)


def _atomic_bytes(path: Path, payload: bytes, mode: int) -> None:
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise CriticDataError(f"stale partial output exists: {temporary}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        os.chmod(path, mode)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _source_hashes() -> dict[str, str]:
    paths = {
        "preparer": Path(__file__).resolve(),
        "split_contract": Path(SPLITS.__file__).resolve(),
        "root_validator": Path(VALIDATE.__file__).resolve(),
        "root_miner": Path(MINE.__file__).resolve(),
        "privileged_features": Path(PF.__file__).resolve(),
        "research_public_features": Path(QF.__file__).resolve(),
        "production_public_features": Path(PRODUCTION_QF.__file__).resolve(),
        "semantic_mapping": Path(TS.__file__).resolve(),
        "engine_wrapper": ROOT / "tools" / "cabt.py",
        "cards_data": ROOT / "data" / "cards.json",
        "attacks_data": ROOT / "data" / "attacks.json",
        "registered_deck": ROOT / "decks" / "deck.csv",
    }
    return {label: _sha256_file(path) for label, path in paths.items()}


def _assert_root_source_lock(manifest: Mapping[str, Any]) -> None:
    recorded = manifest.get("source_files_sha256")
    if not isinstance(recorded, Mapping):
        raise CriticDataError("root manifest has no source-file hash lock")
    current = MINE._source_hashes()
    if dict(recorded) != current:
        raise CriticDataError(
            "root-mining source/data/engine hashes drifted")


def _load_native_validation(
    path: Path,
    root_manifest: Mapping[str, Any],
    public_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Mapping[str, Any]]]:
    report, file_sha256 = _load_json(path, "native validation")
    if report.get("schema") != VALIDATE.SCHEMA:
        raise CriticDataError("native validation schema mismatch")
    _validate_embedded_checksum(
        report, "report_sha256", "native validation")
    if report.get("root_manifest_sha256") != root_manifest.get(
            "manifest_sha256"):
        raise CriticDataError(
            "native validation names a different root manifest")

    count = len(public_records)
    if (report.get("requested_roots") != count
            or report.get("passed_roots") != count
            or report.get("all_roots_passed") is not True):
        raise CriticDataError(
            "native validation is not a complete all-roots pass")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != count:
        raise CriticDataError("native validation result count mismatch")

    by_id: dict[str, Mapping[str, Any]] = {}
    for expected, result in zip(public_records, results):
        if not isinstance(result, Mapping):
            raise CriticDataError("native validation result is malformed")
        root_id = result.get("root_id")
        if (root_id != expected.get("root_id")
                or not _is_sha256(root_id)
                or root_id in by_id):
            raise CriticDataError(
                "native validation root order/identity mismatch")
        if (result.get("native_begin_pass") is not True
                or result.get("semantic_round_trip_pass") is not True
                or result.get("complete_action_panel") is not True):
            raise CriticDataError(
                f"native validation did not pass root {root_id}")
        select = (
            (expected.get("public_observation") or {}).get("select")
            if isinstance(expected.get("public_observation"), Mapping)
            else None
        )
        options = select.get("option") if isinstance(select, Mapping) else None
        if (not isinstance(options, list)
                or result.get("root_options") != len(options)):
            raise CriticDataError(
                f"native option count drifted for root {root_id}")
        by_id[str(root_id)] = result

    engine = report.get("engine_library")
    if not isinstance(engine, Mapping):
        raise CriticDataError("native validation has no engine binding")
    engine_path_raw = engine.get("path")
    engine_sha = engine.get("sha256")
    if not isinstance(engine_path_raw, str) or not _is_sha256(engine_sha):
        raise CriticDataError("native validation engine binding is malformed")
    engine_path = Path(engine_path_raw).expanduser().resolve()
    if _sha256_file(engine_path) != engine_sha:
        raise CriticDataError("native validation engine library drifted")
    recorded_sources = root_manifest.get("source_files_sha256")
    if (not isinstance(recorded_sources, Mapping)
            or recorded_sources.get("engine_library") != engine_sha):
        raise CriticDataError(
            "native validation and root mining used different engines")
    return report, file_sha256, by_id


def _valid_episode_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise CriticDataError("root episode_id must be text or an integer")
    text = str(value)
    if not text:
        raise CriticDataError("root episode_id cannot be empty")
    return text


def _semantic_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_semantic_json(item) for item in value]
    if isinstance(value, list):
        return [_semantic_json(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_json(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise CriticDataError(
        f"semantic action contains unsupported {type(value).__name__}")


def _validate_deck(manifest: Mapping[str, Any]) -> list[int]:
    deck_record = manifest.get("registered_learner_deck")
    if not isinstance(deck_record, Mapping):
        raise CriticDataError("root manifest has no learner deck binding")
    cards = deck_record.get("cards")
    if (not isinstance(cards, list) or len(cards) != 60
            or any(
                not isinstance(card, int) or isinstance(card, bool)
                or not 0 < card < QF.EXPECTED_CARD_VOCAB
                for card in cards
            )
            or _value_sha256(cards) != deck_record.get("sha256")):
        raise CriticDataError("root manifest learner deck is malformed")
    if cards != policy.load_deck():
        raise CriticDataError("registered learner deck drifted")
    return list(cards)


def _encode_row(
    public_record: Mapping[str, Any],
    privileged_record: Mapping[str, Any],
    learner_deck: Sequence[int],
    native_result: Mapping[str, Any],
) -> tuple[str, PF.PrivilegedFeatures, str, int, float]:
    root_id = public_record.get("root_id")
    if (public_record.get("schema") != MINE.PUBLIC_SCHEMA
            or privileged_record.get("schema") != MINE.PRIVILEGED_SCHEMA
            or root_id != privileged_record.get("root_id")
            or not _is_sha256(root_id)):
        raise CriticDataError("paired root schema/identity mismatch")

    public_source = public_record.get("source")
    private_source = privileged_record.get("source")
    if not isinstance(public_source, Mapping) or not isinstance(
            private_source, Mapping):
        raise CriticDataError(f"root {root_id} has malformed source metadata")
    for key in ("episode_id", "source_step", "learner_seat", "replay_sha256"):
        if public_source.get(key) != private_source.get(key):
            raise CriticDataError(
                f"root {root_id} public/private source {key} mismatch")
    episode_id = _valid_episode_id(public_source.get("episode_id"))
    replay_sha = public_source.get("replay_sha256")
    if not _is_sha256(replay_sha):
        raise CriticDataError(f"root {root_id} replay hash is malformed")

    selection = public_record.get("selection")
    if (not isinstance(selection, Mapping)
            or selection.get("mode") != "factual-critic"
            or selection.get("policy")
            != MINE.FACTUAL_CRITIC_SELECTION_POLICY
            or selection.get("factual_terminal_return_label") is not True
            or selection.get("hard_action_label") is not False):
        raise CriticDataError(
            f"root {root_id} is not a factual-critic query")

    reward = public_source.get("learner_reward")
    if (not isinstance(reward, (int, float)) or isinstance(reward, bool)
            or not math.isfinite(float(reward))
            or float(reward) not in (-1.0, 0.0, 1.0)):
        raise CriticDataError(
            f"root {root_id} has no factual terminal return")
    outcome = public_source.get("outcome")
    expected_outcome = (
        "win" if float(reward) > 0
        else "loss" if float(reward) < 0 else "draw"
    )
    if outcome != expected_outcome:
        raise CriticDataError(
            f"root {root_id} outcome disagrees with terminal return")

    logged = public_record.get("logged")
    action = logged.get("action") if isinstance(logged, Mapping) else None
    if (not isinstance(action, list) or len(action) != 1
            or not isinstance(action[0], int) or isinstance(action[0], bool)):
        raise CriticDataError(
            f"root {root_id} must have exactly one logged real action")
    chosen = int(action[0])

    try:
        obs, _ = VALIDATE.reconstruct_observation(
            public_record, privileged_record)
        public_bound = dict(obs)
        public_bound.pop(VALIDATE.CFO.EXACT_HIDDEN_KEY, None)
        features = PF.encode_privileged_observation(
            public_bound, privileged_record["exact_hidden_payload"],
            learner_deck,
        )
    except (ValueError, VALIDATE.ValidationError) as exc:
        raise CriticDataError(
            f"root {root_id} cannot be reconstructed/encoded: {exc}") from exc
    PF.validate_privileged_features(features)

    real_option_count = len(features.public.option_ids) - 1
    if (not 2 <= real_option_count <= 12
            or not 0 <= chosen < real_option_count
            or native_result.get("root_options") != real_option_count):
        raise CriticDataError(
            f"root {root_id} logged action is not a real native option")
    expected_semantic = _semantic_json(TS.semantic_action(obs, action))
    if logged.get("semantic_action") != expected_semantic:
        raise CriticDataError(
            f"root {root_id} logged semantic action drifted")

    identity = public_record.get("identity")
    binding = privileged_record.get("binding")
    public_feature_hash = _sha256_bytes(
        PRODUCTION_QF.feature_fingerprint(features.public))
    privileged_hash = features.canonical_hash()
    if (not isinstance(identity, Mapping)
            or identity.get("qu_v2_feature_fingerprint")
            != public_feature_hash
            or not isinstance(binding, Mapping)
            or binding.get("privileged_feature_sha256") != privileged_hash
            or native_result.get("privileged_feature_sha256")
            != privileged_hash):
        raise CriticDataError(
            f"root {root_id} feature fingerprint drifted")
    return episode_id, features, str(root_id), chosen, float(reward)


def _stack_game(
    rows: Sequence[tuple[PF.PrivilegedFeatures, str, int, float]],
    *,
    split: str,
    episode_id_sha256: str,
    root_manifest_sha256: str,
    native_validation_sha256: str,
) -> dict[str, np.ndarray]:
    if not rows:
        raise CriticDataError("cannot serialize an empty critic game")
    samples = [row[0] for row in rows]
    for sample in samples:
        PF.validate_privileged_features(sample)
    option_counts = np.asarray(
        [len(sample.public.option_ids) for sample in samples],
        dtype=np.int32,
    )
    max_options = int(option_counts.max())
    count = len(samples)
    sample_arrays = [sample.arrays() for sample in samples]
    arrays: dict[str, np.ndarray] = {}
    for name in PF.ARRAY_NAMES:
        values = [record[name] for record in sample_arrays]
        if name in _VARIABLE_PUBLIC_ARRAYS:
            tail = values[0].shape[1:]
            padded = np.zeros(
                (count, max_options, *tail), dtype=values[0].dtype)
            for row_index, value in enumerate(values):
                padded[row_index, :len(value)] = value
            arrays[name] = padded
        else:
            arrays[name] = np.stack(values)

    root_ids = [row[1] for row in rows]
    chosen = np.asarray([row[2] for row in rows], dtype=np.int32)
    returns = np.asarray([row[3] for row in rows], dtype=np.float32)
    weights = np.full(count, 1.0 / count, dtype=np.float32)
    metadata = {
        "schema": np.asarray(GAME_SCHEMA.encode("ascii"), dtype="S64"),
        "feature_schema": np.asarray(QF.SCHEMA.encode("ascii"), dtype="S64"),
        "privileged_feature_schema": np.asarray(
            PF.SCHEMA.encode("ascii"), dtype="S64"),
        "split_schema": np.asarray(SPLITS.SCHEMA.encode("ascii"), dtype="S64"),
        "split": np.asarray(split.encode("ascii"), dtype="S16"),
        "split_seed": np.asarray(SPLIT_SEED, dtype=np.int32),
        "episode_id_sha256": np.asarray(
            episode_id_sha256.encode("ascii"), dtype="S64"),
        "root_manifest_sha256": np.asarray(
            root_manifest_sha256.encode("ascii"), dtype="S64"),
        "native_validation_sha256": np.asarray(
            native_validation_sha256.encode("ascii"), dtype="S64"),
        "record_count": np.asarray(count, dtype=np.int32),
        "root_ids": np.asarray(
            [value.encode("ascii") for value in root_ids], dtype="S64"),
        "option_count": option_counts,
        "chosen_index": chosen,
        "terminal_return": returns,
        "game_weight": weights,
    }
    return {**metadata, **arrays}


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if tuple(arrays) != GAME_ARRAY_NAMES:
        raise CriticDataError("critic game array order/schema drifted")
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise CriticDataError("critic game contains a pickled/object array")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _prepare_output(output: Path) -> tuple[Path, int]:
    output = output.expanduser().resolve()
    if any(_inside(output, tree) for tree in PROTECTED_OUTPUT_TREES):
        raise CriticDataError(
            "refusing to write private critic data in a production tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_dir():
            raise CriticDataError(
                f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise CriticDataError(
                f"output directory is not empty: {output}")
    else:
        output.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    lock = output / ".prepare.lock"
    descriptor = os.open(
        lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    for name in SPLITS.NAMES:
        directory = output / name
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
    return output, descriptor


def _assert_manifest_safe(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _MANIFEST_FORBIDDEN_KEYS:
                raise CriticDataError(
                    f"critic manifest would expose private identity at "
                    f"{path}.{key}")
            _assert_manifest_safe(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_manifest_safe(item, f"{path}[{index}]")


def prepare_dataset(
    root_dir: Path,
    native_validation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate and materialize one complete factual critic dataset."""
    root_dir = root_dir.expanduser().resolve()
    native_validation_path = native_validation_path.expanduser().resolve()
    try:
        root_manifest, public, privileged = VALIDATE.load_root_artifacts(
            root_dir)
    except (OSError, ValueError, VALIDATE.ValidationError) as exc:
        raise CriticDataError(f"invalid root artifacts: {exc}") from exc
    if (root_manifest.get("selection_mode") != "factual-critic"
            or root_manifest.get("selection_policy")
            != MINE.FACTUAL_CRITIC_SELECTION_POLICY):
        raise CriticDataError(
            "critic data requires a factual-critic mined root directory")
    if not public:
        raise CriticDataError("factual critic root set is empty")
    _assert_root_source_lock(root_manifest)
    learner_deck = _validate_deck(root_manifest)
    report, native_file_sha, native_by_id = _load_native_validation(
        native_validation_path, root_manifest, public)
    native_report_sha = str(report["report_sha256"])

    try:
        SPLITS.assert_feature_classes_do_not_cross_splits(
            public, seed=SPLIT_SEED)
    except ValueError as exc:
        raise CriticDataError(
            f"public feature split leakage: {exc}") from exc

    grouped: dict[
        str, list[tuple[PF.PrivilegedFeatures, str, int, float]]
    ] = {}
    replay_hash_by_episode: dict[str, str] = {}
    reward_by_episode: dict[str, float] = {}
    for public_record, privileged_record in zip(public, privileged):
        root_id = public_record.get("root_id")
        native_result = native_by_id.get(str(root_id))
        if native_result is None:
            raise CriticDataError(
                f"root {root_id} has no native validation result")
        episode_id, features, encoded_root_id, chosen, reward = _encode_row(
            public_record, privileged_record, learner_deck, native_result)
        replay_hash = (public_record.get("source") or {}).get("replay_sha256")
        previous = replay_hash_by_episode.setdefault(
            episode_id, str(replay_hash))
        if previous != replay_hash:
            raise CriticDataError(
                f"episode_id {episode_id!r} names multiple replay hashes")
        previous_reward = reward_by_episode.setdefault(episode_id, reward)
        if previous_reward != reward:
            raise CriticDataError(
                f"episode_id {episode_id!r} has inconsistent terminal returns")
        grouped.setdefault(episode_id, []).append(
            (features, encoded_root_id, chosen, reward))
    if not grouped:
        raise CriticDataError("no valid factual critic games were encoded")

    root_manifest_sha = str(root_manifest["manifest_sha256"])
    root_manifest_file_sha = _sha256_file(root_dir / "manifest.json")
    output, lock_descriptor = _prepare_output(output_dir)
    lock_path = output / ".prepare.lock"
    artifact_table: dict[str, dict[str, Any]] = {
        name: {"games": 0, "roots": 0, "files": []}
        for name in SPLITS.NAMES
    }
    try:
        for episode_id in sorted(grouped):
            rows = grouped[episode_id]
            rows.sort(key=lambda row: row[1])
            split = SPLITS.split_for_episode(episode_id, seed=SPLIT_SEED)
            episode_hash = _sha256_bytes(episode_id.encode("utf-8"))
            arrays = _stack_game(
                rows,
                split=split,
                episode_id_sha256=episode_hash,
                root_manifest_sha256=root_manifest_sha,
                native_validation_sha256=native_file_sha,
            )
            payload = _npz_bytes(arrays)
            relative = Path(split) / f"game-{episode_hash}.npz"
            path = output / relative
            _atomic_bytes(path, payload, 0o600)
            entry = {
                "path": relative.as_posix(),
                "sha256": _sha256_bytes(payload),
                "bytes": len(payload),
                "mode": "0600",
                "episode_id_sha256": episode_hash,
                "roots": len(rows),
                "max_option_count": int(arrays["option_count"].max()),
            }
            artifact_table[split]["games"] += 1
            artifact_table[split]["roots"] += len(rows)
            artifact_table[split]["files"].append(entry)

        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "contains_privileged_exact_hidden_features": True,
            "direct_actor_training_eligible": False,
            "label_contract": (
                "one logged real action per root receives only the factual "
                "terminal learner return; no counterfactual label is implied"
            ),
            "privacy": {
                "game_artifact_mode": "0600",
                "pickle_allowed": False,
                "manifest_contains_private_id_material": False,
            },
            "split_contract": SPLITS.contract(seed=SPLIT_SEED),
            "inputs": {
                "root_manifest": {
                    "file_sha256": root_manifest_file_sha,
                    "manifest_sha256": root_manifest_sha,
                    "selection_mode": "factual-critic",
                    "public_roots_sha256": root_manifest[
                        "artifacts"]["public_roots"]["sha256"],
                    "privileged_roots_sha256": root_manifest[
                        "artifacts"]["privileged_roots"]["sha256"],
                },
                "native_validation": {
                    "file_sha256": native_file_sha,
                    "report_sha256": native_report_sha,
                    "complete_all_roots_pass": True,
                },
            },
            "counts": {
                "games": len(grouped),
                "roots": sum(len(rows) for rows in grouped.values()),
                "splits": {
                    name: {
                        "games": artifact_table[name]["games"],
                        "roots": artifact_table[name]["roots"],
                    }
                    for name in SPLITS.NAMES
                },
            },
            "artifacts": artifact_table,
            "source_files_sha256": _source_hashes(),
        }
        _assert_manifest_safe(manifest)
        manifest["manifest_sha256"] = _value_sha256(manifest)
        manifest_bytes = json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        _atomic_bytes(output / "manifest.json", manifest_bytes, 0o644)
    finally:
        os.close(lock_descriptor)
        try:
            lock_path.unlink()
        except OSError:
            pass
    return manifest


def _scalar_bytes(array: np.ndarray, name: str) -> str:
    if array.shape != () or array.dtype.kind != "S":
        raise CriticDataError(f"{name} must be a scalar fixed-width byte string")
    try:
        return bytes(array.item()).decode("ascii")
    except (AttributeError, UnicodeDecodeError) as exc:
        raise CriticDataError(f"{name} is not ASCII metadata") from exc


def _scalar_int(array: np.ndarray, name: str) -> int:
    if array.shape != () or array.dtype != np.dtype(np.int32):
        raise CriticDataError(f"{name} must be scalar int32")
    return int(array)


def _expect_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    value = arrays[name]
    if (value.shape != shape or value.dtype != dtype
            or not value.flags.c_contiguous):
        raise CriticDataError(
            f"{name} has {value.dtype} {value.shape}, expected "
            f"{dtype} {shape}")
    return value


def load_game_npz(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_split: str | None = None,
) -> tuple[tuple[PF.PrivilegedFeatures, ...], CriticLabels]:
    """Strictly load one private game artifact and return records plus labels."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise CriticDataError(f"critic game file does not exist: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise CriticDataError(
            f"critic game mode is {mode:04o}, expected 0600")
    file_sha = _sha256_file(path)
    if expected_sha256 is not None and (
            not _is_sha256(expected_sha256) or file_sha != expected_sha256):
        raise CriticDataError("critic game checksum mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (tuple(archive.files) != GAME_ARRAY_NAMES
                    or len(set(archive.files)) != len(GAME_ARRAY_NAMES)):
                raise CriticDataError(
                    "critic game has missing, extra, reordered, or duplicate arrays")
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    except CriticDataError:
        raise
    except Exception as exc:
        raise CriticDataError(
            f"cannot load pickle-free critic game: {exc}") from exc
    if any(value.dtype.hasobject for value in arrays.values()):
        raise CriticDataError("critic game contains an object/pickled array")

    if _scalar_bytes(arrays["schema"], "schema") != GAME_SCHEMA:
        raise CriticDataError("critic game schema mismatch")
    if _scalar_bytes(arrays["feature_schema"], "feature_schema") != QF.SCHEMA:
        raise CriticDataError("critic game public feature schema mismatch")
    if _scalar_bytes(
            arrays["privileged_feature_schema"],
            "privileged_feature_schema") != PF.SCHEMA:
        raise CriticDataError("critic game privileged feature schema mismatch")
    if _scalar_bytes(
            arrays["split_schema"], "split_schema") != SPLITS.SCHEMA:
        raise CriticDataError("critic game split schema mismatch")
    split = _scalar_bytes(arrays["split"], "split")
    if split not in SPLITS.NAMES:
        raise CriticDataError("critic game split name is invalid")
    if expected_split is not None and split != expected_split:
        raise CriticDataError("critic game is in the wrong requested split")
    if _scalar_int(arrays["split_seed"], "split_seed") != SPLIT_SEED:
        raise CriticDataError("critic game split seed drifted")

    episode_hash = _scalar_bytes(
        arrays["episode_id_sha256"], "episode_id_sha256")
    root_manifest_sha = _scalar_bytes(
        arrays["root_manifest_sha256"], "root_manifest_sha256")
    native_sha = _scalar_bytes(
        arrays["native_validation_sha256"], "native_validation_sha256")
    if not all(_is_sha256(value) for value in (
            episode_hash, root_manifest_sha, native_sha)):
        raise CriticDataError("critic game provenance hash is malformed")

    count = _scalar_int(arrays["record_count"], "record_count")
    if count <= 0:
        raise CriticDataError("critic game is empty")
    root_array = arrays["root_ids"]
    if root_array.shape != (count,) or root_array.dtype != np.dtype("S64"):
        raise CriticDataError("root_ids must be fixed ASCII S64[records]")
    try:
        root_ids = tuple(
            bytes(value).decode("ascii") for value in root_array.tolist())
    except UnicodeDecodeError as exc:
        raise CriticDataError("root_ids are not ASCII") from exc
    if (len(set(root_ids)) != count
            or not all(_is_sha256(value) for value in root_ids)):
        raise CriticDataError("root_ids are malformed or duplicated")

    option_counts = _expect_array(
        arrays, "option_count", (count,), np.dtype(np.int32))
    chosen = _expect_array(
        arrays, "chosen_index", (count,), np.dtype(np.int32))
    returns = _expect_array(
        arrays, "terminal_return", (count,), np.dtype(np.float32))
    weights = _expect_array(
        arrays, "game_weight", (count,), np.dtype(np.float32))
    if (np.any(option_counts < 3) or np.any(option_counts > 13)
            or np.any(chosen < 0)
            or np.any(chosen >= option_counts - 1)):
        raise CriticDataError(
            "option counts/chosen real-action indices are invalid")
    if (not np.isfinite(returns).all()
            or not np.isin(returns, np.asarray(
                [-1.0, 0.0, 1.0], dtype=np.float32)).all()):
        raise CriticDataError("factual terminal returns are invalid")
    expected_weight = np.float32(1.0 / count)
    if (not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.allclose(weights, expected_weight, rtol=0, atol=1e-7)
            or not np.isclose(
                weights.sum(dtype=np.float64), 1.0, rtol=0, atol=1e-6)):
        raise CriticDataError("per-game weights are not normalized")

    max_options = int(option_counts.max())
    public_shapes = {
        "board_ids": (QF.BOARD_SLOTS,),
        "board_energy_ids": (QF.BOARD_SLOTS, QF.ENERGY_SLOTS),
        "board_tool_ids": (QF.BOARD_SLOTS, QF.TOOL_SLOTS),
        "board_evolution_ids": (QF.BOARD_SLOTS, QF.EVOLUTION_SLOTS),
        "board_features": (QF.BOARD_SLOTS, QF.BOARD_FEATURES),
        "hand_ids": (QF.HAND_SLOTS,),
        "my_discard_ids": (QF.DISCARD_SLOTS,),
        "opponent_discard_ids": (QF.DISCARD_SLOTS,),
        "looking_ids": (QF.LOOKING_SLOTS,),
        "stadium_ids": (QF.STADIUM_SLOTS,),
        "prompt_ids": (QF.PROMPT_ID_SLOTS,),
        "prompt_features": (QF.PROMPT_FEATURES,),
        "registered_deck_ids": (QF.REGISTERED_DECK_SLOTS,),
        "option_ids": (max_options,),
        "option_target_ids": (max_options,),
        "option_features": (max_options, QF.OPTION_FEATURES),
        "option_mask": (max_options,),
    }
    integer_public = frozenset({
        "board_ids", "board_energy_ids", "board_tool_ids",
        "board_evolution_ids", "hand_ids", "my_discard_ids",
        "opponent_discard_ids", "looking_ids", "stadium_ids", "prompt_ids",
        "registered_deck_ids", "option_ids", "option_target_ids",
    })
    hidden_shapes = {
        "my_deck_ids": (PF.DECK_SLOTS,),
        "my_deck_mask": (PF.DECK_SLOTS,),
        "my_prize_ids": (PF.PRIZE_SLOTS,),
        "my_prize_mask": (PF.PRIZE_SLOTS,),
        "opponent_deck_ids": (PF.DECK_SLOTS,),
        "opponent_deck_mask": (PF.DECK_SLOTS,),
        "opponent_prize_ids": (PF.PRIZE_SLOTS,),
        "opponent_prize_mask": (PF.PRIZE_SLOTS,),
        "opponent_hand_ids": (PF.OPPONENT_HAND_SLOTS,),
        "opponent_hand_mask": (PF.OPPONENT_HAND_SLOTS,),
        "opponent_active_ids": (PF.OPPONENT_ACTIVE_SLOTS,),
        "opponent_active_mask": (PF.OPPONENT_ACTIVE_SLOTS,),
    }
    for name, tail in public_shapes.items():
        dtype = (
            np.dtype(np.bool_) if name == "option_mask"
            else np.dtype(np.int32) if name in integer_public
            else np.dtype(np.float32)
        )
        _expect_array(arrays, f"public_{name}", (count, *tail), dtype)
    for name, tail in hidden_shapes.items():
        dtype = np.dtype(np.bool_) if name.endswith("_mask") else np.dtype(
            np.int32)
        _expect_array(arrays, name, (count, *tail), dtype)

    option_mask = arrays["public_option_mask"]
    for row, option_count in enumerate(option_counts.tolist()):
        if (not option_mask[row, :option_count].all()
                or option_mask[row, option_count:].any()):
            raise CriticDataError(
                "public option mask is not canonical prefix padding")
        for name in (
                "public_option_ids",
                "public_option_target_ids",
                "public_option_features"):
            if np.any(arrays[name][row, option_count:] != 0):
                raise CriticDataError(
                    f"{name} has nonzero padded option rows")

    records: list[PF.PrivilegedFeatures] = []
    for row, option_count in enumerate(option_counts.tolist()):
        public_kwargs: dict[str, np.ndarray] = {}
        for field in fields(QF.PublicFeatures):
            name = field.name
            value = arrays[f"public_{name}"][row]
            if f"public_{name}" in _VARIABLE_PUBLIC_ARRAYS:
                value = value[:option_count]
            public_kwargs[name] = _readonly(value)
        public_features = QF.PublicFeatures(**public_kwargs)
        hidden_kwargs = {
            name: _readonly(arrays[name][row])
            for name in PF.HIDDEN_ARRAY_NAMES
        }
        record = PF.PrivilegedFeatures(
            public=public_features, **hidden_kwargs)
        PF.validate_privileged_features(record)
        records.append(record)

    labels = CriticLabels(
        split=split,
        episode_id_sha256=episode_hash,
        root_manifest_sha256=root_manifest_sha,
        native_validation_sha256=native_sha,
        root_ids=root_ids,
        option_counts=_readonly(option_counts),
        chosen_indices=_readonly(chosen),
        terminal_returns=_readonly(returns),
        game_weights=_readonly(weights),
    )
    return tuple(records), labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument(
        "--native-validation",
        help=(
            "complete validation report; defaults to "
            "<root-dir>/native-validation.json"
        ),
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root_dir = Path(args.root_dir).expanduser().resolve()
    native = (
        Path(args.native_validation).expanduser().resolve()
        if args.native_validation
        else root_dir / DEFAULT_NATIVE_VALIDATION
    )
    try:
        manifest = prepare_dataset(
            root_dir, native, Path(args.out_dir))
    except (OSError, ValueError, CriticDataError) as exc:
        parser.error(str(exc))
    counts = manifest["counts"]
    print(
        f"Qu-v2C factual critic data: {counts['roots']} roots in "
        f"{counts['games']} complete games",
        flush=True,
    )
    print(f"Artifacts: {Path(args.out_dir).expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
