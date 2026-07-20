"""Distill turn-search JSONL targets into the existing reflex policy net.

The teacher generator stores a distribution over *semantic* root actions.
This trainer maps every semantic action back onto the option indices in the
recorded observation before constructing a target.  Multi-pick distributions
are represented as a trie of conditional pick/STOP targets, matching the
sequential selection semantics in :mod:`tools.train` and :mod:`agent.model`.

Only the policy path is optimized.  Game outcomes and the value head are not
used as RL/value targets; the value layers are deliberately excluded from the
optimizer.  A leaderboard behavior-cloning corpus can optionally be mixed in
as an NLL anchor.

Example::

    ~/.venvs/ptcg-rl/bin/python tools/train_teacher.py data/teacher/*.jsonl \
        --resume tools/checkpoints/ft10/latest.pt \
        --out tools/checkpoints/teacher1/weights.npz \
        --ckpt-dir tools/checkpoints/teacher1 \
        --bc-anchor ~/Desktop/ptcg_episodes
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS_DIR)

from agent import features as FE  # noqa: E402
from agent import model as NPM  # noqa: E402
from agent import policy  # noqa: E402
from agent import turn_search  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from eval_turn_search import planner_provenance  # noqa: E402
from train import (  # noqa: E402
    DEV,
    TorchNet,
    _stack_minibatch,
    export_npz,
    load_bc_samples,
    picks_logprob,
)


TEACHER_SCHEMA = "ptcg.turn_search.teacher.v1"
DEFAULT_ARCH = (16, 256, 128, 128, 64)
REQUIRED_SCHEMA_FIELDS = (
    "schema, game_id, observation.select.option, "
    "root.semantic_actions, root.soft_distribution, config, and provenance"
)


class DatasetError(ValueError):
    """A fail-closed teacher data error with source-line context."""


@dataclass(frozen=True)
class TargetStep:
    """A conditional distribution after ``prefix`` has already been picked."""

    prefix: tuple[int, ...]
    available: np.ndarray
    target: np.ndarray
    reach: float


@dataclass
class TeacherSample:
    state: dict[str, np.ndarray]
    opt_ids: np.ndarray
    opt_feats: np.ndarray
    target_steps: tuple[TargetStep, ...]
    game_id: str
    decision_id: str
    n_opts: int
    n_min: int
    n_max: int
    balance_key: tuple[str, str, str]
    source: str
    weight: float = 1.0


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _freeze(value: Any) -> Any:
    """Restore tuple-shaped semantic fingerprints after JSON decoding."""
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item))
                            for key, item in value.items()))
    return value


def _where(path: str, line_no: int) -> str:
    return f"{path}:{line_no}"


def _fail(where: str, message: str) -> DatasetError:
    return DatasetError(f"{where}: {message}")


def _numeric_sequence(value: Any, where: str, name: str,
                      *, nonnegative: bool = False) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise _fail(where, f"{name} must be a non-empty numeric sequence")
    out: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise _fail(where, f"{name}[{index}] is boolean, not numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise _fail(where, f"{name}[{index}] is not numeric") from exc
        if not math.isfinite(number):
            raise _fail(where, f"{name}[{index}] is not finite")
        if nonnegative and number < 0:
            raise _fail(where, f"{name}[{index}] must be non-negative")
        out.append(number)
    return out


def _as_indices(value: Any, where: str, name: str) -> list[int]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, int) and not isinstance(value, bool):
        return [int(value)]
    if not isinstance(value, (list, tuple)):
        raise _fail(where, f"{name} must be an option index or index list")
    out: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, (int, np.integer)) or isinstance(item, bool):
            raise _fail(where, f"{name}[{index}] is not an integer option index")
        out.append(int(item))
    return out


def _expand_inputs(specs: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for raw in specs:
        expanded = os.path.abspath(os.path.expanduser(raw))
        if glob.has_magic(expanded):
            matches = sorted(glob.glob(expanded, recursive=True))
        elif os.path.isdir(expanded):
            matches = sorted(glob.glob(os.path.join(expanded, "**", "*.jsonl"),
                                       recursive=True))
        elif os.path.isfile(expanded):
            matches = [expanded]
        else:
            matches = []
        if not matches:
            raise DatasetError(f"input {raw!r} matched no JSONL files")
        paths.extend(path for path in matches if os.path.isfile(path))
    unique: list[str] = []
    seen: set[str] = set()
    for path in paths:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            unique.append(path)
    if not unique:
        raise DatasetError("no teacher JSONL inputs")
    return unique


def _bc_anchor_provenance(spec: str | None) -> dict[str, Any] | None:
    if not spec:
        return None
    files: list[dict[str, Any]] = []
    for source_index, part in enumerate(spec.split(",")):
        directory, _, multiplier_text = part.partition(":")
        multiplier = float(multiplier_text) if multiplier_text else 1.0
        expanded = os.path.abspath(os.path.expanduser(directory))
        for path in sorted(glob.glob(os.path.join(expanded, "*.json"))):
            files.append({
                "source_index": source_index,
                "multiplier": multiplier,
                "path": os.path.abspath(path),
                "bytes": os.path.getsize(path),
                "sha256": _sha256_file(path),
            })
    return {
        "spec": spec,
        "files": len(files),
        "corpus_sha256": _sha256_json(files),
        "entries": files,
    }


def _schema(record: Mapping[str, Any]) -> str | None:
    value = record.get("schema", record.get("schema_version"))
    if value == TEACHER_SCHEMA:
        return TEACHER_SCHEMA
    if value in (1, "1", "v1", "teacher.v1"):
        return "compat-v1"
    return None


class ProvenanceGuard:
    """Validate hashes and keep policy/search provenance invariant across shards.

    Opponent deck/policy, game seed, worker and matchup intentionally may vary;
    the learner weights/deck and planner implementation may not.
    """

    def __init__(self, learner_deck: Sequence[int]):
        self.local_deck_sha256 = _sha256_json(list(learner_deck))
        self.expected: dict[str, Any] = {}

    def _same(self, key: str, value: Any, where: str) -> None:
        if key not in self.expected:
            self.expected[key] = value
        elif self.expected[key] != value:
            raise _fail(
                where,
                f"provenance mismatch for {key}: {value!r} != "
                f"{self.expected[key]!r}",
            )

    def check(self, record: Mapping[str, Any], where: str, schema: str) -> None:
        provenance = record.get("provenance")
        config = record.get("config")
        if not isinstance(provenance, Mapping) or not isinstance(config, Mapping):
            raise _fail(where, f"records require mapping-valued config and provenance; "
                              f"required fields: {REQUIRED_SCHEMA_FIELDS}")

        configured_schema = config.get("schema")
        if schema == TEACHER_SCHEMA and configured_schema != TEACHER_SCHEMA:
            raise _fail(where, "config.schema does not match the record schema")

        # Exact generator records are self-authenticating.  Recompute every
        # hash whose payload is present rather than merely trusting its shape.
        if schema == TEACHER_SCHEMA:
            config_hash = provenance.get("config_sha256")
            if not _is_hash(config_hash) or config_hash != _sha256_json(dict(config)):
                raise _fail(where, "provenance.config_sha256 does not match config")

            mandatory_base = {
                "config_sha256", "learner_weights_sha256",
                "learner_deck_sha256", "turn_search_sha256",
                "turn_search_path", "planner_config_sha256",
                "feature_version", "features_sha256",
                "search_policy_sha256", "planner_meta_sha256",
                "engine_sha256", "battle_engine_sha256",
                "cards_data_sha256", "attacks_data_sha256",
                "model_sha256", "obsview_sha256", "policy_sha256",
                "cards_sha256", "cabt_sha256", "teacher_generator_sha256",
                "planner_tooling_sha256",
            }
            base_fields = provenance.get("base_provenance_fields")
            if not isinstance(base_fields, list) or not base_fields or \
                    len(base_fields) != len(set(base_fields)) or \
                    not all(isinstance(key, str) for key in base_fields) or \
                    not mandatory_base.issubset(base_fields):
                raise _fail(where, "provenance has no valid sealed base field list")
            if any(key not in provenance for key in base_fields):
                raise _fail(where, "provenance is missing a sealed base field")
            base_payload = {
                "fields": base_fields,
                "values": {key: provenance.get(key) for key in base_fields},
            }
            base_hash = provenance.get("base_provenance_sha256")
            if not _is_hash(base_hash) or base_hash != _sha256_json(base_payload):
                raise _fail(where, "provenance.base_provenance_sha256 is invalid")

            record_payload = dict(provenance)
            record_hash = record_payload.pop("record_provenance_sha256", None)
            if not _is_hash(record_hash) or record_hash != _sha256_json(record_payload):
                raise _fail(where, "provenance.record_provenance_sha256 is invalid")

        learner_deck_hash = provenance.get(
            "learner_deck_sha256", provenance.get("deck_sha256")
        )
        if not _is_hash(learner_deck_hash):
            raise _fail(where, "provenance has no valid learner deck SHA-256")
        if learner_deck_hash.lower() != self.local_deck_sha256:
            raise _fail(
                where,
                "teacher learner deck does not match the repository learner deck",
            )

        learner_weights_hash = provenance.get(
            "learner_weights_sha256", provenance.get("weights_sha256")
        )
        planner_hash = provenance.get(
            "turn_search_sha256", provenance.get("planner_sha256")
        )
        if not _is_hash(learner_weights_hash) or not _is_hash(planner_hash):
            raise _fail(where, "provenance requires learner-weights and planner hashes")

        local_planner_hash = _sha256_file(str(turn_search.__file__))
        if planner_hash.lower() != local_planner_hash:
            raise _fail(
                where,
                "teacher semantic fingerprints were produced by a different "
                "turn_search.py; regenerate or train with the matching source",
            )

        planner_config = config.get("planner")
        planner_config_hash = provenance.get("planner_config_sha256")
        if not isinstance(planner_config, Mapping) or \
                not _is_hash(planner_config_hash) or \
                planner_config_hash != _sha256_json(dict(planner_config)):
            raise _fail(where, "planner config/hash is missing or inconsistent")

        behavior_hashes = (
            "features_sha256", "search_policy_sha256", "planner_meta_sha256",
            "engine_sha256", "battle_engine_sha256",
            "cards_data_sha256", "attacks_data_sha256",
            "model_sha256", "obsview_sha256", "policy_sha256",
            "cards_sha256", "cabt_sha256", "teacher_generator_sha256",
            "planner_tooling_sha256",
        )
        local_behavior = planner_provenance(turn_search)
        local_behavior["teacher_generator_sha256"] = _sha256_file(
            os.path.join(ROOT, "tools", "selfplay_teacher.py")
        )
        for key in behavior_hashes:
            value = provenance.get(key)
            if not _is_hash(value):
                raise _fail(where, f"provenance.{key} is not a SHA-256")
            self._same(key, value.lower(), where)
            local_value = local_behavior.get(key)
            if not _is_hash(local_value) or value.lower() != local_value.lower():
                raise _fail(
                    where,
                    f"teacher {key} does not match the local training source/data",
                )

        self._same("schema", schema, where)
        self._same("learner_deck_sha256", learner_deck_hash.lower(), where)
        self._same("learner_weights_sha256", learner_weights_hash.lower(), where)
        self._same("turn_search_sha256", planner_hash.lower(), where)
        self._same("planner_config_sha256", planner_config_hash.lower(), where)
        self._same("budget_s", config.get("budget_s", config.get("budget")), where)
        self._same(
            "max_particles",
            config.get("max_particles", config.get("particles")),
            where,
        )
        feature_version = record.get(
            "feature_version", provenance.get("feature_version")
        )
        if feature_version is not None:
            try:
                feature_version = int(feature_version)
            except (TypeError, ValueError) as exc:
                raise _fail(where, "feature_version is not an integer") from exc
            if feature_version != FE.FEAT_VERSION:
                raise _fail(
                    where,
                    f"feature_version {feature_version} != runtime {FE.FEAT_VERSION}",
                )
            self._same("feature_version", feature_version, where)


def _validate_action(action: list[int], view: ObsView, where: str,
                     name: str) -> tuple[int, ...]:
    n_opts = len(view.options)
    n_min = view.min_count
    n_max = view.max_count
    if not isinstance(n_min, int) or isinstance(n_min, bool) or n_min < 0:
        raise _fail(where, "observation select.minCount is invalid")
    if not isinstance(n_max, int) or isinstance(n_max, bool):
        raise _fail(where, "observation select.maxCount is invalid")
    effective_max = n_opts if n_max <= 0 else min(n_max, n_opts)
    if n_min > effective_max:
        raise _fail(where, "observation selection bounds are impossible")
    if len(action) != len(set(action)):
        raise _fail(where, f"{name} contains duplicate option indices")
    if not all(0 <= index < n_opts for index in action):
        raise _fail(where, f"{name} is not aligned to real observation options")
    if not n_min <= len(action) <= effective_max:
        raise _fail(
            where,
            f"{name} length {len(action)} violates [{n_min}, {effective_max}]",
        )
    return tuple(action)


def _semantic_to_indices(value: Any, observation: dict, view: ObsView,
                         where: str, name: str) -> tuple[int, ...]:
    semantic = _freeze(value)
    if not isinstance(semantic, tuple):
        raise _fail(where, f"{name} is not a semantic action sequence")
    try:
        mapped = turn_search.map_semantic_action(observation, semantic)
    except Exception as exc:
        raise _fail(where, f"{name} semantic mapping raised {exc!r}") from exc
    if mapped is None:
        raise _fail(where, f"{name} does not map onto this observation")
    action = _validate_action(mapped, view, where, name)
    try:
        roundtrip = turn_search.semantic_action(observation, action)
    except Exception as exc:
        raise _fail(where, f"{name} failed semantic roundtrip: {exc!r}") from exc
    if roundtrip != semantic:
        raise _fail(where, f"{name} is not canonical for this observation")
    return action


def _root_actions(record: Mapping[str, Any], observation: dict, view: ObsView,
                  where: str, schema: str) -> tuple[list[tuple[int, ...]], Any]:
    root = record.get("root")
    if schema == TEACHER_SCHEMA:
        if not isinstance(root, Mapping):
            raise _fail(where, f"missing root object; required fields: "
                               f"{REQUIRED_SCHEMA_FIELDS}")
        raw_actions = root.get("semantic_actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise _fail(where, "root.semantic_actions must be a non-empty list")
        actions = [
            _semantic_to_indices(value, observation, view, where,
                                 f"root.semantic_actions[{index}]")
            for index, value in enumerate(raw_actions)
        ]
        return actions, root

    # Compatibility path for early shards: semantic actions remain preferred,
    # but explicit current option indices are accepted and checked strictly.
    if not isinstance(root, Mapping):
        root = {}
    raw_actions = (
        root.get("semantic_actions") or root.get("root_actions")
        or record.get("root_actions")
    )
    if not isinstance(raw_actions, list) or not raw_actions:
        raise _fail(where, "no root semantic/actions list; required fields: "
                           f"{REQUIRED_SCHEMA_FIELDS}")
    actions: list[tuple[int, ...]] = []
    for index, entry in enumerate(raw_actions):
        name = f"root_actions[{index}]"
        if isinstance(entry, Mapping):
            semantic_value = next((entry[key] for key in (
                "semantic_action", "semantic", "fingerprint", "key"
            ) if key in entry), None)
            indices_value = next((entry[key] for key in (
                "current_indices", "option_indices", "indices", "action",
                "picks", "selection",
            ) if key in entry), None)
            semantic_action = (
                _semantic_to_indices(semantic_value, observation, view, where, name)
                if semantic_value is not None else None
            )
            index_action = (
                _validate_action(_as_indices(indices_value, where, name),
                                 view, where, name)
                if indices_value is not None else None
            )
            if semantic_action is None and index_action is None:
                raise _fail(where, f"{name} has neither semantics nor indices")
            if semantic_action is not None and index_action is not None \
                    and set(semantic_action) != set(index_action):
                raise _fail(where, f"{name} semantic/index representations disagree")
            action = semantic_action if semantic_action is not None else index_action
        elif isinstance(entry, int) and not isinstance(entry, bool):
            action = _validate_action([entry], view, where, name)
        elif isinstance(entry, (list, tuple)) and all(
                isinstance(item, (int, np.integer)) and not isinstance(item, bool)
                for item in entry):
            action = _validate_action(list(entry), view, where, name)
        else:
            action = _semantic_to_indices(entry, observation, view, where, name)
        assert action is not None
        actions.append(action)
    return actions, root


def _aligned_vector(value: Any, actions: Sequence[tuple[int, ...]], n_opts: int,
                    where: str, name: str, *, nonnegative: bool) -> list[float]:
    values = _numeric_sequence(value, where, name, nonnegative=nonnegative)
    if len(values) == len(actions):
        return values
    # Some prototype shards stored vectors in real-option order rather than
    # semantic-root order.  Gather those vectors only when every root action is
    # a single first pick (or STOP), so the mapping is unambiguous.
    if len(values) in (n_opts, n_opts + 1) and all(len(action) <= 1
                                                   for action in actions):
        gathered: list[float] = []
        for action in actions:
            index = action[0] if action else n_opts
            if index >= len(values):
                raise _fail(where, f"{name} omits the STOP target")
            gathered.append(values[index])
        return gathered
    raise _fail(
        where,
        f"{name} length {len(values)} cannot align with {len(actions)} root "
        f"actions and {n_opts} real options",
    )


def _mapping_vector(value: Mapping[str, Any], actions: Sequence[tuple[int, ...]],
                    where: str, name: str) -> list[float]:
    out: list[float] = []
    for root_index, action in enumerate(actions):
        aliases = [str(root_index), json.dumps(list(action), separators=(",", ":"))]
        if len(action) == 1:
            aliases.insert(0, str(action[0]))
        elif not action:
            aliases = ["STOP", "stop", "[]"] + aliases
        found = next((value[key] for key in aliases if key in value), None)
        if found is None:
            raise _fail(where, f"{name} has no value for root action {root_index}")
        try:
            number = float(found)
        except (TypeError, ValueError) as exc:
            raise _fail(where, f"{name} value for root action {root_index} is invalid") \
                from exc
        if not math.isfinite(number):
            raise _fail(where, f"{name} value for root action {root_index} is not finite")
        out.append(number)
    return out


def _root_distribution(record: Mapping[str, Any], root: Mapping[str, Any],
                       actions: Sequence[tuple[int, ...]], n_opts: int,
                       where: str, schema: str, score_temperature: float) -> np.ndarray:
    target = next((container[name] for container in (root, record)
                   for name in ("soft_distribution", "target_policy", "root_policy",
                                "visit_distribution", "policy")
                   if name in container and container[name] is not None), None)
    source = "soft policy"
    if target is not None:
        if isinstance(target, Mapping):
            weights = _mapping_vector(target, actions, where, source)
        else:
            weights = _aligned_vector(
                target, actions, n_opts, where, source, nonnegative=True,
            )
    else:
        counts = next((container[name] for container in (root, record)
                       for name in ("counts", "visit_counts", "visits")
                       if name in container and container[name] is not None), None)
        if counts is not None:
            weights = _aligned_vector(
                counts, actions, n_opts, where, "search counts", nonnegative=True,
            )
            source = "search counts"
        else:
            scores = next((container[name] for container in (root, record)
                           for name in ("mean_scores", "search_scores", "scores",
                                        "q_values", "values")
                           if name in container and container[name] is not None), None)
            if scores is None:
                raise _fail(
                    where,
                    "record has no soft target, visit counts, or search scores",
                )
            score_values = _aligned_vector(
                scores, actions, n_opts, where, "search scores", nonnegative=False,
            )
            peak = max(score_values)
            weights = [math.exp(max((value - peak) / score_temperature, -80.0))
                       for value in score_values]
            source = "search-score softmax"

    total = float(sum(weights))
    if total <= 0:
        raise _fail(where, f"{source} has zero total mass")
    if schema == TEACHER_SCHEMA and target is not None \
            and abs(total - 1.0) > 1e-3:
        raise _fail(where, f"root.soft_distribution sums to {total}, not 1")
    policy_target = np.asarray(weights, dtype=np.float64) / total

    # Validate diagnostics even though neither scores nor terminal results are
    # training targets.  Misaligned vectors usually indicate a corrupted shard.
    for name, nonnegative in (("mean_scores", False), ("counts", True),
                              ("valid_particle_counts", True)):
        value = root.get(name)
        if value is None:
            continue
        diagnostic = _aligned_vector(
            value, actions, n_opts, where, f"root.{name}",
            nonnegative=nonnegative,
        )
        if name == "valid_particle_counts" and any(v <= 0 for v in diagnostic):
            raise _fail(where, "root.valid_particle_counts must all be positive")
    return policy_target


def _target_trie(actions: Sequence[tuple[int, ...]], distribution: np.ndarray,
                 n_opts: int, n_min: int, n_max: int,
                 where: str) -> tuple[TargetStep, ...]:
    if len(set(actions)) != len(actions):
        raise _fail(where, "multiple semantic root actions map to the same option indices")
    effective_max = n_opts if n_max <= 0 else min(n_max, n_opts)
    edges: dict[tuple[int, ...], defaultdict[int, float]] = {}
    reach: defaultdict[tuple[int, ...], float] = defaultdict(float)
    stop = n_opts
    for action, mass in zip(actions, distribution):
        if mass <= 0:
            continue
        tokens = list(action)
        if len(action) < effective_max:
            tokens.append(stop)
        prefix: tuple[int, ...] = ()
        for token in tokens:
            reach[prefix] += float(mass)
            if prefix not in edges:
                edges[prefix] = defaultdict(float)
            edges[prefix][token] += float(mass)
            if token == stop:
                break
            prefix = prefix + (token,)

    steps: list[TargetStep] = []
    for prefix in sorted(edges, key=lambda item: (len(item), repr(item))):
        available = np.ones(n_opts + 1, dtype=np.bool_)
        for picked in prefix:
            available[picked] = False
        available[stop] = len(prefix) >= n_min
        target = np.zeros(n_opts + 1, dtype=np.float32)
        node_mass = reach[prefix]
        if node_mass <= 0:
            continue
        for token, edge_mass in edges[prefix].items():
            if not 0 <= token <= n_opts or not available[token]:
                raise _fail(where, f"teacher trie contains illegal token {token} "
                                   f"after prefix {prefix}")
            target[token] = edge_mass / node_mass
        if not np.isclose(float(target.sum()), 1.0, atol=1e-5):
            raise _fail(where, "conditional teacher target does not sum to one")
        steps.append(TargetStep(prefix, available, target, node_mass))
    if not steps:
        raise _fail(where, "teacher target contains no positive-mass decisions")
    return tuple(steps)


def _matchup_key(record: Mapping[str, Any]) -> str:
    matchup = record.get("matchup")
    if matchup is not None:
        try:
            return _canonical_json(matchup).decode("utf-8")
        except (TypeError, ValueError):
            return repr(matchup)
    provenance = record.get("provenance")
    if isinstance(provenance, Mapping):
        deck = provenance.get("opponent_deck_sha256", "unknown-deck")
        opponent = provenance.get("opponent_policy", "unknown-policy")
        weights = provenance.get("opponent_weights_sha256", "no-weights")
        return f"{deck}:{opponent}:{weights}"
    return "unknown-matchup"


def _parse_record(record: Any, where: str, guard: ProvenanceGuard,
                  score_temperature: float, turn_bucket_size: int) -> TeacherSample:
    if not isinstance(record, Mapping):
        raise _fail(where, "JSONL row must be an object")
    schema = _schema(record)
    if schema is None:
        raise _fail(
            where,
            f"unsupported/missing schema; expected {TEACHER_SCHEMA!r}; "
            f"required fields: {REQUIRED_SCHEMA_FIELDS}",
        )
    game_id = record.get("game_id")
    if not isinstance(game_id, (str, int)) or isinstance(game_id, bool) \
            or str(game_id) == "":
        raise _fail(where, "game_id must be a non-empty string or integer")
    decision_value = record.get("decision_id", record.get("decision_index"))
    if decision_value is None:
        raise _fail(where, "record requires decision_id or decision_index")
    if not isinstance(decision_value, (str, int)) or isinstance(decision_value, bool):
        raise _fail(where, "decision_id must be a string or integer")

    observation = record.get("observation", record.get("obs"))
    if not isinstance(observation, dict):
        raise _fail(where, f"observation must be an object; required fields: "
                           f"{REQUIRED_SCHEMA_FIELDS}")
    select = observation.get("select")
    if not isinstance(select, dict) or not isinstance(select.get("option"), list) \
            or not select["option"]:
        raise _fail(where, "observation.select.option must be a non-empty list")
    view = ObsView(observation)
    n_opts = len(view.options)
    if not isinstance(view.min_count, int) or not isinstance(view.max_count, int):
        raise _fail(where, "observation selection bounds must be integers")

    guard.check(record, where, schema)
    actions, root = _root_actions(record, observation, view, where, schema)
    target = _root_distribution(
        record, root, actions, n_opts, where, schema, score_temperature,
    )
    if schema == TEACHER_SCHEMA:
        best_index = record.get("target_best_root_index")
        if not isinstance(best_index, int) or isinstance(best_index, bool) or \
                not 0 <= best_index < len(actions):
            raise _fail(where, "target_best_root_index is missing or invalid")
        expected_best = int(np.argmax(target))
        if best_index != expected_best:
            raise _fail(
                where,
                f"target_best_root_index {best_index} != policy argmax {expected_best}",
            )
        target_best = _validate_action(
            _as_indices(record.get("target_best_action"), where,
                        "target_best_action"),
            view, where, "target_best_action",
        )
        if target_best != actions[best_index]:
            raise _fail(
                where,
                "target_best_action does not match its semantic root action",
            )
    target_steps = _target_trie(
        actions, target, n_opts, view.min_count, view.max_count, where,
    )

    selected = record.get("selected_action")
    if selected is not None:
        selected_action = _validate_action(
            _as_indices(selected, where, "selected_action"),
            view, where, "selected_action",
        )
        if selected_action not in actions and set(selected_action) not in (
                set(action) for action in actions):
            raise _fail(where, "selected_action is absent from semantic root actions")

    turn = view.turn
    recorded_turn = record.get("turn")
    if recorded_turn is not None and recorded_turn != turn:
        raise _fail(where, f"record turn {recorded_turn!r} != observation turn {turn!r}")
    turn_number = turn if isinstance(turn, int) and turn >= 0 else 0
    context_key = f"type={view.select_type}:context={view.context}"
    turn_bucket = f"turn={turn_number // turn_bucket_size * turn_bucket_size}+"

    try:
        state = FE.encode_state(view)
        opt_ids, opt_feats = FE.encode_options(view)
    except Exception as exc:
        raise _fail(where, f"feature encoding failed: {exc!r}") from exc
    if opt_feats.shape != (n_opts + 1, FE.OPT_FEATS):
        raise _fail(where, f"feature encoder returned unexpected shape {opt_feats.shape}")
    return TeacherSample(
        state=state,
        opt_ids=opt_ids,
        opt_feats=opt_feats,
        target_steps=target_steps,
        game_id=str(game_id),
        decision_id=str(decision_value),
        n_opts=n_opts,
        n_min=view.min_count,
        n_max=view.max_count,
        balance_key=(_matchup_key(record), context_key, turn_bucket),
        source=where,
    )


def load_teacher_samples(paths: Sequence[str], learner_deck: Sequence[int],
                         score_temperature: float, turn_bucket_size: int
                         ) -> tuple[list[TeacherSample], dict[str, Any]]:
    guard = ProvenanceGuard(learner_deck)
    samples: list[TeacherSample] = []
    seen_decisions: set[tuple[str, str]] = set()
    input_info: list[dict[str, Any]] = []
    for path in paths:
        count = 0
        with open(path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                where = _where(path, line_no)
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _fail(where, f"invalid JSON: {exc.msg}") from exc
                sample = _parse_record(
                    record, where, guard, score_temperature, turn_bucket_size,
                )
                identity = (sample.game_id, sample.decision_id)
                if identity in seen_decisions:
                    raise _fail(where, f"duplicate game/decision id {identity!r}")
                seen_decisions.add(identity)
                samples.append(sample)
                count += 1
        input_info.append({
            "path": os.path.abspath(path),
            "sha256": _sha256_file(path),
            "bytes": os.path.getsize(path),
            "records": count,
        })
    if not samples:
        raise DatasetError("teacher inputs contain no non-empty records")
    metadata = {
        "inputs": input_info,
        "teacher_provenance": dict(guard.expected),
    }
    return samples, metadata


def split_by_game(samples: Sequence[TeacherSample], val_frac: float,
                  seed: int) -> tuple[list[TeacherSample], list[TeacherSample]]:
    games: defaultdict[str, list[TeacherSample]] = defaultdict(list)
    for sample in samples:
        games[sample.game_id].append(sample)
    game_ids = sorted(games)
    if len(game_ids) < 2:
        raise DatasetError(
            "held-out policy KL requires at least two distinct game_id groups"
        )
    random.Random(seed).shuffle(game_ids)
    n_val = max(1, min(len(game_ids) - 1, round(len(game_ids) * val_frac)))
    val_games = set(game_ids[:n_val])
    train = [sample for sample in samples if sample.game_id not in val_games]
    val = [sample for sample in samples if sample.game_id in val_games]
    if not train or not val:
        raise DatasetError("game-grouped train/validation split is empty")
    if {sample.game_id for sample in train} & {sample.game_id for sample in val}:
        raise AssertionError("game_id leakage across train/validation")
    return train, val


def apply_inverse_group_weights(samples: Sequence[TeacherSample]) -> dict[str, Any]:
    counts = Counter(sample.balance_key for sample in samples)
    raw = [1.0 / counts[sample.balance_key] for sample in samples]
    normalizer = len(raw) / max(sum(raw), 1e-12)
    for sample, value in zip(samples, raw):
        sample.weight = value * normalizer
    return {
        "groups": len(counts),
        "smallest_group": min(counts.values()),
        "largest_group": max(counts.values()),
        "min_weight": min(sample.weight for sample in samples),
        "max_weight": max(sample.weight for sample in samples),
    }


def _teacher_batch_objective(net: TorchNet, batch: Sequence[TeacherSample]
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    ids, hand, mdisc, odisc, scalars, opt_ids, opt_feats, mask = \
        _stack_minibatch(batch)
    state_vec = net.state_vec(ids, hand, mdisc, odisc, scalars)
    logits = net.logits(state_vec, opt_ids, opt_feats, mask)
    weighted_ce = torch.zeros((), device=DEV)
    weighted_kl = torch.zeros((), device=DEV)
    denominator = 0.0
    for row, sample in enumerate(batch):
        sample_ce = torch.zeros((), device=DEV)
        sample_kl = torch.zeros((), device=DEV)
        for step in sample.target_steps:
            available = torch.from_numpy(step.available).to(device=DEV)
            target = torch.from_numpy(step.target).to(device=DEV)
            conditional = logits[row, :sample.n_opts + 1].masked_fill(
                ~available, -1e9,
            )
            log_policy = F.log_softmax(conditional, dim=0)
            sample_ce = sample_ce - step.reach * (target * log_policy).sum()
            positive = target > 0
            sample_kl = sample_kl + step.reach * (
                target[positive] * (target[positive].log() - log_policy[positive])
            ).sum()
        weighted_ce = weighted_ce + sample.weight * sample_ce
        weighted_kl = weighted_kl + sample.weight * sample_kl
        denominator += sample.weight
    scale = max(denominator, 1e-12)
    return weighted_ce / scale, weighted_kl / scale


def _anchor_nll(net: TorchNet, anchor: Sequence[Any], rng: np.random.Generator,
                count: int) -> torch.Tensor:
    indices = rng.integers(0, len(anchor), size=min(count, len(anchor)))
    batch = [anchor[int(index)] for index in indices]
    ids, hand, mdisc, odisc, scalars, opt_ids, opt_feats, mask = \
        _stack_minibatch(batch)
    state_vec = net.state_vec(ids, hand, mdisc, odisc, scalars)
    logits = net.logits(state_vec, opt_ids, opt_feats, mask)
    log_probs = [
        picks_logprob(
            logits[row, :sample.n_opts + 1], sample.picks,
            sample.n_opts, sample.n_min, sample.n_max,
        )[0]
        for row, sample in enumerate(batch)
    ]
    weights = torch.tensor(
        [sample.weight for sample in batch], dtype=torch.float32, device=DEV,
    )
    return -(torch.stack(log_probs) * weights).sum() / weights.sum().clamp(min=1e-6)


@torch.no_grad()
def evaluate_policy_kl(net: TorchNet, samples: Sequence[TeacherSample],
                       batch_size: int) -> tuple[float, float]:
    net.eval()
    total_ce = total_kl = total_weight = 0.0
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        ce, kl = _teacher_batch_objective(net, batch)
        weight = sum(sample.weight for sample in batch)
        total_ce += float(ce) * weight
        total_kl += float(kl) * weight
        total_weight += weight
    return total_ce / max(total_weight, 1e-12), total_kl / max(total_weight, 1e-12)


def _policy_parameters(net: TorchNet) -> list[torch.nn.Parameter]:
    # The embedding/state trunk is shared with the value head.  Distillation
    # has no value target, so changing that trunk would silently invalidate the
    # exported value function even with v1/v2 frozen.  Train only the dedicated
    # option head; PPO/BC can later update the shared representation with a
    # legitimate value objective.
    modules = (net.o1, net.o2, net.o3)
    selected = [parameter for module in modules for parameter in module.parameters()]
    selected_ids = {id(parameter) for parameter in selected}
    for parameter in net.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    return selected


def _load_torch(path: str) -> Any:
    try:
        return torch.load(path, map_location=DEV, weights_only=True)
    except TypeError:  # Torch before weights_only was introduced.
        return torch.load(path, map_location=DEV)


def _state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping) or "emb.weight" not in payload \
            and "module.emb.weight" not in payload:
        raise ValueError("checkpoint is not a TorchNet state_dict")
    state: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        clean = str(key)
        for prefix in ("module.", "_orig_mod."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        state[clean] = value
    return state


def _infer_arch(state: Mapping[str, torch.Tensor]) -> tuple[int, int, int, int, int]:
    required = ("emb.weight", "s1.weight", "s2.weight", "o1.weight", "o2.weight")
    if any(key not in state or not hasattr(state[key], "shape") for key in required):
        raise ValueError("checkpoint omits TorchNet architecture tensors")
    return (
        int(state["emb.weight"].shape[1]),
        int(state["s1.weight"].shape[0]),
        int(state["s2.weight"].shape[0]),
        int(state["o1.weight"].shape[0]),
        int(state["o2.weight"].shape[0]),
    )


def _parse_arch(value: str | None) -> tuple[int, int, int, int, int]:
    if value is None:
        return DEFAULT_ARCH
    try:
        arch = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("--arch must contain five comma-separated integers") from exc
    if len(arch) != 5 or any(size <= 0 for size in arch):
        raise ValueError("--arch must contain five positive integers")
    return arch  # type: ignore[return-value]


def make_net(resume: str | None, arch_arg: str | None) -> tuple[TorchNet, tuple[int, ...]]:
    if resume:
        if not os.path.isfile(resume):
            raise ValueError(f"resume checkpoint {resume!r} does not exist")
        state = _state_dict(_load_torch(resume))
        inferred = _infer_arch(state)
        if arch_arg is not None and _parse_arch(arch_arg) != inferred:
            raise ValueError(
                f"--arch {_parse_arch(arch_arg)} does not match checkpoint {inferred}"
            )
        net = TorchNet(*inferred).to(DEV)
        net.load_state_dict(state, strict=True)
        return net, inferred
    arch = _parse_arch(arch_arg)
    return TorchNet(*arch).to(DEV), arch


@torch.no_grad()
def assert_numpy_roundtrip(net: TorchNet, sample: TeacherSample,
                           output_path: str) -> None:
    export_npz(net, output_path)
    with np.load(output_path) as weights:
        numpy_net = NPM.Net(weights)
    np_logits, np_value = numpy_net.forward(
        sample.state, sample.opt_ids, sample.opt_feats,
    )
    ids = torch.from_numpy(sample.state["ids"][None]).long().to(DEV)
    hand = torch.from_numpy(sample.state["hand_ids"][None]).long().to(DEV)
    mdisc = torch.from_numpy(sample.state["my_disc"][None]).long().to(DEV)
    odisc = torch.from_numpy(sample.state["opp_disc"][None]).long().to(DEV)
    scalars = torch.from_numpy(sample.state["scalars"][None]).to(DEV)
    opt_ids = torch.from_numpy(sample.opt_ids[None].astype(np.int64)).to(DEV)
    opt_feats = torch.from_numpy(sample.opt_feats[None]).to(DEV)
    mask = torch.ones(1, sample.opt_feats.shape[0], dtype=torch.bool, device=DEV)
    state_vec = net.state_vec(ids, hand, mdisc, odisc, scalars)
    torch_logits = net.logits(state_vec, opt_ids, opt_feats, mask)[0].cpu().numpy()
    torch_value = float(net.value(state_vec)[0])
    if not np.allclose(np_logits, torch_logits, atol=1e-4, rtol=1e-5):
        raise AssertionError(
            f"Torch/NumPy logit roundtrip mismatch: "
            f"{np.max(np.abs(np_logits - torch_logits))}"
        )
    if not math.isclose(np_value, torch_value, abs_tol=1e-4, rel_tol=1e-5):
        raise AssertionError("Torch/NumPy value-head roundtrip mismatch")


def _git_metadata() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return commit, dirty
    except Exception:
        return None, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "distill turn-search JSONL into the existing TorchNet policy prior "
            "(no PPO or terminal value targets)"
        ),
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="teacher JSONL file(s), directory/directories, or glob(s)",
    )
    parser.add_argument(
        "--resume", required=True,
        help="required TorchNet .pt state_dict; the shared trunk/value path is frozen",
    )
    parser.add_argument(
        "--out", default=os.path.join(ROOT, "tools", "checkpoints", "teacher",
                                      "weights.npz"),
        help="final NumPy inference weights (.npz is appended when omitted)",
    )
    parser.add_argument(
        "--ckpt-dir", default=os.path.join(ROOT, "tools", "checkpoints", "teacher"),
        help="directory for best.pt, latest.pt, metrics, and manifest",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--arch", default=None,
                        help="emb,s1,s2,o1,o2; inferred from --resume when omitted")
    parser.add_argument(
        "--bc-anchor", default=None,
        help="episode dir spec accepted by train.load_bc_samples",
    )
    parser.add_argument(
        "--bc-anchor-coef", "--bc-coef", dest="bc_anchor_coef",
        type=float, default=0.3,
        help="coefficient on expert pick-sequence NLL (default: 0.3)",
    )
    parser.add_argument("--bc-anchor-batch", type=int, default=256)
    parser.add_argument("--w-win", type=float, default=1.0)
    parser.add_argument("--w-draw", type=float, default=0.3)
    parser.add_argument("--w-loss", type=float, default=0.1)
    parser.add_argument(
        "--score-temperature", type=float, default=1.0,
        help="temperature used only when a compatibility record has scores but no policy",
    )
    parser.add_argument(
        "--turn-bucket-size", type=int, default=4,
        help="turns per inverse-frequency balance bucket",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="explicitly replace existing run artifacts in --ckpt-dir/--out",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.out.endswith(".npz"):
        args.out += ".npz"
    if args.epochs <= 0 or args.lr <= 0 or args.batch_size <= 0:
        parser.error("--epochs, --lr, and --batch-size must be > 0")
    if not 0.0 < args.val_frac < 1.0:
        parser.error("--val-frac must be strictly between 0 and 1")
    if args.bc_anchor_coef < 0 or args.bc_anchor_batch <= 0:
        parser.error("BC anchor coefficient must be >= 0 and batch size > 0")
    if args.score_temperature <= 0 or args.turn_bucket_size <= 0:
        parser.error("score temperature and turn bucket size must be > 0")
    if args.grad_clip <= 0:
        parser.error("--grad-clip must be > 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    try:
        input_paths = _expand_inputs(args.inputs)
        learner_deck = policy.load_deck()
        samples, dataset_meta = load_teacher_samples(
            input_paths, learner_deck, args.score_temperature,
            args.turn_bucket_size,
        )
        train_samples, val_samples = split_by_game(
            samples, args.val_frac, args.seed,
        )
        train_balance = apply_inverse_group_weights(train_samples)
        val_balance = apply_inverse_group_weights(val_samples)
        net, arch = make_net(args.resume, args.arch)
    except (DatasetError, ValueError, OSError) as exc:
        parser.error(str(exc))

    anchor = None
    if args.bc_anchor:
        anchor = load_bc_samples(
            args.bc_anchor, args.w_win, args.w_draw, args.w_loss,
        )
        if not anchor:
            parser.error("--bc-anchor yielded no usable expert decisions")

    frozen_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in net.state_dict().items()
        if not name.startswith(("o1.", "o2.", "o3."))
    }

    output_parent = os.path.dirname(os.path.abspath(args.out))
    manifest_path = os.path.join(args.ckpt_dir, "run_manifest.json")
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    latest_path = os.path.join(args.ckpt_dir, "latest.pt")
    metrics_path = os.path.join(args.ckpt_dir, "metrics.jsonl")
    artifact_paths = {
        os.path.abspath(path)
        for path in (manifest_path, best_path, latest_path, metrics_path, args.out)
    }
    existing = sorted(path for path in artifact_paths if os.path.exists(path))
    if existing and not args.overwrite:
        parser.error(
            "refusing to overwrite existing run artifacts; choose a new "
            "--ckpt-dir/--out or pass --overwrite: " + ", ".join(existing)
        )
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    git_commit, git_dirty = _git_metadata()
    manifest: dict[str, Any] = {
        "tool": "tools/train_teacher.py",
        "teacher_schema": TEACHER_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "inputs": dataset_meta["inputs"],
        "initialization": {
            "resume_path": os.path.abspath(args.resume) if args.resume else None,
            "resume_sha256": _sha256_file(args.resume) if args.resume else None,
        },
        "bc_anchor_provenance": _bc_anchor_provenance(args.bc_anchor),
        "teacher_provenance": dataset_meta["teacher_provenance"],
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "learner_deck_sha256": _sha256_json(list(learner_deck)),
        "feature_version": FE.FEAT_VERSION,
        "device": str(DEV),
        "torch_version": torch.__version__,
        "architecture": list(arch),
        "samples": {
            "total": len(samples),
            "train": len(train_samples),
            "validation": len(val_samples),
            "train_games": len({sample.game_id for sample in train_samples}),
            "validation_games": len({sample.game_id for sample in val_samples}),
        },
        "balance": {"train": train_balance, "validation": val_balance},
        "objective": {
            "teacher": "soft conditional pick/STOP cross-entropy",
            "selection_metric": "held-out game-grouped policy KL",
            "bc_anchor": "pick-sequence NLL" if anchor else None,
            "value_or_terminal_target": None,
            "trainable_modules": ["o1", "o2", "o3"],
            "value_preservation": "shared trunk and value head frozen",
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)

    policy_parameters = _policy_parameters(net)
    optimizer = torch.optim.Adam(policy_parameters, lr=args.lr)
    initial_ce, initial_kl = evaluate_policy_kl(net, val_samples, args.batch_size)
    best_kl = initial_kl
    best_epoch = 0
    torch.save(net.state_dict(), best_path)
    print(
        f"teacher: {len(train_samples)} train / {len(val_samples)} val decisions "
        f"from {len({sample.game_id for sample in samples})} games; "
        f"device={DEV} arch={arch}",
        flush=True,
    )
    print(
        f"epoch 000: val_ce={initial_ce:.6f} val_policy_kl={initial_kl:.6f}",
        flush=True,
    )

    order = np.arange(len(train_samples))
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps({
            "epoch": 0, "val_ce": initial_ce, "val_policy_kl": initial_kl,
        }, sort_keys=True) + "\n")
        for epoch in range(1, args.epochs + 1):
            net.train()
            rng.shuffle(order)
            train_ce_sum = anchor_sum = train_weight = 0.0
            for start in range(0, len(order), args.batch_size):
                batch = [train_samples[int(index)]
                         for index in order[start:start + args.batch_size]]
                teacher_ce, _ = _teacher_batch_objective(net, batch)
                loss = teacher_ce
                anchor_nll = None
                if anchor and args.bc_anchor_coef > 0:
                    anchor_nll = _anchor_nll(
                        net, anchor, rng, args.bc_anchor_batch,
                    )
                    loss = loss + args.bc_anchor_coef * anchor_nll
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_parameters, args.grad_clip)
                optimizer.step()

                batch_weight = sum(sample.weight for sample in batch)
                train_ce_sum += float(teacher_ce.detach()) * batch_weight
                train_weight += batch_weight
                if anchor_nll is not None:
                    anchor_sum += float(anchor_nll.detach()) * batch_weight

            val_ce, val_kl = evaluate_policy_kl(net, val_samples, args.batch_size)
            train_ce = train_ce_sum / max(train_weight, 1e-12)
            anchor_metric = anchor_sum / max(train_weight, 1e-12) \
                if anchor and args.bc_anchor_coef > 0 else None
            row = {
                "epoch": epoch,
                "train_teacher_ce": train_ce,
                "train_anchor_nll": anchor_metric,
                "val_ce": val_ce,
                "val_policy_kl": val_kl,
            }
            metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
            metrics_file.flush()
            torch.save(net.state_dict(), latest_path)
            improved = math.isfinite(val_kl) and val_kl < best_kl - 1e-12
            if improved:
                best_kl = val_kl
                best_epoch = epoch
                torch.save(net.state_dict(), best_path)
            anchor_text = (f" anchor_nll={anchor_metric:.6f}"
                           if anchor_metric is not None else "")
            print(
                f"epoch {epoch:03d}: train_ce={train_ce:.6f}{anchor_text} "
                f"val_ce={val_ce:.6f} val_policy_kl={val_kl:.6f}" +
                (" *" if improved else ""),
                flush=True,
            )

    net.load_state_dict(_state_dict(_load_torch(best_path)), strict=True)
    net.eval()
    for name, expected in frozen_state.items():
        actual = net.state_dict()[name].detach().cpu()
        if not torch.equal(actual, expected):
            raise AssertionError(f"frozen parameter changed during distillation: {name}")
    assert_numpy_roundtrip(net, val_samples[0], args.out)

    manifest["best"] = {
        "epoch": best_epoch,
        "validation_policy_kl": best_kl,
        "checkpoint": os.path.abspath(best_path),
        "output_npz": os.path.abspath(args.out),
        "output_sha256": _sha256_file(args.out),
        "torch_numpy_roundtrip": True,
        "frozen_trunk_value_preserved": True,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    print(
        f"best epoch {best_epoch}: held-out policy KL={best_kl:.6f}; "
        f"exported {args.out} (Torch/NumPy roundtrip ok)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
