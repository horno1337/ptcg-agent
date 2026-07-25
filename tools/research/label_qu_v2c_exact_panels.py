"""Label mined Qu-v2C roots with exact-state terminal outcome panels.

This is a research-only bridge between the privilege-separated root miner and
an eventual asymmetric critic.  It reconstructs each selected ladder root with
the native search ABI, evaluates every supported one-pick semantic action, and
continues both seats with the exact frozen production Qu-v2B public policy.

The emitted artifact contains derived terminal returns and public semantic
identities only.  Exact zones and serialized native state never enter the
output.  These panels are deliberately *not* direct actor-distillation labels:
they were produced from privileged exact state and must first be validated
through a separately held-out teacher/critic experiment.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import features as BASE_FEATURES  # noqa: E402
from agent import model  # noqa: E402
from agent import qu_v2_features as QF  # noqa: E402
from agent import turn_search as TS  # noqa: E402
from agent.obsview import ObsView  # noqa: E402
from tools.cabt import AgentSearch, _LIB_PATH  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools import il_dataset  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402
from tools.research import qu_v2c_splits as QU_V2C_SPLITS  # noqa: E402
from tools.research import select_qu_v2c_reliability_roots as RELIABILITY  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.exact-terminal-panels.v1"
DEFAULT_ROOT_DIR = MINE.DEFAULT_OUT
DEFAULT_WEIGHTS = MINE.DEFAULT_QU_V2B
DEFAULT_ROLLOUTS = 4
DEFAULT_HOP_CAP = 500
MIN_STABLE_REFLEX_MARGIN = 1e-6
SPLIT_SEED = QU_V2C_SPLITS.DEFAULT_SEED
SPLIT_NAMES = ("all", *QU_V2C_SPLITS.NAMES)

SENSITIVE_OUTPUT_KEYS = frozenset({
    CFO.EXACT_HIDDEN_KEY,
    "exact_hidden_payload",
    "search_begin_input",
})


class PanelError(RuntimeError):
    """An input, policy, or native rollout violated the panel contract."""


class PanelIncomplete(RuntimeError):
    """One root lacked a complete terminal panel without corrupting the run."""


class PanelRejected(RuntimeError):
    """One root is scientifically ineligible without corrupting the shard."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _json_semantic(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_semantic(item) for item in value]
    if isinstance(value, list):
        return [_json_semantic(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _json_semantic(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise PanelError(f"semantic value has unsupported type {type(value)}")


def _find_sensitive_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in SENSITIVE_OUTPUT_KEYS:
                return f"{path}.{key}"
            found = _find_sensitive_key(item, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_sensitive_key(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _release(search: Any, state: Mapping[str, Any] | None) -> None:
    if not isinstance(state, Mapping):
        return
    search_id = state.get("searchId")
    if isinstance(search_id, int) and not isinstance(search_id, bool):
        search.release(search_id)


def _registered_deck(value: Any, label: str) -> list[int]:
    if (not isinstance(value, list) or len(value) != 60
            or any(
                not isinstance(card_id, int) or isinstance(card_id, bool)
                or not 0 < card_id < BASE_FEATURES.N_CARD_IDS
                for card_id in value
            )):
        raise PanelError(f"{label} is not a valid 60-card registration")
    return list(value)


def _source_input(
        manifest: Mapping[str, Any], public_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    source = public_record.get("source")
    mining = manifest.get("mining")
    inputs = mining.get("inputs") if isinstance(mining, Mapping) else None
    if not isinstance(source, Mapping) or not isinstance(inputs, list):
        raise PanelError("root manifest/source provenance is malformed")
    candidates = [
        record for record in inputs
        if isinstance(record, Mapping)
        and record.get("sha256") == source.get("replay_sha256")
        and str(record.get("episode_id")) == str(source.get("episode_id"))
        and record.get("source") == source.get("source_submission")
        and record.get("learner_seat") == source.get("learner_seat")
    ]
    if len(candidates) != 1:
        raise PanelError(
            f"root {public_record.get('root_id')} does not resolve to exactly "
            "one source replay in the mining manifest"
        )
    return candidates[0]


def recover_registered_decks(
        manifest: Mapping[str, Any],
        public_record: Mapping[str, Any],
        replay_cache: dict[str, tuple[dict[str, Any], tuple[list[int], list[int]]]],
) -> tuple[list[int], list[int]]:
    """Recover and content-verify both seat registrations for one root."""
    source = public_record.get("source")
    source_input = _source_input(manifest, public_record)
    if not isinstance(source, Mapping):
        raise PanelError("root source is malformed")
    replay_sha = source_input.get("sha256")
    replay_path_raw = source_input.get("path")
    if not isinstance(replay_sha, str) or not isinstance(replay_path_raw, str):
        raise PanelError("source replay path/hash is malformed")

    cached = replay_cache.get(replay_sha)
    if cached is None:
        replay_path = Path(replay_path_raw).expanduser().resolve()
        try:
            raw = replay_path.read_bytes()
            replay = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise PanelError(f"cannot load source replay {replay_path}: {exc}") from exc
        if not isinstance(replay, dict) or _sha256_bytes(raw) != replay_sha:
            raise PanelError("source replay content hash drifted")
        decks = il_dataset.decks_from_document(replay)
        if set(decks) != {0, 1}:
            raise PanelError("source replay does not register exactly two decks")
        registrations = (
            _registered_deck(decks[0], "seat-0 source deck"),
            _registered_deck(decks[1], "seat-1 source deck"),
        )
        cached = (replay, registrations)
        replay_cache[replay_sha] = cached
    replay, registrations = cached

    learner_seat = source.get("learner_seat")
    if learner_seat not in (0, 1):
        raise PanelError("root has an invalid learner seat")
    learner_record = manifest.get("registered_learner_deck")
    if not isinstance(learner_record, Mapping):
        raise PanelError("manifest has no registered learner deck")
    learner = _registered_deck(
        learner_record.get("cards"), "manifest learner deck")
    if (_value_sha256(learner) != learner_record.get("sha256")
            or registrations[learner_seat] != learner):
        raise PanelError("source replay learner registration drifted")
    opponent = registrations[1 - learner_seat]
    if _value_sha256(opponent) != source.get("opponent_deck_sha256"):
        raise PanelError("source replay opponent registration hash drifted")

    source_step = source.get("source_step")
    steps = replay.get("steps")
    try:
        replay_obs = steps[source_step][learner_seat]["observation"]
    except (IndexError, KeyError, TypeError):
        raise PanelError("source replay does not contain the named root step")
    identity = public_record.get("identity")
    if (not isinstance(replay_obs, Mapping)
            or not isinstance(identity, Mapping)
            or CFO.public_root_fingerprint(replay_obs)
            != identity.get("public_root_fingerprint")):
        raise PanelError("source replay public root fingerprint drifted")
    return registrations


def game_split(public_record: Mapping[str, Any]) -> str:
    """Assign every root from one Kaggle episode to one deterministic split."""
    source = public_record.get("source")
    episode_id = source.get("episode_id") if isinstance(source, Mapping) else None
    try:
        return QU_V2C_SPLITS.split_for_episode(episode_id, SPLIT_SEED)
    except ValueError as exc:
        raise PanelError("root has no valid episode identity for game split") from exc


def public_actor_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only observation a Qu-v2B continuation may encode."""
    public = CFO.public_rollout_observation(obs)
    sensitive = _find_sensitive_key(public)
    if sensitive is not None:
        raise PanelError(f"public actor observation retains {sensitive}")
    return public


def strict_qu_v2b_action(
        net: Any, obs: Mapping[str, Any],
        registered_decks: Sequence[Sequence[int]],
) -> list[int]:
    """Run strict frozen Qu-v2B inference with the selecting seat's deck."""
    current = obs.get("current")
    selecting = current.get("yourIndex") if isinstance(current, Mapping) else None
    if selecting not in (0, 1) or len(registered_decks) != 2:
        raise PanelError("rollout observation/registration routing is malformed")
    public = public_actor_observation(obs)
    try:
        view = ObsView(public)
        if not view.options:
            raise PanelError("non-terminal rollout has no public options")
        sample = QF.encode_public_observation(
            public, registered_decks[selecting])
        QF.validate_public_features(sample)
        logits, _ = net.forward(sample)
        logits = np.asarray(logits)
        expected = len(view.options) + 1
        if (logits.shape != (expected,) or logits.dtype.kind != "f"
                or not np.isfinite(logits).all()):
            raise PanelError("frozen Qu-v2B returned invalid logits")
        action = model.decode_qu_v2(
            logits, len(view.options), view.min_count, view.max_count)
        normalized = CFO._valid_action(public, action)
        if normalized is None or normalized != action:
            raise PanelError("frozen Qu-v2B emitted an invalid action")
        return action
    except PanelError:
        raise
    except Exception as exc:
        raise PanelError(
            f"strict frozen Qu-v2B inference failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _rollout(
        search: Any,
        state: dict[str, Any],
        root_player: int,
        net: Any,
        registered_decks: Sequence[Sequence[int]],
        hop_cap: int,
) -> tuple[float, int]:
    for hop in range(hop_cap + 1):
        obs = state.get("observation") if isinstance(state, Mapping) else None
        if not isinstance(obs, dict):
            _release(search, state)
            raise PanelError("native rollout state has no observation")
        terminal = CFO._terminal_value(obs, root_player)
        if terminal is not None:
            _release(search, state)
            return float(terminal), hop
        if hop >= hop_cap:
            _release(search, state)
            raise PanelIncomplete("native rollout exceeded the hop cap")
        try:
            action = strict_qu_v2b_action(net, obs, registered_decks)
        except Exception:
            _release(search, state)
            raise
        search_id = state.get("searchId")
        if not isinstance(search_id, int) or isinstance(search_id, bool):
            _release(search, state)
            raise PanelError("native rollout state has no integer search ID")
        try:
            child = search.step(search_id, action)
        finally:
            _release(search, state)
        if not isinstance(child, dict):
            raise PanelError("native SearchStep failed during continuation")
        state = child
    raise AssertionError("unreachable rollout loop")


def _standard_errors(matrix: np.ndarray, reflex_index: int) -> tuple[
        list[float], list[float]]:
    count = matrix.shape[0]
    mean_se = matrix.std(axis=0, ddof=1) / math.sqrt(count)
    deltas = matrix - matrix[:, [reflex_index]]
    advantage_se = deltas.std(axis=0, ddof=1) / math.sqrt(count)
    return (
        [float(value) for value in mean_se],
        [float(value) for value in advantage_se],
    )


def evaluate_root(
        public_record: Mapping[str, Any],
        privileged_record: Mapping[str, Any],
        registered_decks: Sequence[Sequence[int]],
        net: Any,
        search: Any,
        rollouts: int,
        hop_cap: int = DEFAULT_HOP_CAP,
) -> dict[str, Any]:
    """Return one complete exact-state terminal action panel."""
    if rollouts < 4 or rollouts % 4:
        raise PanelError("rollouts must be a positive multiple of four")
    if hop_cap < 1:
        raise PanelError("hop cap must be positive")
    obs, hidden = VALIDATE.reconstruct_observation(
        public_record, privileged_record)
    root_id = public_record.get("root_id")
    source = public_record.get("source")
    if not isinstance(root_id, str) or not isinstance(source, Mapping):
        raise PanelError("root identity/source is malformed")
    root_player = source.get("learner_seat")
    if root_player not in (0, 1):
        raise PanelError("root learner seat is invalid")

    learner_deck = registered_decks[root_player]
    public_bound = copy.deepcopy(obs)
    public_bound.pop(CFO.EXACT_HIDDEN_KEY, None)
    privileged_features = PF.encode_privileged_observation(
        public_bound, privileged_record["exact_hidden_payload"], learner_deck)
    binding = privileged_record.get("binding")
    if (not isinstance(binding, Mapping)
            or binding.get("privileged_feature_sha256")
            != privileged_features.canonical_hash()):
        raise PanelError("privileged feature binding drifted")

    semantic_options = TS.semantic_options(obs)
    option_count = len(semantic_options)
    if (not 2 <= option_count <= 12
            or option_count
            != len(((obs.get("select") or {}).get("option") or ()))):
        raise PanelError("root no longer has a complete supported action panel")
    recorded_b = public_record.get("qu_v2b")
    recorded_semantic = (
        recorded_b.get("semantic_action")
        if isinstance(recorded_b, Mapping) else None)
    recorded_margin = (
        recorded_b.get("margin") if isinstance(recorded_b, Mapping) else None)
    if (not isinstance(recorded_b, Mapping)
            or not isinstance(recorded_margin, (int, float))
            or isinstance(recorded_margin, bool)
            or not math.isfinite(float(recorded_margin))
            or float(recorded_margin) < 0):
        raise PanelError("frozen Qu-v2B root record is malformed")
    if float(recorded_margin) <= MIN_STABLE_REFLEX_MARGIN:
        raise PanelRejected(
            "unstable_qu_v2b_numeric_tie",
            "frozen Qu-v2B top-two logit margin is too small to define a "
            "stable reflex baseline",
        )

    root_actions = tuple((token,) for token in semantic_options)
    reflex = strict_qu_v2b_action(net, obs, registered_decks)
    if len(reflex) != 1 or not 0 <= reflex[0] < option_count:
        raise PanelError("frozen Qu-v2B root action is not one-pick")
    reflex_index = reflex[0]
    if (recorded_b.get("action") != reflex
            or recorded_semantic
            != _json_semantic(TS.semantic_action(obs, reflex))):
        raise PanelError("frozen Qu-v2B root action drifted from mining")

    order_seed = int.from_bytes(
        hashlib.sha256(root_id.encode("ascii")).digest()[:8], "big")
    rows: list[list[float]] = []
    root_orders: list[list[int]] = []
    branch_orders: list[list[int]] = []
    rollout_hops: list[int] = []

    for repetition in range(rollouts):
        root_order = CFO.balanced_action_order(
            option_count, repetition, order_seed)
        branch_order = CFO.balanced_action_order(
            option_count, repetition, order_seed ^ 0x9E3779B97F4A7C15)
        root: Mapping[str, Any] | None = None
        children: list[dict[str, Any] | None] = [None] * option_count
        consumed: set[int] = set()
        try:
            # Revalidate every vector immediately before the ctypes boundary.
            hidden = CFO.validate_hidden_payload(
                obs, obs.get(CFO.EXACT_HIDDEN_KEY))
            root = search.begin(
                obs,
                hidden["my_deck"], hidden["my_prize"],
                hidden["opponent_deck"], hidden["opponent_prize"],
                hidden["opponent_hand"], hidden["opponent_active"],
                manual_coin=False,
            )
            if not isinstance(root, Mapping):
                raise PanelError("native SearchBegin failed")
            root_obs = root.get("observation")
            root_search_id = root.get("searchId")
            if (not isinstance(root_obs, dict)
                    or not isinstance(root_search_id, int)
                    or isinstance(root_search_id, bool)
                    or CFO.public_root_fingerprint(root_obs)
                    != CFO.public_root_fingerprint(obs)):
                raise PanelError("native SearchBegin reconstructed a different root")
            for action_index in root_order:
                mapped = TS.map_semantic_action(
                    root_obs, root_actions[action_index])
                if mapped is None or len(mapped) != 1:
                    raise PanelError(
                        "semantic root action did not round-trip exactly")
                child = search.step(root_search_id, mapped)
                if not isinstance(child, dict):
                    raise PanelError("native SearchStep failed at root")
                children[action_index] = child
            _release(search, root)
            root = None

            row: list[float | None] = [None] * option_count
            for action_index in branch_order:
                child = children[action_index]
                if not isinstance(child, dict):
                    raise PanelError("root action panel is incomplete")
                consumed.add(action_index)
                value, hops = _rollout(
                    search, child, root_player, net, registered_decks, hop_cap)
                row[action_index] = value
                rollout_hops.append(hops)
            if any(value is None for value in row):
                raise PanelError("terminal outcome row is incomplete")
            rows.append([float(value) for value in row])
            root_orders.append(list(root_order))
            branch_orders.append(list(branch_order))
        finally:
            _release(search, root)
            for action_index, child in enumerate(children):
                if action_index not in consumed:
                    _release(search, child)
            search.end()

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (rollouts, option_count) or not np.isfinite(matrix).all():
        raise PanelError("terminal outcome panel is incomplete or non-finite")
    means = matrix.mean(axis=0)
    advantages = means - means[reflex_index]
    mean_se, advantage_se = _standard_errors(matrix, reflex_index)
    reason, candidate_index, holdout = CFO.choose_with_holdout(
        matrix, reflex_index)
    visits = np.bincount(
        np.argmax(matrix, axis=1), minlength=option_count)

    return {
        "root_id": root_id,
        "source": {
            key: source.get(key)
            for key in (
                "episode_id", "source_submission", "source_step",
                "learner_seat", "learner_reward", "outcome",
                "replay_sha256", "opponent_archetype",
                "opponent_deck_sha256",
            )
        },
        "public_root_fingerprint": (
            public_record.get("identity") or {}
        ).get("public_root_fingerprint"),
        "semantic_root_actions": [
            _json_semantic(action) for action in root_actions
        ],
        "qu_v2b_root_action": {
            "index": reflex_index,
            "action": reflex,
            "semantic_action": _json_semantic(
                TS.semantic_action(obs, reflex)),
            "top_two_logit_margin": float(recorded_margin),
        },
        "raw_outcomes": matrix.tolist(),
        "root_step_orders": root_orders,
        "branch_rollout_orders": branch_orders,
        "mean_scores": [float(value) for value in means],
        "mean_standard_errors": mean_se,
        "advantages_over_qu_v2b": [
            float(value) for value in advantages
        ],
        "advantage_standard_errors": advantage_se,
        "argmax_visits": [int(value) for value in visits],
        "holdout_selection": {
            "reason": reason,
            "candidate_index": candidate_index,
            "diagnostics": holdout,
        },
        "rollout_diagnostics": {
            "requested_rollouts": rollouts,
            "completed_rollouts": len(rows),
            "rollout_hops_mean": (
                float(np.mean(rollout_hops)) if rollout_hops else 0.0),
            "rollout_hops_max": max(rollout_hops) if rollout_hops else 0,
            "engine_rng_seedable": False,
            "stochastic_transitions_common_random_number_paired": False,
        },
        "label_eligibility": {
            "asymmetric_critic_research": True,
            "direct_actor_distillation": False,
            "reason": (
                "terminal panel conditions on privileged exact hidden state; "
                "teacher transfer must pass a public held-out gate first"
            ),
        },
    }


def _source_hashes() -> dict[str, str]:
    paths = {
        "labeler": Path(__file__).resolve(),
        "root_validator": Path(VALIDATE.__file__).resolve(),
        "root_miner": Path(MINE.__file__).resolve(),
        "game_split": Path(QU_V2C_SPLITS.__file__).resolve(),
        "reliability_selector": Path(RELIABILITY.__file__).resolve(),
        "counterfactual_contract": Path(CFO.__file__).resolve(),
        "privileged_features": Path(PF.__file__).resolve(),
        "production_model": Path(model.__file__).resolve(),
        "production_public_features": Path(QF.__file__).resolve(),
        "semantic_mapping": Path(TS.__file__).resolve(),
        "engine_wrapper": ROOT / "tools" / "cabt.py",
        "cards_data": ROOT / "data" / "cards.json",
        "attacks_data": ROOT / "data" / "attacks.json",
    }
    return {label: _sha256_file(path) for label, path in paths.items()}


def validate_root_manifest_route(manifest: Mapping[str, Any]) -> str:
    """Accept only the original critical corpus or locked reliability subset."""
    selection_mode = manifest.get("selection_mode")
    selection_policy = manifest.get("selection_policy")
    if (
        selection_mode in (None, "critical")
        and selection_policy == MINE.CRITICAL_SELECTION_POLICY
    ):
        return "critical"
    derivation = manifest.get("derivation")
    parent = manifest.get("parent")
    diagnostics = (
        derivation.get("diagnostics")
        if isinstance(derivation, Mapping) else None)
    if (
        selection_mode == RELIABILITY.SELECTION_MODE
        and selection_policy == RELIABILITY.SELECTION_POLICY
        and manifest.get("development_only") is True
        and manifest.get("sealed_test") is False
        and isinstance(derivation, Mapping)
        and derivation.get("schema") == RELIABILITY.SCHEMA
        and derivation.get("selection_seed") == RELIABILITY.SELECTION_SEED
        and derivation.get("root_count") == RELIABILITY.ROOT_COUNT
        and derivation.get("unique_game_requirement")
        == RELIABILITY.ROOT_COUNT
        and isinstance(parent, Mapping)
        and parent.get("selection_mode") == "factual-critic"
        and parent.get("selection_policy")
        == MINE.FACTUAL_CRITIC_SELECTION_POLICY
        and _is_sha256(parent.get("manifest_sha256"))
    ):
        return "label-reliability-development"
    if (
        selection_mode == RELIABILITY.GENERALIZATION_SELECTION_MODE
        and selection_policy
        == RELIABILITY.GENERALIZATION_SELECTION_POLICY
        and manifest.get("development_only") is True
        and manifest.get("sealed_test") is False
        and isinstance(derivation, Mapping)
        and derivation.get("schema")
        == RELIABILITY.GENERALIZATION_SCHEMA
        and derivation.get("selection_seed")
        == RELIABILITY.GENERALIZATION_SELECTION_SEED
        and derivation.get("root_count") == RELIABILITY.ROOT_COUNT
        and derivation.get("unique_game_requirement")
        == RELIABILITY.ROOT_COUNT
        and derivation.get("exact_marginal_quotas") is False
        and isinstance(parent, Mapping)
        and parent.get("selection_mode") == "factual-critic"
        and parent.get("selection_policy")
        == MINE.FACTUAL_CRITIC_SELECTION_POLICY
        and _is_sha256(parent.get("manifest_sha256"))
    ):
        return "label-generalization-heldout"
    if (
        selection_mode == RELIABILITY.GENERALIZATION_SELECTION_MODE
        and selection_policy
        == RELIABILITY.GENERALIZATION_SELECTION_POLICY
        and manifest.get("development_only") is True
        and manifest.get("sealed_test") is False
        and isinstance(derivation, Mapping)
        and derivation.get("schema")
        == RELIABILITY.GENERALIZATION_SCHEMA
        and derivation.get("selection_seed")
        == RELIABILITY.GENERALIZATION_SELECTION_SEED
        and isinstance(derivation.get("root_count"), int)
        and derivation.get("root_count") >= 50
        and derivation.get("unique_game_requirement")
        == derivation.get("root_count")
        and derivation.get("exact_marginal_quotas") is False
        and isinstance(diagnostics, Mapping)
        and diagnostics.get("all_remaining_fresh_games_selected") is True
        and diagnostics.get("minimum_common_complete_games") == 50
        and _is_sha256(diagnostics.get("ordered_root_ids_sha256"))
        and isinstance(parent, Mapping)
        and parent.get("selection_mode") == "factual-critic"
        and parent.get("selection_policy")
        == MINE.FACTUAL_CRITIC_SELECTION_POLICY
        and _is_sha256(parent.get("manifest_sha256"))
    ):
        return "label-generalization-heldout"
    if (
        selection_mode
        == RELIABILITY.REPLICATION_CANDIDATE_SELECTION_MODE
        and selection_policy
        == RELIABILITY.REPLICATION_CANDIDATE_SELECTION_POLICY
        and manifest.get("development_only") is True
        and manifest.get("sealed_test") is False
        and isinstance(derivation, Mapping)
        and derivation.get("schema")
        == RELIABILITY.REPLICATION_CANDIDATE_SCHEMA
        and derivation.get("selection_seed")
        == RELIABILITY.REPLICATION_CANDIDATE_SELECTION_SEED
        and derivation.get("root_count")
        == RELIABILITY.REPLICATION_CANDIDATE_ROOT_COUNT
        and derivation.get("unique_game_requirement")
        == RELIABILITY.REPLICATION_CANDIDATE_ROOT_COUNT
        and derivation.get("exact_marginal_quotas") is False
        and isinstance(parent, Mapping)
        and parent.get("selection_mode") == "factual-critic"
        and parent.get("selection_policy")
        == MINE.FACTUAL_CRITIC_SELECTION_POLICY
        and _is_sha256(parent.get("manifest_sha256"))
    ):
        return "label-generalization-replication-candidates"
    if (
        selection_mode
        == RELIABILITY.REPLICATION_CANDIDATE_V2_SELECTION_MODE
        and selection_policy
        == RELIABILITY.REPLICATION_CANDIDATE_V2_SELECTION_POLICY
        and manifest.get("development_only") is True
        and manifest.get("sealed_test") is False
        and isinstance(derivation, Mapping)
        and derivation.get("schema")
        == RELIABILITY.REPLICATION_CANDIDATE_V2_SCHEMA
        and derivation.get("selection_seed")
        == RELIABILITY.REPLICATION_CANDIDATE_SELECTION_SEED
        and derivation.get("root_count")
        == RELIABILITY.REPLICATION_CANDIDATE_ROOT_COUNT
        and derivation.get("unique_game_requirement")
        == RELIABILITY.REPLICATION_CANDIDATE_ROOT_COUNT
        and derivation.get("exact_marginal_quotas") is False
        and isinstance(parent, Mapping)
        and parent.get("selection_mode") == "factual-critic"
        and parent.get("selection_policy")
        == MINE.FACTUAL_CRITIC_SELECTION_POLICY
        and _is_sha256(parent.get("manifest_sha256"))
    ):
        return "label-generalization-replication-candidates-v2"
    raise PanelError(
        "exact-panel v1 accepts only pre-registered critical roots or the "
        "locked label-reliability/generalization subsets")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise PanelError(f"stale partial output exists: {temporary}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--json-out")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="number of roots (0 means every root after --offset)",
    )
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument(
        "--split", choices=SPLIT_NAMES, default="all",
        help=(
            "deterministic game-level shard; seed 23 with "
            "train/validation/test thresholds 70/15/15"
        ),
    )
    parser.add_argument("--overwrite-result", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.rollouts < 4 or args.rollouts % 4:
        parser.error("--rollouts must be a positive multiple of 4")

    root_dir = Path(args.root_dir).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    output = (
        Path(args.json_out).expanduser().resolve()
        if args.json_out else root_dir / (
            f"exact-panels-{args.split}-o{args.offset}-"
            f"n{args.limit or 'all'}-r{args.rollouts}.json"
        )
    )
    if output.exists() and not args.overwrite_result:
        parser.error(f"result already exists: {output}")

    try:
        manifest, public, privileged = VALIDATE.load_root_artifacts(root_dir)
        root_route = validate_root_manifest_route(manifest)
        weights = manifest.get("weights")
        b_record = weights.get("qu_v2b") if isinstance(weights, Mapping) else None
        weights_sha = _sha256_file(weights_path)
        if (not isinstance(b_record, Mapping)
                or b_record.get("sha256") != MINE.FROZEN_QU_V2B_SHA256
                or weights_sha != MINE.FROZEN_QU_V2B_SHA256):
            raise PanelError("weights are not the frozen production Qu-v2B")
        net = model.load(str(weights_path))
        if (not isinstance(net, model.QuV2Net)
                or not getattr(net, "is_qu_v2", False)
                or getattr(net, "has_deck_adapter", False)):
            raise PanelError("could not load strict unadapted production Qu-v2B")

        QU_V2C_SPLITS.assert_feature_classes_do_not_cross_splits(
            public, SPLIT_SEED)
        paired = list(zip(public, privileged))
        if args.split != "all":
            paired = [
                pair for pair in paired if game_split(pair[0]) == args.split
            ]
        end = len(paired) if not args.limit else args.offset + args.limit
        selected = paired[args.offset:end]
        selected_public = [pair[0] for pair in selected]
        selected_privileged = [pair[1] for pair in selected]
        if not selected_public:
            raise PanelError("selected root shard is empty")
        if len(selected_public) != len(selected_privileged):
            raise PanelError("public/privileged root shard diverged")

        replay_cache: dict[
            str, tuple[dict[str, Any], tuple[list[int], list[int]]]
        ] = {}
        search = AgentSearch()
        panels: list[dict[str, Any]] = []
        rejected_roots: list[dict[str, Any]] = []
        for public_record, privileged_record in zip(
                selected_public, selected_privileged):
            registrations = recover_registered_decks(
                manifest, public_record, replay_cache)
            try:
                panels.append(evaluate_root(
                    public_record, privileged_record, registrations, net, search,
                    args.rollouts,
                ))
            except PanelIncomplete as exc:
                rejected_roots.append({
                    "root_id": public_record.get("root_id"),
                    "reason": "incomplete_terminal_panel",
                    "detail": str(exc),
                })
            except PanelRejected as exc:
                rejected_roots.append({
                    "root_id": public_record.get("root_id"),
                    "reason": exc.reason,
                    "detail": str(exc),
                })
        if not panels:
            raise PanelError("every selected root lacked a complete terminal panel")

        root_artifacts = manifest.get("artifacts")
        payload = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "contains_exact_hidden_card_ids": False,
            "contains_native_search_bytes": False,
            "derived_from_privileged_exact_hidden_state": True,
            "direct_actor_distillation_eligible": False,
            "authorized_use": (
                "asymmetric critic/teacher research followed by a separate "
                "public held-out transfer gate"
            ),
            "root_manifest_sha256": manifest.get("manifest_sha256"),
            "root_route": root_route,
            "root_artifacts": {
                label: {
                    "sha256": record.get("sha256"),
                    "records": record.get("records"),
                }
                for label, record in (
                    root_artifacts.items()
                    if isinstance(root_artifacts, Mapping) else ()
                )
                if isinstance(record, Mapping)
            },
            "shard": {
                "split": args.split,
                "eligible_roots_before_offset_limit": len(paired),
                "offset": args.offset,
                "limit": args.limit,
                "start_root_id": selected_public[0].get("root_id"),
                "end_root_id": selected_public[-1].get("root_id"),
                "requested_roots": len(selected_public),
                "completed_roots": len(panels),
                "rejected_roots": len(rejected_roots),
            },
            "rollout_contract": {
                "rollouts_per_root": args.rollouts,
                "all_supported_one_pick_actions": True,
                "root_order": "cyclic/reversed balanced pairs",
                "branch_order": "independently cyclic/reversed balanced pairs",
                "continuation": (
                    "strict frozen production Qu-v2B public inference for "
                    "both seats, registered deck routed by selecting seat"
                ),
                "controller_fallback": "none; any error invalidates the shard",
                "terminal_return_perspective": "learner/root seat",
                "holdout_split": (
                    "alternating complete forward/reverse pairs; disjoint "
                    "selection and confirmation"
                ),
                "game_split": {
                    **QU_V2C_SPLITS.contract(SPLIT_SEED),
                    "filter_before_offset_limit": True,
                },
                "hop_cap": DEFAULT_HOP_CAP,
            },
            "weights": {
                "qu_v2b_sha256": weights_sha,
            },
            "engine": {
                "library_sha256": _sha256_file(Path(_LIB_PATH).resolve()),
                "rng_seedable": False,
            },
            "source_files_sha256": _source_hashes(),
            "panels": panels,
            "root_rejections": rejected_roots,
        }
        sensitive = _find_sensitive_key(payload)
        if sensitive is not None:
            raise PanelError(f"output would expose sensitive field {sensitive}")
        payload["report_sha256"] = _value_sha256(payload)
        _atomic_json(output, payload)
    except (
        OSError, ValueError, VALIDATE.ValidationError, PanelError,
    ) as exc:
        parser.error(str(exc))

    print(
        f"Qu-v2C exact panels: {len(panels)} roots x "
        f"{args.rollouts} rollouts labeled; "
        f"{len(rejected_roots)} incomplete roots rejected",
        flush=True,
    )
    print(f"Report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
