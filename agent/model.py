"""Numpy inference for the RL policy net (no torch in the submission).

Architecture (mirrored exactly by the torch twin in tools/train.py):
  card embedding [N_CARD_IDS x EMB]
  state trunk:  13 slot embeds + pooled hand/discard embeds + scalars
                -> 256 relu -> 128 relu = state_vec
  value head:   state_vec -> 64 relu -> 1 tanh
  option head:  [opt_feats | opt card embed | state_vec] -> 128 relu
                -> 64 relu -> 1 logit  (per option row, incl. virtual STOP)

Optional deck adapters are stored in the same archive under
``deck_adapter_*`` keys.  A canonical registered-deck match activates tiny
zero-initialized deltas on the existing semantic option/value heads.  Archives
without the complete optional group take the original path byte-for-byte.

Weights ship as agent/weights.npz (exported by tools/train.py). Loading is
lazy and failure-tolerant: `load()` returns None if the file is missing or
incompatible, and policy.py then keeps using the rule-based fallback.
"""

import os
from typing import Mapping

import numpy as np

from . import features

_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")

QU_V2_MODEL_SCHEMA = "ptcg.qu-v2a.model.v2"
QU_V2_FEATURE_SCHEMA = "ptcg.qu-v2a.public-relational.v4"
# The candidate records the research Torch/NumPy twin that produced its
# arrays.  Production contains only the NumPy half, so pin the reviewed source
# identity here and prove numerical parity in tests before packaging.
QU_V2_TRAINING_MODEL_SHA256 = (
    "1e8c957ad8abb4ef60e7cc4299d50188c6c32b0927e365b4f78a764bf1338f9b"
)

_QU_V2_LINEAR_NAMES = (
    "board1", "board_relation", "state1", "state2", "option1",
    "context1", "policy", "value1", "value2",
)
_QU_V2_KEYS = frozenset({
    "schema", "feature_schema", "feature_dependency_fingerprint",
    "model_implementation_sha256", "architecture", "embedding",
    *(f"{name}_{suffix}" for name in _QU_V2_LINEAR_NAMES
      for suffix in ("weight", "bias")),
})

_KEYS = ["emb", "s1w", "s1b", "s2w", "s2b",
         "v1w", "v1b", "v2w", "v2b",
         "o1w", "o1b", "o2w", "o2b", "o3w", "o3b"]

DECK_ADAPTER_VERSION = 2
_SUPPORTED_DECK_ADAPTER_VERSIONS = (1, 2)
_DECK_ADAPTER_KEYS = [
    "learner_deck",
    "deck_adapter_o3w",
    "deck_adapter_v2w", "deck_adapter_v2b",
]
_DECK_ADAPTER_V1_GROUP = ["deck_adapter_version", *_DECK_ADAPTER_KEYS]
_DECK_ADAPTER_V2_GROUP = [*_DECK_ADAPTER_V1_GROUP, "deck_adapter_select_type"]


class Net:
    def __init__(self, w: dict):
        for k in _KEYS:
            setattr(self, k, np.asarray(w[k], dtype=np.float32))
        self.emb_dim = self.emb.shape[1]  # all layer sizes derive from the file
        self.feat_version = int(w.get("feat_version", features.FEAT_VERSION))
        adapterish = {key for key in w.keys()
                      if key == "learner_deck"
                      or str(key).startswith("deck_adapter_")}
        self.deck_adapter_version = 0
        self.deck_adapter_select_type = None
        if adapterish:
            if "deck_adapter_version" not in adapterish:
                raise ValueError("deck adapter group is missing its version")
            raw_version = np.asarray(w["deck_adapter_version"])
            if (raw_version.shape != () or raw_version.dtype.kind not in "iu"
                    or raw_version.dtype.kind == "b"):
                raise ValueError("deck adapter version must be an integer scalar")
            self.deck_adapter_version = int(raw_version)
            if self.deck_adapter_version not in _SUPPORTED_DECK_ADAPTER_VERSIONS:
                raise ValueError(
                    f"unsupported deck adapter version {self.deck_adapter_version}")
            expected_group = set(
                _DECK_ADAPTER_V2_GROUP if self.deck_adapter_version >= 2
                else _DECK_ADAPTER_V1_GROUP)
            if adapterish != expected_group:
                missing = sorted(expected_group - adapterish)
                unknown = sorted(adapterish - expected_group)
                raise ValueError(
                    f"invalid deck adapter group; missing={missing}, "
                    f"unknown={unknown}")
            for k in _DECK_ADAPTER_KEYS:
                if k not in w:
                    raise ValueError(f"deck adapter is missing {k}")
            raw_target = np.asarray(w["learner_deck"])
            if raw_target.dtype.kind not in "iu" or raw_target.dtype.kind == "b":
                raise ValueError("deck adapter target must contain integer card IDs")
            target = raw_target.astype(np.int32, copy=False).reshape(-1)
            if (target.shape != (60,) or np.any(target <= 0)
                    or np.any(target >= self.emb.shape[0])):
                raise ValueError("deck adapter target must contain 60 valid card IDs")
            self.learner_deck = np.sort(target)
            for k in _DECK_ADAPTER_KEYS[1:]:
                raw = np.asarray(w[k])
                if raw.dtype != np.float32 or not np.isfinite(raw).all():
                    raise ValueError(
                        f"deck adapter {k} must be finite float32")
                setattr(self, k, raw)
            self._validate_adapter_shapes()
            self._policy_adapter_nonzero = bool(
                np.any(self.deck_adapter_o3w))
            self._value_adapter_nonzero = bool(
                np.any(self.deck_adapter_v2w)
                or np.any(self.deck_adapter_v2b))
            if self.deck_adapter_version >= 2:
                raw_select_type = np.asarray(w["deck_adapter_select_type"])
                if (raw_select_type.shape != ()
                        or raw_select_type.dtype.kind not in "iu"
                        or raw_select_type.dtype.kind == "b"):
                    raise ValueError(
                        "deck adapter select type must be an integer scalar")
                select_type = int(raw_select_type)
                if not -1 <= select_type <= 10:
                    raise ValueError("deck adapter select type is out of range")
                self.deck_adapter_select_type = (
                    None if select_type == -1 else select_type)

    @property
    def has_deck_adapter(self) -> bool:
        return self.deck_adapter_version in _SUPPORTED_DECK_ADAPTER_VERSIONS

    def supports_policy_context(self, opt_feats: np.ndarray) -> bool:
        """Whether the policy residual owns this SelectType context."""
        if self.deck_adapter_select_type is None:
            return True
        features_array = np.asarray(opt_feats)
        column = 17 + self.deck_adapter_select_type
        return (features_array.ndim == 2 and features_array.shape[0] > 0
                and features_array.shape[1] > column
                and bool(np.all(features_array[:, column] == 1.0)))

    def _validate_adapter_shapes(self) -> None:
        expected = {
            "deck_adapter_o3w": self.o3w.shape,
            "deck_adapter_v2w": self.v2w.shape,
            "deck_adapter_v2b": self.v2b.shape,
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"deck adapter {name} has shape {getattr(self, name).shape}, "
                    f"expected {shape}")

    def supports_deck(self, deck_ids) -> bool:
        """Whether this checkpoint's residual is registered for ``deck_ids``.

        Deck registration is a multiset; JSON/deck-file ordering must not
        change behavior.  Missing or malformed registrations fail closed into
        the frozen base policy.
        """
        if not self.has_deck_adapter or deck_ids is None:
            return False
        try:
            raw = np.asarray(deck_ids)
        except (TypeError, ValueError):
            return False
        if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
            return False
        deck = raw.astype(np.int32, copy=False).reshape(-1)
        return (deck.shape == (60,) and np.array_equal(
            np.sort(deck), self.learner_deck))

    def _state_vec(self, st: dict) -> np.ndarray:
        emb = self.emb

        def pool(ids):
            mask = ids > 0
            if not mask.any():
                return np.zeros(self.emb_dim, dtype=np.float32)
            return emb[ids[mask]].mean(axis=0)

        scalars = st["scalars"]
        if self.feat_version < 2:
            # v1 nets: prize scalars were dead (always 0) and 42-47 absent;
            # scalar growth is append-only so truncation restores the layout
            scalars = scalars[:42].copy()
            scalars[5] = scalars[6] = 0.0
        x = np.concatenate([
            emb[st["ids"]].reshape(-1),
            pool(st["hand_ids"]), pool(st["my_disc"]), pool(st["opp_disc"]),
            scalars,
        ])
        h = np.maximum(x @ self.s1w + self.s1b, 0.0)
        return np.maximum(h @ self.s2w + self.s2b, 0.0)

    def forward(self, st: dict, opt_ids: np.ndarray, opt_feats: np.ndarray,
                deck_ids=None):
        """Return option logits and value; ``deck_ids`` is the registration.

        ``deck_ids`` is deliberately separate from the observation feature
        contract.  A matching optional adapter sees it; old checkpoints,
        missing registrations and non-target decks execute the frozen base.
        """
        sv = self._state_vec(st)
        vh = np.maximum(sv @ self.v1w + self.v1b, 0.0)
        value_pre = vh @ self.v2w + self.v2b

        x = np.concatenate([
            opt_feats,
            self.emb[opt_ids],
            np.broadcast_to(sv, (opt_feats.shape[0], sv.shape[0])),
        ], axis=1)
        h = np.maximum(x @ self.o1w + self.o1b, 0.0)
        h = np.maximum(h @ self.o2w + self.o2b, 0.0)
        logits = (h @ self.o3w + self.o3b).reshape(-1)

        if self.supports_deck(deck_ids):
            if (self._policy_adapter_nonzero
                    and self.supports_policy_context(opt_feats)):
                logits = logits + (h @ self.deck_adapter_o3w).reshape(-1)
            if self._value_adapter_nonzero:
                value_pre = (value_pre + vh @ self.deck_adapter_v2w
                             + self.deck_adapter_v2b)

        value = float(np.tanh(value_pre)[0])
        return logits, value


def _qu_v2_scalar_text(weights, name: str) -> str:
    if name not in weights:
        raise ValueError(f"Qu-v2 artifact is missing {name}")
    value = np.asarray(weights[name])
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError(f"Qu-v2 artifact {name} is not a scalar string")
    return str(value.item())


def _qu_v2_float_array(weights, name: str,
                       shape: tuple[int, ...]) -> np.ndarray:
    if name not in weights:
        raise ValueError(f"Qu-v2 artifact is missing {name}")
    value = np.asarray(weights[name])
    if value.dtype != np.dtype(np.float32) or value.shape != shape:
        raise ValueError(
            f"invalid Qu-v2 {name}: got {value.dtype} {value.shape}, "
            f"expected float32 {shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"Qu-v2 {name} contains a non-finite value")
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    result.setflags(write=False)
    return result


def _qu_v2_expected_shapes(architecture: tuple[int, ...], qf):
    embedding, board_hidden, state_hidden, option_hidden, context_hidden = (
        architecture)
    board_input = embedding * 4 + qf.BOARD_FEATURES
    state_input = (
        board_hidden * 2 + embedding * 6
        + embedding * qf.PROMPT_ID_SLOTS + qf.PROMPT_FEATURES
    )
    option_input = qf.OPTION_FEATURES + embedding * 2 + state_hidden
    contextual_input = option_hidden * 3 + state_hidden
    value_hidden = max(state_hidden // 2, 1)
    return {
        "embedding": (features.N_CARD_IDS, embedding),
        "board1_weight": (board_input, board_hidden),
        "board1_bias": (board_hidden,),
        "board_relation_weight": (board_hidden * 2, board_hidden),
        "board_relation_bias": (board_hidden,),
        "state1_weight": (state_input, state_hidden),
        "state1_bias": (state_hidden,),
        "state2_weight": (state_hidden, state_hidden),
        "state2_bias": (state_hidden,),
        "option1_weight": (option_input, option_hidden),
        "option1_bias": (option_hidden,),
        "context1_weight": (contextual_input, context_hidden),
        "context1_bias": (context_hidden,),
        "policy_weight": (context_hidden, 1),
        "policy_bias": (1,),
        "value1_weight": (state_hidden, value_hidden),
        "value1_bias": (value_hidden,),
        "value2_weight": (value_hidden, 1),
        "value2_bias": (1,),
    }


def _qu_v2_masked_mean(values: np.ndarray, mask: np.ndarray,
                       axis: int) -> np.ndarray:
    weights = np.expand_dims(mask.astype(np.float32), -1)
    # ``min=``/``max=`` aliases were added to ndarray.clip in NumPy 2.1.
    # Kaggle may provide NumPy 1.x, whose stable signature is (a_min, a_max).
    denominator = weights.sum(axis=axis).clip(1.0, None)
    return (values * weights).sum(axis=axis) / denominator


def _qu_v2_masked_max(values: np.ndarray, mask: np.ndarray,
                      axis: int) -> np.ndarray:
    expanded = np.expand_dims(mask, -1)
    result = np.where(expanded, values, -np.inf).max(axis=axis)
    return np.where(np.isfinite(result), result, 0.0).astype(np.float32)


class QuV2Net:
    """Strict Torch-free production twin of the evaluated Qu-v2A network."""

    is_qu_v2 = True
    has_deck_adapter = False

    def __init__(self, weights: Mapping[str, np.ndarray]):
        # Keep the Torch-free production encoder inside the agent package.
        # Its embedded fingerprint is checked against the evaluated artifact;
        # runtime imports never depend on a top-level research/tools package.
        from . import qu_v2_features as qf

        names = frozenset(weights.keys())
        if names != _QU_V2_KEYS:
            missing = sorted(_QU_V2_KEYS - names)
            extra = sorted(names - _QU_V2_KEYS)
            raise ValueError(
                f"invalid Qu-v2 array set; missing={missing}, extra={extra}")
        if (_qu_v2_scalar_text(weights, "schema") != QU_V2_MODEL_SCHEMA
                or _qu_v2_scalar_text(weights, "feature_schema")
                != QU_V2_FEATURE_SCHEMA
                or qf.SCHEMA != QU_V2_FEATURE_SCHEMA):
            raise ValueError("incompatible Qu-v2 model/feature schema")
        if (_qu_v2_scalar_text(
                weights, "feature_dependency_fingerprint")
                != qf.assert_feature_dependency_lock()):
            raise ValueError("Qu-v2 feature dependency fingerprint drifted")
        if (_qu_v2_scalar_text(
                weights, "model_implementation_sha256")
                != QU_V2_TRAINING_MODEL_SHA256):
            raise ValueError("Qu-v2 training model identity drifted")

        architecture = np.asarray(weights["architecture"])
        if (architecture.shape != (5,)
                or architecture.dtype != np.dtype(np.int32)
                or np.any(architecture <= 0) or np.any(architecture > 4096)):
            raise ValueError("invalid Qu-v2 architecture metadata")
        self.architecture = tuple(int(value) for value in architecture)
        shapes = _qu_v2_expected_shapes(self.architecture, qf)
        self.embedding = _qu_v2_float_array(
            weights, "embedding", shapes["embedding"])
        if np.any(self.embedding[0] != 0.0):
            raise ValueError("Qu-v2 embedding padding row is not zero")
        for name in _QU_V2_LINEAR_NAMES:
            for suffix in ("weight", "bias"):
                key = f"{name}_{suffix}"
                setattr(self, key, _qu_v2_float_array(weights, key, shapes[key]))
        self._qf = qf

    @staticmethod
    def _relu(value: np.ndarray) -> np.ndarray:
        return np.maximum(value, np.float32(0.0))

    def _linear(self, value: np.ndarray, name: str) -> np.ndarray:
        return value @ getattr(self, f"{name}_weight") + getattr(
            self, f"{name}_bias")

    def _pool_ids(self, ids: np.ndarray) -> np.ndarray:
        mask = ids > 0
        embedded = self.embedding[ids]
        weights = np.expand_dims(mask.astype(np.float32), -1)
        denominator = weights.sum(axis=-2).clip(1.0, None)
        return (embedded * weights).sum(axis=-2) / denominator

    def _state_vector(self, sample) -> np.ndarray:
        board_mask = sample.board_features[:, 0] > 0.0
        board_input = np.concatenate([
            self.embedding[sample.board_ids],
            self._pool_ids(sample.board_energy_ids),
            self._pool_ids(sample.board_tool_ids),
            self._pool_ids(sample.board_evolution_ids),
            sample.board_features,
        ], axis=-1)
        board = self._relu(self._linear(board_input, "board1"))
        board = board * board_mask[:, None].astype(np.float32)
        board_global = _qu_v2_masked_mean(board, board_mask, axis=0)
        related = self._relu(self._linear(np.concatenate([
            board, np.broadcast_to(board_global, board.shape),
        ], axis=-1), "board_relation"))
        related = related * board_mask[:, None].astype(np.float32)
        board_mean = _qu_v2_masked_mean(related, board_mask, axis=0)
        board_max = _qu_v2_masked_max(related, board_mask, axis=0)
        state_input = np.concatenate([
            board_mean, board_max,
            self._pool_ids(sample.hand_ids),
            self._pool_ids(sample.my_discard_ids),
            self._pool_ids(sample.opponent_discard_ids),
            self._pool_ids(sample.looking_ids),
            self._pool_ids(sample.stadium_ids),
            self._pool_ids(sample.registered_deck_ids),
            self.embedding[sample.prompt_ids].reshape(-1),
            sample.prompt_features,
        ]).astype(np.float32, copy=False)
        hidden = self._relu(self._linear(state_input, "state1"))
        return self._relu(self._linear(hidden, "state2"))

    def forward(self, sample):
        self._qf.validate_public_features(sample)
        state = self._state_vector(sample)
        option_input = np.concatenate([
            sample.option_features,
            self.embedding[sample.option_ids],
            self.embedding[sample.option_target_ids],
            np.broadcast_to(state, (len(sample.option_ids), len(state))),
        ], axis=-1)
        option = self._relu(self._linear(option_input, "option1"))
        mask = sample.option_mask.astype(np.bool_)
        option_mean = _qu_v2_masked_mean(option, mask, axis=0)
        option_max = _qu_v2_masked_max(option, mask, axis=0)
        contextual = np.concatenate([
            option,
            np.broadcast_to(option_mean, option.shape),
            np.broadcast_to(option_max, option.shape),
            np.broadcast_to(state, (len(option), len(state))),
        ], axis=-1)
        logits = self._linear(
            self._relu(self._linear(contextual, "context1")), "policy",
        ).reshape(-1)
        logits = np.where(mask, logits, np.float32(-1e9)).astype(np.float32)
        value_hidden = self._relu(self._linear(state, "value1"))
        value = float(np.tanh(self._linear(value_hidden, "value2"))[0])
        if not np.isfinite(logits).all() or not np.isfinite(value):
            raise FloatingPointError("Qu-v2 inference produced non-finite output")
        return logits, value


def decode_qu_v2(logits: np.ndarray, n_options: int,
                 min_count: int, max_count: int) -> list[int]:
    """Greedy no-replacement decode with the evaluated virtual-STOP rules."""
    for name, value in (("n_options", n_options), ("min_count", min_count),
                        ("max_count", max_count)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    n_options, min_count, max_count = (
        int(n_options), int(min_count), int(max_count))
    if n_options < 0 or min_count < 0 or max_count < 0:
        raise ValueError("selection sizes cannot be negative")
    if max_count > 0 and min_count > max_count:
        raise ValueError("min_count exceeds max_count")
    if (not isinstance(logits, np.ndarray) or logits.ndim != 1
            or logits.shape != (n_options + 1,)
            or logits.dtype.kind != "f" or not np.isfinite(logits).all()):
        raise ValueError("invalid Qu-v2 logits")

    effective_min = min(min_count, n_options)
    effective_max = min(max_count, n_options) if max_count > 0 else n_options
    selected: list[int] = []
    available = np.ones(n_options + 1, dtype=np.bool_)
    stop = n_options
    while True:
        mask = available.copy()
        if len(selected) >= effective_max:
            mask[:n_options] = False
        mask[stop] = len(selected) >= effective_min
        if not mask.any():
            raise ValueError("selection contract has no legal action")
        choice = int(np.argmax(np.where(mask, logits, -np.inf)))
        if choice == stop:
            return selected
        selected.append(choice)
        available[choice] = False


def forward_registered(net, st: dict, opt_ids: np.ndarray,
                       opt_feats: np.ndarray, deck_ids):
    """Call a real deck adapter without breaking legacy three-argument nets.

    Several safety/evaluation harnesses intentionally use tiny duck-typed test
    doubles.  Preserve that contract for every unadapted net.
    """
    if getattr(net, "has_deck_adapter", False):
        return net.forward(st, opt_ids, opt_feats, deck_ids)
    return net.forward(st, opt_ids, opt_feats)


_cached = None
_cached_path = None


def load(path: str = _WEIGHTS_PATH) -> Net | None:
    global _cached, _cached_path
    resolved = os.path.abspath(path)
    if _cached is not None and _cached_path == resolved:
        return _cached
    try:
        with np.load(resolved, allow_pickle=False) as w:
            raw_schema = w.get("schema")
            schema = None
            if raw_schema is not None:
                value = np.asarray(raw_schema)
                if value.shape == () and value.dtype.kind in "US":
                    schema = str(value.item())
            if schema == QU_V2_MODEL_SCHEMA:
                loaded = QuV2Net(w)
            else:
                if not 1 <= int(w.get("feat_version", -1)) <= features.FEAT_VERSION:
                    return None
                loaded = Net(w)
        _cached, _cached_path = loaded, resolved
    except Exception:
        return None
    return _cached


def select_indices(logits: np.ndarray, n_opts: int, n_min: int, n_max: int) -> list[int]:
    """Greedy sequential pick with the virtual STOP action (index n_opts).

    Picks options in descending logit order; once n_min is satisfied, STOP
    competes with the remaining options. Result always respects
    [n_min, n_max] up to what's available (safety repairs the rest).
    """
    eff_max = min(n_max, n_opts) if n_max > 0 else n_opts
    order = np.argsort(-logits[:n_opts]).tolist()
    stop_logit = float(logits[n_opts])
    picks: list[int] = []
    for i in order:
        if len(picks) >= eff_max:
            break
        if len(picks) >= n_min and float(logits[i]) < stop_logit:
            break
        picks.append(int(i))
    return picks
