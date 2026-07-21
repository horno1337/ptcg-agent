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

import numpy as np

from . import features

_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")

_KEYS = ["emb", "s1w", "s1b", "s2w", "s2b",
         "v1w", "v1b", "v2w", "v2b",
         "o1w", "o1b", "o2w", "o2b", "o3w", "o3b"]

DECK_ADAPTER_VERSION = 1
_DECK_ADAPTER_KEYS = [
    "learner_deck",
    "deck_adapter_o3w",
    "deck_adapter_v2w", "deck_adapter_v2b",
]
_DECK_ADAPTER_GROUP = ["deck_adapter_version", *_DECK_ADAPTER_KEYS]


class Net:
    def __init__(self, w: dict):
        for k in _KEYS:
            setattr(self, k, np.asarray(w[k], dtype=np.float32))
        self.emb_dim = self.emb.shape[1]  # all layer sizes derive from the file
        self.feat_version = int(w.get("feat_version", features.FEAT_VERSION))
        adapterish = {key for key in w.keys()
                      if key == "learner_deck"
                      or str(key).startswith("deck_adapter_")}
        unknown = adapterish - set(_DECK_ADAPTER_GROUP)
        if unknown:
            raise ValueError(f"unknown deck adapter keys: {sorted(unknown)}")
        present = adapterish & set(_DECK_ADAPTER_GROUP)
        if present and present != set(_DECK_ADAPTER_GROUP):
            missing = sorted(set(_DECK_ADAPTER_GROUP) - present)
            raise ValueError(f"partial deck adapter group; missing {missing}")
        self.deck_adapter_version = 0
        if present:
            raw_version = np.asarray(w["deck_adapter_version"])
            if (raw_version.shape != () or raw_version.dtype.kind not in "iu"
                    or raw_version.dtype.kind == "b"):
                raise ValueError("deck adapter version must be an integer scalar")
            self.deck_adapter_version = int(raw_version)
            if self.deck_adapter_version != DECK_ADAPTER_VERSION:
                raise ValueError(
                    f"unsupported deck adapter version {self.deck_adapter_version}")
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

    @property
    def has_deck_adapter(self) -> bool:
        return self.deck_adapter_version == DECK_ADAPTER_VERSION

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
            if self._policy_adapter_nonzero:
                logits = logits + (h @ self.deck_adapter_o3w).reshape(-1)
            if self._value_adapter_nonzero:
                value_pre = (value_pre + vh @ self.deck_adapter_v2w
                             + self.deck_adapter_v2b)

        value = float(np.tanh(value_pre)[0])
        return logits, value


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
