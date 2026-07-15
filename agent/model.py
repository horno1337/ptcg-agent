"""Numpy inference for the RL policy net (no torch in the submission).

Architecture (mirrored exactly by the torch twin in tools/train.py):
  card embedding [N_CARD_IDS x EMB]
  state trunk:  13 slot embeds + pooled hand/discard embeds + scalars
                -> 256 relu -> 128 relu = state_vec
  value head:   state_vec -> 64 relu -> 1 tanh
  option head:  [opt_feats | opt card embed | state_vec] -> 128 relu
                -> 64 relu -> 1 logit  (per option row, incl. virtual STOP)

Weights ship as agent/weights.npz (exported by tools/train.py). Loading is
lazy and failure-tolerant: `load()` returns None if the file is missing or
incompatible, and policy.py then keeps using the rule-based fallback.
"""

import os

import numpy as np

from . import features

EMB = 16
_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")

_KEYS = ["emb", "s1w", "s1b", "s2w", "s2b",
         "v1w", "v1b", "v2w", "v2b",
         "o1w", "o1b", "o2w", "o2b", "o3w", "o3b"]


class Net:
    def __init__(self, w: dict):
        for k in _KEYS:
            setattr(self, k, np.asarray(w[k], dtype=np.float32))

    def _state_vec(self, st: dict) -> np.ndarray:
        emb = self.emb

        def pool(ids):
            mask = ids > 0
            if not mask.any():
                return np.zeros(EMB, dtype=np.float32)
            return emb[ids[mask]].mean(axis=0)

        x = np.concatenate([
            emb[st["ids"]].reshape(-1),
            pool(st["hand_ids"]), pool(st["my_disc"]), pool(st["opp_disc"]),
            st["scalars"],
        ])
        h = np.maximum(x @ self.s1w + self.s1b, 0.0)
        return np.maximum(h @ self.s2w + self.s2b, 0.0)

    def forward(self, st: dict, opt_ids: np.ndarray, opt_feats: np.ndarray):
        """-> (logits [M], value scalar) where M includes the STOP row."""
        sv = self._state_vec(st)
        vh = np.maximum(sv @ self.v1w + self.v1b, 0.0)
        value = float(np.tanh(vh @ self.v2w + self.v2b)[0])

        x = np.concatenate([
            opt_feats,
            self.emb[opt_ids],
            np.broadcast_to(sv, (opt_feats.shape[0], sv.shape[0])),
        ], axis=1)
        h = np.maximum(x @ self.o1w + self.o1b, 0.0)
        h = np.maximum(h @ self.o2w + self.o2b, 0.0)
        logits = (h @ self.o3w + self.o3b).reshape(-1)
        return logits, value


_cached = None


def load(path: str = _WEIGHTS_PATH) -> Net | None:
    global _cached
    if _cached is not None:
        return _cached
    try:
        w = np.load(path)
        if int(w.get("feat_version", -1)) != features.FEAT_VERSION:
            return None
        _cached = Net(w)
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
