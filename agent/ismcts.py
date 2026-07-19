"""PUCT-ISMCTS: AlphaGo-style search over information sets.

Option C for breaking the reflex-net ceiling. Our PIMC search lost on the ladder
(strategy fusion); vanilla UCB-ISMCTS needs thousands of iterations we can't
afford on the slow competition CPU. An RL engineer's advice: "AlphaGo style,
just fancy MCTS, is reasonable if you can't search like crazy." So this is PUCT
(policy-prior-guided MCTS) adapted for our constraints:

  * POLICY PRIOR from our existing net (softmax of its option logits) focuses the
    few hundred affordable iterations on plausible moves -- the AlphaZero trick.
  * LEAF VALUE from the DETERMINISTIC eval (prize/damage), NOT the net's value
    head -- that head degraded off-distribution and lost us v2/v3. Net for
    policy (what it's good at), determinism for value (what the net is bad at).
  * INFORMATION SETS via determinization: a fresh guessed world per iteration
    (search_policy._predict + SearchBegin), tree keyed by our action sequence,
    opponent/effects played as environment (reflex). This handles hidden info
    and dodges strategy fusion. No rollout -- expand-and-evaluate, so each
    iteration is cheap (a descent + one eval), fitting more of them in budget.

We have a LARGE time bank (~5-9s/decision available; see CLAUDE.md), so spend it.
PROTOTYPE: not wired into the dispatcher; validate with an eval harness first.
Fails soft (returns None) on any missing piece, like search_policy.
"""

import math
import os
import random
import time

import numpy as np

from . import features as FE
from . import search_policy as SP
from .obsview import ObsView

C_PUCT = 1.4                # exploration constant in the PUCT term


class _Node:
    __slots__ = ("n", "w", "prior")

    def __init__(self, prior):
        self.n = {}                     # action -> visits
        self.w = {}                     # action -> summed value (root's view)
        self.prior = prior              # action -> policy prior P(a)

    def puct(self, legal: list[int]) -> int:
        total = sum(self.n.values())
        sqrt_total = math.sqrt(total + 1)
        best, best_v = legal[0], -1e18
        for a in legal:
            na = self.n.get(a, 0)
            q = self.w.get(a, 0.0) / na if na else 0.0
            u = C_PUCT * self.prior.get(a, 1.0 / len(legal)) * sqrt_total / (1 + na)
            v = q + u
            if v > best_v:
                best, best_v = a, v
        return best


def _legal(state) -> list[int]:
    sel = (state.get("observation") or {}).get("select") or {}
    return list(range((len(sel.get("option") or []))))


def _priors(net, obs) -> dict:
    """Softmax of the net's option logits -> P(a) over legal single-pick actions."""
    v = ObsView(obs)
    st = FE.encode_state(v)
    cids, feats = FE.encode_options(v)
    logits, _ = net.forward(st, cids, feats)
    n_opts = feats.shape[0] - 1
    x = np.asarray(logits[:n_opts], dtype=np.float64)
    x -= x.max()
    e = np.exp(x)
    p = e / e.sum()
    return {a: float(p[a]) for a in range(n_opts)}


def _step(L, state, act):
    return SP._parse(L.SearchStep(SP._agent_ptr, state["searchId"],
                                  SP._arr(act), len(act)))


def decide(view: ObsView, net, my_deck_list: list[int],
           budget_s: float, max_iters: int = 600) -> list[int] | None:
    """PUCT-ISMCTS for a single-pick select. None -> caller falls back to reflex."""
    sel = view.select
    if sel is None or view.max_count != 1:
        return None
    n_root = len(view.options)
    if not 2 <= n_root <= SP.MAX_OPTS:
        return None
    obs = view.obs
    if not obs.get("search_begin_input"):
        return None
    L = SP._load_lib()
    if L is None or net is None:
        return None

    root_player = view.my_index
    sbi = obs["search_begin_input"]
    rng = random.Random(int(time.monotonic() * 1e6) & 0xFFFFFF)
    tree: dict[tuple, _Node] = {}
    t0 = time.monotonic()
    iters = 0

    try:
        while iters < max_iters and time.monotonic() - t0 < budget_s:
            preds = SP._predict(view, my_deck_list, rng)
            state = SP._parse(L.SearchBegin(
                SP._agent_ptr, sbi.encode("ascii"), len(sbi),
                SP._arr(preds[0]), SP._arr(preds[1]), SP._arr(preds[2]),
                SP._arr(preds[3]), SP._arr(preds[4]), SP._arr([]), 0))
            if not state or len(_legal(state)) != n_root:
                continue

            path, key, leaf_val, hops = [], (), 0.0, 0
            while state and hops < 80:
                hops += 1
                cobs = state["observation"]
                cur = cobs.get("current") or {}
                if cur.get("result", -1) != -1:
                    leaf_val = SP._value_det(cobs, root_player)
                    break
                legal = _legal(state)
                if not legal:
                    leaf_val = SP._value_det(cobs, root_player)
                    break
                if cur.get("yourIndex") != root_player:      # opponent = environment
                    act = SP._reflex(net, cobs)
                    if act is None:
                        break
                    state = _step(L, state, act)
                    continue
                node = tree.get(key)
                if node is None:                             # EXPAND + evaluate leaf
                    tree[key] = _Node(_priors(net, cobs))
                    leaf_val = SP._value_det(cobs, root_player)
                    break
                a = node.puct(legal)                         # SELECT via PUCT
                path.append((key, a))
                state = _step(L, state, [a])
                key = key + (a,)

            for k, a in path:                                # BACKPROP
                nd = tree[k]
                nd.n[a] = nd.n.get(a, 0) + 1
                nd.w[a] = nd.w.get(a, 0.0) + leaf_val
            iters += 1
    except Exception:
        if os.environ.get("PTCG_ISMCTS_DEBUG"):
            import traceback
            traceback.print_exc()
        return None
    finally:
        try:
            L.SearchEnd(SP._agent_ptr)
        except Exception:
            pass

    root = tree.get(())
    if not root or not root.n:
        return None
    global last_iters
    last_iters = iters
    return [max(root.n, key=lambda a: root.n[a])]            # most-visited action


last_iters = 0
