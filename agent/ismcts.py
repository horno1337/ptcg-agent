"""SO-ISMCTS: Single-Observer Information Set Monte Carlo Tree Search.

Option C: fix the reason our determinized (PIMC) search lost on the ladder --
STRATEGY FUSION (planning as if we'll know the opponent's hidden cards). ISMCTS
builds ONE tree over information sets (keyed by the action sequence from the
root), sampling a fresh determinization each iteration but sharing statistics
across them, so it can't make decisions that depend on info it won't have.

Key detail (Cowling/Powley/Whitehouse 2012): in the UCB selection term, an
action's AVAILABILITY count (how many iterations it was legal) replaces the
parent visit count -- otherwise actions legal in few determinizations get all
exploration and no exploitation.

Reuses the engine plumbing + determinizer + deterministic leaf eval from
search_policy. PROTOTYPE: not wired into the dispatcher yet; driven by
tools/eval_search.py-style harness for validation first. Fails soft (returns
None) on any missing piece, exactly like search_policy.
"""

import math
import random
import time

from . import search_policy as SP
from .obsview import ObsView

C_UCB = 0.7                 # exploration constant (paper uses ~sqrt(2)/2)
ROLLOUT_CAP = 24            # reflex-net rollout depth cap after leaving the tree


class _Node:
    """One information-set node: per-action visit/reward/availability stats."""
    __slots__ = ("n", "w", "avail")

    def __init__(self):
        self.n = {}          # action-index -> visit count
        self.w = {}          # action-index -> summed reward (root player's view)
        self.avail = {}      # action-index -> availability count

    def ucb(self, legal: list[int]) -> int:
        """Pick among `legal` (legal in THIS determinization). Untried first,
        else max UCB with availability replacing the parent visit count."""
        untried = [a for a in legal if a not in self.n]
        if untried:
            return random.choice(untried)
        best, best_v = legal[0], -1e18
        for a in legal:
            q = self.w[a] / self.n[a]
            v = q + C_UCB * math.sqrt(math.log(max(self.avail[a], 1)) / self.n[a])
            if v > best_v:
                best, best_v = a, v
        return best


def _legal(state) -> list[int]:
    """Legal action indices at an engine search state (single-pick only)."""
    sel = (state.get("observation") or {}).get("select") or {}
    opts = sel.get("option") or []
    return list(range(len(opts)))


def decide(view: ObsView, net, my_deck_list: list[int],
           budget_s: float, max_iters: int = 400) -> list[int] | None:
    """SO-ISMCTS for a single-pick select. None -> caller falls back to reflex."""
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
    tree: dict[tuple, _Node] = {(): _Node()}
    t0 = time.monotonic()
    iters = 0

    try:
        while iters < max_iters and time.monotonic() - t0 < budget_s:
            preds = SP._predict(view, my_deck_list, rng)
            st = SP._parse(L.SearchBegin(
                SP._agent_ptr, sbi.encode("ascii"), len(sbi),
                SP._arr(preds[0]), SP._arr(preds[1]), SP._arr(preds[2]),
                SP._arr(preds[3]), SP._arr(preds[4]), SP._arr([]), 0))
            if not st or len(_legal(st)) != n_root:
                continue

            # --- descend: tree nodes ONLY at our decisions; the opponent and
            #     effects are environment (reflex), so the tree keys on our
            #     action sequence and aggregates over sampled opponent replies ---
            path, state, key, hops, expanded = [], st, (), 0, False
            while state and hops < 60:
                hops += 1
                cobs = state["observation"]
                cur = cobs.get("current") or {}
                if cur.get("result", -1) != -1:
                    break
                legal = _legal(state)
                if not legal:
                    break
                if cur.get("yourIndex") == root_player:
                    node = tree.setdefault(key, _Node())
                    untried = [a for a in legal if a not in node.n]
                    a = node.ucb(legal)
                    path.append((key, a, legal))
                    state = SP._parse(L.SearchStep(SP._agent_ptr,
                                                   state["searchId"], SP._arr([a]), 1))
                    key = key + (a,)
                    if untried:                 # expansion -> rollout
                        tree.setdefault(key, _Node())
                        expanded = True
                        break
                else:                           # opponent / effect = environment
                    act = SP._reflex(net, cobs)
                    if act is None:
                        break
                    state = SP._parse(L.SearchStep(SP._agent_ptr,
                                                   state["searchId"], SP._arr(act), len(act)))

            # --- rollout to a leaf, then deterministic eval (root player's view) ---
            depth = 0
            while state:
                cobs = state["observation"]
                cur = cobs.get("current") or {}
                if cur.get("result", -1) != -1 or depth >= ROLLOUT_CAP \
                        or not (cobs.get("select") or {}).get("option"):
                    break
                act = SP._reflex(net, cobs)
                if act is None:
                    break
                state = SP._parse(L.SearchStep(SP._agent_ptr,
                                               state["searchId"], SP._arr(act), len(act)))
                depth += 1
            reward = SP._value_det(state["observation"], root_player) if state else 0.0

            # --- backprop: visits/reward on the chosen path; availability for
            #     EVERY action legal in this determinization at each node ---
            for k, a, legal in path:
                node = tree[k]
                node.n[a] = node.n.get(a, 0) + 1
                node.w[a] = node.w.get(a, 0.0) + reward
                for la in legal:
                    node.avail[la] = node.avail.get(la, 0) + 1
            iters += 1
    except Exception:
        return None
    finally:
        try:
            L.SearchEnd(SP._agent_ptr)
        except Exception:
            pass

    root = tree[()]
    if not root.n:
        return None
    global last_iters
    last_iters = iters
    return [max(root.n, key=lambda a: root.n[a])]   # most-visited root action


last_iters = 0
