"""Determinized value search at inference (2-ply, degrading gracefully).

For single-pick selects, instead of trusting the policy net's reflex, we:
  1. predict every hidden zone (our deck order/prizes from our known list;
     the opponent's hand/deck/prizes by matching their revealed cards
     against the mined meta-deck library in meta_decks.json),
  2. build K concrete worlds via the engine's SearchBegin,
  3. for each legal option, step it, roll forward with the reflex net —
     through the opponent's full reply (2-ply) when the clock affords it,
     to the end of our own turn (1-ply) when a det proves too expensive,
  4. pick the option with the best mean value across worlds — but only if
     at least MIN_DETS complete worlds fit the budget. A decision backed by
     one sampled world is an all-in bet on a single guess about hidden
     cards; measured worse than reflex, so we defer (return None) instead.

Self-contained: binds the engine lib directly (cg/ bundle on kaggle,
engine/libcg.so locally) and fails soft — any missing piece or exhausted
time budget returns None and the caller falls back to reflex play.
"""

import ctypes
import json
import os
import random
import time

import numpy as np

from . import features as FE
from . import model as NPM
from .obsview import ObsView

_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_CANDIDATES = [
    os.path.join(_DIR, "..", "cg", "libcg.so"),        # kaggle bundle (linux x86)
    os.path.join(_DIR, "..", "engine", "libcg.so"),    # local build
]
_META_PATH = os.path.join(_DIR, "meta_decks.json")

# Runtime search never beat reflex on ladder CPUs (v1 633, v2 465, v3 524 vs
# reflex v0 651; v3 replays: it overrode the net on 63% of contested picks).
# Retired at inference; the machinery stays for training-time generation and
# tools/eval_search.py, which force-enables it.
ENABLED = False

BUDGET_S = float(os.environ.get("PTCG_SEARCH_BUDGET", "1.5"))
_RESERVE_S = 150.0        # stop searching when overage drops below this
MAX_DETS = 16             # determinized worlds per decision (budget-bound)
MIN_DETS = 3              # evidence floor: fewer complete worlds -> reflex
_CAP_MULT = 2.0           # hard per-decision cap, in budgets (runaway det)
MAX_OPTS = 24             # only search single-pick selects up to this width
ROLLOUT_CAP = 48          # max reflex steps rolled per line (spans opp's reply)

_lib = None
_agent_ptr = None
_meta = None
last_dets = 0     # complete worlds behind the most recent decide() (diagnostics)


def _load_lib():
    global _lib, _agent_ptr
    if _lib is not None:
        return _lib
    for path in _LIB_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            L = ctypes.CDLL(path)
            # the engine aborts on double GameInitialize; the same .so may
            # already be loaded by tools/cabt.py in this process (PID-stamped:
            # a flag inherited by a child process must not skip its init)
            import hashlib
            flag = "_PTCG_INIT_" + hashlib.md5(
                os.path.realpath(path).encode()).hexdigest()[:12]
            if os.environ.get(flag) != str(os.getpid()):
                L.GameInitialize()
                os.environ[flag] = str(os.getpid())
            L.AgentStart.restype = ctypes.c_void_p
            L.SearchBegin.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
            ]
            L.SearchBegin.restype = ctypes.c_char_p
            L.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                     ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            L.SearchStep.restype = ctypes.c_char_p
            L.SearchEnd.argtypes = [ctypes.c_void_p]
            _agent_ptr = L.AgentStart()
            _lib = L
            return _lib
        except Exception:
            continue
    return None


def _meta_decks() -> list[list[int]]:
    global _meta
    if _meta is None:
        try:
            with open(_META_PATH) as f:
                _meta = [e["deck"] for e in json.load(f)]
        except Exception:
            _meta = []
    return _meta


def _arr(xs):
    return (ctypes.c_int * max(len(xs), 1))(*xs)


def _parse(raw):
    try:
        r = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except Exception:
        return None
    return None if r.get("error") else r.get("state")


def _seen_ids(p: dict, with_hand: bool) -> list[int]:
    ids = [c.get("id") for c in p.get("discard") or [] if isinstance(c, dict)]
    if with_hand:
        ids += [c.get("id") for c in p.get("hand") or [] if isinstance(c, dict)]
    for e in (p.get("active") or []) + (p.get("bench") or []):
        if isinstance(e, dict):
            ids.append(e.get("id"))
            for k in ("energyCards", "tools", "preEvolution"):
                ids += [x.get("id") for x in e.get(k) or [] if isinstance(x, dict)]
    return [i for i in ids if isinstance(i, int)]


def _remainder(full_list: list[int], seen: list[int]) -> list[int]:
    pool = list(full_list)
    for cid in seen:
        if cid in pool:
            pool.remove(cid)
    return pool


def _match_meta(seen: list[int]) -> tuple[list[int], float]:
    """Best-covering known decklist for an opponent's revealed cards and the
    coverage ratio (matched / seen). Low coverage means the opponent is NOT
    playing anything we know — searching a phantom world then hurts badly."""
    best, best_cover = None, 0
    for deck in _meta_decks():
        pool = list(deck)
        cover = 0
        for cid in seen:
            if cid in pool:
                pool.remove(cid)
                cover += 1
        if cover > best_cover or best is None:
            best, best_cover = deck, cover
    ratio = best_cover / len(seen) if seen else 0.0
    return best or [], ratio


def _opponent_model_confident(view: ObsView) -> bool:
    """Gate: search only when the opponent's revealed cards pin them to a
    known list. Measured: search is +25pts vs known decks, -42pts vs unknown."""
    opp = view.opp or {}
    seen = _seen_ids(opp, with_hand=False)
    if len(seen) < 4:
        return False          # too early to identify anyone: stay reflex
    _, ratio = _match_meta(seen)
    return ratio >= 0.9


def _sized(pool: list[int], sizes: list[int], rng) -> list[list[int]]:
    rng.shuffle(pool)
    need = sum(sizes)
    pool = (pool + [5] * need)[:need]   # pad with basic psychic energy
    out, i = [], 0
    for s in sizes:
        out.append(pool[i:i + s])
        i += s
    return out


def _predict(view: ObsView, my_deck_list: list[int], rng):
    me, opp = view.me or {}, view.opp or {}
    my_pool = _remainder(my_deck_list, _seen_ids(me, with_hand=True))
    my_prize, my_deck = _sized(my_pool, [len(me.get("prize") or []),
                                         me.get("deckCount") or 0], rng)
    opp_seen = _seen_ids(opp, with_hand=False)
    opp_list = _match_meta(opp_seen)[0] or my_deck_list
    opp_pool = _remainder(opp_list, opp_seen)
    opp_prize, opp_hand, opp_deck = _sized(
        opp_pool, [len(opp.get("prize") or []), opp.get("handCount") or 0,
                   opp.get("deckCount") or 0], rng)
    return my_deck, my_prize, opp_deck, opp_prize, opp_hand


def _reflex(net, obs: dict) -> list[int] | None:
    v = ObsView(obs)
    if not v.options:
        return None
    st = FE.encode_state(v)
    cids, feats = FE.encode_options(v)
    logits, _ = net.forward(st, cids, feats)
    picks = NPM.select_indices(logits, feats.shape[0] - 1, v.min_count, v.max_count)
    return picks or [0]


def _value(net, obs: dict, root_player: int) -> float:
    cur = obs.get("current") or {}
    res = cur.get("result", -1)
    if res != -1:
        return 0.0 if res == 2 else (1.0 if res == root_player else -1.0)
    v = ObsView(obs)
    st = FE.encode_state(v)
    cids, feats = FE.encode_options(v) if v.options else (
        np.zeros(1, dtype=np.int32), np.zeros((1, FE.OPT_FEATS), dtype=np.float32))
    _, val = net.forward(st, cids, feats)
    return val if v.my_index == root_player else -val


def decide(view: ObsView, net, my_deck_list: list[int]) -> list[int] | None:
    """One-ply determinized value search. None -> caller falls back."""
    if not ENABLED:
        return None
    sel = view.select
    if sel is None or view.max_count != 1:
        return None
    n = len(view.options)
    if not 2 <= n <= MAX_OPTS:
        return None
    obs = view.obs
    if not obs.get("search_begin_input"):
        return None
    rot = obs.get("remainingOverageTime")
    if isinstance(rot, (int, float)) and rot < _RESERVE_S:
        return None
    if not _opponent_model_confident(view):
        return None
    L = _load_lib()
    if L is None or net is None:
        return None

    t0 = time.monotonic()
    root_player = view.my_index
    rng = random.Random(int(t0 * 1e6) & 0xFFFFFF)
    scores = np.zeros(n)
    n_complete = 0
    two_ply = True

    try:
        det_cost = 0.0
        for _ in range(MAX_DETS):
            # only start a determinization we can finish for EVERY option:
            # partial dets compare options across different worlds and the
            # bias is worse than fewer, fully-paired samples
            spent = time.monotonic() - t0
            if spent + max(det_cost * 1.2, 0.02) > BUDGET_S:
                break
            d0 = time.monotonic()
            preds = _predict(view, my_deck_list, rng)
            sbi = obs["search_begin_input"]
            st = _parse(L.SearchBegin(
                _agent_ptr, sbi.encode("ascii"), len(sbi),
                _arr(preds[0]), _arr(preds[1]), _arr(preds[2]),
                _arr(preds[3]), _arr(preds[4]), _arr([]), 0))
            if not st:
                continue
            root_sel = (st["observation"].get("select") or {})
            if len(root_sel.get("option") or []) != n:
                continue
            det_scores = np.zeros(n)
            complete = True
            for i in range(n):
                if time.monotonic() - t0 > _CAP_MULT * BUDGET_S:
                    complete = False   # runaway det must not eat the clock
                    break
                child = _parse(L.SearchStep(_agent_ptr, st["searchId"], _arr([i]), 1))
                depth = 0
                flipped = False
                while child:
                    cobs = child["observation"]
                    cur = cobs.get("current") or {}
                    yi = cur.get("yourIndex")
                    if yi != root_player:
                        if not two_ply:
                            break        # 1-ply: score at the end of our turn
                        flipped = True   # the reflex net now plays their reply
                    if cur.get("result", -1) != -1 or depth >= ROLLOUT_CAP \
                            or (flipped and yi == root_player) \
                            or not (cobs.get("select") or {}).get("option"):
                        break            # terminal, cap, or our next turn began
                    act = _reflex(net, cobs)
                    if act is None:
                        break
                    child = _parse(L.SearchStep(_agent_ptr, child["searchId"],
                                                _arr(act), len(act)))
                    depth += 1
                if not child:
                    complete = False
                    break
                det_scores[i] = _value(net, child["observation"], root_player)
            if complete:
                scores += det_scores
                n_complete += 1
            det_cost = time.monotonic() - d0
            if two_ply and det_cost > BUDGET_S / MIN_DETS:
                # this hardware can't fit MIN_DETS 2-ply worlds in budget:
                # drop the opponent-reply ply rather than starve the sample
                # (1-ply ≈ half the rollout; re-estimate cost accordingly)
                two_ply = False
                det_cost *= 0.5
    except Exception:
        return None
    finally:
        try:
            L.SearchEnd(_agent_ptr)
        except Exception:
            pass

    global last_dets
    last_dets = n_complete
    if n_complete < MIN_DETS:
        return None   # never act on evidence from fewer worlds than the floor
    return [int((scores / n_complete).argmax())]
