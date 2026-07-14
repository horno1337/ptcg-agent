"""Sanity tests for the safety wrapper — run with: python tests/test_safety.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent import agent
from agent.safety import _repair

def sel(n_opts, min_c=1, max_c=1, st=0, ctx=0, types=None):
    opts = [{"type": (types[i] if types else 3), "index": i, "cardId": None} for i in range(n_opts)]
    return {"select": {"type": st, "context": ctx, "minCount": min_c, "maxCount": max_c, "option": opts},
            "current": {"yourIndex": 0, "turn": 1, "players": [{}, {}]}, "logs": []}

# deck selection
a = agent({"select": None, "current": None, "logs": []})
assert isinstance(a, list) and len(a) == 60, "deck must be 60 ids"

# every select type returns a legal count
for st in range(11):
    for n, mn, mx in [(1,1,1),(5,1,1),(5,2,3),(8,0,8),(3,3,3)]:
        obs = sel(n, mn, mx, st=st)
        a = agent(obs)
        assert all(0 <= i < n for i in a), (st, n, a)
        assert len(set(a)) == len(a), "duplicate indices"
        assert len(a) >= min(mn, n) and (mx == 0 or len(a) <= mx), (st, mn, mx, a)

# repair: garbage in, legal out
obs = sel(4, 1, 2)
assert _repair("garbage", obs)
assert _repair([99, -1, 2, 2, 3], obs) == [2, 3]
assert _repair([], obs) == [0]

# policy exception path: malformed observation must not raise
a = agent({"select": {"type": 0, "option": [{"type": 13}], "minCount": 1, "maxCount": 1}, "current": None})
assert a == [0]

print("all safety tests passed")
