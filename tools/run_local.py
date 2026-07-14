"""Run one game on the local engine and write the replay JSON to replay.json.

Usage: python tools/run_local.py [opponent: random|first|self]
The JSON is the engine's VisualizeData output (one vis state per select) —
the same format the competition's battle viewer consumes.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cabt import run_battle, random_agent, first_agent
from agent import agent as my_agent

opp_name = sys.argv[1] if len(sys.argv) > 1 else "random"
opp = {"random": random_agent, "first": first_agent, "self": my_agent}[opp_name]

r = run_battle(my_agent, opp, collect_replay=True)
outcome = {0: "win (P0/me)", 1: f"loss (P1/{opp_name})", 2: "draw"}[r["result"]]
print(f"result: {outcome}  selects={r['selects']}  error={r['error']}")
if r["replay"]:
    with open("replay.json", "w") as f:
        f.write(r["replay"])
    print(f"wrote replay.json ({len(r['replay'])} bytes, "
          f"{len(json.loads(r['replay']))} states)")
