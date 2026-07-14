"""Local evaluation: our agent vs baselines, N games, win-rate summary.

Usage: python tools/eval.py [n_games] [opponent: random|first|self]
Runs on the locally built official engine (tools/build_engine.sh), no
kaggle-environments needed.

Caveat from other competitors: local results do NOT reliably predict ladder
rank. Use this only to catch crashes/regressions, not to compare decks.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cabt import run_battle, random_agent, first_agent
from agent import agent as my_agent


def run(n_games=10, opponent="random"):
    opp = {"random": random_agent, "first": first_agent, "self": my_agent}[opponent]
    results = {"win": 0, "loss": 0, "draw": 0, "error": 0}
    t0 = time.time()
    for g in range(n_games):
        # alternate seats to cancel first-player advantage
        me = g % 2
        agents = [my_agent, opp] if me == 0 else [opp, my_agent]
        r = run_battle(*agents)
        if r["error"] is not None and r["error"][0] == me:
            results["error"] += 1
        elif r["result"] == me:
            results["win"] += 1
        elif r["result"] == 1 - me:
            results["loss"] += 1
        else:
            results["draw"] += 1
        note = f" error={r['error']}" if r["error"] else ""
        print(f"game {g+1}/{n_games}: result={r['result']} me={me} "
              f"selects={r['selects']}{note}")
    dt = time.time() - t0
    n = max(n_games, 1)
    print(f"\nvs {opponent}: {results}  winrate={results['win']/n:.1%}  ({dt/n:.2f}s/game)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    opp = sys.argv[2] if len(sys.argv) > 2 else "random"
    run(n, opp)
