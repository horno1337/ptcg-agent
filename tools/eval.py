"""Local evaluation: our agent vs baselines, N games, win-rate summary.

Usage: python tools/eval.py [n_games] [opponent: random|first|self]
Caveat from other competitors: local results do NOT reliably predict ladder
rank. Use this only to catch crashes/regressions, not to compare decks.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kaggle_environments import make
from kaggle_environments.envs.cabt.cabt import random_agent, first_agent
from agent import agent as my_agent

def run(n_games=10, opponent="random"):
    opp = {"random": random_agent, "first": first_agent, "self": my_agent}[opponent]
    results = {"win": 0, "loss": 0, "draw": 0, "error": 0}
    t0 = time.time()
    for g in range(n_games):
        env = make("cabt", debug=True)
        # alternate seats to cancel first-player advantage
        agents = [my_agent, opp] if g % 2 == 0 else [opp, my_agent]
        me = 0 if g % 2 == 0 else 1
        steps = env.run(agents)
        r = steps[-1][me].reward
        status = steps[-1][me].status
        if status in ("ERROR", "INVALID", "TIMEOUT"):
            results["error"] += 1
        elif r == 1:
            results["win"] += 1
        elif r == -1:
            results["loss"] += 1
        else:
            results["draw"] += 1
        print(f"game {g+1}/{n_games}: reward={r} status={status}")
    dt = time.time() - t0
    n = max(n_games, 1)
    print(f"\nvs {opponent}: {results}  winrate={results['win']/n:.1%}  ({dt/n:.1f}s/game)")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    opp = sys.argv[2] if len(sys.argv) > 2 else "random"
    run(n, opp)
