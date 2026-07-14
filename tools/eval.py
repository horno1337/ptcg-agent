"""Local evaluation: our agent vs baselines, N games, win-rate summary.

Usage: python tools/eval.py [n_games] [opponent: random|first|self] [--save-losses]
Runs on the locally built official engine (tools/build_engine.sh), no
kaggle-environments needed. --save-losses writes each lost game's replay to
replays/*.json; review them via tools/visualizer.html (official viewer).

Caveat from other competitors: local results do NOT reliably predict ladder
rank. Use this only to catch crashes/regressions, not to compare decks.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cabt import run_battle, random_agent, first_agent
from agent import agent as my_agent

_REPLAY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replays")


def run(n_games=10, opponent="random", save_losses=False):
    opp = {"random": random_agent, "first": first_agent, "self": my_agent}[opponent]
    results = {"win": 0, "loss": 0, "draw": 0, "error": 0}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    saved = 0
    t0 = time.time()
    for g in range(n_games):
        # alternate seats to cancel first-player advantage
        me = g % 2
        agents = [my_agent, opp] if me == 0 else [opp, my_agent]
        r = run_battle(*agents, collect_replay=save_losses)
        if r["error"] is not None and r["error"][0] == me:
            results["error"] += 1
        elif r["result"] == me:
            results["win"] += 1
        elif r["result"] == 1 - me:
            results["loss"] += 1
            if save_losses and r["replay"]:
                os.makedirs(_REPLAY_DIR, exist_ok=True)
                path = os.path.join(_REPLAY_DIR, f"{stamp}_{opponent}_g{g+1}_L.json")
                with open(path, "w") as f:
                    f.write(r["replay"])
                saved += 1
        else:
            results["draw"] += 1
        note = f" error={r['error']}" if r["error"] else ""
        print(f"game {g+1}/{n_games}: result={r['result']} me={me} "
              f"selects={r['selects']}{note}")
    dt = time.time() - t0
    n = max(n_games, 1)
    print(f"\nvs {opponent}: {results}  winrate={results['win']/n:.1%}  ({dt/n:.2f}s/game)")
    if saved:
        print(f"saved {saved} loss replays to replays/{stamp}_{opponent}_g*_L.json")


if __name__ == "__main__":
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 10
    opp = args[1] if len(args) > 1 else "random"
    run(n, opp, save_losses="--save-losses" in flags)
