"""Flywheel generation: the SEARCH agent self-plays and its games are
written in the kaggle episode format, so il_dataset/train.py consume them
unchanged (winner-seat weighting then clones the winning searcher's moves).

Usage:
  PTCG_SEARCH_BUDGET=0.3 python tools/selfplay_search.py OUT_DIR N_GAMES [WORKER_ID]

Run several instances in parallel (one per core); each writes
OUT_DIR/sp_<worker>_<n>.json. Only the fields il_dataset reads are written:
steps[t][p].{observation, action}, rewards, info.TeamNames.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cabt import Battle  # noqa: E402
from agent import policy  # noqa: E402
from agent.safety import agent as search_agent  # noqa: E402


def play_one(deck):
    """Returns (steps, rewards) in episode-log layout: the action recorded at
    step t answers the observation at step t-1, matching real kaggle logs."""
    b = Battle(deck, deck)
    steps = [[{"observation": None, "action": None} for _ in range(2)]]
    result = 2
    try:
        for _ in range(2000):
            obs, sp = b.obs()
            if obs["current"]["result"] != -1:
                result = obs["current"]["result"]
                break
            # per-game clock so safety never trips its process-lifetime fallback
            obs["remainingOverageTime"] = 600.0
            action = search_agent(obs)
            # store WITHOUT search_begin_input (large base64 blob)
            slim = {"select": obs.get("select"), "current": obs.get("current")}
            row = [{"observation": None, "action": []},
                   {"observation": None, "action": []}]
            # obs at step t-1 pairs with action at step t: store obs on the
            # PREVIOUS row, action on this one (same convention as kaggle)
            steps[-1][sp]["observation"] = slim
            row[sp]["action"] = list(action)
            steps.append(row)
            if b.select(list(action)):
                result = 1 - sp
                break
    finally:
        b.close()
    rewards = [0, 0] if result == 2 else [1 if result == 0 else -1,
                                          1 if result == 1 else -1]
    return steps, rewards


def main(out_dir, n_games, worker):
    os.makedirs(out_dir, exist_ok=True)
    deck = policy.load_deck()
    for g in range(n_games):
        steps, rewards = play_one(deck)
        ep = {"steps": steps, "rewards": rewards,
              "info": {"TeamNames": [f"sp{worker}", f"sp{worker}"]}}
        path = os.path.join(out_dir, f"sp_{worker}_{g:05d}.json")
        with open(path, "w") as f:
            json.dump(ep, f)
        if (g + 1) % 10 == 0:
            print(f"worker {worker}: {g + 1}/{n_games}", flush=True)
    print(f"worker {worker}: done ({n_games} games)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "0")
