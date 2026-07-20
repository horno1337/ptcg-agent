"""Synthetic anti-passive demos: champion reflex vs random, kaggle episode
format. Only the champion seat's actions are recorded (the random seat's
rows stay empty so il_dataset never imitates them). Wins only — the point
is demonstrations of closing out a passive opponent."""
import json, os, sys
ROOT = "/home/horn/Desktop/GIT-repo/ptcg-agent"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/tools")
import numpy as np
from cabt import Battle, random_agent
from agent import features as FE, model, policy
from agent.obsview import ObsView

net = model.Net(np.load(os.path.join(ROOT, "agent", "weights.npz")))

def reflex_move(obs):
    v = ObsView(obs)
    if not v.options:
        return policy.decide_rules(obs)
    st = FE.encode_state(v)
    cids, feats = FE.encode_options_for_net(v, net)
    logits, _ = net.forward(st, cids, feats)
    picks = model.select_indices(logits, feats.shape[0] - 1,
                                 v.min_count, v.max_count)
    return picks

def play_one(deck, champ_seat):
    b = Battle(deck, deck)
    steps = [[{"observation": None, "action": None} for _ in range(2)]]
    result = 2
    try:
        for _ in range(2000):
            obs, sp = b.obs()
            if obs["current"]["result"] != -1:
                result = obs["current"]["result"]
                break
            if sp == champ_seat:
                action = reflex_move(obs)
                slim = {"select": obs.get("select"), "current": obs.get("current")}
                row = [{"observation": None, "action": []},
                       {"observation": None, "action": []}]
                steps[-1][sp]["observation"] = slim
                row[sp]["action"] = list(action)
                steps.append(row)
            else:
                action = random_agent(obs)   # not recorded
            if b.select(list(action)):
                result = 1 - sp
                break
    finally:
        b.close()
    return steps, result

def main(out_dir, n_eps):
    os.makedirs(out_dir, exist_ok=True)
    deck = policy.load_deck()
    kept = tries = 0
    while kept < n_eps and tries < n_eps * 3:
        seat = tries % 2
        steps, result = play_one(deck, seat)
        tries += 1
        if result != seat:      # keep wins only
            continue
        rewards = [0, 0]
        rewards[seat], rewards[1 - seat] = 1, -1
        ep = {"steps": steps, "rewards": rewards,
              "info": {"TeamNames": ["champ" if i == seat else "chaos"
                                     for i in range(2)]}}
        with open(os.path.join(out_dir, f"ap_{kept:04d}.json"), "w") as f:
            json.dump(ep, f)
        kept += 1
    print(f"kept {kept} winning episodes from {tries} games")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
