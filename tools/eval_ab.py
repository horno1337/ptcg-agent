"""Reflex-vs-reflex A/B between two weight files — the candidate gate.

Both seats run the deployable reflex config (policy net + rules on any gap)
piloting the frozen deck; only the weights differ, so the delta is the
training change and nothing else. Seats alternate each game; shard with
--seed across parallel workers and sum the RESULT lines.

  python tools/eval_ab.py 150 tools/checkpoints/bc-scout-v1.npz
  python tools/eval_ab.py 75 CAND.npz --seed 1   # second shard, other parity
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cabt import Battle                     # noqa: E402
from agent import features as FE            # noqa: E402
from agent import model                     # noqa: E402
from agent import policy                    # noqa: E402
from agent.obsview import ObsView           # noqa: E402


def load_net(path: str) -> model.Net:
    w = np.load(path)
    ver = int(w.get("feat_version", -1))
    if not 1 <= ver <= FE.FEAT_VERSION:
        raise SystemExit(f"{path}: incompatible feat_version {ver}")
    return model.Net(w)


def reflex_move(net: model.Net, obs: dict) -> list[int]:
    v = ObsView(obs)
    if not v.options:
        return policy.decide_rules(obs)
    st = FE.encode_state(v)
    cids, feats = FE.encode_options(v)
    logits, _ = net.forward(st, cids, feats)
    picks = model.select_indices(logits, feats.shape[0] - 1,
                                 v.min_count, v.max_count)
    return picks or policy.decide_rules(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", type=int)
    ap.add_argument("candidate", help="weights.npz for the candidate seat")
    ap.add_argument("--base", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "agent", "weights.npz"))
    ap.add_argument("--seed", type=int, default=0,
                    help="seat-parity shard id for parallel workers")
    a = ap.parse_args()

    cand, base = load_net(a.candidate), load_net(a.base)
    deck = policy.load_deck()

    w = l = d = 0
    for g in range(a.games):
        seat = (g + a.seed) % 2
        b = Battle(deck, deck)
        result = 2
        try:
            for _ in range(2000):
                obs, sp = b.obs()
                if obs["current"]["result"] != -1:
                    result = obs["current"]["result"]
                    break
                act = reflex_move(cand if sp == seat else base, obs)
                if b.select(list(act)):
                    result = 1 - sp
                    break
        finally:
            b.close()
        out = "D" if result == 2 else ("W" if result == seat else "L")
        if out == "W":
            w += 1
        elif out == "L":
            l += 1
        else:
            d += 1
        print(f"g{g} seat{seat} {out}", flush=True)

    n = max(w + l, 1)
    print(f"RESULT cand={a.candidate} W{w} L{l} D{d} "
          f"winrate={100.0 * w / n:.1f}%")


if __name__ == "__main__":
    main()
