"""Reflex-vs-reflex A/B between two weight files — the candidate gate.

Both seats run the deployable reflex config (policy net + rules on any gap)
piloting the frozen deck; only the weights differ, so the delta is the
training change and nothing else. Seats alternate each game; shard with
--seed across parallel workers and sum the RESULT lines.

  python tools/eval_ab.py 150 tools/checkpoints/bc-scout-v1.npz
  python tools/eval_ab.py 75 CAND.npz --seed 1   # second shard, other parity
  python tools/eval_ab.py 60 CAND.npz --opp meta:2   # generalization: each
      # net separately vs rules piloting meta deck 2; mirror games can't see
      # gains vs other archetypes, this can
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
    ap.add_argument("--opp", default="mirror",
                    help="mirror (cand vs base) or meta:<i> (each net "
                         "separately vs rules piloting meta deck i)")
    a = ap.parse_args()

    cand, base = load_net(a.candidate), load_net(a.base)
    deck = policy.load_deck()

    def series(tag, my_move, opp_move, opp_deck):
        w = l = d = 0
        for g in range(a.games):
            seat = (g + a.seed) % 2
            decks = [opp_deck, opp_deck]
            decks[seat] = deck
            b = Battle(decks[0], decks[1])
            result = 2
            try:
                for _ in range(2000):
                    obs, sp = b.obs()
                    if obs["current"]["result"] != -1:
                        result = obs["current"]["result"]
                        break
                    act = my_move(obs) if sp == seat else opp_move(obs)
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
            print(f"{tag} g{g} seat{seat} {out}", flush=True)
        n = max(w + l, 1)
        print(f"RESULT {tag} W{w} L{l} D{d} winrate={100.0 * w / n:.1f}%")
        return w, l

    if a.opp == "mirror":
        series(f"cand={a.candidate}",
               lambda o: reflex_move(cand, o),
               lambda o: reflex_move(base, o), deck)
        return

    import json
    idx = int(a.opp.split(":")[1])
    meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "agent", "meta_decks.json")
    with open(meta_path) as f:
        opp_deck = json.load(f)[idx]["deck"]
    cw, cl = series("cand-vs-meta", lambda o: reflex_move(cand, o),
                    policy.decide_rules, opp_deck)
    bw, bl = series("base-vs-meta", lambda o: reflex_move(base, o),
                    policy.decide_rules, opp_deck)
    print(f"DELTA cand {100.0 * cw / max(cw + cl, 1):.1f}% vs "
          f"base {100.0 * bw / max(bw + bl, 1):.1f}% "
          f"(meta deck {idx}, n={a.games} each)")


if __name__ == "__main__":
    main()
