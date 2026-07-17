"""Search-vs-reflex A/B eval at configurable compute — the pre-ship check.

The ladder runs on much slower CPUs than the dev box, and determinized
search degrades with hardware: fewer complete worlds fit the per-decision
budget. cvkpaper-v2 shipped a config that had only ever been measured at
dev-box compute and inverted on the ladder (single-det decisions are worse
than reflex). Emulate slow hardware with --budget before every ship.

  python tools/eval_search.py 20 --budget 0.35 --seed 0     # kaggle-like
  python tools/eval_search.py 20 --budget 0.35 --min-dets 1 # v2-ship repro
  python tools/eval_search.py 20 --opp meta:3               # vs known deck

The search seat runs the full dispatcher (search -> reflex -> rules) with a
realistic 600s per-game clock; the opponent seat is reflex (mirror) or
rules piloting a meta-library deck. Seats alternate each game; shard with
--seed across parallel workers and sum the RESULT lines.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cabt import Battle                     # noqa: E402
from agent import features as FE            # noqa: E402
from agent import model                     # noqa: E402
from agent import policy                    # noqa: E402
from agent import search_policy as SP       # noqa: E402
from agent.obsview import ObsView           # noqa: E402


def reflex_move(obs):
    """The v0 baseline: policy net only, rules on any gap."""
    net = model.load()
    v = ObsView(obs)
    if net is None or not v.options:
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
    ap.add_argument("--budget", type=float, default=SP.BUDGET_S,
                    help="per-decision search budget; lower = slower hardware")
    ap.add_argument("--min-dets", type=int, default=None,
                    help="override the evidence floor (1 reproduces v2)")
    ap.add_argument("--cap-mult", type=float, default=None,
                    help="override the per-decision hard cap (v2 had none: "
                         "use a huge value so the first det always completes)")
    ap.add_argument("--opp", default="mirror",
                    help="mirror (reflex opponent) or meta:<i> (rules pilot)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seat-parity shard id for parallel workers")
    a = ap.parse_args()

    SP.ENABLED = True   # the harness measures search; the ladder default is off
    SP.BUDGET_S = a.budget
    if a.min_dets is not None:
        SP.MIN_DETS = a.min_dets
    if a.cap_mult is not None:
        SP._CAP_MULT = a.cap_mult
    my_deck = policy.load_deck()
    if a.opp == "mirror":
        opp_deck, opp_move = my_deck, reflex_move
    else:
        idx = int(a.opp.split(":")[1])
        with open(SP._META_PATH) as f:
            opp_deck = json.load(f)[idx]["deck"]
        opp_move = policy.decide_rules

    w = l = d = 0
    acted = deferred = 0
    hist: dict[int, int] = {}
    think = []
    for g in range(a.games):
        seat = (g + a.seed) % 2
        decks = [opp_deck, opp_deck]
        decks[seat] = my_deck
        b = Battle(decks[0], decks[1])
        spent = 0.0
        result = 2
        try:
            for _ in range(2000):
                obs, sp = b.obs()
                if obs["current"]["result"] != -1:
                    result = obs["current"]["result"]
                    break
                if sp == seat:
                    obs["remainingOverageTime"] = max(600.0 - spent, 0.0)
                    SP.last_dets = -1
                    t = time.monotonic()
                    act = policy.decide(obs)
                    spent += time.monotonic() - t
                    if SP.last_dets >= 0:      # det loop actually ran
                        hist[SP.last_dets] = hist.get(SP.last_dets, 0) + 1
                        if SP.last_dets >= SP.MIN_DETS:
                            acted += 1
                        else:
                            deferred += 1
                else:
                    act = opp_move(obs)
                if b.select(list(act)):
                    result = 1 - sp
                    break
        finally:
            b.close()
        think.append(spent)
        out = "D" if result == 2 else ("W" if result == seat else "L")
        if out == "W":
            w += 1
        elif out == "L":
            l += 1
        else:
            d += 1
        print(f"g{g} seat{seat} {out} think={spent:.0f}s", flush=True)
    n = max(w + l + d, 1)
    print(f"RESULT w={w} l={l} d={d} winrate={(w + 0.5 * d) / n:.3f} "
          f"acted={acted} deferred={deferred} "
          f"dets_hist={sorted(hist.items())} "
          f"think_avg={sum(think) / max(len(think), 1):.0f}s", flush=True)


if __name__ == "__main__":
    main()
