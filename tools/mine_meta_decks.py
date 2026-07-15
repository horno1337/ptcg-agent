"""Mine registered decklists out of episode logs -> agent/meta_decks.json.

Every episode records both players' 60-card deck registrations. The search
policy uses this library to predict an opponent's hidden cards: match their
revealed cards against known lists, fill the unseen zones with the best
match's remainder.

Usage: python tools/mine_meta_decks.py [episode_dir]
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import il_dataset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "agent", "meta_decks.json")


def main(ep_dir: str):
    import glob
    lists: dict[tuple, dict] = {}
    for f in sorted(glob.glob(os.path.join(ep_dir, "*.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
            teams = (d.get("info") or {}).get("TeamNames") or ["?", "?"]
            for p, deck in il_dataset.decks(f).items():
                key = tuple(sorted(deck))
                e = lists.setdefault(key, {"deck": sorted(deck), "count": 0, "teams": set()})
                e["count"] += 1
                e["teams"].add(teams[p] if p < len(teams) else "?")
        except Exception as ex:
            print(f"skip {f}: {ex}")
    out = sorted(lists.values(), key=lambda e: -e["count"])
    for e in out:
        e["teams"] = sorted(e["teams"])
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print(f"{len(out)} unique decklists from {ep_dir} -> {OUT}")
    for e in out[:8]:
        c = Counter(e["deck"])
        print(f"  x{e['count']:3d} {e['teams'][:3]} top cards {c.most_common(4)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Desktop/ptcg_episodes"))
