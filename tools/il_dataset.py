"""Kaggle episode JSONs -> imitation-learning samples.

Episode logs (downloaded from leaderboard game pages) are
kaggle-environments records: steps[t][p] = {observation, action, ...}.
The action recorded at step t answers the observation at step t-1
(verified empirically: 100% action legality under that pairing, ~79%
under same-step). Deck registrations (60-int actions) are skipped as
policy samples but exposed via `decks()` for list mining.

Yields (obs_dict, action_indices, final_reward_for_actor) triples; the
obs is the acting player's legitimate view (own hand visible, opponent
hidden), directly consumable by agent/features.py.
"""

import glob
import json
import os


def iter_episode(path: str):
    with open(path) as f:
        d = json.load(f)
    steps = d.get("steps") or []
    rewards = d.get("rewards") or [0, 0]
    for t in range(1, len(steps)):
        for p in (0, 1):
            row = steps[t][p]
            act = row.get("action")
            if not act or len(act) == 60:
                continue
            obs = (steps[t - 1][p].get("observation")) or {}
            sel = obs.get("select")
            if not sel or not sel.get("option"):
                continue
            n = len(sel["option"])
            mn, mx = sel.get("minCount", 1), sel.get("maxCount", 1)
            if not all(isinstance(a, int) and 0 <= a < n for a in act):
                continue
            if len(act) < min(mn, n) or (mx > 0 and len(act) > mx):
                continue
            r = rewards[p] if p < len(rewards) and rewards[p] is not None else 0
            yield obs, list(act), float(r)


def iter_dir(dir_path: str):
    for f in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
        yield from iter_episode(f)


def decks(path: str) -> dict[int, list[int]]:
    """Player index -> registered 60-card deck, if present in the log."""
    with open(path) as f:
        d = json.load(f)
    out = {}
    for row in d.get("steps") or []:
        for p in (0, 1):
            act = row[p].get("action")
            if act and len(act) == 60 and p not in out:
                out[p] = list(act)
    return out


if __name__ == "__main__":
    import sys
    n, by_type = 0, {}
    for obs, act, r in iter_dir(sys.argv[1] if len(sys.argv) > 1 else "."):
        n += 1
        st = obs["select"].get("type")
        by_type[st] = by_type.get(st, 0) + 1
    print(f"{n} samples; by select type: {dict(sorted(by_type.items()))}")
