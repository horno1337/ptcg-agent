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
import math
import os


def iter_document(d: dict):
    """Yield validated decisions from an already-decoded episode document."""
    steps = d.get("steps") or []
    rewards = d.get("rewards")
    if (not isinstance(rewards, list) or len(rewards) != 2
            or any(not isinstance(r, (int, float)) or isinstance(r, bool)
                   or not math.isfinite(r) or float(r) not in (-1.0, 0.0, 1.0)
                   for r in rewards)
            or float(rewards[0]) != -float(rewards[1])):
        # Corrupt/partial episodes are not neutral demonstrations.  Fail the
        # shard closed instead of silently inventing draw rewards.
        return
    for t in range(1, len(steps)):
        for p in (0, 1):
            row = steps[t][p]
            source_row = steps[t - 1][p]
            # Kaggle writes [] for the non-acting seat too.  Preserve a real
            # optional STOP only for the seat whose preceding observation
            # requested the action.  row.action answers source_row.observation,
            # so the relevant ACTIVE/INACTIVE marker also lives on source_row;
            # compact locally-generated episodes omit status altogether.
            if source_row.get("status") == "INACTIVE":
                continue
            act = row.get("action")
            # [] is a real action for optional selects (minCount == 0), not a
            # missing action.  Only None means this row has no paired action.
            if not isinstance(act, list) or len(act) == 60:
                continue
            obs = source_row.get("observation") or {}
            sel = obs.get("select")
            if not sel or not sel.get("option"):
                continue
            current = obs.get("current")
            if not isinstance(current, dict) or current.get("yourIndex") != p:
                # Action rows are seat-indexed.  Training the action against
                # another player's view is both wrong supervision and a
                # possible hidden-information leak.
                continue
            n = len(sel["option"])
            mn, mx = sel.get("minCount", 1), sel.get("maxCount", 1)
            if (not all(isinstance(a, int) and not isinstance(a, bool)
                        and 0 <= a < n for a in act)
                    or len(set(act)) != len(act)):
                continue
            if len(act) < min(mn, n) or (mx > 0 and len(act) > mx):
                continue
            yield obs, list(act), float(rewards[p])


def iter_episode(path: str):
    with open(path) as f:
        d = json.load(f)
    yield from iter_document(d)


def iter_dir(dir_path: str):
    for f in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
        yield from iter_episode(f)


def decks_from_document(d: dict) -> dict[int, list[int]]:
    """Player index -> registered 60-card deck in a decoded document."""
    out = {}
    for row in d.get("steps") or []:
        for p in (0, 1):
            act = row[p].get("action")
            if act and len(act) == 60 and p not in out:
                out[p] = list(act)
    return out


def decks(path: str) -> dict[int, list[int]]:
    """Player index -> registered 60-card deck, if present in the log."""
    with open(path) as f:
        d = json.load(f)
    return decks_from_document(d)


if __name__ == "__main__":
    import sys
    n, by_type = 0, {}
    for obs, act, r in iter_dir(sys.argv[1] if len(sys.argv) > 1 else "."):
        n += 1
        st = obs["select"].get("type")
        by_type[st] = by_type.get(st, 0) + 1
    print(f"{n} samples; by select type: {dict(sorted(by_type.items()))}")
