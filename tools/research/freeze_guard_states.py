"""Freeze the three ladder-loss states into a committed test fixture.

The replays live under tools/checkpoints/, which is gitignored, so the tests
cannot read them on a fresh clone. This extracts exactly the prompts the guards
are meant to fix, records where each came from, and writes a small JSON fixture.

Run once; the fixture is the durable artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from tools.il_dataset import decks_from_document, iter_document  # noqa: E402

REPLAYS = ROOT / "tools/checkpoints/cage-guards-20260816/replays"
OUT = ROOT / "tests/fixtures/alakazam_guard_states.json"
TARGET = "4b090895e20d39512f1469048d57d4df181202c002ff5e38b98b49e9b5a838ee"

# (name, episode id, our seat, prompt index within the document)
WANTED = [
    ("suicide_93586883", 93586883, 0, 43),
    ("suicide_93588738", 93588738, 0, 15),
    ("lethal_93591463", 93591463, 1, 71),
]


def deck_sha(deck) -> str:
    return hashlib.sha256(
        ",".join(str(int(c)) for c in sorted(int(x) for x in deck)).encode()
    ).hexdigest()


def main() -> int:
    states, registration = [], None
    for name, episode_id, seat, prompt in WANTED:
        document = json.loads((REPLAYS / f"{episode_id}.json").read_text())
        deck = (decks_from_document(document) or {})[seat]
        if deck_sha(deck) != TARGET:
            raise SystemExit(f"{episode_id} seat {seat} is not the target list")
        registration = [int(c) for c in deck]
        for index, (obs, action, _reward) in enumerate(iter_document(document)):
            if index != prompt:
                continue
            if (obs.get("current") or {}).get("yourIndex") != seat:
                raise SystemExit(f"{episode_id} prompt {prompt} is not our seat")
            states.append({
                "name": name, "episode_id": episode_id, "seat": seat,
                "prompt_index": prompt, "logged_action": list(action),
                "obs": obs,
            })
            break
        else:
            raise SystemExit(f"{episode_id} has no prompt {prompt}")
    payload = {
        "schema": "ptcg.alakazam-guard-states.v1",
        "source": "Kaggle replays of our own 4b090895 ladder losses",
        "registration": registration,
        "registration_sha256": TARGET,
        "states": states,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"{len(states)} states -> {OUT}")
    for row in states:
        print(f"  {row['name']:<20} ep {row['episode_id']} "
              f"prompt {row['prompt_index']} logged {row['logged_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
