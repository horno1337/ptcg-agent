"""Read-only replay audit of the locked Phantom protection candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D, model, qu_v2_features as FEATURES  # noqa: E402
from agent.obsview import ST_CARD, ObsView  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import decisions, registered_decks  # noqa: E402


def audit(directory: Path, team: str) -> dict:
    D._CARD_PATH = str(ROOT / "agent/dragapult_elite_card_weights.npz")
    D.CARD_WEIGHTS_SHA256 = D._sha256(D._CARD_PATH)
    D._card = None; D._attempted.discard("card")
    net = D._load_head("card")
    counts = Counter(); examples = []
    seen = set()
    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        decks = registered_decks(document)
        teams = (document.get("info") or {}).get("TeamNames") or ()
        rewards = document.get("rewards") or ()
        episode = int((document.get("info") or {}).get("EpisodeId") or -1)
        for seat, deck in sorted(decks.items()):
            if tuple(sorted(deck)) != D.TARGET_DECK or seat >= len(teams) or teams[seat] != team:
                continue
            if (episode, seat) in seen:
                continue
            seen.add((episode, seat)); counts["exact_seats"] += 1
            result = "win" if rewards[seat] == 1 else "loss" if rewards[seat] == -1 else "draw"
            for obs, logged in decisions(document, seat):
                view = ObsView(obs)
                if view.select_type != ST_CARD:
                    continue
                sample = FEATURES.encode_public_observation(obs, deck)
                logits, _ = net.forward(sample)
                base = model.decode_qu_v2(logits, len(view.options), view.min_count, view.max_count)
                secured = D._guard_phantom_secure_prize(view, base)
                candidate = D._guard_phantom_protected_target(view, logits, secured)
                if candidate == secured:
                    continue
                counts["interventions"] += 1
                counts[f"interventions_{result}"] += 1
                counts["candidate_matches_logged"] += candidate == logged
                counts["parent_matches_logged"] += secured == logged
                chosen = view.option_board_entry(view.options[secured[0]])
                replacement = view.option_board_entry(view.options[candidate[0]])
                counts[f"source_{(chosen or {}).get('id')}"] += 1
                if len(examples) < 40:
                    examples.append({
                        "episode_id": episode, "result": result, "turn": view.turn,
                        "remain": view.select.get("remainDamageCounter"),
                        "source": (chosen or {}).get("id"),
                        "replacement": (replacement or {}).get("id"),
                        "replacement_hp": (replacement or {}).get("hp"),
                        "parent_logged": secured == logged,
                        "candidate_logged": candidate == logged,
                    })
    return {"source": str(directory.resolve()), "team": team,
            "counts": dict(counts), "examples": examples,
            "promotion_authority": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path); parser.add_argument("--team", required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(); value = audit(args.directory, args.team)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(value["counts"], indent=2, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
