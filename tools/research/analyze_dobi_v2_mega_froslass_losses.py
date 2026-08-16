"""Audit mechanical failure patterns in frozen Dobi-v2 Mega Froslass losses."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import cards  # noqa: E402
from agent.obsview import AREA_ACTIVE, AREA_BENCH, ST_MAIN, ObsView  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    available_buckets,
    coarse,
)
from tools.research.compare_current_grim_to_dobi_v2 import decisions  # noqa: E402


BATTLE_CAGE = 1264
SPIKEMUTH = 1259


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def stadium_id(view: ObsView) -> int | None:
    stadium = (view.current or {}).get("stadium") or []
    entry = stadium[0] if isinstance(stadium, list) and stadium else stadium
    return entry.get("id") if isinstance(entry, dict) else None


def hand_size(side: Any) -> int | None:
    hand = (side or {}).get("hand")
    if isinstance(hand, list):
        return len(hand)
    count = (side or {}).get("handCount")
    return count if isinstance(count, int) else None


def card_name(card_id: int | None) -> str | None:
    info = cards.card(card_id) if isinstance(card_id, int) else None
    return info.get("name") if isinstance(info, dict) else None


def effect_target(rows: list[tuple[int, dict, list[int]]], start: int,
                  opponent: int) -> dict[str, Any] | None:
    for _, observation, logged in rows[start + 1:]:
        view = ObsView(observation)
        if view.select_type == ST_MAIN:
            break
        for pick in logged:
            if not 0 <= pick < len(view.options):
                continue
            option = view.options[pick]
            area = option.get("area")
            if option.get("playerIndex") != opponent or area not in {
                    AREA_ACTIVE, AREA_BENCH}:
                continue
            return {
                "area": "active" if area == AREA_ACTIVE else "bench",
                "card": card_name(view.semantic_option_card_id(option)),
            }
    return None


def main() -> int:
    history = json.loads((
        ROOT / "tools/checkpoints/dobi-v2-all-frozen-20260815/"
        "history-analysis-v3.json"
    ).read_text(encoding="utf-8"))
    games = [
        row for cohort in history["cohorts"].values()
        for row in cohort["games"]
        if row["opponent_archetype"] == "Mega Froslass"
    ]
    totals: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for game in games:
        document = json.loads(Path(game["path"]).read_text(encoding="utf-8"))
        learner = int(game["seat"])
        per_game: Counter[str] = Counter()
        events: list[dict[str, Any]] = []
        learner_rows = list(decisions(document, learner))
        for row_index, (_, observation, logged) in enumerate(learner_rows):
            view = ObsView(observation)
            if view.select_type != ST_MAIN:
                continue
            label = coarse(view, logged)
            cage = stadium_id(view) == BATTLE_CAGE
            offered = set(available_buckets(view))
            relevant = label in {
                "ability:Munkidori", "attack:Shadow Bullet",
                "play:Spikemuth Gym", "play:Boss’s Orders",
            }
            if label == "ability:Munkidori":
                target = effect_target(learner_rows, row_index, 1 - learner)
                per_game["munkidori_uses"] += 1
                per_game["munkidori_under_cage"] += int(cage)
                if cage and target is not None:
                    per_game[f"munkidori_under_cage_to_{target['area']}"] += 1
                    per_game[
                        f"munkidori_under_cage_target:{target['card']}"
                    ] += 1
                per_game["munkidori_under_cage_with_spikemuth_offered"] += int(
                    cage and "play:Spikemuth Gym" in offered)
            elif label == "attack:Shadow Bullet":
                target = effect_target(learner_rows, row_index, 1 - learner)
                per_game["shadow_attacks"] += 1
                per_game["shadow_under_cage"] += int(cage)
                per_game["shadow_under_cage_with_spikemuth_offered"] += int(
                    cage and "play:Spikemuth Gym" in offered)
            elif label == "play:Spikemuth Gym":
                per_game["spikemuth_plays"] += 1
            elif label == "play:Boss’s Orders":
                per_game["boss_plays"] += 1
            if relevant:
                active = ((view.opp or {}).get("active") or [None])[0]
                events.append({
                    "turn": view.turn,
                    "action": label,
                    "stadium": card_name(stadium_id(view)),
                    "own_hand": hand_size(view.me),
                    "opponent_active": card_name(
                        active.get("id") if isinstance(active, dict) else None),
                    "opponent_active_hp": (
                        active.get("hp") if isinstance(active, dict) else None),
                    "spikemuth_offered": "play:Spikemuth Gym" in offered,
                    "effect_target": target if label in {
                        "ability:Munkidori", "attack:Shadow Bullet"} else None,
                })

        for _, observation, logged in decisions(document, 1 - learner):
            view = ObsView(observation)
            if view.select_type != ST_MAIN:
                continue
            label = coarse(view, logged)
            if label == "play:Wally's Compassion":
                per_game["opponent_wally"] += 1
            if label == "attack:Resentful Refrain":
                per_game["resentful_refrain"] += 1
                own_hand = hand_size(view.opp)
                if own_hand is not None:
                    per_game[f"refrain_against_{own_hand}_cards"] += 1
            if label == "attack:Absolute Snow":
                per_game["absolute_snow"] += 1
            if label in {"attack:Gale Thrust", "attack:Spiky Hopper"}:
                per_game["lopunny_attacks"] += 1

        totals.update(per_game)
        rows.append({
            "episode_id": game["episode_id"],
            "opponent": game["opponent_team_name"],
            "went_first": game["went_first"],
            "prizes_taken": game["resource_conversion"]["own_prizes_taken"],
            "counts": dict(sorted(per_game.items())),
            "events": events,
        })

    payload = {
        "schema": "ptcg.dobi-v2.mega-froslass-loss-audit.v1",
        "games": len(rows),
        "totals": dict(sorted(totals.items())),
        "rows": rows,
        "limits": [
            "All observed Dobi-v2 games were losses, so this identifies repeated "
            "mechanical waste and correlations, not a causal win-rate effect.",
            "Battle Cage arithmetic is mechanical: opponent attack/Ability damage "
            "counters placed on the Bench are prevented while it is active.",
        ],
    }
    payload["result_sha256"] = canonical_sha256(payload)
    out = ROOT / "tools/checkpoints/dobi-v2-all-frozen-20260815/mega-froslass-loss-audit.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"games": len(rows), **payload["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
