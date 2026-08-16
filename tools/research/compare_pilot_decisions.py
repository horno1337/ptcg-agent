"""Compare our seat's decisions against a strong pilot's on the SAME 60 cards.

kenkoooo is 18-1 on 4b090895, the exact registration our Cage agent plays, so
the comparison carries no deck confound at all. Everything measured here is a
public, countable decision property -- the point is to find a divergence that
could become a rule, not to admire the win rate.

Cohorts: our wins, our losses, and the pilot's wins.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from tools.il_dataset import decks_from_document, iter_document  # noqa: E402
from agent.obsview import ObsView  # noqa: E402

OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ABILITY, OT_ATTACK, OT_END = 7, 8, 9, 10, 13, 14
POWERFUL_HAND = 1072
ALAKAZAM, KADABRA, ABRA = 743, 742, 741
DUDUNSPARCE, FEZANDIPITI = 66, 140
BATTLE_CAGE, RARE_CANDY = 1264, 1079


def deck_sha(deck) -> str:
    return hashlib.sha256(
        ",".join(str(int(c)) for c in sorted(int(x) for x in deck)).encode()
    ).hexdigest()


def one(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value if isinstance(value, dict) else None


def measure_game(document, seat) -> dict | None:
    """Countable properties of one seat's play across a whole game."""
    m = {
        "powerful_hand_sizes": [], "attacks": 0, "turns_seen": set(),
        "first_alakazam_turn": None, "bench_at_attack": [],
        "draw_abilities": 0, "battle_cage_played": 0, "rare_candy": 0,
        "end_without_attack": 0, "decisions": 0, "min_deck": None,
        "prizes_taken": None,
    }
    attacked_this_turn = set()
    for obs, action, _reward in iter_document(document):
        current = obs.get("current") or {}
        if current.get("yourIndex") != seat:
            continue
        view = ObsView(obs)
        me = view.me or {}
        turn = current.get("turn")
        if isinstance(turn, int):
            m["turns_seen"].add(turn)
        deck_count = me.get("deckCount")
        if isinstance(deck_count, int):
            m["min_deck"] = (deck_count if m["min_deck"] is None
                             else min(m["min_deck"], deck_count))
        prizes = me.get("prize")
        if isinstance(prizes, list):
            m["prizes_taken"] = 6 - len(prizes) if len(prizes) <= 6 else None
        if view.select_type != 0:
            continue
        m["decisions"] += 1
        options = view.options
        for index in action:
            if index >= len(options):
                continue
            option = options[index]
            kind = option.get("type")
            card_id = view.option_card_id(option) or view.semantic_option_card_id(option)
            if kind == OT_ATTACK:
                m["attacks"] += 1
                if option.get("attackId") == POWERFUL_HAND:
                    m["powerful_hand_sizes"].append(len(me.get("hand") or []))
                bench = me.get("bench") or []
                m["bench_at_attack"].append(
                    sum(1 for b in bench if isinstance(b, dict)))
                if isinstance(turn, int):
                    attacked_this_turn.add(turn)
            elif kind == OT_EVOLVE and card_id == ALAKAZAM:
                if m["first_alakazam_turn"] is None and isinstance(turn, int):
                    m["first_alakazam_turn"] = turn
            elif kind == OT_ABILITY and card_id in (DUDUNSPARCE, FEZANDIPITI):
                m["draw_abilities"] += 1
            elif kind == OT_PLAY and card_id == BATTLE_CAGE:
                m["battle_cage_played"] += 1
            elif kind == OT_PLAY and card_id == RARE_CANDY:
                m["rare_candy"] += 1
            elif kind == OT_END and isinstance(turn, int):
                if turn not in attacked_this_turn:
                    m["end_without_attack"] += 1
    m["turns"] = len(m["turns_seen"])
    del m["turns_seen"]
    return m


def summarize(name, games) -> dict:
    def col(key):
        return [g[key] for g in games if g.get(key) is not None]

    def flat(key):
        return [v for g in games for v in g.get(key, [])]

    ph = flat("powerful_hand_sizes")
    out = {
        "games": len(games),
        "turns_mean": round(statistics.fmean(col("turns")), 2) if games else 0,
        "attacks_per_game": round(statistics.fmean(col("attacks")), 2) if games else 0,
        "powerful_hand_n": len(ph),
        "powerful_hand_mean_cards": round(statistics.fmean(ph), 2) if ph else None,
        "powerful_hand_mean_damage": round(statistics.fmean(ph) * 20, 1) if ph else None,
        "powerful_hand_median": statistics.median(ph) if ph else None,
        "bench_at_attack_mean": round(statistics.fmean(flat("bench_at_attack")), 2)
                                 if flat("bench_at_attack") else None,
        "first_alakazam_turn_mean": round(
            statistics.fmean(col("first_alakazam_turn")), 2)
            if col("first_alakazam_turn") else None,
        "reached_alakazam_pct": round(
            100.0 * len(col("first_alakazam_turn")) / max(len(games), 1), 1),
        "draw_abilities_per_game": round(
            statistics.fmean(col("draw_abilities")), 2) if games else 0,
        "battle_cage_per_game": round(
            statistics.fmean(col("battle_cage_played")), 2) if games else 0,
        "rare_candy_per_game": round(statistics.fmean(col("rare_candy")), 2) if games else 0,
        "ends_without_attack_per_game": round(
            statistics.fmean(col("end_without_attack")), 2) if games else 0,
        "min_deck_mean": round(statistics.fmean(col("min_deck")), 2)
                          if col("min_deck") else None,
        "decisions_per_game": round(statistics.fmean(col("decisions")), 2) if games else 0,
    }
    return out


def collect(paths, team, target_sha, want_win=None):
    games = []
    for path in paths:
        if not path.stem.isdigit():
            continue
        try:
            document = json.loads(path.read_text())
        except Exception:
            continue
        names = (document.get("info") or {}).get("TeamNames") or []
        rewards = document.get("rewards") or []
        decks = decks_from_document(document) or {}
        seats = [s for s, n in enumerate(names) if n == team]
        if len(seats) != 1 or len(rewards) != 2:
            continue
        seat = seats[0]
        if deck_sha(decks.get(seat, [])) != target_sha:
            continue
        won = float(rewards[seat]) > 0
        if want_win is not None and won != want_win:
            continue
        measured = measure_game(document, seat)
        if measured:
            measured["episode_id"] = int(path.stem)
            games.append(measured)
    return games


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ours", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--our-team", default="増殖するG")
    p.add_argument("--pilot-team", default="kenkoooo @ estie, inc.")
    p.add_argument("--deck-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ours = sorted(args.ours.glob("*.json"))
    pilot = sorted(args.pilot.glob("*.json"))
    cohorts = {
        "our_wins": collect(ours, args.our_team, args.deck_sha256, True),
        "our_losses": collect(ours, args.our_team, args.deck_sha256, False),
        "pilot_wins": collect(pilot, args.pilot_team, args.deck_sha256, True),
        "pilot_losses": collect(pilot, args.pilot_team, args.deck_sha256, False),
    }
    report = {name: summarize(name, games) for name, games in cohorts.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"summary": report,
         "per_game": {k: v for k, v in cohorts.items()}}, indent=1,
        sort_keys=True, default=str) + "\n")

    keys = [k for k in report["our_losses"] if k != "games"]
    order = ["our_wins", "our_losses", "pilot_wins", "pilot_losses"]
    width = max(len(k) for k in keys) + 2
    print(f"{'metric':<{width}}" + "".join(f"{c:>16}" for c in order))
    print(f"{'games':<{width}}" + "".join(
        f"{report[c]['games']:>16}" for c in order))
    print("-" * (width + 16 * len(order)))
    for key in keys:
        row = f"{key:<{width}}"
        for c in order:
            v = report[c].get(key)
            row += f"{'-' if v is None else v:>16}"
        print(row)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
