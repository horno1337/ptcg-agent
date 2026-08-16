"""Opening-conditioned Phantom-readiness audit for exact-list Dragapult.

This is a causal-screening diagnostic, not a causal estimate.  It compares the
deployed probe with historical top exact-list pilots after deterministic
matching on a public turn-3 opening signature.  If the timing gap disappears,
no readiness policy is justified.  If it persists in supported strata, those
states become candidates for later learner-reached branching.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dragapult_bc as D  # noqa: E402
from agent.obsview import OT_ATTACK, ST_MAIN, ObsView  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    attack_name, decisions, registered_decks,
)


SEARCH_CARDS = frozenset((1086, 1097, 1121, 1152, 1198, 1231))
ATTACK_ENERGY = frozenset((D.FIRE_ENERGY, D.PSYCHIC_ENERGY))


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def cards_in(player: dict, area: str) -> list[int]:
    result = []
    for value in player.get(area) or ():
        card_id = value.get("id") if isinstance(value, dict) else value
        if isinstance(card_id, int) and not isinstance(card_id, bool):
            result.append(card_id)
    return result


def entries(player: dict) -> list[dict]:
    return [
        value for value in list(player.get("active") or ()) + list(player.get("bench") or ())
        if isinstance(value, dict)
    ]


def energy_ids(entry: dict) -> set[int]:
    result = {x for x in entry.get("energies") or () if isinstance(x, int)}
    result.update(
        x.get("id") for x in entry.get("energyCards") or ()
        if isinstance(x, dict) and isinstance(x.get("id"), int)
    )
    return result


def opening(view: ObsView) -> dict[str, Any]:
    mine = view.me or {}; board = entries(mine); hand = cards_in(mine, "hand")
    line = [row for row in board if row.get("id") in D.DRAGAPULT_LINE]
    energized = [row for row in line if energy_ids(row) & ATTACK_ENERGY]
    complementary = [row for row in line if ATTACK_ENERGY <= energy_ids(row)]
    return {
        "turn": view.turn,
        "seat": view.my_index,
        "line_count": len(line),
        "drakloak_count": sum(row.get("id") == D.DRAKLOAK for row in line),
        "dragapult_count": sum(row.get("id") == D.DRAGAPULT_EX for row in line),
        "energized_line_count": len(energized),
        "ready_line_count": len(complementary),
        "distinct_attack_energy_on_line": len(set().union(*(energy_ids(row) & ATTACK_ENERGY for row in line)) if line else set()),
        "fire_in_hand": hand.count(D.FIRE_ENERGY),
        "psychic_in_hand": hand.count(D.PSYCHIC_ENERGY),
        "drakloak_in_hand": hand.count(D.DRAKLOAK),
        "dragapult_in_hand": hand.count(D.DRAGAPULT_EX),
        "search_in_hand": sum(card_id in SEARCH_CARDS for card_id in hand),
        "hand_count": int(mine.get("handCount") or len(hand)),
        "active_budew": bool(
            (mine.get("active") or ())
            and isinstance((mine.get("active") or ())[0], dict)
            and (mine.get("active") or ())[0].get("id") == 235
        ),
        "my_prizes_left": len(mine.get("prize") or ()),
        "opponent_prizes_left": len((view.opp or {}).get("prize") or ()),
    }


def signature(row: dict[str, Any]) -> tuple:
    """Coarsened public opening-quality signature fixed before outcomes."""
    return (
        row["seat"], min(row["line_count"], 3), min(row["drakloak_count"], 2),
        min(row["energized_line_count"], 2), row["distinct_attack_energy_on_line"],
        bool(row["fire_in_hand"]), bool(row["psychic_in_hand"]),
        bool(row["drakloak_in_hand"] or row["dragapult_in_hand"]),
        min(row["search_in_hand"], 2), row["active_budew"],
    )


def quality(row: dict[str, Any]) -> int:
    return (
        min(row["line_count"], 3) + row["drakloak_count"]
        + 2 * row["energized_line_count"] + row["distinct_attack_energy_on_line"]
        + int(bool(row["fire_in_hand"])) + int(bool(row["psychic_in_hand"]))
        + int(bool(row["drakloak_in_hand"] or row["dragapult_in_hand"]))
        + min(row["search_in_hand"], 2) + int(row["active_budew"])
    )


def cohort(directory: Path, teams: set[str], label: str) -> list[dict[str, Any]]:
    result = []; seen = set()
    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict): continue
        decks = registered_decks(document); names = (document.get("info") or {}).get("TeamNames") or ()
        rewards = document.get("rewards") or (); episode = int((document.get("info") or {}).get("EpisodeId") or -1)
        for seat, deck in sorted(decks.items()):
            if tuple(sorted(deck)) != D.TARGET_DECK or seat >= len(names) or names[seat] not in teams:
                continue
            if tuple(sorted(decks.get(1 - seat, ()))) == D.TARGET_DECK or (episode, seat) in seen:
                continue
            seen.add((episode, seat)); rows = [(ObsView(obs), action) for obs, action in decisions(document, seat)]
            own_turns = sorted({view.turn for view, _ in rows if view.select_type == ST_MAIN})
            if not own_turns: continue
            target_turn = own_turns[1] if len(own_turns) > 1 else own_turns[0]
            start = next((view for view, _ in rows if view.select_type == ST_MAIN and view.turn == target_turn), None)
            if start is None: continue
            first_phantom = None
            for view, action in rows:
                if view.select_type != ST_MAIN or not action: continue
                option = view.options[action[0]]
                if option.get("type") == OT_ATTACK and attack_name(option.get("attackId")) == "Phantom Dive":
                    first_phantom = view.turn; break
            state = opening(start); sig = signature(state)
            result.append({
                "cohort": label, "episode_id": episode, "seat": seat,
                "result": "win" if rewards[seat] == 1 else "loss" if rewards[seat] == -1 else "draw",
                "opening_turn": target_turn, "opening": state,
                "signature": list(sig), "quality": quality(state),
                "first_phantom_turn": first_phantom,
                "delay_from_opening": (first_phantom - target_turn) if first_phantom is not None else 20 - target_turn,
                "phantom_by_two_own_turns": first_phantom is not None and first_phantom <= target_turn + 4,
            })
    return result


def summarize(rows: list[dict]) -> dict[str, Any]:
    delays = [row["delay_from_opening"] for row in rows]
    return {
        "games": len(rows), "median_delay": float(statistics.median(delays)) if delays else None,
        "phantom_by_two_own_turns": sum(row["phantom_by_two_own_turns"] for row in rows),
        "phantom_by_two_own_turns_rate": sum(row["phantom_by_two_own_turns"] for row in rows) / len(rows) if rows else None,
        "no_phantom": sum(row["first_phantom_turn"] is None for row in rows),
        "mean_quality": sum(row["quality"] for row in rows) / len(rows) if rows else None,
    }


def analyze(probe_dir: Path, expert_dir: Path) -> dict[str, Any]:
    probes = cohort(probe_dir, {"増殖するG"}, "probe")
    experts = cohort(expert_dir, {"やる気元気ミワハルキ", "flg"}, "expert")
    by_sig: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in probes + experts:
        by_sig[tuple(row["signature"])][row["cohort"]].append(row)
    supported = {sig: value for sig, value in by_sig.items() if value["probe"] and value["expert"]}
    matched_probe = [row for value in supported.values() for row in value["probe"]]
    weighted_expert = []
    stratum_rows = []
    for sig, value in supported.items():
        p, e = value["probe"], value["expert"]
        p_rate = sum(x["phantom_by_two_own_turns"] for x in p) / len(p)
        e_rate = sum(x["phantom_by_two_own_turns"] for x in e) / len(e)
        stratum_rows.append({"signature": list(sig), "probe_n": len(p), "expert_n": len(e),
                             "probe_rate": p_rate, "expert_rate": e_rate,
                             "expert_minus_probe": e_rate - p_rate})
        weighted_expert.extend([e] * len(p))
    matched_expert_rate = (
        sum(sum(x["phantom_by_two_own_turns"] for x in group) / len(group) for group in weighted_expert) / len(weighted_expert)
        if weighted_expert else None
    )
    probe_rate = (
        sum(x["phantom_by_two_own_turns"] for x in matched_probe) / len(matched_probe)
        if matched_probe else None
    )
    payload = {
        "schema": "ptcg.dragapult.opening-conditioned-readiness.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "outcome_definition": "first Phantom Dive within two subsequent own turns after second own-turn opening snapshot",
        "opening_signature": "seat, line/evolution/energy counts, Fire/Psychic/evolution/search hand access, active Budew",
        "unconditioned": {"probe": summarize(probes), "expert": summarize(experts)},
        "exact_signature_support": {
            "strata": len(supported), "probe_games": len(matched_probe),
            "probe_coverage": len(matched_probe) / len(probes) if probes else None,
            "probe_rate": probe_rate, "expert_standardized_rate": matched_expert_rate,
            "expert_minus_probe": matched_expert_rate - probe_rate if probe_rate is not None and matched_expert_rate is not None else None,
        },
        "quality_buckets": {
            str(bucket): {
                label: summarize([row for row in values if row["cohort"] == label])
                for label in ("probe", "expert")
            }
            for bucket, values in sorted({
                q: [row for row in probes + experts if row["quality"] == q]
                for q in {row["quality"] for row in probes + experts}
            }.items())
        },
        "supported_strata": sorted(stratum_rows, key=lambda row: (-row["probe_n"], row["signature"])),
        "interpretation_rule": (
            "Build nothing if exact-signature probe coverage < 0.50 or the standardized expert-minus-probe rate <= 0.05; "
            "otherwise only learner-reached branching is authorized, not integration."
        ),
        "readiness_branching_authorized": bool(
            len(matched_probe) >= 0.5 * len(probes)
            and matched_expert_rate is not None and probe_rate is not None
            and matched_expert_rate - probe_rate > 0.05
        ),
        "training_authority": False, "integration_authority": False,
        "rows": probes + experts,
    }
    payload["result_sha256"] = canonical(payload); return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--expert-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(); value = analyze(args.probe_dir, args.expert_dir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"unconditioned": value["unconditioned"],
                      "support": value["exact_signature_support"],
                      "authorized": value["readiness_branching_authorized"],
                      "sha256": value["result_sha256"]}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
