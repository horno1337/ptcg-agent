"""Analyze recent frozen Dobi-v2 ladder games and loss hypotheses.

The report is descriptive.  It separates actual Dobi-v2 replays from the
current-expert corpus and does not authorize a policy change.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import cards  # noqa: E402
from agent.obsview import OT_ATTACK, OT_PLAY, ST_MAIN, ObsView  # noqa: E402
from tools import analyze_ladder_replays as LADDER  # noqa: E402
from tools.research import compare_grimmsnarl_replay_cohorts as COHORT  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import coarse  # noqa: E402
from tools.research.compare_current_grim_to_dobi_v2 import decisions  # noqa: E402


BOSS = 1182
SHADOW_BULLET = 937
TEAM = "増殖するG"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def mean(rows: Iterable[float]) -> float | None:
    values = list(rows)
    return statistics.mean(values) if values else None


def prize_value(entry: Mapping[str, Any]) -> int:
    info = cards.card(entry.get("id")) or {}
    return 3 if info.get("megaEx") else 2 if info.get("ex") else 1


def nominal_boss_root(view: ObsView) -> dict[str, Any] | None:
    """Return a conservative visible multi-Prize Boss opportunity.

    This deliberately ignores weakness and counts only a benched two-/three-
    Prize target with at most 180 HP while the current Active has over 180 HP.
    It remains a hypothesis because protection and future prize mapping can
    make the locally immediate KO strategically inferior.
    """
    if view.select_type != ST_MAIN:
        return None
    boss = any(
        option.get("type") == OT_PLAY
        and view.semantic_option_card_id(option) == BOSS
        for option in view.options
    )
    shadow = any(
        option.get("type") == OT_ATTACK
        and option.get("attackId") == SHADOW_BULLET
        for option in view.options
    )
    active = ((view.opp or {}).get("active") or [None])[0]
    if (
        not boss or not shadow or not isinstance(active, Mapping)
        or not isinstance(active.get("hp"), int) or active["hp"] <= 180
    ):
        return None
    candidates = []
    for entry in (view.opp or {}).get("bench") or []:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("hp"), int)
            or entry["hp"] <= 0 or entry["hp"] > 180
            or prize_value(entry) < 2
        ):
            continue
        info = cards.card(entry.get("id")) or {}
        candidates.append({
            "card_id": entry.get("id"),
            "name": info.get("name"),
            "hp": entry["hp"],
            "prize_value": prize_value(entry),
        })
    if not candidates:
        return None
    return {
        "engine_turn": view.turn,
        "active_card_id": active.get("id"),
        "active_hp": active["hp"],
        "learner_prizes_left": len((view.me or {}).get("prize") or []),
        "targets": sorted(
            candidates,
            key=lambda row: (-row["prize_value"], row["hp"], row["card_id"]),
        ),
    }


def boss_roots(action_rows, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    turns: dict[int, dict[str, Any]] = {}
    for view, chosen in action_rows:
        if view.select_type != ST_MAIN:
            continue
        row = turns.setdefault(view.turn, {
            "root": None, "played_boss": False, "used_shadow_bullet": False,
        })
        label = coarse(view, chosen)
        row["played_boss"] |= label == "play:Boss’s Orders"
        row["used_shadow_bullet"] |= label == "attack:Shadow Bullet"
        if row["root"] is None:
            row["root"] = nominal_boss_root(view)
    return [
        {**dict(metadata), **row["root"],
         "played_boss_this_turn": row["played_boss"],
         "used_shadow_bullet_this_turn": row["used_shadow_bullet"]}
        for row in turns.values() if row["root"] is not None
    ]


def root_summary(roots: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_boss = Counter(
        f"{row['outcome']}:{'boss' if row['played_boss_this_turn'] else 'no_boss'}"
        for row in roots
    )
    return {
        "roots": len(roots),
        "distinct_games": len({row["episode_id"] for row in roots}),
        "played_boss": sum(row["played_boss_this_turn"] for row in roots),
        "played_boss_rate": (
            sum(row["played_boss_this_turn"] for row in roots) / len(roots)
            if roots else None
        ),
        "by_outcome_and_action": dict(sorted(outcome_boss.items())),
        "rows": roots,
    }


def outcome_metrics(games: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for outcome in ("win", "loss", "draw"):
        rows = [row for row in games if row["outcome"] == outcome]
        if not rows:
            continue
        result[outcome] = {
            "games": len(rows),
            "own_prizes_taken_mean": mean(
                row["resource_conversion"]["own_prizes_taken"] for row in rows
            ),
            "attacks_mean": mean(
                row["resource_conversion"]["attacks"] for row in rows
            ),
            "prizes_per_attack_mean": mean(
                row["resource_conversion"]["prizes_per_attack"] for row in rows
            ),
            "max_line_points_mean": mean(
                row["resource_conversion"]["max_line_points"] for row in rows
            ),
            "max_energy_in_play_mean": mean(
                row["resource_conversion"]["max_energy_in_play"] for row in rows
            ),
        }
    return result


def matchup_summary(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for game in games:
        buckets[game["opponent_archetype"]][game["outcome"]] += 1
    return [{
        "matchup": matchup,
        "games": sum(counts.values()),
        "wins": counts["win"],
        "losses": counts["loss"],
        "draws": counts["draw"],
        "win_rate": counts["win"] / sum(counts.values()),
    } for matchup, counts in sorted(
        buckets.items(), key=lambda item: (-sum(item[1].values()), item[0])
    )]


def concise_game(game: Mapping[str, Any]) -> dict[str, Any]:
    turns = game["turn_rows"]
    first_grim = next((
        row["own_turn"] for row in turns
        if row["metrics"]["me.grimmsnarl"] > 0
    ), None)
    first_attack = next((
        row["own_turn"] for row in turns if row["metrics"]["attacks"] > 0
    ), None)
    return {
        "episode_id": game["episode_id"],
        "outcome": game["outcome"],
        "opponent": game["opponent_team_name"],
        "matchup": game["opponent_archetype"],
        "went_first": game["went_first"],
        "own_turns": game["turns"],
        "first_grimmsnarl_own_turn": first_grim,
        "first_attack_own_turn": first_attack,
        "attacks": game["attacks_by_name"],
        "resources": game["resource_conversion"],
        "turn_timeline": [{
            "own_turn": row["own_turn"],
            "engine_turn": row["engine_turn"],
            "grimmsnarl": row["metrics"]["me.grimmsnarl"],
            "munkidori": row["metrics"]["me.munkidori"],
            "powered_munkidori": row["metrics"]["me.munkidori_dark"],
            "froslass": row["metrics"]["me.froslass"],
            "energy_in_play": row["metrics"]["me.energy_in_play"],
            "attacks": row["metrics"]["attacks"],
            "own_prizes_taken": row["metrics"]["observed_cumulative_prizes_taken"],
            "opponent_prizes_taken": row["metrics"]["observed_cumulative_opponent_prizes_taken"],
        } for row in turns],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--expert-result", type=Path, required=True)
    parser.add_argument("--recent-episode", type=int, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    spec = COHORT.CohortSpec(
        "dobi-v2-live",
        (args.replay_dir.resolve(),),
        COHORT.read_exact_grimmsnarl_deck(ROOT / "decks/md_v1_grimmsnarl.csv"),
        frozenset((TEAM,)),
    )
    games, diagnostics = COHORT._collect_cohort(spec)
    by_id = {game["episode_id"]: game for game in games}
    recent = [by_id[value] for value in args.recent_episode if value in by_id]

    live_roots = []
    for game in games:
        document = json.loads(Path(game["path"]).read_text(encoding="utf-8"))
        live_roots.extend(boss_roots(
            LADDER.action_rows(document, game["seat"]),
            {key: game[key] for key in (
                "episode_id", "outcome", "opponent_archetype",
            )},
        ))

    expert_result = json.loads(args.expert_result.read_text(encoding="utf-8"))
    expert_roots = []
    for game in expert_result["cohort"]["games"]:
        document = json.loads(Path(game["path"]).read_text(encoding="utf-8"))
        rows = (
            (ObsView(observation), action)
            for _, observation, action in decisions(document, game["seat"])
        )
        expert_roots.extend(boss_roots(rows, {
            "episode_id": game["episode_id"],
            "outcome": game["outcome"],
            "teacher": game["teacher"],
            "opponent_deck_sha256": game["opponent_deck_sha256"],
        }))

    payload = {
        "schema": "ptcg.dobi-v2.recent-loss-analysis.v1",
        "design": {
            "descriptive_only": True,
            "training_authorized": False,
            "package_authorized": False,
            "upload_authorized": False,
        },
        "live_cohort": {
            "diagnostics": diagnostics,
            "outcomes": dict(Counter(game["outcome"] for game in games)),
            "matchups": matchup_summary(games),
            "outcome_metrics": outcome_metrics(games),
        },
        "recent": {
            "requested_episode_ids": args.recent_episode,
            "resolved_games": len(recent),
            "outcomes": dict(Counter(game["outcome"] for game in recent)),
            "games": [concise_game(game) for game in recent],
        },
        "nominal_multi_prize_boss": {
            "definition": (
                "Boss and Shadow Bullet legal, current Active nominally survives "
                "180, and a benched two-/three-Prize target has <=180 HP"
            ),
            "actual_dobi_v2": root_summary(live_roots),
            "current_experts": root_summary(expert_roots),
            "interpretation_limit": (
                "This local arithmetic condition does not prove that Boss is "
                "globally optimal; prize map, protection, and later turns matter."
            ),
        },
        "findings": [
            "Shadow Bullet completion is not re-estimated here; use the separate live completion audit.",
            "Recent losses split into blowouts and one-Prize finishes, so there is no single setup-only explanation.",
            "Powered-Munkidori board count is outcome-conditioned and can fall because opponents remove Munkidori; it is not by itself an attachment-policy label.",
            "Nominal immediate multi-Prize Boss roots are the narrowest remaining causal hypothesis and require a separate prospective gameplay gate.",
        ],
    }
    payload["result_sha256"] = canonical_sha256(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "live_games": len(games),
        "recent": payload["recent"]["outcomes"],
        "live_boss_roots": len(live_roots),
        "expert_boss_roots": len(expert_roots),
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
