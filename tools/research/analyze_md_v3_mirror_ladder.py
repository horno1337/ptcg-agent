"""Describe setup and routing in the content-locked MD-v3 mirror cohort.

This is diagnostic only.  Outcomes are game-level associations, not labels
that declare individual logged actions correct or incorrect.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cards import card  # noqa: E402
from agent.obsview import OT_ATTACK, OT_END, OT_PLAY, ObsView, ST_CARD, ST_MAIN  # noqa: E402
from tools import analyze_ladder_replays as LADDER  # noqa: E402
from tools import index_corpus  # noqa: E402


TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
TEAM_ALIAS = "増殖するG"
GRIM_IDS = frozenset((646, 647, 648))


class DiagnosticError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_id(entry: Any) -> int | None:
    if isinstance(entry, Mapping):
        value = entry.get("id")
        return value if isinstance(value, int) else None
    return entry if isinstance(entry, int) else None


def _board(view: ObsView) -> list[Mapping[str, Any]]:
    me = view.me or {}
    return [
        entry for entry in (me.get("active") or []) + (me.get("bench") or [])
        if isinstance(entry, Mapping)
    ]


def _has_dark(entry: Mapping[str, Any]) -> bool:
    values: list[Any] = []
    values.extend(entry.get("energyCards") or [])
    values.extend(entry.get("energies") or [])
    return 7 in {_entry_id(value) for value in values}


def _card_name(card_id: int | None) -> str:
    info = card(card_id)
    return str(info.get("name")) if info else str(card_id)


def _seat(game: Mapping[str, Any]) -> int | None:
    seats = [
        row for row in game.get("seats", [])
        if row.get("registered_deck_sha256") == TARGET_DECK_SHA256
    ]
    if len(seats) == 1:
        return int(seats[0]["seat"])
    named = [row for row in seats if row.get("team_name") == TEAM_ALIAS]
    return int(named[0]["seat"]) if len(named) == 1 else None


def _opponent_is_grim(game: Mapping[str, Any], learner_seat: int) -> bool:
    opponents = [
        row for row in game.get("seats", [])
        if int(row.get("seat", -1)) == 1 - learner_seat
    ]
    if len(opponents) != 1:
        return False
    deck = opponents[0].get("registered_deck") or []
    return LADDER.archetype(deck) == "Grimmsnarl"


def _effect_family(view: ObsView) -> str:
    return f"{view.context}:{_card_name(view.effect_card_id)}"


def _game_metrics(
    replay: Mapping[str, Any], seat: int, result: str
) -> dict[str, Any]:
    turns: dict[int, dict[str, Any]] = {}
    forced_ends = 0
    optional_ends = 0
    rare_candy_plays = 0
    card_families: Counter[str] = Counter()
    card_prompts = 0
    decisions = 0
    first_grim_turn = None
    first_two_lines_turn = None
    first_dark_munk_turn = None
    first_morgrem_turn = None
    first_froslass_turn = None

    for view, action in LADDER.action_rows(dict(replay), seat):
        decisions += 1
        board = _board(view)
        ids = [_entry_id(entry) for entry in board]
        state = {
            "grim": ids.count(648),
            "morgrem": ids.count(647),
            "impidimp": ids.count(646),
            "munkidori": ids.count(112),
            "dark_munkidori": sum(
                _entry_id(entry) == 112 and _has_dark(entry)
                for entry in board
            ),
            "froslass": ids.count(104),
            "snorunt": ids.count(860),
        }
        state["grim_lines"] = sum(state[key] for key in (
            "grim", "morgrem", "impidimp"
        ))
        turns.setdefault(int(view.turn), state)
        if state["grim"] and first_grim_turn is None:
            first_grim_turn = int(view.turn)
        if state["grim_lines"] >= 2 and first_two_lines_turn is None:
            first_two_lines_turn = int(view.turn)
        if state["dark_munkidori"] and first_dark_munk_turn is None:
            first_dark_munk_turn = int(view.turn)
        if state["morgrem"] and first_morgrem_turn is None:
            first_morgrem_turn = int(view.turn)
        if state["froslass"] and first_froslass_turn is None:
            first_froslass_turn = int(view.turn)

        if view.select_type == ST_CARD:
            card_prompts += 1
            card_families[_effect_family(view)] += 1
        if view.select_type != ST_MAIN:
            continue
        option_types = [option.get("type") for option in view.options]
        chosen_types = [
            option_types[index] for index in action if 0 <= index < len(option_types)
        ]
        if OT_END in chosen_types:
            if OT_ATTACK in option_types:
                optional_ends += 1
            else:
                forced_ends += 1
        for index in action:
            if not 0 <= index < len(view.options):
                continue
            option = view.options[index]
            if (
                option.get("type") == OT_PLAY
                and view.semantic_option_card_id(option) == 1079
            ):
                rare_candy_plays += 1

    ordered_turns = sorted(turns)
    opening = [turns[key] for key in ordered_turns[:3]]
    return {
        "result": result,
        "decisions": decisions,
        "learner_turns_observed": len(ordered_turns),
        "first_grim_turn": first_grim_turn,
        "first_morgrem_turn": first_morgrem_turn,
        "first_two_lines_turn": first_two_lines_turn,
        "first_dark_munk_turn": first_dark_munk_turn,
        "first_froslass_turn": first_froslass_turn,
        "forced_ends": forced_ends,
        "optional_ends": optional_ends,
        "rare_candy_plays": rare_candy_plays,
        "card_prompts": card_prompts,
        "card_families": dict(card_families),
        "opening_states": opening,
    }


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(row.get(key) is not None for row in rows) / len(rows) if rows else 0.0


def _average(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families: Counter[str] = Counter()
    for row in rows:
        families.update(row["card_families"])
    return {
        "games": len(rows),
        "first_grim_observed_rate": _rate(rows, "first_grim_turn"),
        "first_morgrem_observed_rate": _rate(rows, "first_morgrem_turn"),
        "two_lines_observed_rate": _rate(rows, "first_two_lines_turn"),
        "dark_munkidori_observed_rate": _rate(rows, "first_dark_munk_turn"),
        "froslass_observed_rate": _rate(rows, "first_froslass_turn"),
        "mean_first_grim_turn_when_observed": _average(rows, "first_grim_turn"),
        "mean_first_morgrem_turn_when_observed": _average(
            rows, "first_morgrem_turn"
        ),
        "mean_first_dark_munk_turn_when_observed": _average(
            rows, "first_dark_munk_turn"
        ),
        "mean_forced_ends": _average(rows, "forced_ends"),
        "mean_optional_ends": _average(rows, "optional_ends"),
        "mean_rare_candy_plays": _average(rows, "rare_candy_plays"),
        "mean_card_prompts": _average(rows, "card_prompts"),
        "st_card_families": [
            {"family": family, "prompts": count}
            for family, count in families.most_common()
        ],
    }


def analyze(index_path: Path) -> dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != "ptcg-corpus-index-v2":
        raise DiagnosticError("unexpected corpus-index schema")
    rows: list[dict[str, Any]] = []
    unresolved = Counter()
    replay_hashes: dict[str, str] = {}
    for game in index.get("games", []):
        seat = _seat(game)
        if seat is None:
            unresolved["learner_seat"] += 1
            continue
        if not _opponent_is_grim(game, seat):
            continue
        aliases = game.get("aliases") or []
        if not aliases:
            raise DiagnosticError("indexed game has no replay alias")
        path = Path(aliases[0]["resolved_path"])
        if _sha256(path) != game.get("content_sha256"):
            raise DiagnosticError(f"indexed replay drifted: {path}")
        replay = json.loads(path.read_text(encoding="utf-8"))
        reward = float(game["seats"][seat]["reward"])
        result = "win" if reward > 0 else ("loss" if reward < 0 else "draw")
        metrics = _game_metrics(replay, seat, result)
        metrics.update({
            "episode_id": game.get("episode_id"),
            "seat": seat,
            "split": game.get("split"),
            "exact_mirror": all(
                row.get("registered_deck_sha256") == TARGET_DECK_SHA256
                for row in game.get("seats", [])
            ),
        })
        rows.append(metrics)
        replay_hashes[str(game.get("episode_id"))] = game["content_sha256"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["result"]].append(row)
    payload = {
        "schema": "ptcg.md-v3.mirror-ladder-diagnostic.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "Descriptive associations in fixed ladder outcomes. No individual "
            "logged action is treated as a correct or counterfactual label."
        ),
        "input": {
            "index_path": str(index_path.resolve()),
            "index_sha256": _sha256(index_path),
            "manifest_sha256": index.get("manifest_sha256"),
            "corpus_content_sha256": index.get("corpus_content_sha256"),
        },
        "games": {
            "resolved_grimmsnarl": len(rows),
            "exact_mirror": sum(row["exact_mirror"] for row in rows),
            "by_result": dict(Counter(row["result"] for row in rows)),
            "unresolved": dict(unresolved),
        },
        "summary": {
            result: _summarize(group)
            for result, group in sorted(grouped.items())
        },
        "records": rows,
        "replay_sha256_by_episode": replay_hashes,
        "interpretation_contract": {
            "may_support": (
                "choosing a narrow separately preregistered mirror experiment"
            ),
            "may_not_support": (
                "behavior-cloning losses, declaring winner actions optimal, "
                "or promoting a runtime"
            ),
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    payload["diagnostic_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = analyze(args.index)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (DiagnosticError, OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "diagnostic_sha256": payload["diagnostic_sha256"],
        "games": payload["games"],
        "summary": payload["summary"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
