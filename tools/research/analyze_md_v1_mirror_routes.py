"""Summarize a locked MD-v1 mirror replay cohort without inventing labels.

The local evaluator's replay capture is a JSON list of public observations.
This tool uses the evaluator report to resolve the learner seat and terminal
result, then measures which prompt families still route to frozen Qu-v2B.
Losing actions remain descriptive states; they are never called mistakes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.obsview import ObsView, ST_MAIN  # noqa: E402


SCHEMA = "ptcg.md-v1.mirror-route-diagnostic.v1"
SELECT_TYPES = {
    0: "main",
    1: "card",
    2: "attached_card",
    3: "card_or_attached",
    4: "energy",
    5: "skill",
    6: "attack",
    7: "evolve",
    8: "count",
    9: "yes_no",
    10: "special_condition",
}
OPTION_TYPES = {
    0: "number",
    1: "yes",
    2: "no",
    3: "card",
    4: "tool_card",
    5: "energy_card",
    6: "energy",
    7: "play",
    8: "attach",
    9: "evolve",
    10: "ability",
    11: "discard",
    12: "retreat",
    13: "attack",
    14: "end",
    15: "skill",
    16: "special_condition",
}
SERIALIZED_SELECT_TYPES = {
    "Main": 0,
    "Card": 1,
    "AttachedCard": 2,
    "CardOrAttached": 3,
    "Energy": 4,
    "Skill": 5,
    "Attack": 6,
    "Evolve": 7,
    "Count": 8,
    "YesNo": 9,
    "SpecialCondition": 10,
}


class DiagnosticError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_type(view: ObsView) -> int:
    raw = view.select_type
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return SERIALIZED_SELECT_TYPES.get(str(raw), -1)


def _option_type_name(raw: Any) -> str:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return OPTION_TYPES.get(raw, f"type{raw}")
    return str(raw)


def _family(view: ObsView) -> str:
    select = view.select or {}
    option_types = Counter(
        _option_type_name(option.get("type"))
        for option in view.options
    )
    context_card = select.get("contextCard")
    if isinstance(context_card, Mapping):
        context_card = context_card.get("name") or context_card.get("id")
    record = {
        "select_type": SELECT_TYPES.get(
            _select_type(view), f"type{view.select_type}"
        ),
        "context": select.get("context"),
        "context_card": context_card,
        "min_count": view.min_count,
        "max_count": view.max_count,
        "option_types": dict(sorted(option_types.items())),
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _rows(counter: Counter[str], games: int) -> list[dict[str, Any]]:
    result = []
    for encoded, count in counter.most_common():
        family = json.loads(encoded)
        family["prompts"] = count
        family["prompts_per_game"] = count / games if games else 0.0
        result.append(family)
    return result


def analyze(eval_path: Path, replay_dir: Path) -> dict[str, Any]:
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    if report.get("schema") != "ptcg-eval-ab-v2":
        raise DiagnosticError("unexpected evaluator schema")
    if report.get("args", {}).get("candidate_select_type") != ST_MAIN:
        raise DiagnosticError("cohort is not scoped to MD-v1 at ST_MAIN")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise DiagnosticError("expected one mirror result arm")
    summary = results[0].get("summary") or {}
    records = results[0].get("records")
    if (
        not summary.get("gate_valid")
        or summary.get("invalid") != 0
        or not isinstance(records, list)
        or len(records) != report.get("args", {}).get("games")
    ):
        raise DiagnosticError("mirror cohort is incomplete or invalid")

    by_result_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_result_route: dict[str, Counter[str]] = defaultdict(Counter)
    non_main_families: Counter[str] = Counter()
    loss_families: Counter[str] = Counter()
    replay_hashes = {}
    prompt_total = 0

    for record in records:
        episode_id = int(record["episode_id"])
        path = replay_dir / f"candidate-vs-base-episode-{episode_id:06d}.json"
        if not path.is_file():
            raise DiagnosticError(f"missing replay for episode {episode_id}")
        observations = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(observations, list):
            raise DiagnosticError(f"replay {path} is not an observation list")
        replay_hashes[str(episode_id)] = _sha256(path)
        result = str(record["result"])
        learner_seat = int(record["learner_seat"])
        for observation in observations:
            if not isinstance(observation, Mapping):
                continue
            current = observation.get("current")
            select = observation.get("select")
            if (
                not isinstance(current, Mapping)
                or current.get("yourIndex") != learner_seat
                or not isinstance(select, Mapping)
                or not select.get("option")
            ):
                continue
            view = ObsView(dict(observation))
            normalized_type = _select_type(view)
            select_name = SELECT_TYPES.get(
                normalized_type, f"type{view.select_type}"
            )
            route = "md_v1" if normalized_type == ST_MAIN else "qu_v2b"
            by_result_type[result][select_name] += 1
            by_result_route[result][route] += 1
            prompt_total += 1
            if route == "qu_v2b":
                family = _family(view)
                non_main_families[family] += 1
                if result == "loss":
                    loss_families[family] += 1

    result_games = Counter(str(record["result"]) for record in records)
    route_totals = Counter()
    type_totals = Counter()
    for counts in by_result_route.values():
        route_totals.update(counts)
    for counts in by_result_type.values():
        type_totals.update(counts)
    if prompt_total != sum(route_totals.values()):
        raise DiagnosticError("prompt accounting drift")

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "Descriptive public-observation route attribution only. Loss "
            "states and logged actions are not action-quality labels."
        ),
        "inputs": {
            "eval_path": str(eval_path),
            "eval_sha256": _sha256(eval_path),
            "replay_dir": str(replay_dir),
            "replay_sha256_by_episode": replay_hashes,
        },
        "games": {
            "total": len(records),
            "by_result": dict(sorted(result_games.items())),
        },
        "prompts": {
            "total": prompt_total,
            "route_counts": dict(sorted(route_totals.items())),
            "route_fractions": {
                key: value / prompt_total
                for key, value in sorted(route_totals.items())
            },
            "select_type_counts": dict(sorted(type_totals.items())),
            "by_result_and_route": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_result_route.items())
            },
            "by_result_and_select_type": {
                key: dict(sorted(value.items()))
                for key, value in sorted(by_result_type.items())
            },
        },
        "non_st_main_families": {
            "all": _rows(non_main_families, len(records)),
            "losses": _rows(loss_families, result_games.get("loss", 0)),
        },
        "interpretation_contract": {
            "may_support": (
                "selection of high-volume prompt families for a separately "
                "locked deck-matched data or counterfactual-label study"
            ),
            "may_not_support": (
                "action correction, model promotion, deck promotion, or "
                "winner-action behavior cloning"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(
            args.eval.expanduser().resolve(),
            args.replays.expanduser().resolve(),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            DiagnosticError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "games": report["games"],
        "prompts": report["prompts"],
        "top_loss_families":
            report["non_st_main_families"]["losses"][:10],
    }, indent=2, sort_keys=True))
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
