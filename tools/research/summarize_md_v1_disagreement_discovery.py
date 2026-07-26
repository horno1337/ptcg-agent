"""Summarize a raw MD-v1 disagreement attribution report into scrubbed counts.

Deduplicates repeated ST_MAIN situations (identical semantic option sets) to
one representative root each, so a common early-turn decision that recurs
across many games is not counted -- or later weighted -- as if it were many
independent data points. Reports structural MD/Qu/logged agreement shape
only; it never claims one action is better than another. That claim requires
the separate paired-rollout confirmation stage.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _root_signature(prompt: dict[str, Any]) -> str:
    payload = json.dumps(
        prompt["semantic_options"], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _category(prompt: dict[str, Any]) -> str:
    md_vs_logged = prompt["secondary_md_vs_logged_disagreement"]
    qu_vs_logged = prompt["secondary_qu_vs_logged_disagreement"]
    if not md_vs_logged and qu_vs_logged:
        return "md_matches_logged_qu_differs"
    if md_vs_logged and not qu_vs_logged:
        return "qu_matches_logged_md_differs"
    if md_vs_logged and qu_vs_logged:
        return "three_way_distinct"
    return "inconsistent"  # md==logged and qu==logged would imply md==qu


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    prompts = report["prompts"]
    games = report["games"]

    unique_games = {g["episode_id"] for g in games}
    unique_teams = {g["team_name"] for g in games if g["team_name"]}

    all_roots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        all_roots[_root_signature(prompt)].append(prompt)
    disagreement_roots: dict[str, list[dict[str, Any]]] = {
        sig: rows for sig, rows in all_roots.items()
        if rows[0]["primary_md_vs_qu_disagreement"]
    }

    category_counts = Counter(
        _category(rows[0]) for rows in disagreement_roots.values())
    teams_per_disagreement_root = Counter()
    games_with_disagreement = set()
    team_disagreement_roots: dict[str, set[str]] = defaultdict(set)
    for sig, rows in disagreement_roots.items():
        for row in rows:
            games_with_disagreement.add(row["episode_id"])
            if row["team_name"]:
                team_disagreement_roots[row["team_name"]].add(sig)

    return {
        "schema": "ptcg.md-v2.disagreement-discovery-summary.v1",
        "source_counts": report["counts"],
        "exact_deck_games": len(unique_games),
        "exact_deck_teams": sorted(unique_teams),
        "unique_st_main_roots": len(all_roots),
        "unique_md_vs_qu_disagreement_roots": len(disagreement_roots),
        "raw_md_vs_qu_disagreement_prompts": sum(
            len(rows) for rows in disagreement_roots.values()),
        "deduplication": {
            "raw_eligible_prompts": report["counts"]["eligible_prompts"],
            "unique_roots_after_dedup": len(all_roots),
            "repeated_prompt_instances_collapsed": (
                report["counts"]["eligible_prompts"] - len(all_roots)),
        },
        "disagreement_categories": dict(sorted(category_counts.items())),
        "disagreement_games": len(games_with_disagreement),
        "disagreement_teams": len(team_disagreement_roots),
        "disagreement_roots_per_team": {
            team: len(sigs)
            for team, sigs in sorted(
                team_disagreement_roots.items(),
                key=lambda kv: -len(kv[1]))
        },
        "errors_and_unresolved": {
            key: value for key, value in report["counts"].items()
            if key not in (
                "eligible_prompts", "primary_md_vs_qu_agreement",
                "primary_md_vs_qu_disagreement", "resolved_action_rows")
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = summarize(report)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {k: v for k, v in summary.items()
         if not isinstance(v, (dict, list)) or len(str(v)) < 200},
        indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
