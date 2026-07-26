"""Confirm MD-v1/Qu-v2B/logged-action advantages across two independent
rollout panels and apply the pre-registered 200-root/30-game/5-team gate.

For every root present in both the discovery and confirmation panel, reads
the three action indices already recorded at mining time (MD-v1, frozen
Qu-v2B, logged top-pilot) and their rollout mean scores from each
independent panel. A comparison confirms only if both panels agree in sign
on which action scored higher; unresolved (near-zero) deltas in either panel
never count as agreement. The five interpretable categories are derived only
from confirmed comparisons -- an unconfirmed root is discarded, never
guessed at.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_md_v2_rollout_cohort as LOCK  # noqa: E402


SCHEMA = "ptcg.md-v2.confirmation-gate.v1"
NEAR_ZERO = 1e-9


class ConfirmationError(RuntimeError):
    """The confirmation gate could not be evaluated."""


def _sign(value: float) -> int:
    if value > NEAR_ZERO:
        return 1
    if value < -NEAR_ZERO:
        return -1
    return 0


def _panel_index(panel: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["root_id"]: row for row in panel["panels"]}


def _action_scores(
    panel_row: Mapping[str, Any], public_record: Mapping[str, Any],
) -> dict[str, float] | None:
    means = panel_row["mean_scores"]
    md_index = public_record["parent"]["action"][0]
    qu_index = public_record["qu_v2b"]["action"][0]
    logged_index = public_record["logged"]["action"][0]
    if not all(0 <= i < len(means) for i in (md_index, qu_index, logged_index)):
        return None
    return {
        "md": means[md_index], "qu": means[qu_index], "logged": means[logged_index],
    }


def _confirmed_comparison(
    discovery_scores: Mapping[str, float],
    confirmation_scores: Mapping[str, float],
    left: str,
    right: str,
) -> int | None:
    """Return the confirmed sign of ``left - right`` or ``None``."""
    d_sign = _sign(discovery_scores[left] - discovery_scores[right])
    c_sign = _sign(confirmation_scores[left] - confirmation_scores[right])
    if d_sign == 0 or c_sign == 0 or d_sign != c_sign:
        return None
    return d_sign


def _categories(
    md_vs_qu: int | None,
    md_vs_logged: int | None,
    qu_vs_logged: int | None,
    md_equals_logged: bool,
    qu_equals_logged: bool,
) -> list[str]:
    """Independent, non-exclusive tags a confirmed root can carry.

    A root can honestly carry more than one tag (e.g. logged beats Qu and
    MD simultaneously beats Qu while differing from logged); tags are
    reported as separate boolean facts rather than forced into one bucket.
    """
    if md_vs_qu is None:
        return ["no_stable_advantage_discard"]
    tags = []
    md_loses_to_logged = md_equals_logged or md_vs_logged == -1
    md_beats_logged = (not md_equals_logged) and md_vs_logged == 1
    logged_beats_qu = (not qu_equals_logged) and qu_vs_logged == -1
    if md_vs_qu == -1 and md_loses_to_logged:
        tags.append("md_loses_to_both_strong_correction_target")
    if md_vs_qu == 1 and not md_equals_logged:
        tags.append("md_beats_qu_but_differs_from_logged")
    if md_vs_qu == 1 and md_equals_logged:
        tags.append("md_matches_logged_and_beats_qu")
    if md_vs_qu == 1 and md_beats_logged:
        tags.append("md_beats_both_qu_and_logged")
    if logged_beats_qu and md_loses_to_logged and md_vs_qu == -1:
        tags.append("logged_beats_both_valuable_pilot_signal")
    if not tags:
        tags.append("confirmed_other_ordering")
    return tags


def _team_by_game(cohort_lock: Mapping[str, Any]) -> dict[str, str]:
    return {
        row["game_key"]: row["team_name"]
        for row in cohort_lock["candidate_pool"]["ordered_roots"]
    }


def confirm(
    discovery_panel: Mapping[str, Any],
    confirmation_panel: Mapping[str, Any],
    public_records_by_root: Mapping[str, Mapping[str, Any]],
    cohort_lock: Mapping[str, Any],
) -> dict[str, Any]:
    team_by_game = _team_by_game(cohort_lock)
    discovery_by_root = _panel_index(discovery_panel)
    confirmation_by_root = _panel_index(confirmation_panel)
    common_roots = sorted(
        set(discovery_by_root) & set(confirmation_by_root))
    confirmed_roots: list[dict[str, Any]] = []
    unconfirmed = 0
    for root_id in common_roots:
        public_record = public_records_by_root.get(root_id)
        if public_record is None:
            continue
        d_scores = _action_scores(discovery_by_root[root_id], public_record)
        c_scores = _action_scores(confirmation_by_root[root_id], public_record)
        if d_scores is None or c_scores is None:
            continue
        md_vs_qu = _confirmed_comparison(d_scores, c_scores, "md", "qu")
        md_equals_logged = (
            public_record["parent"]["action"] == public_record["logged"]["action"])
        md_vs_logged = (
            None if md_equals_logged
            else _confirmed_comparison(d_scores, c_scores, "md", "logged"))
        qu_equals_logged = (
            public_record["qu_v2b"]["action"] == public_record["logged"]["action"])
        qu_vs_logged = (
            None if qu_equals_logged
            else _confirmed_comparison(d_scores, c_scores, "qu", "logged"))
        tags = _categories(
            md_vs_qu, md_vs_logged, qu_vs_logged,
            md_equals_logged, qu_equals_logged)
        if tags == ["no_stable_advantage_discard"]:
            unconfirmed += 1
            continue
        episode_id = public_record["source"]["episode_id"]
        confirmed_roots.append({
            "root_id": root_id,
            "episode_id": episode_id,
            "team_name": team_by_game.get(episode_id, "?"),
            "categories": tags,
            "md_vs_qu_sign": md_vs_qu,
            "md_vs_logged_sign": md_vs_logged,
            "qu_vs_logged_sign": qu_vs_logged,
            "discovery_scores": d_scores,
            "confirmation_scores": c_scores,
        })
    total_scored = len(confirmed_roots) + unconfirmed
    sign_agreement = (
        len(confirmed_roots) / total_scored if total_scored else 0.0)
    confirmed_games = {row["episode_id"] for row in confirmed_roots}
    confirmed_teams = {row["team_name"] for row in confirmed_roots}
    category_counts: Counter[str] = Counter()
    for row in confirmed_roots:
        category_counts.update(row["categories"])
    gate = {
        "minimum_sign_agreement": LOCK.MIN_SIGN_AGREEMENT,
        "minimum_confirmed_roots": LOCK.MIN_CONFIRMED_ROOTS,
        "minimum_confirmed_games": LOCK.MIN_CONFIRMED_GAMES,
        "minimum_confirmed_teams": LOCK.MIN_CONFIRMED_TEAMS,
        "observed_sign_agreement": sign_agreement,
        "observed_confirmed_roots": len(confirmed_roots),
        "observed_confirmed_games": len(confirmed_games),
        "observed_confirmed_teams": len(confirmed_teams),
        "sign_agreement_passed": sign_agreement >= LOCK.MIN_SIGN_AGREEMENT,
        "roots_passed": len(confirmed_roots) >= LOCK.MIN_CONFIRMED_ROOTS,
        "games_passed": len(confirmed_games) >= LOCK.MIN_CONFIRMED_GAMES,
        "teams_passed": len(confirmed_teams) >= LOCK.MIN_CONFIRMED_TEAMS,
    }
    gate["gate_passed"] = all([
        gate["sign_agreement_passed"], gate["roots_passed"],
        gate["games_passed"], gate["teams_passed"],
    ])
    return {
        "schema": SCHEMA,
        "roots_scored": total_scored,
        "roots_unconfirmed": unconfirmed,
        "category_counts": dict(sorted(category_counts.items())),
        "gate": gate,
        "confirmed_roots": confirmed_roots,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-panel", type=Path, required=True)
    parser.add_argument("--confirmation-panel", type=Path, required=True)
    parser.add_argument("--public-roots", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        cohort_lock = LOCK.load_lock(args.cohort_lock)
        discovery_panel = json.loads(
            args.discovery_panel.read_text(encoding="utf-8"))
        confirmation_panel = json.loads(
            args.confirmation_panel.read_text(encoding="utf-8"))
        public_records_by_root = {
            json.loads(line)["root_id"]: json.loads(line)
            for line in args.public_roots.read_text(
                encoding="utf-8").splitlines() if line.strip()
        }
        result = confirm(
            discovery_panel, confirmation_panel, public_records_by_root,
            cohort_lock)
    except (
        LOCK.CohortLockError, OSError, ValueError, json.JSONDecodeError,
        KeyError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {"category_counts": result["category_counts"], "gate": result["gate"]},
        indent=2, sort_keys=True))
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
