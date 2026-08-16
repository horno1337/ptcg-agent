"""Describe expert-before-Dobi MAIN ordering on the AlphaStarmie Grim cohort.

This is a read-only diagnostic.  It replays frozen Dobi-v2 on the already
locked public observations and, when the expert later takes Dobi's proposed
coarse action in the same turn, records the action the expert placed before
it.  Frequency and directionality are not promotion evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.obsview import ObsView  # noqa: E402
from tools.audit_submission_runtime import _safe_extract  # noqa: E402
from tools.research import compare_current_grim_to_dobi_v2 as COMPARE  # noqa: E402
from tools.research.analyze_top_dragapult_divergence import (  # noqa: E402
    SELECT_TYPES,
    coarse,
)


DEFAULT_RUN = ROOT / "tools/checkpoints/alphatcg-grim-scout-20260813"
DEFAULT_ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
DEFAULT_OUT = DEFAULT_RUN / "sequencing-audit.json"


class AuditError(RuntimeError):
    """The fixed descriptive sequencing audit failed closed."""


def matchup_hashes(run: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for name in ("lucario", "dragapult", "mirror"):
        path = run / f"result-{name}.json"
        if not path.exists():
            result[name] = set()
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[name] = {
            str(game["opponent_deck_sha256"])
            for game in payload["cohort"]["games"]
            if game.get("opponent_deck_sha256")
        }
    return result


def classify(opponent_hash: str | None, hashes: dict[str, set[str]]) -> str:
    if opponent_hash == COMPARE.TARGET_DECK_SHA256:
        return "mirror"
    for name in ("lucario", "dragapult"):
        if opponent_hash in hashes[name]:
            return name
    return "other"


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise AuditError(f"refusing to overwrite {args.out}")
    if COMPARE.file_sha256(args.archive) != COMPARE.EXPECTED_ARCHIVE_SHA256:
        raise AuditError("frozen Dobi-v2 archive identity mismatch")

    source_path = args.run / "result.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    games = source["cohort"]["games"]
    hashes = matchup_hashes(args.run)
    edge_counts: Counter[tuple[str, str]] = Counter()
    edge_outcomes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    edge_matchups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    lag_counts: Counter[int] = Counter()
    lags: list[int] = []
    totals: Counter[str] = Counter()
    per_game_edges: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)

    with tempfile.TemporaryDirectory(prefix="alphastarmie-sequence-") as temporary:
        extracted = Path(temporary) / "submission"
        extracted.mkdir()
        _safe_extract(args.archive, extracted)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", COMPARE.WORKER],
            cwd=extracted,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            for game_index, game in enumerate(games, start=1):
                document = json.loads(Path(game["path"]).read_text(encoding="utf-8"))
                rows: list[dict[str, Any]] = []
                for step, observation, logged in COMPARE.decisions(document, game["seat"]):
                    view = ObsView(observation)
                    if SELECT_TYPES.get(view.select_type, str(view.select_type)) != "MAIN":
                        continue
                    expert = COMPARE.valid_action(view, logged)
                    if expert is None:
                        totals["invalid_expert"] += 1
                        continue
                    process.stdin.write(json.dumps(
                        {"observation": observation}, separators=(",", ":"),
                    ) + "\n")
                    process.stdin.flush()
                    line = process.stdout.readline()
                    if not line:
                        raise AuditError("Dobi-v2 worker terminated early")
                    predicted = COMPARE.valid_action(
                        view, json.loads(line).get("action"),
                    )
                    if predicted is None:
                        totals["invalid_dobi"] += 1
                        continue
                    rows.append({
                        "step": step,
                        "turn": int(view.turn),
                        "expert": coarse(view, list(expert)),
                        "dobi": coarse(view, list(predicted)),
                        "agree": (
                            COMPARE.semantic_set(view, expert)
                            == COMPARE.semantic_set(view, predicted)
                        ),
                    })
                    totals["main_decisions"] += 1
                matchup = classify(game.get("opponent_deck_sha256"), hashes)
                for position, row in enumerate(rows):
                    if row["agree"] or row["expert"] == row["dobi"]:
                        continue
                    totals["coarse_disagreements"] += 1
                    later_position = next((
                        index for index in range(position + 1, len(rows))
                        if rows[index]["turn"] == row["turn"]
                        and rows[index]["expert"] == row["dobi"]
                    ), None)
                    if later_position is None:
                        totals["never_this_turn"] += 1
                        continue
                    totals["later_same_turn"] += 1
                    edge = (row["expert"], row["dobi"])
                    lag = later_position - position
                    edge_counts[edge] += 1
                    edge_outcomes[edge][game["outcome"]] += 1
                    edge_matchups[edge][matchup] += 1
                    per_game_edges[edge].add((int(game["episode_id"]), int(game["seat"])))
                    lag_counts[lag] += 1
                    lags.append(lag)
                if game_index % 25 == 0:
                    print(f"audited {game_index}/{len(games)} seat-games", file=sys.stderr)
            process.stdin.close()
            return_code = process.wait(timeout=120)
            stderr = process.stderr.read() if process.stderr is not None else ""
            if return_code != 0:
                raise AuditError(f"Dobi-v2 worker failed: {stderr[-4000:]}")
        finally:
            if process.poll() is None:
                process.kill()

    edges = []
    for (before, after), count in edge_counts.most_common():
        reverse = edge_counts[(after, before)]
        pair_total = count + reverse
        edges.append({
            "expert_before": before,
            "dobi_proposed_then_expert_took": after,
            "occurrences": count,
            "seat_games": len(per_game_edges[(before, after)]),
            "reverse_occurrences": reverse,
            "direction_share_within_pair": rate(count, pair_total),
            "by_outcome": dict(edge_outcomes[(before, after)]),
            "by_matchup": dict(edge_matchups[(before, after)]),
        })
    stable = [
        edge for edge in edges
        if edge["occurrences"] >= 10
        and edge["direction_share_within_pair"] is not None
        and edge["direction_share_within_pair"] >= 0.8
        and edge["by_outcome"].get("win", 0) > 0
        and edge["by_outcome"].get("loss", 0) > 0
    ]
    payload: dict[str, Any] = {
        "schema": "ptcg.alphastarmie-grim-sequencing-audit.v1",
        "design": {
            "read_only": True,
            "causal": False,
            "same_public_observations": True,
            "source_result": str(source_path.resolve()),
            "source_result_sha256": COMPARE.file_sha256(source_path),
            "archive_sha256": COMPARE.file_sha256(args.archive),
            "definition": (
                "At a coarse MAIN disagreement, record expert_current -> "
                "Dobi_proposal only if the expert later takes that Dobi action "
                "in the same turn. The edge describes ordering, not value."
            ),
        },
        "cohort": {
            "seat_games": len(games),
            "outcomes": dict(Counter(game["outcome"] for game in games)),
        },
        "totals": {
            **dict(totals),
            "later_same_turn_share": rate(
                totals["later_same_turn"],
                totals["later_same_turn"] + totals["never_this_turn"],
            ),
        },
        "lag_in_subsequent_main_decisions": {
            "counts": {str(key): value for key, value in sorted(lag_counts.items())},
            "median": statistics.median(lags) if lags else None,
            "mean": statistics.mean(lags) if lags else None,
        },
        "ordered_edges": edges,
        "stable_descriptive_edges": stable,
        "stable_edge_screen": {
            "minimum_occurrences": 10,
            "minimum_direction_share": 0.8,
            "requires_presence_in_both_outcomes": True,
            "purpose": "prioritize inspection only; does not authorize a rule",
        },
        "mechanical_condition_screen": {
            "passed_edges": [],
            "reason": (
                "A coarse A-before-B edge does not itself establish an "
                "irreversible legality, arithmetic, or resource-preservation "
                "condition from public state. Each edge would require a "
                "separate state-level mechanical predicate or causal test."
            ),
        },
        "promotion_authority": False,
        "training_authority": False,
    }
    payload["result_sha256"] = COMPARE.canonical_sha256(payload)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "later_same_turn": totals["later_same_turn"],
        "never_this_turn": totals["never_this_turn"],
        "stable_edges": len(stable),
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
