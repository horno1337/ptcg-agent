"""Reconstruct exact-hidden-state roots for the locked MD-v2 rollout cohort.

Reuses the existing Qu-v2C privileged root-mining machinery
(:mod:`tools.research.mine_qu_v2c_roots`) exactly as built, parametrized to
MD-v1's target Grimmsnarl deck instead of our own registered learner deck,
and with MD-v1's own network standing in as the "parent" whose disagreement
with frozen Qu-v2B is detected. For each locked candidate root (already
identified at the discovery stage), the source game is mined in
``factual-critic`` mode (no reward/disagreement pre-filter, so every eligible
row in that game round-trips through the same exact-hidden-state alignment
used everywhere else in this project) and the one root matching the locked
candidate's exact source step is kept.

This is deliberately per-candidate, not a corpus-wide sweep: the rollout
cohort was already pre-registered before any outcome was inspected, so
mining only re-derives the exact-hidden-state material for roots already
selected on structural (MD != Qu) grounds -- no new selection happens here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import md_v1 as MD  # noqa: E402
from agent import model  # noqa: E402
from tools.research import lock_md_v2_rollout_cohort as LOCK  # noqa: E402
from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402


SCHEMA = "ptcg.md-v2.root-reconstruction.v1"


class ReconstructionError(RuntimeError):
    """A locked candidate root could not be reconstructed."""


def reconstruct(
    cohort_lock: Mapping[str, Any],
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    qu_v2b = model.load()
    if qu_v2b is None or not getattr(qu_v2b, "is_qu_v2", False):
        raise ReconstructionError("frozen Qu-v2B failed to load")
    md_v1_net = MD._load()
    if md_v1_net is None:
        raise ReconstructionError("MD-v1 weights failed to load")
    learner_deck = list(MD.TARGET_DECK)

    ordered_roots = cohort_lock["candidate_pool"]["ordered_roots"]
    if limit is not None:
        ordered_roots = ordered_roots[:limit]

    public_records: list[dict[str, Any]] = []
    privileged_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for candidate in ordered_roots:
        replay_path = Path(candidate["replay_path"])
        entry = {
            "root_id": candidate["root_id"],
            "game_key": candidate["game_key"],
            "team_name": candidate["team_name"],
            "replay_path": str(replay_path),
        }
        try:
            public_batch, privileged_batch, _summary = MINE.mine(
                [replay_path],
                learner_deck,
                qu_v2b,
                md_v1_net,
                aliases=set(),
                selection_mode="factual-critic",
            )
        except MINE.MiningError as exc:
            entry["status"] = "mining_error"
            entry["detail"] = str(exc)
            diagnostics.append(entry)
            continue
        matches = [
            (public_record, privileged_record)
            for public_record, privileged_record in zip(
                public_batch, privileged_batch)
            if (
                public_record["source"]["source_step"]
                == candidate["source_step"]
                and public_record["source"]["learner_seat"]
                == candidate["acting_seat"]
            )
        ]
        if len(matches) != 1:
            entry["status"] = "no_unique_match"
            entry["matches_found"] = len(matches)
            diagnostics.append(entry)
            continue
        public_record, privileged_record = matches[0]
        if not public_record["selection"]["frozen_b_parent_disagreement"]:
            entry["status"] = "disagreement_did_not_reproduce"
            diagnostics.append(entry)
            continue
        public_records.append(public_record)
        privileged_records.append(privileged_record)
        entry["status"] = "reconstructed"
        diagnostics.append(entry)

    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_lock_sha256": cohort_lock["lock_sha256"],
        "requested_candidates": len(ordered_roots),
        "reconstructed_roots": len(public_records),
        "status_counts": {
            status: sum(1 for d in diagnostics if d["status"] == status)
            for status in {d["status"] for d in diagnostics}
        },
        "candidate_diagnostics": diagnostics,
    }
    return public_records, privileged_records, summary


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False)
        for record in records
    ) + ("\n" if records else "")
    path.write_text(payload, encoding="utf-8")
    os.chmod(path, mode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--public-out", type=Path, required=True)
    parser.add_argument("--privileged-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        cohort_lock = LOCK.load_lock(args.cohort_lock)
        public_records, privileged_records, summary = reconstruct(
            cohort_lock, limit=args.limit)
    except (
        LOCK.CohortLockError, ReconstructionError, OSError, ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _write_jsonl(args.public_out, public_records, 0o644)
    _write_jsonl(args.privileged_out, privileged_records, 0o600)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["status_counts"], indent=2))
    print(f"reconstructed {summary['reconstructed_roots']} "
          f"of {summary['requested_candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
