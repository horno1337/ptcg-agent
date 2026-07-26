"""Run one independent balanced-rollout panel over reconstructed MD-v2 roots.

Reuses :func:`tools.research.label_qu_v2c_exact_panels.evaluate_root`
unchanged -- it already evaluates every supported semantic option at a root
in one balanced-rollout panel continuing with frozen Qu-v2B policy for both
seats, which is exactly what isolates the value of the root decision alone.
MD-v1's action, frozen Qu-v2B's action, and the logged top-pilot action are
three of the evaluated options; this script does not add a second rollout
mechanism, it reads the three relevant entries out of the same panel.

Run this script twice per locked cohort (independent ``--panel-tag`` values)
to get the discovery and confirmation panels the pre-registered protocol
requires; the native engine RNG is unseedable, so two calls against the same
roots are already two independent stochastic draws.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model  # noqa: E402
from tools import il_dataset  # noqa: E402
from tools.cabt import AgentSearch  # noqa: E402
from tools.research import label_qu_v2c_exact_panels as LABEL  # noqa: E402
from tools.research import lock_md_v2_rollout_cohort as LOCK  # noqa: E402


SCHEMA = "ptcg.md-v2.rollout-panel.v1"


def _replay_path_lookup(cohort_lock: Mapping[str, Any]) -> dict[str, str]:
    return {
        row["replay_sha256"]: row["replay_path"]
        for row in cohort_lock["candidate_pool"]["ordered_roots"]
    }


def run_panel(
    public_roots: Sequence[Mapping[str, Any]],
    privileged_roots: Sequence[Mapping[str, Any]],
    cohort_lock: Mapping[str, Any],
    *,
    rollouts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    net = model.load()
    if net is None or not getattr(net, "is_qu_v2", False):
        raise LABEL.PanelError("frozen Qu-v2B failed to load")
    lookup = _replay_path_lookup(cohort_lock)
    search = AgentSearch()
    panels: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    deck_cache: dict[str, tuple[list[int], list[int]]] = {}
    for public_record, privileged_record in zip(public_roots, privileged_roots):
        replay_sha = public_record["source"]["replay_sha256"]
        replay_path = lookup.get(replay_sha)
        if replay_path is None:
            rejected.append({
                "root_id": public_record.get("root_id"),
                "reason": "replay_path_not_in_cohort_lock",
                "detail": replay_sha,
            })
            continue
        if replay_sha not in deck_cache:
            decks = il_dataset.decks(replay_path)
            if set(decks) != {0, 1}:
                rejected.append({
                    "root_id": public_record.get("root_id"),
                    "reason": "malformed_deck_registration",
                    "detail": replay_path,
                })
                continue
            deck_cache[replay_sha] = (decks[0], decks[1])
        registered_decks = deck_cache[replay_sha]
        try:
            panels.append(LABEL.evaluate_root(
                public_record, privileged_record, registered_decks, net,
                search, rollouts,
            ))
        except LABEL.PanelIncomplete as exc:
            rejected.append({
                "root_id": public_record.get("root_id"),
                "reason": "incomplete_terminal_panel",
                "detail": str(exc),
            })
        except LABEL.PanelRejected as exc:
            rejected.append({
                "root_id": public_record.get("root_id"),
                "reason": exc.reason,
                "detail": str(exc),
            })
    return panels, rejected


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-roots", type=Path, required=True)
    parser.add_argument("--privileged-roots", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--panel-tag", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        cohort_lock = LOCK.load_lock(args.cohort_lock)
        public_roots = _load_jsonl(args.public_roots)
        privileged_roots = _load_jsonl(args.privileged_roots)
        if len(public_roots) != len(privileged_roots):
            raise LABEL.PanelError("public/privileged root counts diverged")
        panels, rejected = run_panel(
            public_roots, privileged_roots, cohort_lock,
            rollouts=args.rollouts,
        )
        if not panels:
            raise LABEL.PanelError("every root lacked a complete panel")
    except (
        LOCK.CohortLockError, LABEL.PanelError, OSError, ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "panel_tag": args.panel_tag,
        "cohort_lock_sha256": cohort_lock["lock_sha256"],
        "rollouts": args.rollouts,
        "requested_roots": len(public_roots),
        "completed_roots": len(panels),
        "rejected_roots": rejected,
        "panels": panels,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"panel_tag={args.panel_tag} completed={len(panels)} "
          f"rejected={len(rejected)}")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
