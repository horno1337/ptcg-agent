"""Exploratory exact-state panel for the seven rare Petrel/Lillie roots.

The preregistered supporter experiment collected only seven roots and its hash
split placed every root in confirmation, so its formal panel cannot open.  This
separate readout locks all seven roots before terminal branching.  It may reject
the hypothesis; a positive result has no promotion authority and requires a
fresh independent confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools.cabt import AgentSearch, _LIB_PATH  # noqa: E402
from tools.research import (  # noqa: E402
    eval_dobi_v2_lucario_boss_counterfactual as BASE,
    eval_dobi_v2_lucario_supporter_counterfactual as SUPPORT,
    eval_dobi_v2_public_lucario_dragapult as PUBLIC,
)


RUN = SUPPORT.RUN / "exploratory-all-seven"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
ROLLOUTS = 16


class ExploratoryError(RuntimeError):
    pass


def load_collection() -> dict[str, Any]:
    return BASE.load_self(
        SUPPORT.COLLECTION_RESULT,
        "ptcg.dobi-v2-lucario-supporter-collection-result.v1",
        "result_sha256",
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise ExploratoryError("exploratory panel already consumed")
    collection = load_collection()
    roots = BASE.read_jsonl(SUPPORT.PUBLIC_ROOTS)
    if len(roots) != 7:
        raise ExploratoryError(f"expected exactly seven collected roots, got {len(roots)}")
    root_ids = sorted(row["root_id"] for row in roots)
    payload = {
        "schema": "ptcg.dobi-v2-lucario-supporter-exploratory-lock.v1",
        "created_at": BASE.now(),
        "written_before_first_terminal_rollout": True,
        "reason_formal_panel_cannot_open": (
            "only seven roots were collected and the locked SHA split assigned "
            "all seven to confirmation; the required 4+4 panel is impossible"
        ),
        "selection": {
            "root_ids": root_ids,
            "root_count": len(root_ids),
            "order": "ascending root_id",
            "outcomes_not_used_for_selection": True,
        },
        "rollouts": {
            "per_action_per_root": ROLLOUTS,
            "all_legal_root_actions": True,
            "continuations": "frozen Dobi-v2 versus exactly restored public Lucario",
        },
        "interpretation": {
            "reject_if": "aggregate Lillie-minus-Petrel <=0 or positive on <=3/7 roots",
            "provisional_if": (
                "aggregate >=0.10, positive on >=5/7 roots, negative on <=1/7; "
                "requires fresh independent confirmation"
            ),
        },
        "artifacts": {
            "explorer": BASE.artifact(Path(__file__)),
            "supporter_collector": BASE.artifact(Path(SUPPORT.__file__)),
            "collection_lock": BASE.artifact(SUPPORT.COLLECTION_LOCK),
            "collection_result": BASE.artifact(SUPPORT.COLLECTION_RESULT),
            "public_roots": BASE.artifact(SUPPORT.PUBLIC_ROOTS),
            "privileged_roots": BASE.artifact(SUPPORT.PRIVILEGED_ROOTS),
            "stateful_worker": BASE.artifact(BASE.STATEFUL_WORKER),
            "engine": BASE.artifact(Path(_LIB_PATH)),
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = BASE.canonical(payload)
    BASE.write_new(LOCK, payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = BASE.load_self(
        LOCK,
        "ptcg.dobi-v2-lucario-supporter-exploratory-lock.v1",
        "lock_sha256",
    )
    for row in value["artifacts"].values():
        if PUBLIC.file_sha256(Path(row["path"])) != row["sha256"]:
            raise ExploratoryError(f"locked artifact drifted: {row['path']}")
    return value


def screen(panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["lillie_minus_petrel"]) for row in panels]
    aggregate = float(np.mean(deltas))
    positive = sum(value > 0 for value in deltas)
    negative = sum(value < 0 for value in deltas)
    if aggregate <= 0 or positive <= 3:
        verdict = "reject"
    elif aggregate >= 0.10 and positive >= 5 and negative <= 1:
        verdict = "provisional; requires fresh independent confirmation"
    else:
        verdict = "inconclusive; do not build a rule"
    return {
        "aggregate_mean_delta": aggregate,
        "positive_roots": positive,
        "zero_roots": sum(value == 0 for value in deltas),
        "negative_roots": negative,
        "verdict": verdict,
    }


def run(lock: Mapping[str, Any], quiet: bool) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise ExploratoryError("exploratory panel already consumed")
    BASE.write_new(ATTEMPT, {
        "schema": "ptcg.dobi-v2-lucario-supporter-exploratory-attempt.v1",
        "created_at": BASE.now(),
        "lock_sha256": lock["lock_sha256"],
        "written_before_first_terminal_rollout": True,
    })
    public = {row["root_id"]: row for row in BASE.read_jsonl(SUPPORT.PUBLIC_ROOTS)}
    privileged = {
        row["root_id"]: row for row in BASE.read_jsonl(SUPPORT.PRIVILEGED_ROOTS)
    }
    controller, learner_deck = PUBLIC.load_dobi()
    collection_lock = SUPPORT.load_collection_lock()
    opponent_deck = tuple(collection_lock["opponent"]["deck"])
    archive = Path(collection_lock["opponent"]["archive"]["path"])
    panels: list[dict[str, Any]] = []
    search = AgentSearch()
    with tempfile.TemporaryDirectory(prefix="dobi-supporter-explore-") as raw_temp:
        extracted = Path(raw_temp) / "lucario"
        extracted.mkdir()
        PUBLIC.safe_extract(archive, extracted)
        for root_id in lock["selection"]["root_ids"]:
            panel = BASE.evaluate_root(
                public[root_id], privileged[root_id], extracted,
                learner_deck, opponent_deck, controller, search,
            )
            panel["petrel_index"] = panel.pop("current_index")
            panel["lillie_index"] = panel.pop("fallback_index")
            panel["petrel_label"] = panel.pop("current_label")
            panel["lillie_label"] = panel.pop("fallback_label")
            panel["lillie_minus_petrel"] = panel.pop("fallback_minus_boss")
            panels.append(panel)
            if not quiet:
                print(f"exploratory panel {len(panels)}/7", flush=True)
    result = {
        "schema": "ptcg.dobi-v2-lucario-supporter-exploratory-result.v1",
        "created_at": BASE.now(),
        "lock_sha256": lock["lock_sha256"],
        "panels": panels,
        "screen": screen(panels),
        "contains_exact_hidden_card_ids": False,
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    result["result_sha256"] = BASE.canonical(result)
    BASE.write_new(RESULT, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = build_lock() if args.stage == "lock" else run(load_lock(), args.quiet)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
