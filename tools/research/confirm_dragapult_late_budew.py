"""Confirm the preregistered late-Budew tempo hypothesis on held-out roots.

The discovery screen found one state where the deployed policy's Ultra Ball
lost every rollout while Itchy Pollen won every rollout.  This gate selects
only held-out source episodes, using public information and a fixed condition
written before any confirmation outcomes are generated:

* Budew is Active;
* the opponent has at most two Prizes remaining;
* the deployed action is Ultra Ball; and
* an attack is legal.

Exact hidden state is used only by the offline native simulator.  A runtime
policy may use only the public condition, and this single-root confirmation is
diagnostic rather than promotion authority.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (ROOT, TOOLS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools.cabt import AgentSearch
from tools.research import collect_dragapult_tempo_roots as COLLECT
from tools.research import label_dragapult_tempo_roots as PANEL
from tools.research import eval_dragapult_route_guards as RG
from tools.research import eval_md_v2_scaled_gameplay as COMMON


RUN = COLLECT.RUN / "late-budew-confirm-v1"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
ROLLOUTS = 16


def eligible(row: Mapping[str, Any]) -> bool:
    snapshot = row.get("tempo_snapshot") or {}
    source = row.get("source") or {}
    return bool(
        not PANEL.discovery_episode(int(source.get("episode_id", -1)))
        and row.get("current_policy", {}).get("family") == "ultra_ball"
        and snapshot.get("my_active_budew") is True
        and int(snapshot.get("opponent_prizes_left", 99)) <= 2
        and "attack" in row.get("option_families", [])
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise PANEL.PanelError("late-Budew confirmation already locked or consumed")
    collection = json.loads(COLLECT.RESULT.read_text(encoding="utf-8"))
    collection_sha = collection.pop("result_sha256", None)
    if collection_sha != PANEL.canonical(collection):
        raise PANEL.PanelError("collection result self-hash failed")
    roots = sorted(
        row["root_id"] for row in PANEL.read_jsonl(COLLECT.PUBLIC) if eligible(row)
    )
    if not roots:
        raise PANEL.PanelError("no held-out roots satisfy the preregistered condition")
    paths = {
        "confirmation": Path(__file__).resolve(),
        "panel_evaluator": Path(PANEL.__file__).resolve(),
        "collection_lock": COLLECT.LOCK,
        "collection_result": COLLECT.RESULT,
        "public_roots": COLLECT.PUBLIC,
        "privileged_roots": COLLECT.PRIVILEGED,
        "main": RG.PATHS["main"], "card": RG.PATHS["card"],
        "qu": RG.PATHS["qu"], "policy": RG.PATHS["guards"],
    }
    payload = {
        "schema": "ptcg.dragapult-late-budew-confirm.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_confirmation_outcomes": True,
        "collection_result_sha256": collection_sha,
        "public_condition": {
            "split": "confirmation episode",
            "my_active_budew": True,
            "opponent_prizes_left_max": 2,
            "current_family": "ultra_ball",
            "required_legal_family": "attack",
        },
        "root_ids": roots,
        "rollouts_per_action": ROLLOUTS,
        "all_legal_root_actions": True,
        "learner_continuation": "exact dragapult-v2",
        "opponent_continuation": "frozen Qu-v2B",
        "success_definition": (
            "attack-family mean outcome exceeds current Ultra Ball and its "
            "paired repetition advantage is positive in at least 12/16 rollouts"
        ),
        "artifacts": {name: PANEL.artifact(path) for name, path in paths.items()},
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = PANEL.canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if (
        value.get("schema") != "ptcg.dragapult-late-budew-confirm.lock.v1"
        or claimed != PANEL.canonical(value)
    ):
        raise PANEL.PanelError("late-Budew confirmation lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or PANEL.sha256_file(path) != row["sha256"]:
            raise PANEL.PanelError(f"locked artifact drifted: {path}")
    return value


def selected_records(lock: Mapping[str, Any]):
    public = {row["root_id"]: row for row in PANEL.read_jsonl(COLLECT.PUBLIC)}
    privileged = {
        row["root_id"]: row for row in PANEL.read_jsonl(COLLECT.PRIVILEGED)
    }
    rows = []
    for root_id in lock["root_ids"]:
        if root_id not in public or root_id not in privileged:
            raise PANEL.PanelError(f"selected root is absent: {root_id}")
        if not eligible(public[root_id]):
            raise PANEL.PanelError(f"selected root no longer satisfies condition: {root_id}")
        rows.append((public[root_id], privileged[root_id]))
    return rows


def summarize(panel: Mapping[str, Any]) -> dict[str, Any]:
    current = int(panel["current_action_index"])
    attack_indices = [
        index for index, family in enumerate(panel["option_families"])
        if family == "attack"
    ]
    if not attack_indices:
        raise PANEL.PanelError("confirmed root has no attack option")
    attack = max(attack_indices, key=lambda index: panel["mean_scores"][index])
    paired = [
        row[attack] - row[current] for row in panel["raw_outcomes"]
    ]
    positives = sum(delta > 0 for delta in paired)
    ties = sum(delta == 0 for delta in paired)
    passed = bool(
        panel["mean_scores"][attack] > panel["mean_scores"][current]
        and positives >= 12
    )
    return {
        "root_id": panel["root_id"],
        "current_index": current,
        "attack_index": attack,
        "current_mean": panel["mean_scores"][current],
        "attack_mean": panel["mean_scores"][attack],
        "paired_positive": positives,
        "paired_ties": ties,
        "paired_negative": len(paired) - positives - ties,
        "passed": passed,
    }


def run(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise PANEL.PanelError("late-Budew confirmation attempt already consumed")
    PANEL.write_new(ATTEMPT, {
        "schema": "ptcg.dragapult-late-budew-confirm.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_confirmation_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    main = COMMON._load_net(Path(lock["artifacts"]["main"]["path"]), "v2 MAIN")
    card = COMMON._load_net(Path(lock["artifacts"]["card"]["path"]), "v2 CARD")
    qu = COMMON._load_net(Path(lock["artifacts"]["qu"]["path"]), "Qu-v2B")
    search = AgentSearch()
    panels = []
    for public, privileged in selected_records(lock):
        decks = PANEL.registrations(lock, public)
        panels.append(PANEL.evaluate_root(
            public, privileged, decks, main, card, qu, search,
            rollout_count=ROLLOUTS,
        ))
    summaries = [summarize(panel) for panel in panels]
    payload = {
        "schema": "ptcg.dragapult-late-budew-confirm.result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "summaries": summaries,
        "all_roots_passed": all(row["passed"] for row in summaries),
        "panels": panels,
        "contains_exact_hidden_card_ids": False,
        "promotion_authority": False,
        "package_authority": False,
    }
    payload["result_sha256"] = PANEL.canonical(payload)
    PANEL.write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        lock = build_lock() if not LOCK.exists() else load_lock()
        if not LOCK.exists():
            PANEL.write_new(LOCK, lock)
        if args.lock_only or not args.run:
            print(json.dumps({
                "lock_sha256": lock["lock_sha256"],
                "roots": len(lock["root_ids"]),
            }, sort_keys=True))
            return 0
        result = run(lock)
    except (PANEL.PanelError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "all_roots_passed": result["all_roots_passed"],
        "result_sha256": result["result_sha256"],
        "summaries": result["summaries"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
