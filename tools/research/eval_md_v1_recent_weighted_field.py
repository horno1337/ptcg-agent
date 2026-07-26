"""Compare MD-v1/Grim and frozen Qu-v2B/Alakazam on a weighted field."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model, policy  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


SCHEMA = "ptcg.md-v1.recent-weighted-field-eval.v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_field(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ptcg.recent-frequency-weighted-field.v1":
        raise ValueError("weighted field schema mismatch")
    field = payload.get("field")
    if not isinstance(field, list) or not field:
        raise ValueError("weighted field is empty")
    total = 0.0
    for index, item in enumerate(field):
        deck = item.get("deck") if isinstance(item, Mapping) else None
        weight = item.get("field_weight") if isinstance(item, Mapping) else None
        if (
            not isinstance(deck, list) or len(deck) != 60
            or not isinstance(weight, (int, float))
            or isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0
        ):
            raise ValueError(f"invalid weighted field entry {index}")
        total += float(weight)
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise ValueError("weighted field weights do not sum to one")
    return payload, field


def build_opponents(
    field: Sequence[Mapping[str, Any]], base_net: model.Net,
) -> tuple[list[OpponentSpec], EVAL.DeployableReflex]:
    controller = EVAL.DeployableReflex(
        base_net, f"field-qu-v2b:{file_sha256(ROOT / 'agent/weights.npz')}")
    opponents = []
    for item in field:
        deck = tuple(int(card_id) for card_id in item["deck"])

        def move(obs, rng, registered_deck=deck):
            del rng
            return controller.act(obs, registered_deck)

        opponents.append(OpponentSpec(
            key=f"{item['archetype']}/qu-v2b",
            deck=deck,
            move=move,
            weight=float(item["field_weight"]),
            policy_id=controller.name,
            schedule_group="recent-weighted-field",
        ))
    return opponents, controller


def independent_delta_ci(
    left: EVAL.SeriesResult, right: EVAL.SeriesResult,
) -> tuple[float, float, float]:
    """Unpooled normal CI for bounded per-game score (win=1, draw=.5)."""
    left_values = [
        1.0 if row.result == "win" else 0.5 if row.result == "draw" else 0.0
        for row in left.records
    ]
    right_values = [
        1.0 if row.result == "win" else 0.5 if row.result == "draw" else 0.0
        for row in right.records
    ]
    if len(left_values) < 2 or len(right_values) < 2:
        raise ValueError("delta interval needs at least two games per arm")
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    left_var = sum((x - left_mean) ** 2 for x in left_values) / (
        len(left_values) - 1)
    right_var = sum((x - right_mean) ** 2 for x in right_values) / (
        len(right_values) - 1)
    se = math.sqrt(
        left_var / len(left_values) + right_var / len(right_values))
    delta = left_mean - right_mean
    return delta, delta - 1.959963984540054 * se, (
        delta + 1.959963984540054 * se)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--md-weights", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--grim-deck", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number")

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    recorded_lock_hash = lock.pop("lock_sha256", None)
    if recorded_lock_hash != hashlib.sha256(json.dumps(
        lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest():
        parser.error("gate lock self-hash mismatch")
    if args.games != lock["protocol"]["games_per_arm"] \
            or args.seed != lock["protocol"]["schedule_seed"]:
        parser.error("games/seed differ from the prospective lock")

    field_payload, field = load_field(args.field.resolve())
    base_net = EVAL.load_net(str(args.base_weights))
    md_net = EVAL.load_net(str(args.md_weights))
    grim_deck = [
        int(line) for line in args.grim_deck.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    alakazam_deck = policy.load_deck()
    if len(grim_deck) != 60 or len(alakazam_deck) != 60:
        parser.error("learner deck is not 60 cards")

    arm_specs = (
        (
            "md-v1-grimmsnarl",
            grim_deck,
            EVAL.DeployableReflex(
                md_net, f"md-v1:{file_sha256(args.md_weights)}", grim_deck,
                fallback_net=base_net, candidate_select_type=0,
            ),
        ),
        (
            "qu-v2b-alakazam",
            alakazam_deck,
            EVAL.DeployableReflex(
                base_net, f"qu-v2b:{file_sha256(args.base_weights)}",
                alakazam_deck,
            ),
        ),
    )
    results = []
    schedules = []
    opponent_sets = []
    opponent_diagnostics = []
    environments = []
    for tag, learner_deck, controller in arm_specs:
        opponents, field_controller = build_opponents(field, base_net)
        schedule = build_paired_schedule(
            opponents, args.games, seed=args.seed)
        result = EVAL.run_series(
            tag, controller, learner_deck, opponents, schedule,
            max_selects=5000, time_bank_s=600.0,
            verbose=not args.quiet,
        )
        EVAL.print_result(result)
        results.append(result)
        schedules.append(schedule)
        opponent_sets.append(opponents)
        opponent_diagnostics.append(field_controller.diagnostics())
        environments.append(environment_manifest(
            learner_deck, opponents, str(args.field)))

    md_result, alakazam_result = results
    delta, low, high = independent_delta_ci(md_result, alakazam_result)
    all_valid = md_result.gate_valid and alakazam_result.gate_valid
    decision = (
        "md-v1-grimmsnarl"
        if all_valid and low > 0
        else "qu-v2b-alakazam"
        if all_valid and high < 0
        else "inconclusive"
    )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": recorded_lock_hash,
        "field": {
            "path": str(args.field.resolve()),
            "sha256": file_sha256(args.field),
            "included_share": field_payload["selection"]["included_share"],
        },
        "metric": "(wins + 0.5 * official draws) / scheduled games",
        "all_valid": all_valid,
        "md_minus_alakazam": {
            "delta": delta,
            "independent_unpooled_ci95": [low, high],
        },
        "decision": decision,
        "results": [
            {
                "summary": result.summary(),
                "records": [asdict(record) for record in result.records],
            }
            for result in results
        ],
        "schedules": [
            schedule_manifest(schedule, opponents)
            for schedule, opponents in zip(schedules, opponent_sets)
        ],
        "environments": environments,
        "opponent_controllers": opponent_diagnostics,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.json_out.with_suffix(args.json_out.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.json_out)
    print(
        f"WEIGHTED_FIELD md={100*md_result.score:.1f}% "
        f"alakazam={100*alakazam_result.score:.1f}% "
        f"delta={100*delta:+.1f}pp ci95=[{100*low:+.1f},{100*high:+.1f}] "
        f"decision={decision}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
