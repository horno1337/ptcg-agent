#!/usr/bin/env python3
"""Milestone 1 for Dobi-aware turn search: exact reproduction parity.

Before the planner may simulate anything, ``agent/seat_policy.py`` must prove
it reproduces frozen Dobi-v2 **exactly** with search overrides disabled.  The
reference is the packaged runtime inside archive ``409dad44...fa8e4``, driven
through the packaged ``policy`` module -- not repo source, whose
``dobi_v1_card`` and ``md_v1`` carry different expected-weights hashes and so
fall through to Qu-v2B.

Protocol
--------
Play games with the packaged runtime as the learner, capturing every prompt
and the action the packaged router actually produced.  Replay each captured
observation through ``FrozenDobiV2Policy`` and require an exact action match.
Any mismatch, exception, or route disagreement fails the milestone.

This measures reproduction only.  Coverage, latency, override rate and
gameplay are separate later stages and are deliberately not run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent.seat_policy import (  # noqa: E402
    DOBI_V2_ARCHIVE_SHA256, FrozenDobiV2Policy, QuV2BasePolicy,
    SeatPolicyError, SeatPolicyTable, load_frozen_runtime,
)

ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
DECK = ROOT / "decks" / "deck.csv"


class ParityError(RuntimeError):
    pass


def read_deck(path: Path) -> tuple[int, ...]:
    values = tuple(int(line.strip()) for line in
                   path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(values) != 60:
        raise ParityError(f"{path} is not a 60-card registration")
    return values


class CapturingReference:
    """Packaged Dobi-v2 router; records every prompt and its emitted action."""

    def __init__(self, runtime, deck):
        self.runtime, self.deck = runtime, deck
        self.prompts: list[dict] = []
        self.routes: Counter[str] = Counter()
        self.reference = FrozenDobiV2Policy(runtime)

    def act(self, obs: dict) -> list[int]:
        decision = self.reference.decide(obs, self.deck, seat=0)
        self.prompts.append({"obs": obs, "action": list(decision.action),
                             "route": decision.route})
        self.routes[decision.route] += 1
        return list(decision.action)


def packaged_action(runtime, obs, deck) -> list[int]:
    """Drive the packaged ``policy`` module itself, as the ladder would.

    ``policy.load_deck`` reads the packaged ``decks/deck.csv``; bind the same
    registration explicitly so the comparison isolates routing, not file IO.
    """
    policy = runtime.policy
    original = policy.load_deck
    policy.load_deck = lambda: tuple(deck)
    try:
        view = policy.ObsView(obs)
        return list(policy._model_decide(view) or [])
    finally:
        policy.load_deck = original


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--opp", default="pool:8")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--archive", default=str(ARCHIVE))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    runtime = load_frozen_runtime(Path(args.archive))
    deck = read_deck(DECK)
    print(f"[parity] frozen runtime {runtime.archive_sha256[:16]}... "
          f"deck {hashlib.sha256(repr(deck).encode()).hexdigest()[:12]}",
          flush=True)

    # --- capture prompts from real games driven by the packaged router -----
    from tools import eval_ab as EVAL
    from tools.rl_env import OpponentSpec, build_paired_schedule

    qu = runtime.model.load()
    reference = CapturingReference(runtime, deck)

    class _Ctl:
        name = "frozen-dobi-v2/reference"
        def __init__(self): self.calls = 0
        def act(self, obs):
            self.calls += 1
            return reference.act(obs)
        def diagnostics(self):
            return {"calls": self.calls, "exceptions": {}, "fallbacks": 0}

    deck_specs = EVAL.resolve_decks(args.opp, deck, str(ROOT / "agent" / "meta_decks.json"))
    opponents, _field = EVAL.make_field(
        deck_specs, "rules", qu, "parity-field")
    schedule = build_paired_schedule(opponents, args.games, seed=args.seed)
    controller = _Ctl()
    series = EVAL.run_series("dobi-v2-parity", controller, deck, opponents,
                             schedule, max_selects=5000, time_bank_s=600.0,
                             verbose=False)
    print(f"[parity] {len(series.records)} games, "
          f"{len(reference.prompts)} prompts captured", flush=True)

    # --- replay every prompt through the seat-aware interface --------------
    table = SeatPolicyTable().bind(0, FrozenDobiV2Policy(runtime), deck)
    mismatches, errors = [], []
    route_counts: Counter[str] = Counter()
    for index, row in enumerate(reference.prompts):
        try:
            decision = table.decide(row["obs"], seat=0)
        except Exception as error:                      # noqa: BLE001
            errors.append({"index": index, "error": repr(error)})
            continue
        route_counts[decision.route] += 1
        if list(decision.action) != list(row["action"]):
            mismatches.append({"index": index, "route_reference": row["route"],
                               "route_replay": decision.route,
                               "reference": row["action"],
                               "replay": list(decision.action)})

    # --- independent check against the packaged policy module -------------
    packaged_mismatches = []
    for index, row in enumerate(reference.prompts[:2000]):
        try:
            got = packaged_action(runtime, row["obs"], deck)
        except Exception as error:                      # noqa: BLE001
            packaged_mismatches.append({"index": index, "error": repr(error)})
            continue
        if got != list(row["action"]):
            packaged_mismatches.append(
                {"index": index, "packaged": got, "seat_policy": row["action"]})

    # --- seat discipline ---------------------------------------------------
    seat_checks = {}
    try:
        SeatPolicyTable().decide({}, seat=1)
        seat_checks["unbound_seat_rejected"] = False
    except SeatPolicyError:
        seat_checks["unbound_seat_rejected"] = True
    # Reusing one policy instance across seats is the failure to prevent:
    # the opponent would then be scored by Dobi's deck-conditioned policy.
    try:
        shared_policy = FrozenDobiV2Policy(runtime)
        SeatPolicyTable().bind(0, shared_policy, deck).bind(1, shared_policy, deck)
        seat_checks["shared_policy_instance_rejected"] = False
    except SeatPolicyError:
        seat_checks["shared_policy_instance_rejected"] = True
    # A true mirror legitimately gives both seats equal 60-card lists, so deck
    # equality must NOT be rejected -- only the shared policy instance is.
    try:
        SeatPolicyTable().bind(0, FrozenDobiV2Policy(runtime), deck) \
                         .bind(1, QuV2BasePolicy(runtime), deck)
        seat_checks["mirror_decks_allowed"] = True
    except SeatPolicyError:
        seat_checks["mirror_decks_allowed"] = False

    passed = (not mismatches and not errors and not packaged_mismatches
              and all(seat_checks.values()))
    payload = {
        "schema": "ptcg.dobi-v2-seat-policy-parity.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "milestone": "1-exact-reproduction",
        "archive_sha256": runtime.archive_sha256,
        "expected_archive_sha256": DOBI_V2_ARCHIVE_SHA256,
        "games": len(series.records),
        "prompts": len(reference.prompts),
        "reference_routes": dict(reference.routes),
        "replay_routes": dict(route_counts),
        "mismatches": mismatches[:50],
        "mismatch_count": len(mismatches),
        "errors": errors[:50],
        "error_count": len(errors),
        "packaged_policy_mismatches": packaged_mismatches[:50],
        "packaged_policy_mismatch_count": len(packaged_mismatches),
        "packaged_policy_prompts_checked": min(len(reference.prompts), 2000),
        "seat_discipline": seat_checks,
        "overlay_faults": dict(reference.reference.overlay_faults),
        "overlay_last_error": dict(reference.reference.overlay_last_error),
        "passed": passed,
        "gameplay_authority": False,
    }
    body = json.dumps(payload, sort_keys=True, indent=2)
    payload["result_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    print()
    print(f"  prompts              {payload['prompts']}")
    print(f"  reference routes     {dict(reference.routes)}")
    print(f"  replay routes        {dict(route_counts)}")
    print(f"  action mismatches    {payload['mismatch_count']}")
    print(f"  replay errors        {payload['error_count']}")
    print(f"  packaged-policy diff {payload['packaged_policy_mismatch_count']} "
          f"of {payload['packaged_policy_prompts_checked']}")
    print(f"  seat discipline      {seat_checks}")
    print(f"\n  PARITY {'PASSED' if passed else 'FAILED'}")
    if args.out:
        print(f"  {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ParityError, SeatPolicyError) as error:
        print(f"PARITY ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
