"""JSON-lines worker for the audited Kiyota Mega-Lucario public policy.

Unlike the generic one-game worker, this experiment-only adapter can snapshot
and restore the policy's three documented mutable globals.  It is executed in
the same restricted bubblewrap filesystem boundary as the public-agent gate.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import resource
import sys


PLAN_FIELDS = ("attacker", "target", "attack_index", "remain_hp", "energy")


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))


def _emit(protocol, value: object) -> None:
    protocol.write(json.dumps(value, separators=(",", ":"), allow_nan=False))
    protocol.write("\n")
    protocol.flush()


def _snapshot(module) -> dict[str, object]:
    plan = module.plan
    return {
        "pre_turn": int(module.pre_turn),
        "ability_used": bool(module.ability_used),
        "plan": {name: getattr(plan, name) for name in PLAN_FIELDS},
    }


def _restore(module, value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "pre_turn", "ability_used", "plan",
    }:
        raise ValueError("invalid Lucario policy-state envelope")
    plan_value = value["plan"]
    if not isinstance(plan_value, dict) or set(plan_value) != set(PLAN_FIELDS):
        raise ValueError("invalid Lucario AttackPlan state")
    plan = module.AttackPlan()
    for name in PLAN_FIELDS:
        item = plan_value[name]
        if name == "energy":
            if not isinstance(item, bool):
                raise ValueError("invalid AttackPlan.energy")
        elif not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"invalid AttackPlan.{name}")
        setattr(plan, name, item)
    if (not isinstance(value["pre_turn"], int)
            or isinstance(value["pre_turn"], bool)
            or not isinstance(value["ability_used"], bool)):
        raise ValueError("invalid Lucario scalar policy state")
    module.plan = plan
    module.pre_turn = value["pre_turn"]
    module.ability_used = value["ability_used"]


def main() -> int:
    _limits()
    os.chdir("/agent")
    sys.path.insert(0, "/agent")
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        module = importlib.import_module("main")
        policy = getattr(module, "agent")
        with open("/agent/deck.csv", encoding="utf-8") as handle:
            deck = [int(row) for row in handle.read().split()]
    _emit(protocol, {"ready": True, "deck": deck, "state": _snapshot(module)})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request == {"close": True}:
                return 0
            if request == {"snapshot": True}:
                _emit(protocol, {"state": _snapshot(module)})
                continue
            if "restore" in request:
                _restore(module, request["restore"])
                _emit(protocol, {"restored": True, "state": _snapshot(module)})
                continue
            with contextlib.redirect_stdout(sys.stderr):
                action = policy(request["observation"])
            _emit(protocol, {"action": action})
        except Exception as error:
            _emit(protocol, {
                "error": type(error).__name__, "detail": str(error)[:1000],
            })
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
