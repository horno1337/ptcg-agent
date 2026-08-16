"""Run one audited public competition agent behind a JSON-lines boundary.

This helper is launched inside a bubblewrap filesystem sandbox by
``eval_dobi_v2_public_lucario_dragapult.py``.  One process serves exactly one
game, which prevents module-level state leaking between games.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import resource
import sys


def _limit_process() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))


def _emit(protocol, value: object) -> None:
    protocol.write(json.dumps(value, separators=(",", ":"), allow_nan=False))
    protocol.write("\n")
    protocol.flush()


def main() -> int:
    _limit_process()
    os.chdir("/agent")
    sys.path.insert(0, "/agent")
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        module = importlib.import_module("main")
        policy = getattr(module, "agent")
        with open("/agent/deck.csv", encoding="utf-8") as handle:
            deck = [int(row) for row in handle.read().split()]
    _emit(protocol, {"ready": True, "deck": deck})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request == {"close": True}:
                return 0
            with contextlib.redirect_stdout(sys.stderr):
                action = policy(request["observation"])
            _emit(protocol, {"action": action})
        except Exception as error:  # Parent treats this as an invalid cell.
            _emit(protocol, {
                "error": type(error).__name__,
                "detail": str(error)[:1000],
            })
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
