"""Wait for locked v1 diagnostics, then conditionally launch Lucario-v2."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import lock_lucario_majkel_bc_v2 as LOCKER  # noqa: E402
from tools.research import run_lucario_majkel_bc_v2 as RUNNER  # noqa: E402


RESULTS = tuple(
    LOCKER.V1 / f"head-isolation/{head}/result.json"
    for head in ("main", "card")
)
TIMEOUT_S = 3_600


def main() -> int:
    started = time.monotonic()
    while not all(path.is_file() for path in RESULTS):
        if time.monotonic() - started > TIMEOUT_S:
            raise SystemExit("timed out waiting for Lucario-v1 isolation results")
        time.sleep(10)
    decisions = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))["decision"]
        for path in RESULTS
    }
    print(json.dumps({"event": "isolation_complete", "decisions": decisions}, sort_keys=True), flush=True)
    return RUNNER.main()


if __name__ == "__main__":
    raise SystemExit(main())
