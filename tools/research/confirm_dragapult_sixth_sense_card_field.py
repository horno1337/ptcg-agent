"""Fresh strict confirmation of the loss060 CARD field-screen winner."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import eval_dragapult_sixth_sense_card_field as E  # noqa: E402


E.OUT = E.RUN / "exact-field-confirmation"
E.LOCK = E.OUT / "lock.json"
E.RESULT = E.OUT / "result.json"
E.GAMES = 512
E.SEED = 2_026_081_235


def main() -> int:
    E.OUT.mkdir(parents=True, exist_ok=True)
    if not E.LOCK.exists():
        lock = E.build_lock()
        lock.pop("lock_sha256", None)
        lock["protocol"]["eligibility"] = (
            "valid, overall paired CI95 lower > 0, and Dragapult and Mega "
            "Lucario slice point deltas >= 0"
        )
        lock["lock_sha256"] = E.BASE.canonical_sha256(lock)
        E.LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    result = E.run()
    result["strict_confirmed"] = bool(
        result["valid"] and result["comparison"]["ci95"][0] > 0
        and result["slices"]["Dragapult"]["mean_delta"] >= 0
        and result["slices"]["Mega Lucario"]["mean_delta"] >= 0
    )
    E.RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "strict_confirmed": result["strict_confirmed"],
        "comparison": result["comparison"], "slices": result["slices"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
