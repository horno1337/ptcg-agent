"""Run the locked 200-game random smoke from the exact PPO-300k tarball."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import smoke_dobi_v1 as smoke


RUN = ROOT / "tools/checkpoints/ppo-300k"

smoke.RUN = RUN
smoke.ARCHIVE = ROOT / "submission-ppo-300k-unsigned.tar.gz"
smoke.MANIFEST = RUN / "package-manifest.json"
smoke.LOCK = RUN / "random-smoke-lock.json"
smoke.RESULT = RUN / "random-smoke-result.json"
smoke.EXPECTED_CANDIDATE_HASH = (
    "425a2a360aebf380541be4075eac059c8f58b09be2e80433ff140cca7723dad3"
)
smoke.__file__ = __file__
smoke.SCRIPT = (
    smoke.SCRIPT
    .replace("dobi-v1", "ppo-300k")
    .replace(
        "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c",
        "425a2a360aebf380541be4075eac059c8f58b09be2e80433ff140cca7723dad3",
    )
)


if __name__ == "__main__":
    raise SystemExit(smoke.main())
