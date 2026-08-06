"""Run the locked 200-game random smoke from the exact BC-PPO-v1 tarball."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import smoke_dobi_v1 as smoke


RUN = ROOT / "tools/checkpoints/dobi-v1-ppo300k-bc-repair/ladder-package"
smoke.RUN = RUN
smoke.ARCHIVE = ROOT / "submission-bc-ppo-v1-unsigned.tar.gz"
smoke.MANIFEST = RUN / "package-manifest.json"
smoke.LOCK = RUN / "random-smoke-lock.json"
smoke.RESULT = RUN / "random-smoke-result.json"
smoke.EXPECTED_CANDIDATE_HASH = (
    "007dedcd24b34e3145846f86fd2a55ee7d5d4c1999420102d21dd019e96d45df"
)
smoke.__file__ = __file__
smoke.SCRIPT = (
    smoke.SCRIPT
    .replace("dobi-v1", "bc-ppo-v1")
    .replace(
        "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c",
        smoke.EXPECTED_CANDIDATE_HASH,
    )
)


if __name__ == "__main__":
    raise SystemExit(smoke.main())
