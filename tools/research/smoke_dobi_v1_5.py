"""Run the locked 200-game random smoke from the Dobi-v1.5 tarball."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import smoke_dobi_v1 as smoke


RUN = ROOT / "tools/checkpoints/dobi-v1.5"
smoke.RUN = RUN
smoke.ARCHIVE = ROOT / "submission-dobi-v1.5-unsigned.tar.gz"
smoke.MANIFEST = RUN / "package-manifest.json"
smoke.LOCK = RUN / "random-smoke-lock.json"
smoke.RESULT = RUN / "random-smoke-result.json"
smoke.EXPECTED_CANDIDATE_HASH = (
    "7ad4fafaa6cf48627163aa3527fa6507c179a20ceaeb4f3ad23fd5fb057f1a59"
)
smoke.__file__ = __file__
smoke.SCRIPT = (
    smoke.SCRIPT
    .replace("dobi-v1", "dobi-v1.5")
    .replace(
        "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c",
        "7ad4fafaa6cf48627163aa3527fa6507c179a20ceaeb4f3ad23fd5fb057f1a59",
    )
)


if __name__ == "__main__":
    raise SystemExit(smoke.main())
