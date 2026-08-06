"""Cross-UID exact-tarball audit for Dobi-v1.5."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research import audit_dobi_v1 as audit


audit.ARCHIVE = ROOT / "submission-dobi-v1.5-unsigned.tar.gz"
audit.OUTPUT = ROOT / "tools/checkpoints/dobi-v1.5/exact-tarball-audit.json"
audit.WEIGHTS_SHA256 = (
    "7ad4fafaa6cf48627163aa3527fa6507c179a20ceaeb4f3ad23fd5fb057f1a59"
)
audit.SCRIPT = audit.SCRIPT.replace("dobi-v1", "dobi-v1.5")


if __name__ == "__main__":
    raise SystemExit(audit.main())
