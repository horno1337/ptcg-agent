"""Continue the locked BC repair after the extraction publishes its inventory."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "tools/checkpoints/dobi-v1-ppo300k-bc-repair/extraction-inventory.json"
STEPS = (
    ROOT / "tools/research/build_dobi_v1_ppo300k_bc_repair_corpus.py",
    ROOT / "tools/research/run_dobi_v1_ppo300k_bc_repair.py",
)


def main() -> int:
    while not INVENTORY.is_file():
        time.sleep(10)
    for script in STEPS:
        completed = subprocess.run(
            [sys.executable, str(script)], cwd=ROOT, check=False,
        )
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
