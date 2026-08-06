"""Run exact-mirror league generation two with runtime-compatible snapshots."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model
from tools.research import lock_md_v3_mirror_league_100k_v2 as LOCK
from tools.research import run_md_v3_mirror_league_100k as V1


def main(argv: Sequence[str] | None = None) -> None:
    # Exported Qu-v2A arrays use the deployed schema.  Snapshot opponents must
    # therefore use the deployed runtime reader, not the training feature
    # reader used for Torch/NumPy optimization parity.
    V1.QM.NumpyQuV2A = model.QuV2Net
    V1.POP.QM.NumpyQuV2A = model.QuV2Net
    V1.LOCK = LOCK
    V1.main(argv)


if __name__ == "__main__":
    main()
