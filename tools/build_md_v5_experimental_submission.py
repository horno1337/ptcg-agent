"""Build the user-authorized PPO-v2 MD-v5 canary from frozen MD-v3.

The locked PPO gameplay gate had a positive point estimate (50.82%) but its
95% lower confidence bound did not exceed 50%.  This package therefore grants
no promotion authority: it is an explicit ladder experiment.  Relative to the
exact MD-v3 archive, only the ST_MAIN weights and their pinned checksum change.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission-md-v2-card-v1-experimental-unsigned.tar.gz"
BASE_SHA256 = "adb0e8c1da13a56468f6855bee197eab3d4b58f378e0a0a9b1ea6cc8b604ac42"
WEIGHTS = ROOT / "tools/checkpoints/md-v3-ppo-v2/training/terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
WEIGHTS_SHA256 = "1f21df7a6b26df01e15720c2b2f3b70d6e463d8498db9e45346d8505138f4110"
GAMEPLAY = ROOT / "tools/checkpoints/md-v3-ppo-v2/direct-gameplay-result.json"
OUTPUT = ROOT / "submission-md-v5-experimental-unsigned.tar.gz"
MANIFEST = ROOT / "tools/checkpoints/md-v5-experimental/package-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if sha256(BASE) != BASE_SHA256 or sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise SystemExit("frozen MD-v3 base or fixed PPO-v2 weights drifted")
    result = json.loads(GAMEPLAY.read_text(encoding="utf-8"))
    decision = result.get("decision", {})
    if decision.get("passed") is not False or decision.get("score") != 0.508203125:
        raise SystemExit("locked PPO-v2 failed-gate evidence drifted")
    with tempfile.TemporaryDirectory(prefix="md-v5-build-") as temporary:
        stage = Path(temporary)
        with tarfile.open(BASE, "r:gz") as archive:
            archive.extractall(stage, filter="data")
        module = stage / "agent/md_v1.py"
        source = module.read_text(encoding="utf-8")
        old = "76420fc2e031127143f5816e64973ec49e150da22b84604308b649c9a4dc8df8"
        if source.count(old) != 1:
            raise SystemExit("MD-v3 ST_MAIN checksum anchor drifted")
        module.write_text(source.replace(old, WEIGHTS_SHA256), encoding="utf-8")
        (stage / "agent/md_v1_weights.npz").write_bytes(WEIGHTS.read_bytes())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w:gz") as archive:
            for path in sorted(stage.rglob("*")):
                archive.add(path, arcname=str(path.relative_to(stage)), recursive=False)
    payload = {
        "schema": "ptcg.md-v5.experimental-ppo-v2-package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": "md-v5",
        "base_archive_sha256": BASE_SHA256,
        "candidate_archive_sha256": sha256(args.output),
        "st_main_weights_sha256": WEIGHTS_SHA256,
        "change": "replace frozen MD-v3 ST_MAIN weights with fixed PPO-v2 update 16",
        "gameplay": {
            "games": 2560,
            "wins": 1296,
            "losses": 1254,
            "draws": 10,
            "score": decision["score"],
            "wilson_ci95": decision["wilson_ci95"],
            "preregistered_gate_passed": False,
        },
        "authorization": {
            "experimental_user_override": True,
            "promotion_authority": False,
            "user_approved_name": "md-v5",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
