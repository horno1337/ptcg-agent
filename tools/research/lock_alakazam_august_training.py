"""Freeze bounded Alakazam head training after the novelty gate passes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.audit_alakazam_august_novelty import PARENT_SHA256  # noqa: E402


SCHEMA = "ptcg.alakazam-august-bc-training-lock.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-lock", required=True, type=Path)
    parser.add_argument("--novelty", required=True, type=Path)
    parser.add_argument("--parent", type=Path, default=ROOT / "agent/weights.npz")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    split = json.loads(args.split_lock.read_text())
    novelty = json.loads(args.novelty.read_text())
    heads = [
        head.lower() for head in ("MAIN", "CARD")
        if novelty.get("training_authority", {}).get(head, False)
    ]
    if not heads:
        raise SystemExit("novelty failed: no training lock may be written")
    if sha256_file(args.parent) != PARENT_SHA256:
        raise SystemExit("Qu-v2B parent drifted")
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_dataset_build_or_training": True,
        "authorized_heads": heads,
        "inputs": {
            "split_lock_path": str(args.split_lock.resolve()),
            "split_lock_sha256": split.get("lock_sha256"),
            "novelty_path": str(args.novelty.resolve()),
            "novelty_audit_sha256": novelty.get("audit_sha256"),
            "parent_path": str(args.parent.resolve()),
            "parent_sha256": sha256_file(args.parent),
        },
        "dataset": {
            "source": "new eligible August games only",
            "old_qu_v2b_corpus_retrained": False,
            "test_split_opened": False,
            "train_seat_cap": 8_000,
            "validation_seat_cap": 2_000,
            "selection": "deterministic uniform seat sample",
            "weights": "win 1.0, loss 0.6, draw 0.8; per-game decision normalization",
        },
        "training": {
            "initialization": "frozen Qu-v2B",
            "trainable_layers": ["option1", "context1", "policy"],
            "frozen": "embedding, board/state trunk, value head",
            "epochs": 4, "learning_rate": 2e-5,
            "parent_kl_coefficient": 0.7, "batch": 256,
            "seed": 2_026_081_601,
        },
        "deployment": {
            "scope": "exact 3f451509 Alakazam registration and authorized head only",
            "preflight_requires_each_authorized_route_to_fire": True,
            "non_authorized_heads_remain_Qu_v2B": True,
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "authorized_heads": heads, "lock_sha256": payload["lock_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
