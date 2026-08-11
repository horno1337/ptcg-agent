"""Build the deterministic exact-07bed Dragapult BC benchmark submission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_lucario_benchmark_1_submission as COMMON  # noqa: E402


BASE = ROOT / "submission-dobi-v1-unsigned.tar.gz"
MODULE = ROOT / "agent/dragapult_bc.py"
DECK = ROOT / "decks/dragapult_07bed.csv"
MAIN = ROOT / "agent/dragapult_main_weights.npz"
CARD = ROOT / "agent/dragapult_card_weights.npz"
FIELD_LOCK = ROOT / "tools/checkpoints/dragapult-bc-20260810/gameplay/lock.json"
FIELD_RESULT = ROOT / "tools/checkpoints/dragapult-bc-20260810/gameplay/result.json"
OUTPUT = ROOT / "submission-dragapult-benchmark-1-unsigned.tar.gz"
MANIFEST = ROOT / "tools/checkpoints/dragapult-benchmark-1/package-manifest.json"

BASE_SHA256 = "fdd50192ab1bf4fdb2097e4f2bd3c015a841ec1099aa6fbe806ea760db5c9c3f"
MAIN_SHA256 = "6e2183c318e0b41753aa629ffce61706332628740e22613f4120967e0ebb2e55"
CARD_SHA256 = "135696e3a5b080f1a3b7bee4bb882f12a0dfbefc1d43a7cd3913b29c571781df"
DECK_SHA256 = "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725"
SUBMISSION_NAME = "dragapult-benchmark-1"


class BuildError(RuntimeError):
    """A package input, field record, or deterministic diff failed."""


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != COMMON.canonical(value):
        raise BuildError(f"evidence self-hash failed: {path}")
    value[key] = claimed
    return value


def build(output: Path, manifest: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    if output.exists() or manifest.exists():
        raise BuildError("refusing to overwrite package or manifest")
    required = (BASE, MODULE, DECK, MAIN, CARD, FIELD_LOCK, FIELD_RESULT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise BuildError(f"required artifacts missing: {missing}")
    if (
        COMMON.sha256(BASE) != BASE_SHA256
        or COMMON.sha256(MAIN) != MAIN_SHA256
        or COMMON.sha256(CARD) != CARD_SHA256
        or COMMON.deck_sha256(DECK) != DECK_SHA256
    ):
        raise BuildError("frozen archive, weights, or deck identity drifted")
    field_lock = load_self(
        FIELD_LOCK, "ptcg.dragapult-bc.gameplay-lock.v1", "lock_sha256"
    )
    field = load_self(
        FIELD_RESULT, "ptcg.dragapult-bc.gameplay-result.v1", "result_sha256"
    )
    decision = field.get("decision", {})
    if (
        field.get("lock_sha256") != field_lock["lock_sha256"]
        or field_lock.get("learner_deck_sha256") != DECK_SHA256
        or decision.get("valid") is not True
        or decision.get("supported_superiority") is not True
        or decision.get("earns_runtime_integration") is not True
        or decision.get("candidate_minus_control", {}).get("ci95", [0])[0] <= 0
    ):
        raise BuildError("passing Dragapult field evidence is absent")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    expected = [
        "agent/dragapult_bc.py", "agent/dragapult_card_weights.npz",
        "agent/dragapult_main_weights.npz", "agent/policy.py", "decks/deck.csv",
    ]
    try:
        with tempfile.TemporaryDirectory(prefix="dragapult-package-") as temporary:
            parent = Path(temporary) / "parent"
            candidate = Path(temporary) / "candidate"
            parent.mkdir()
            with tarfile.open(BASE, "r:gz") as archive:
                members = archive.getmembers()
                if any(member.issym() or member.islnk() for member in members):
                    raise BuildError("base archive contains links")
                archive.extractall(parent, members=members, filter="data")
            shutil.copytree(parent, candidate)
            before = COMMON.tree_files(parent)
            policy_path = candidate / "agent/policy.py"
            policy = policy_path.read_text(encoding="utf-8")
            anchor = "    # Deterministic KO: override the net ONLY on a provable Powerful Hand lethal.\n"
            if policy.count(anchor) != 1 or "dragapult_bc" in policy:
                raise BuildError("base policy injection anchor drifted")
            injection = (
                "    # Exact-registration Dragapult MAIN/CARD BC specialist.\n"
                "    try:\n"
                "        from . import dragapult_bc as _dragapult_bc\n"
                "        dragapult_action = _dragapult_bc.decide(view, load_deck())\n"
                "        if dragapult_action is not None:\n"
                "            return dragapult_action\n"
                "    except Exception:\n"
                "        pass\n\n"
            )
            policy_path.write_text(
                policy.replace(anchor, injection + anchor), encoding="utf-8"
            )
            shutil.copy2(DECK, candidate / "decks/deck.csv")
            shutil.copy2(MODULE, candidate / "agent/dragapult_bc.py")
            shutil.copy2(MAIN, candidate / "agent/dragapult_main_weights.npz")
            shutil.copy2(CARD, candidate / "agent/dragapult_card_weights.npz")
            after = COMMON.tree_files(candidate)
            changed = sorted(
                name for name in set(before) | set(after)
                if before.get(name) != after.get(name)
            )
            if changed != expected:
                raise BuildError(f"unexpected package diff: {changed}")
            with partial.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0,
                ) as compressed:
                    with tarfile.open(
                        fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT,
                    ) as archive:
                        COMMON.add_tree(archive, candidate)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)

    payload = {
        "schema": "ptcg.dragapult-benchmark-1.package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_name": SUBMISSION_NAME,
        "benchmark_only": True,
        "parent": {"archive": str(BASE.resolve()), "sha256": BASE_SHA256},
        "candidate": {
            "archive": str(output), "sha256": COMMON.sha256(output),
            "deck_sha256": DECK_SHA256,
            "main_weights_sha256": MAIN_SHA256,
            "card_weights_sha256": CARD_SHA256,
            "runtime": "exact 07bed Dragapult Day-1 MAIN/CARD BC plus Qu-v2B residual",
            "modified_files": expected,
        },
        "evidence": {
            "field_lock_sha256": field_lock["lock_sha256"],
            "field_result_sha256": field["result_sha256"],
            "games_per_arm": field["summaries"]["candidate"]["scheduled_games"],
            "candidate_score": field["summaries"]["candidate"]["score"],
            "control_score": field["summaries"]["control"]["score"],
            "paired_delta": decision["candidate_minus_control"]["mean_delta"],
            "paired_ci95": decision["candidate_minus_control"]["ci95"],
        },
        "determinism": {
            "gzip_mtime": 0, "tar_mtime": 0, "uid": 0, "gid": 0,
            "sorted_members": True,
        },
        "authorization": {
            "upload_authorized": True,
            "basis": "user requested ongoing Kaggle benchmark submissions",
            "kaggle_message": SUBMISSION_NAME,
        },
        "required_release_audits": [
            "deployment tests", "200-game exact-archive route smoke",
            "non-owner exact-archive runtime parity",
        ],
    }
    payload["manifest_sha256"] = COMMON.canonical(payload)
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        value = build(args.output, args.manifest)
    except (BuildError, COMMON.BuildError, OSError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
