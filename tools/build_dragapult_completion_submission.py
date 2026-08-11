"""Build the deterministic elite-Dragapult Phantom-completion candidate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
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


BASE = ROOT / "submission-dragapult-elite-1-unsigned.tar.gz"
MODULE = ROOT / "agent/dragapult_bc.py"
LOCK = ROOT / (
    "tools/checkpoints/dragapult-phantom-completion-v1-20260811/"
    "gameplay/lock.json"
)
RESULT = ROOT / (
    "tools/checkpoints/dragapult-phantom-completion-v1-20260811/"
    "gameplay/result.json"
)
OUTPUT = ROOT / "submission-dragapult-completion-1-unsigned.tar.gz"
MANIFEST = ROOT / (
    "tools/checkpoints/dragapult-phantom-completion-v1-20260811/"
    "package/manifest.json"
)

BASE_SHA256 = "599b6d6e872c420f699f536088ccbf9fbbdb59d0b978a13cc73c1ab72531762e"
SOURCE_MAIN_SHA256 = "6e2183c318e0b41753aa629ffce61706332628740e22613f4120967e0ebb2e55"
SOURCE_CARD_SHA256 = "135696e3a5b080f1a3b7bee4bb882f12a0dfbefc1d43a7cd3913b29c571781df"
ELITE_MAIN_SHA256 = "793b230dbf9c67c3ece2b53b3f1a8b0f284765830ec397c6b746b35db2f966e0"
ELITE_CARD_SHA256 = "1a9b3867e81e791e35d68f0a147b9c338162ed707e5657633a06ff867cdb9d43"
SOURCE_MODULE_SHA256 = "14fef2a6472972bdf41182da595022807fde072fa5149983cbb1190328e2dae6"
SUBMISSION_NAME = "dragapult-completion-1"


class BuildError(RuntimeError):
    """A package input, evidence contract, or deterministic diff failed."""


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != COMMON.canonical(value):
        raise BuildError(f"evidence self-hash failed: {path}")
    value[key] = claimed
    return value


def packaged_module() -> bytes:
    if COMMON.sha256(MODULE) != SOURCE_MODULE_SHA256:
        raise BuildError("repository Dragapult policy drifted from the gameplay lock")
    text = MODULE.read_text(encoding="utf-8")
    replacements = (
        (SOURCE_MAIN_SHA256, ELITE_MAIN_SHA256),
        (SOURCE_CARD_SHA256, ELITE_CARD_SHA256),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise BuildError(f"runtime weight hash anchor drifted: {old}")
        text = text.replace(old, new)
    return text.encode("utf-8")


def build(output: Path, manifest: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    if output.exists() or manifest.exists():
        raise BuildError("refusing to overwrite package or manifest")
    for path in (BASE, MODULE, LOCK, RESULT):
        if not path.is_file():
            raise BuildError(f"required artifact missing: {path}")
    if COMMON.sha256(BASE) != BASE_SHA256:
        raise BuildError("frozen elite Dragapult parent archive drifted")

    lock = load_self(
        LOCK, "ptcg.dragapult-phantom-completion.gameplay-lock.v1", "lock_sha256"
    )
    result = load_self(
        RESULT,
        "ptcg.dragapult-phantom-completion.gameplay-result.v1",
        "result_sha256",
    )
    decision = result.get("decision", {})
    delta = decision.get("candidate_minus_control", {})
    if (
        result.get("lock_sha256") != lock["lock_sha256"]
        or lock.get("artifacts", {}).get("policy", {}).get("sha256")
        != SOURCE_MODULE_SHA256
        or decision.get("valid") is not True
        or decision.get("passed") is not True
        or result.get("promotion_authority") is not True
        or delta.get("ci95", [0])[0] <= 0
        or result.get("diagnostics", {}).get("candidate", {}).get("valid") is not True
        or result.get("diagnostics", {}).get("control", {}).get("valid") is not True
    ):
        raise BuildError("strict passing Phantom-completion evidence is absent")

    module_bytes = packaged_module()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    changed_expected = ["agent/dragapult_bc.py"]
    try:
        with tempfile.TemporaryDirectory(prefix="dragapult-completion-package-") as temp:
            parent = Path(temp) / "parent"
            candidate = Path(temp) / "candidate"
            parent.mkdir()
            with tarfile.open(BASE, "r:gz") as archive:
                members = archive.getmembers()
                if any(member.issym() or member.islnk() for member in members):
                    raise BuildError("parent archive contains links")
                archive.extractall(parent, members=members, filter="data")
            shutil.copytree(parent, candidate)
            before = COMMON.tree_files(parent)
            (candidate / "agent/dragapult_bc.py").write_bytes(module_bytes)
            after = COMMON.tree_files(candidate)
            changed = sorted(
                name
                for name in set(before) | set(after)
                if before.get(name) != after.get(name)
            )
            if changed != changed_expected:
                raise BuildError(f"unexpected package diff: {changed}")
            if after.get("agent/dragapult_main_weights.npz") != ELITE_MAIN_SHA256:
                raise BuildError("parent MAIN weights are not the frozen elite weights")
            if after.get("agent/dragapult_card_weights.npz") != ELITE_CARD_SHA256:
                raise BuildError("parent CARD weights are not the frozen elite weights")
            with partial.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
                ) as compressed:
                    with tarfile.open(
                        fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                    ) as archive:
                        COMMON.add_tree(archive, candidate)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)

    payload = {
        "schema": "ptcg.dragapult-completion-1.package.v1",
        "created_at": result["created_at"],
        "submission_name": SUBMISSION_NAME,
        "benchmark_only": True,
        "parent": {"archive": str(BASE.resolve()), "sha256": BASE_SHA256},
        "candidate": {
            "archive": str(output),
            "sha256": COMMON.sha256(output),
            "modified_files": changed_expected,
            "repository_module_sha256": SOURCE_MODULE_SHA256,
            "packaged_module_sha256": hashlib.sha256(module_bytes).hexdigest(),
            "main_weights_sha256": ELITE_MAIN_SHA256,
            "card_weights_sha256": ELITE_CARD_SHA256,
            "runtime": (
                "exact 07bed elite MAIN/CARD BC plus the completion-only MAIN guard; "
                "Boss, early-Dark, and Phantom-target experiments disabled"
            ),
        },
        "evidence": {
            "lock_sha256": lock["lock_sha256"],
            "result_sha256": result["result_sha256"],
            "games_per_arm": delta["paired_units"],
            "candidate_score": result["summaries"]["candidate"]["score"],
            "control_score": result["summaries"]["control"]["score"],
            "paired_delta": delta["mean_delta"],
            "paired_ci95": delta["ci95"],
            "completion_guard_fires": decision["completion_guards"],
        },
        "determinism": {
            "gzip_mtime": 0,
            "tar_mtime": 0,
            "uid": 0,
            "gid": 0,
            "sorted_members": True,
        },
        "authorization": {
            "upload_authorized": False,
            "basis": "candidate package prepared; Kaggle upload requires explicit approval",
            "suggested_kaggle_message": SUBMISSION_NAME,
        },
        "required_release_audits": [
            "deployment tests",
            "200-game exact-archive route smoke",
            "non-owner exact-archive runtime parity",
            "deterministic rebuild parity",
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
        payload = build(args.output, args.manifest)
    except (BuildError, OSError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
