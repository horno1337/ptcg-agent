"""Build the user-approved Dobi-v1.5 two-slot ladder candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission-dobi-v1-unsigned.tar.gz"
BASE_SHA256 = "fdd50192ab1bf4fdb2097e4f2bd3c015a841ec1099aa6fbe806ea760db5c9c3f"
WEIGHTS = ROOT / (
    "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/training/"
    "terminal-update-130-candidate/candidate-qu-v2a-weights.npz"
)
WEIGHTS_SHA256 = "7ad4fafaa6cf48627163aa3527fa6507c179a20ceaeb4f3ad23fd5fb057f1a59"
OLD_WEIGHTS_SHA256 = "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c"
FINALIZATION = ROOT / (
    "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/"
    "finalization-manifest.json"
)
MIRROR_RESULT = ROOT / (
    "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/"
    "mirror-gate/result.json"
)
FIELD_RESULT = ROOT / (
    "tools/checkpoints/dobi-v1-munkidori-control-ppo-v1/"
    "field-gate/result.json"
)
OUTPUT = ROOT / "submission-dobi-v1.5-unsigned.tar.gz"
MANIFEST = ROOT / "tools/checkpoints/dobi-v1.5/package-manifest.json"


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def portable(member: tarfile.TarInfo):
    if "__pycache__" in member.name or member.name.endswith((".pyc", ".pyo")):
        return None
    member.mode = 0o755 if member.isdir() else 0o644
    return member


def build(output: Path, manifest: Path) -> dict:
    output = output.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    if output.exists() or manifest.exists():
        raise BuildError("refusing to overwrite Dobi-v1.5 release artifacts")
    for path in (BASE, WEIGHTS, FINALIZATION, MIRROR_RESULT, FIELD_RESULT):
        if not path.is_file():
            raise BuildError(f"bound artifact missing: {path}")
    if sha256(BASE) != BASE_SHA256 or sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise BuildError("frozen Dobi-v1 archive or candidate weights drifted")

    finalization = json.loads(FINALIZATION.read_text(encoding="utf-8"))
    mirror = json.loads(MIRROR_RESULT.read_text(encoding="utf-8"))
    field = json.loads(FIELD_RESULT.read_text(encoding="utf-8"))
    if (
        finalization.get("selection", {}).get("eligible") is not True
        or finalization.get("selection", {}).get("selected_update") != 130
        or mirror.get("decision", {}).get("valid") is not True
        or mirror.get("decision", {}).get("positive_evidence") is not True
        or field.get("decision", {}).get("valid") is not True
        or field.get("decision", {}).get("passed_noninferiority") is not False
    ):
        raise BuildError("candidate evidence does not match the user override")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    try:
        with tempfile.TemporaryDirectory(prefix="dobi-v1.5-package-") as temporary:
            parent = Path(temporary) / "parent"
            candidate = Path(temporary) / "candidate"
            parent.mkdir()
            with tarfile.open(BASE, "r:gz") as archive:
                members = archive.getmembers()
                if any(member.issym() or member.islnk() for member in members):
                    raise BuildError("base archive contains links")
                archive.extractall(parent, members=members, filter="data")
            shutil.copytree(parent, candidate)
            before = files(parent)
            module = candidate / "agent/md_v1.py"
            source = module.read_text(encoding="utf-8")
            if source.count(OLD_WEIGHTS_SHA256) != 1:
                raise BuildError("frozen Dobi-v1 checksum anchor drifted")
            module.write_text(
                source.replace(OLD_WEIGHTS_SHA256, WEIGHTS_SHA256),
                encoding="utf-8",
            )
            shutil.copy2(WEIGHTS, candidate / "agent/md_v1_weights.npz")
            after = files(candidate)
            changed = sorted(
                name for name in before if before[name] != after.get(name)
            )
            if changed != ["agent/md_v1.py", "agent/md_v1_weights.npz"]:
                raise BuildError(f"unexpected package diff: {changed}")
            with tarfile.open(partial, "w:gz") as archive:
                for name in ("main.py", "agent", "data", "decks", "cg"):
                    path = candidate / name
                    if path.exists():
                        archive.add(path, arcname=name, filter=portable)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)

    payload = {
        "schema": "ptcg.dobi-v1.5.package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "names": ["dobi-v1.5", "dobi-v1.5"],
        "parent": {"archive": str(BASE.resolve()), "sha256": BASE_SHA256},
        "candidate": {
            "archive": str(output),
            "sha256": sha256(output),
            "st_main_weights_sha256": WEIGHTS_SHA256,
            "modified_files": ["agent/md_v1.py", "agent/md_v1_weights.npz"],
            "runtime_change": (
                "replace only frozen Dobi-v1 ST_MAIN weights with fixed "
                "Munkidori-control PPO v1 update 130"
            ),
        },
        "evidence": {
            "mirror": mirror["decision"],
            "field": field["decision"],
            "field_gate_passed": False,
            "user_directed_ladder_override": True,
        },
        "authorization": {
            "user_approved_name": "dobi-v1.5",
            "two_identical_uploads_authorized": True,
        },
        "required_release_audits": [
            "tests/test_safety.py",
            "200-game exact-archive random smoke",
            "non-owner exact-tarball runtime audit",
        ],
    }
    payload["manifest_sha256"] = canonical(payload)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
