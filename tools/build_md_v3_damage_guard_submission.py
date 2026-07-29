"""Derive an MD-v3 damage-guard canary from the exact live MD-v3 archive.

The parent archive is never rebuilt from the worktree.  Its extracted bytes
are preserved except for adding the already locked guard module and changing
the existing fail-soft guard flag from explicit opt-in to default-on.
Uploading remains a separate, user-authorized action.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import grim_damage_guard as GUARD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402


BASE_ARCHIVE = ROOT / "submission-md-v2-card-v1-experimental-unsigned.tar.gz"
BASE_ARCHIVE_SHA256 = (
    "adb0e8c1da13a56468f6855bee197eab3d4b58f378e0a0a9b1ea6cc8b604ac42"
)
BASE_AUDIT = ROOT / "tools/checkpoints/md-v2-card-v1/package-audit.json"
RULE_LOCK = ROOT / "tools/checkpoints/grim-damage-guard-v1/rule-lock.json"
RESERVED_RESULT = (
    ROOT / "tools/checkpoints/grim-damage-guard-v1/reserved-result.json"
)
SIZING_RESULT = (
    ROOT
    / "tools/checkpoints/grim-damage-guard-v1/"
    "md-v3-firing-rate-result.json"
)
DEFAULT_MANIFEST = (
    ROOT
    / "tools/checkpoints/grim-damage-guard-v1/"
    "md-v3-ladder-canary-package.json"
)
MANIFEST_SCHEMA = "ptcg.md-v3.damage-guard-ladder-canary-package.v1"
OLD_ACTIVATION = 'os.environ.get("PTCG_GRIM_DAMAGE_GUARD") == "1"'
NEW_ACTIVATION = 'os.environ.get("PTCG_GRIM_DAMAGE_GUARD", "1") == "1"'


class BuildError(RuntimeError):
    """The live parent, locked guard, or exact package diff drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_self(path: Path, schema: str, hash_key: str) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(
            path.expanduser().resolve(), schema=schema, hash_key=hash_key
        )
    except COMMON.EvaluationError as error:
        raise BuildError(str(error)) from error


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"cannot load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BuildError(f"{path} is not an object")
    return payload


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as error:
                raise BuildError(
                    f"archive member escapes extraction root: {member.name}"
                ) from error
            if member.issym() or member.islnk():
                raise BuildError(f"parent archive contains a link: {member.name}")
        handle.extractall(destination, members=members, filter="data")


def _file_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _portable_member(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in member.name or member.name.endswith((".pyc", ".pyo")):
        return None
    if member.isdir():
        member.mode = 0o755
    elif member.isfile():
        member.mode = 0o644
    return member


def _authorize() -> dict[str, Any]:
    if sha256_file(BASE_ARCHIVE) != BASE_ARCHIVE_SHA256:
        raise BuildError("exact live MD-v3 parent archive drifted")
    audit = _load_json(BASE_AUDIT)
    rule = _load_self(
        RULE_LOCK, "ptcg.grim-damage-guard.rule-lock.v1", "lock_sha256"
    )
    reserved = _load_self(
        RESERVED_RESULT,
        "ptcg.grim-damage-guard.reserved-result.v1",
        "result_sha256",
    )
    sizing = _load_self(
        SIZING_RESULT,
        "ptcg.grim-damage-guard.md-v3-firing-result.v1",
        "result_sha256",
    )
    if (
        audit.get("gate_passed") is not True
        or audit.get("archive", {}).get("sha256") != BASE_ARCHIVE_SHA256
        or rule.get("artifacts", {}).get("guard", {}).get("sha256")
            != sha256_file(Path(GUARD.__file__))
        or reserved.get("rule_lock_sha256") != rule["lock_sha256"]
        or reserved.get("decision", {}).get("passed") is not True
        or sizing.get("promotion_authority") is not False
        or sizing.get("decision", {}).get("proceed_to_gameplay_ab") is not False
    ):
        raise BuildError("parent or guard evidence drifted")
    return {
        "base_audit_sha256": sha256_file(BASE_AUDIT),
        "rule_lock_sha256": rule["lock_sha256"],
        "reserved_result_sha256": reserved["result_sha256"],
        "sizing_result_sha256": sizing["result_sha256"],
    }


def build(
    output: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[Path, dict[str, Any]]:
    evidence = _authorize()
    output = output.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise BuildError("output must end in .tar.gz")
    if output.exists() or manifest_path.exists():
        raise BuildError("refusing to overwrite package output or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.partial")

    try:
        with tempfile.TemporaryDirectory(
            prefix="md-v3-damage-guard-package-"
        ) as temporary:
            work = Path(temporary)
            parent = work / "parent"
            candidate = work / "candidate"
            parent.mkdir()
            _safe_extract(BASE_ARCHIVE, parent)
            shutil.copytree(parent, candidate)
            before = _file_manifest(parent)

            policy_path = candidate / "agent/policy.py"
            source = policy_path.read_text(encoding="utf-8")
            if (
                source.count(OLD_ACTIVATION) != 1
                or NEW_ACTIVATION in source
                or 'os.environ.get("PTCG_MD_V2_CARD", "1") == "1"'
                    not in source
            ):
                raise BuildError("parent guard/card activation source drifted")
            policy_path.write_text(
                source.replace(OLD_ACTIVATION, NEW_ACTIVATION),
                encoding="utf-8",
            )
            guard_path = candidate / "agent/grim_damage_guard.py"
            if guard_path.exists():
                raise BuildError("parent unexpectedly already contains guard")
            shutil.copy2(Path(GUARD.__file__), guard_path)

            after = _file_manifest(candidate)
            added = sorted(after.keys() - before.keys())
            removed = sorted(before.keys() - after.keys())
            modified = sorted(
                name
                for name in before.keys() & after.keys()
                if before[name] != after[name]
            )
            if (
                added != ["agent/grim_damage_guard.py"]
                or removed
                or modified != ["agent/policy.py"]
                or after["agent/grim_damage_guard.py"]
                    != sha256_file(Path(GUARD.__file__))
            ):
                raise BuildError("candidate is not the exact two-file guard diff")

            with tarfile.open(temporary_output, "w:gz") as archive:
                for name in ("main.py", "agent", "data", "decks", "cg"):
                    path = candidate / name
                    if path.exists():
                        archive.add(
                            path, arcname=name, filter=_portable_member
                        )
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)

    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "archive": str(BASE_ARCHIVE.resolve()),
            "sha256": BASE_ARCHIVE_SHA256,
            "live_submission_ids": [55064789, 55064928],
        },
        "candidate": {
            "archive": str(output),
            "sha256": sha256_file(output),
            "added_files": ["agent/grim_damage_guard.py"],
            "modified_files": ["agent/policy.py"],
            "removed_files": [],
            "runtime_change": (
                "default-enable the locked fail-soft visible-KO damage "
                "destination guard ahead of otherwise frozen MD-v3"
            ),
        },
        "evidence": evidence,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "safety_contract": {
            "fail_soft": (
                "any import, scope, or decision exception falls through to "
                "the byte-frozen MD-v3 path"
            ),
            "upload_authorized": False,
            "required_before_upload": [
                "tests/test_safety.py",
                "200-game random smoke",
                "exact-extracted-tarball audit under non-owner UID",
                "user-supplied tag and explicit upload approval",
            ],
        },
    }
    payload["manifest_sha256"] = COMMON.canonical_sha256(payload)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    try:
        archive, payload = build(args.out, manifest_path=args.manifest)
    except (BuildError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "archive": str(archive),
        "archive_sha256": payload["candidate"]["sha256"],
        "manifest_sha256": payload["manifest_sha256"],
        "upload_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
