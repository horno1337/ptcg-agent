"""Build an explicitly experimental MD-v4 override from the exact live MD-v3.

This builder does not reinterpret MD-v4's failed preregistered behavior-size
screen and grants no promotion or upload authority.  It exists because the
user explicitly requested a ladder experiment with the unchanged epoch-4
NumPy mapping.

The exact live MD-v3 archive is extracted into a temporary stage.  Every
existing member is preserved byte-for-byte except ``agent/policy.py``.  Four
new runtime members are added:

* a Torch-free vendored public feature encoder;
* a Torch-free strict NumPy MD-v4 model;
* a fail-soft exact-deck ST_MAIN overlay; and
* the exact staged epoch-4 NPZ.

The policy edit default-enables the overlay ahead of the existing MD-v3
``md_v1`` ST_MAIN route.  Any import, scope, load, feature, inference, or
decode failure returns to that unchanged frozen route.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BASE_ARCHIVE = ROOT / "submission-md-v2-card-v1-experimental-unsigned.tar.gz"
BASE_ARCHIVE_SHA256 = (
    "adb0e8c1da13a56468f6855bee197eab3d4b58f378e0a0a9b1ea6cc8b604ac42"
)
DEFAULT_WEIGHTS = Path("/tmp/mdv4-authoritative.npz")
WEIGHTS_FILE_SHA256 = (
    "5ecfbece28822545102f046450d8b8008837d135875542ddf991ad4021a45d1f"
)
WEIGHTS_MAPPING_SHA256 = (
    "e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01"
)
RECOVERY = (
    ROOT
    / "tools/checkpoints/md-v4-public-window-v1/model/"
    "candidate-md-v4-recovery.pt"
)
RECOVERY_SHA256 = (
    "ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad"
)
RECOVERY_STATE_SHA256 = (
    "6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908"
)
FAILED_RESULT = (
    ROOT
    / "tools/checkpoints/md-v4-public-window-v1/model/"
    "deployed-parent-correction-evaluation-result.json"
)
FAILED_RESULT_FILE_SHA256 = (
    "62f14e86428d283abb7dad6df1f258097d05962cef310e6df4f7d483d2955436"
)
FAILED_RESULT_SELF_SHA256 = (
    "4c3ddc4abfe5fc087f2d563b7660272465be07db870bde20c966edbe48c25fb9"
)

FEATURE_SOURCE = ROOT / "tools/research/md_v4_features.py"
MODEL_SOURCE = ROOT / "tools/research/md_v4_vendor_model.py"
OVERLAY_SOURCE = ROOT / "tools/research/md_v4_vendor_overlay.py"
DEFAULT_MANIFEST = (
    ROOT
    / "tools/checkpoints/md-v4-experimental-override-v1/"
    "package-manifest.json"
)
MANIFEST_SCHEMA = "ptcg.md-v4.experimental-override-package.v1"

_POLICY_ANCHOR = """            sample = _qu_v2_features.encode_public_observation(
                view.obs, registration)
            # Candidate-only exact-deck ST_CARD overlay."""
_POLICY_INSERT = """            sample = _qu_v2_features.encode_public_observation(
                view.obs, registration)
            # User-authorized experimental MD-v4 override.  It is default-on
            # only for the exact registered deck and ST_MAIN; every failure
            # returns None and leaves the frozen MD-v3 route below unchanged.
            if (
                view.select_type == ST_MAIN
                and os.environ.get("PTCG_MD_V4", "1") == "1"
            ):
                try:
                    from . import md_v4 as _md_v4
                    md_v4_action = _md_v4.decide(view, registration)
                    if md_v4_action is not None:
                        return md_v4_action
                except Exception:
                    pass
            # Candidate-only exact-deck ST_CARD overlay."""


class BuildError(RuntimeError):
    """The exact parent, candidate mapping, or one-change package drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise BuildError(f"{path} is not a JSON object")
    return value


def _array_mapping_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ptcg.md-v4.array-mapping.v1\0")
    for name in sorted(values):
        value = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_authority_inputs(weights: Path) -> dict[str, Any]:
    if sha256_file(BASE_ARCHIVE) != BASE_ARCHIVE_SHA256:
        raise BuildError("exact live MD-v3 base archive drifted")
    if sha256_file(RECOVERY) != RECOVERY_SHA256:
        raise BuildError("fixed MD-v4 recovery checkpoint drifted")
    if sha256_file(FAILED_RESULT) != FAILED_RESULT_FILE_SHA256:
        raise BuildError("MD-v4 deployed-parent result file drifted")
    failed = _load_json(FAILED_RESULT)
    if (
        failed.get("schema")
        != "ptcg.md-v4.deployed-parent-correction-result.v1"
        or failed.get("result_sha256") != FAILED_RESULT_SELF_SHA256
        or failed.get("state_dict_sha256") != RECOVERY_STATE_SHA256
        or failed.get("numpy_array_mapping_sha256")
            != WEIGHTS_MAPPING_SHA256
        or failed.get("passed") is not False
        or failed.get("promotion_authority") is not False
        or failed.get("upload_authority") is not False
        or failed.get("final_validation", {}).get(
            "greedy_disagreement_rate"
        ) != 0.023772837332159368
        or failed.get("offline_rejection_gates", {}).get(
            "behavior_size_screen", {}
        ).get("passed") is not False
    ):
        raise BuildError("failed MD-v4 gate evidence drifted")
    if not weights.is_file() or sha256_file(weights) != WEIGHTS_FILE_SHA256:
        raise BuildError("exact staged MD-v4 NumPy file is missing or drifted")
    try:
        with np.load(weights, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except (OSError, ValueError, TypeError) as error:
        raise BuildError(f"cannot load fixed MD-v4 arrays: {error}") from error
    if _array_mapping_sha256(arrays) != WEIGHTS_MAPPING_SHA256:
        raise BuildError("fixed MD-v4 NumPy mapping drifted")
    return {
        "failed_gate_result_file_sha256": FAILED_RESULT_FILE_SHA256,
        "failed_gate_result_sha256": FAILED_RESULT_SELF_SHA256,
        "failed_behavior_disagreement_rate": 0.023772837332159368,
        "failed_behavior_minimum": 0.03,
        "recovery_file_sha256": RECOVERY_SHA256,
        "recovery_state_dict_sha256": RECOVERY_STATE_SHA256,
        "weights_file_sha256": WEIGHTS_FILE_SHA256,
        "weights_mapping_sha256": WEIGHTS_MAPPING_SHA256,
    }


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        names: set[str] = set()
        for member in members:
            if member.name in names:
                raise BuildError(f"duplicate base member: {member.name}")
            names.add(member.name)
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as error:
                raise BuildError(
                    f"base member escapes extraction root: {member.name}"
                ) from error
            if member.issym() or member.islnk():
                raise BuildError(f"base archive contains a link: {member.name}")
        handle.extractall(destination, members=members, filter="data")


def _file_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _portable_member(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in member.name or member.name.endswith((".pyc", ".pyo")):
        return None
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    if member.isdir():
        member.mode = 0o755
    elif member.isfile():
        member.mode = 0o644
    return member


def _vendor_feature_source() -> str:
    source = FEATURE_SOURCE.read_text(encoding="utf-8")
    replacements = {
        "from agent.obsview import (": "from .obsview import (",
        "from tools.research import qu_v2a_features as QF":
            "from . import qu_v2_features as QF",
        "Path(__file__).resolve().parents[2]":
            "Path(__file__).resolve().parents[1]",
        '"tools/research/md_v4_features.py"':
            '"agent/md_v4_features.py"',
        '"tools/research/qu_v2a_features.py"':
            '"agent/qu_v2_features.py"',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise BuildError(f"feature vendor transform is ambiguous: {old}")
        source = source.replace(old, new)
    if "import torch" in source or "from torch" in source:
        raise BuildError("Torch leaked into vendored MD-v4 features")
    return source


def _vendor_model_source() -> str:
    source = MODEL_SOURCE.read_text(encoding="utf-8")
    replacements = {
        "from agent import model": "from . import model",
        "from tools.research import md_v4_features as features":
            "from . import md_v4_features as features",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise BuildError(f"model vendor transform is ambiguous: {old}")
        source = source.replace(old, new)
    if "import torch" in source or "from torch" in source:
        raise BuildError("Torch leaked into vendored MD-v4 model")
    return source


def _vendor_overlay_source() -> str:
    source = OVERLAY_SOURCE.read_text(encoding="utf-8")
    replacements = {
        "from agent.obsview import ObsView, ST_MAIN":
            "from .obsview import ObsView, ST_MAIN",
        "from agent import model": "from . import model",
        "from tools.research import md_v4_features as features":
            "from . import md_v4_features as features",
        "from tools.research import md_v4_vendor_model as md_v4_model":
            "from . import md_v4_model",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise BuildError(f"overlay vendor transform is ambiguous: {old}")
        source = source.replace(old, new)
    if "import torch" in source or "from torch" in source:
        raise BuildError("Torch leaked into vendored MD-v4 overlay")
    return source


def _patch_policy(source: str) -> str:
    if (
        source.count(_POLICY_ANCHOR) != 1
        or "PTCG_MD_V4" in source
        or source.count("from . import md_v1 as _md_v1") != 1
    ):
        raise BuildError("live MD-v3 policy insertion point drifted")
    patched = source.replace(_POLICY_ANCHOR, _POLICY_INSERT)
    if (
        patched.count('os.environ.get("PTCG_MD_V4", "1") == "1"') != 1
        or patched.index("from . import md_v4 as _md_v4")
            > patched.index("from . import md_v1 as _md_v1")
    ):
        raise BuildError("MD-v4 policy ordering patch failed")
    return patched


def build(
    output: Path,
    *,
    weights_path: Path = DEFAULT_WEIGHTS,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[Path, dict[str, Any]]:
    output = output.expanduser().resolve()
    weights_path = weights_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise BuildError("output must end in .tar.gz")
    if output.exists() or manifest_path.exists():
        raise BuildError("refusing to overwrite package output or manifest")
    evidence = _validate_authority_inputs(weights_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")

    before: dict[str, str]
    after: dict[str, str]
    try:
        with tempfile.TemporaryDirectory(
            prefix="md-v4-experimental-package-"
        ) as temporary:
            root = Path(temporary)
            parent = root / "parent"
            candidate = root / "candidate"
            parent.mkdir()
            _safe_extract(BASE_ARCHIVE, parent)
            shutil.copytree(parent, candidate)
            before = _file_manifest(parent)

            policy_path = candidate / "agent/policy.py"
            policy_path.write_text(
                _patch_policy(policy_path.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            additions = {
                "agent/md_v4_features.py": _vendor_feature_source(),
                "agent/md_v4_model.py": _vendor_model_source(),
                "agent/md_v4.py": _vendor_overlay_source(),
            }
            for relative, source in additions.items():
                path = candidate / relative
                if path.exists():
                    raise BuildError(f"base unexpectedly contains {relative}")
                path.write_text(source, encoding="utf-8")
            target_weights = candidate / "agent/md_v4_weights.npz"
            if target_weights.exists():
                raise BuildError("base unexpectedly contains MD-v4 weights")
            shutil.copy2(weights_path, target_weights)

            after = _file_manifest(candidate)
            expected_added = sorted([
                "agent/md_v4.py",
                "agent/md_v4_features.py",
                "agent/md_v4_model.py",
                "agent/md_v4_weights.npz",
            ])
            added = sorted(after.keys() - before.keys())
            removed = sorted(before.keys() - after.keys())
            modified = sorted(
                name
                for name in before.keys() & after.keys()
                if before[name] != after[name]
            )
            if (
                added != expected_added
                or removed
                or modified != ["agent/policy.py"]
                or after["agent/md_v4_weights.npz"] != WEIGHTS_FILE_SHA256
                or any(
                    after[name] != before[name]
                    for name in before
                    if name != "agent/policy.py"
                )
            ):
                raise BuildError("candidate is not the exact MD-v4 overlay diff")

            with tarfile.open(partial, "w:gz") as archive:
                for name in ("main.py", "agent", "data", "decks", "cg"):
                    path = candidate / name
                    if path.exists():
                        archive.add(
                            path, arcname=name, filter=_portable_member
                        )
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)

    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent": {
            "archive": str(BASE_ARCHIVE.resolve()),
            "sha256": BASE_ARCHIVE_SHA256,
            "description": "exact live frozen MD-v3 archive",
        },
        "candidate": {
            "archive": str(output),
            "sha256": sha256_file(output),
            "added_files": sorted(after.keys() - before.keys()),
            "modified_files": ["agent/policy.py"],
            "removed_files": [],
            "weights_file_sha256": WEIGHTS_FILE_SHA256,
            "weights_mapping_sha256": WEIGHTS_MAPPING_SHA256,
            "torch_free_vendor_sources": True,
            "runtime_change": (
                "default-on, exact-deck ST_MAIN MD-v4 NumPy overlay ahead "
                "of byte-frozen MD-v3 main; fail-soft to frozen MD-v3"
            ),
            "original_member_byte_mismatches_outside_policy": 0,
        },
        "evidence": evidence,
        "gate_status": {
            "preregistered_behavior_size_gate_passed": False,
            "experimental_user_override": True,
            "promotion_claim": False,
        },
        "safety_contract": {
            "user_authorized_experimental_packaging": True,
            "fail_soft_to_frozen_md_v3": True,
            "upload_authorized": False,
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "feature_source": {
                "path": str(FEATURE_SOURCE),
                "sha256": sha256_file(FEATURE_SOURCE),
            },
            "model_source": {
                "path": str(MODEL_SOURCE),
                "sha256": sha256_file(MODEL_SOURCE),
            },
            "overlay_source": {
                "path": str(OVERLAY_SOURCE),
                "sha256": sha256_file(OVERLAY_SOURCE),
            },
        },
        "authorization": {
            "research_package_only": True,
            "promotion_authority": False,
            "upload_authority": False,
            "required_before_upload": [
                "full research-vs-vendored callback conformance",
                "tests/test_safety.py against exact extracted archive",
                "200-game random smoke with zero faults/fallbacks/repairs",
                "exact-extracted-tarball audit under non-owner UID",
                "user-supplied tag and explicit upload approval",
            ],
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    try:
        archive, payload = build(
            args.out,
            weights_path=args.weights,
            manifest_path=args.manifest,
        )
    except (BuildError, OSError, ValueError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps({
        "archive": str(archive),
        "archive_sha256": payload["candidate"]["sha256"],
        "manifest_sha256": payload["manifest_sha256"],
        "promotion_authority": False,
        "upload_authority": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
