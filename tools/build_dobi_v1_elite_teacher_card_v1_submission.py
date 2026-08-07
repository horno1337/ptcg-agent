"""Deterministically package the gated selective ST_CARD overlay.

This builder grants neither a ladder name nor upload authority.  It refuses to
run until the behavior, direct-mirror, and field gates pass with exact lineage.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from tools.research import (  # noqa: E402
    eval_dobi_v1_elite_teacher_card_v1_behavior as BEHAVIOR,
    eval_dobi_v1_elite_teacher_card_v1_field as FIELD,
    eval_dobi_v1_elite_teacher_card_v1_gameplay as MIRROR,
    lock_dobi_v1_elite_teacher_card_v1 as SOURCE,
    prepare_dobi_v1_elite_teacher_card_v1 as PREP,
)


BASE = SOURCE.DOBI_ARCHIVE
CARD_MODULE = ROOT / "agent/dobi_v1_card.py"
OUTPUT = ROOT / (
    "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
)
MANIFEST = SOURCE.RUN / "package-manifest.json"
SENTINEL = "UNBOUND_CANDIDATE_REQUIRES_SUCCESSFUL_GATES"
POLICY_ANCHOR = "            # Candidate-only exact-deck ST_CARD overlay."
POLICY_INSERT = '''            # Gated selective Dobi ST_CARD correction.
            if (
                view.select_type == ST_CARD
                and os.environ.get("PTCG_DOBI_V1_CARD", "1") == "1"
            ):
                try:
                    from . import dobi_v1_card as _dobi_v1_card
                    dobi_card_action = _dobi_v1_card.decide(
                        sample, view, registration)
                    if dobi_card_action is not None:
                        return dobi_card_action
                except Exception:
                    pass
'''


class BuildError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    try:
        return MIRROR.COMMON.load_self_hashed_json(path, schema=schema, hash_key=key)
    except MIRROR.COMMON.EvaluationError as error:
        raise BuildError(str(error)) from error


def verify_evidence() -> dict[str, Any]:
    """Return the selected weights only after verifying the full gate chain."""
    try:
        source = PREP.load_and_verify_lock(SOURCE.OUTPUT)
        PREP.verify_bound_artifacts(source)
        screen = _load_self(
            SOURCE.SCREEN_RESULT, BEHAVIOR.SCREEN_SCHEMA, "result_sha256",
        )
        direct_lock, direct_paths = MIRROR.load_lock()
        direct_result = _load_self(
            MIRROR.RESULT, MIRROR.RESULT_SCHEMA, "result_sha256",
        )
        field_lock, _field_paths = FIELD.load_lock()
        field_result = _load_self(
            FIELD.RESULT, FIELD.RESULT_SCHEMA, "result_sha256",
        )
    except (OSError, TypeError, ValueError, PREP.ExtractionError,
            MIRROR.GameplayError, FIELD.FieldError) as error:
        raise BuildError(str(error)) from error
    selected = screen.get("selected_arm")
    weights = direct_paths["candidate_weights"]
    if (
        screen.get("decision") != "advance_to_direct_mirror_gate"
        or not isinstance(selected, str)
        or direct_lock.get("source_lock_sha256") != source["lock_sha256"]
        or direct_lock.get("behavior_screen_sha256") != screen["result_sha256"]
        or direct_lock.get("candidate", {}).get("selected_arm") != selected
        or direct_result.get("gameplay_lock_sha256") != direct_lock["lock_sha256"]
        or direct_result.get("decision", {}).get("valid") is not True
        or direct_result.get("decision", {}).get("passed") is not True
        or field_lock.get("source_lock_sha256") != source["lock_sha256"]
        or field_lock.get("direct_mirror_result_sha256")
            != direct_result["result_sha256"]
        or field_result.get("field_lock_sha256") != field_lock["lock_sha256"]
        or field_result.get("decision", {}).get("valid") is not True
        or field_result.get("decision", {}).get("passed_noninferiority") is not True
        or sha256_file(weights)
            != direct_lock.get("candidate", {}).get("weights_sha256")
    ):
        raise BuildError("passing gate lineage is incomplete or inconsistent")
    return {
        "source": source,
        "behavior": screen,
        "direct_lock": direct_lock,
        "direct_result": direct_result,
        "field_lock": field_lock,
        "field_result": field_result,
        "weights": weights,
    }


def packaged_card_module(weights_sha256: str) -> bytes:
    source = CARD_MODULE.read_text(encoding="utf-8")
    if source.count(SENTINEL) != 1 or len(weights_sha256) != 64:
        raise BuildError("candidate hash sentinel or selected SHA drifted")
    return source.replace(SENTINEL, weights_sha256).encode("utf-8")


def packaged_policy(parent: bytes) -> bytes:
    source = parent.decode("utf-8")
    if source.count(POLICY_ANCHOR) != 1 or "PTCG_DOBI_V1_CARD" in source:
        raise BuildError("frozen Dobi policy insertion anchor drifted")
    return source.replace(POLICY_ANCHOR, POLICY_INSERT + POLICY_ANCHOR).encode("utf-8")


def _read_parent() -> tuple[dict[str, tarfile.TarInfo], dict[str, bytes]]:
    infos: dict[str, tarfile.TarInfo] = {}
    content: dict[str, bytes] = {}
    with tarfile.open(BASE, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name in infos or member.issym() or member.islnk():
                raise BuildError("frozen Dobi archive has duplicate names or links")
            infos[member.name] = member
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is None:
                    raise BuildError(f"cannot read frozen member: {member.name}")
                content[member.name] = handle.read()
    return infos, content


def _write_deterministic(
    path: Path, infos: Mapping[str, tarfile.TarInfo], content: Mapping[str, bytes],
) -> None:
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.GNU_FORMAT) as out:
                for name in sorted(infos):
                    old = infos[name]
                    info = tarfile.TarInfo(name)
                    info.type = old.type
                    info.mode = 0o755 if old.isdir() else 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    data = content.get(name, b"")
                    info.size = len(data) if old.isfile() else 0
                    out.addfile(info, io.BytesIO(data) if old.isfile() else None)


def build(output: Path = OUTPUT, manifest: Path = MANIFEST) -> dict[str, Any]:
    output = output.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    if output.exists() or manifest.exists():
        raise BuildError("refusing to overwrite package output or manifest")
    SOURCE.verify_frozen_dobi_archive()
    evidence = verify_evidence()
    weights = Path(evidence["weights"])
    weights_sha = sha256_file(weights)
    infos, content = _read_parent()
    before = {name: sha256_bytes(data) for name, data in content.items()}
    content["agent/policy.py"] = packaged_policy(content["agent/policy.py"])
    content["agent/dobi_v1_card.py"] = packaged_card_module(weights_sha)
    content["agent/dobi_v1_card_weights.npz"] = weights.read_bytes()
    for name in ("agent/dobi_v1_card.py", "agent/dobi_v1_card_weights.npz"):
        info = tarfile.TarInfo(name)
        info.type = tarfile.REGTYPE
        infos[name] = info
    changed = sorted(
        name for name in before if sha256_bytes(content[name]) != before[name]
    )
    added = sorted(set(content) - set(before))
    if changed != ["agent/policy.py"] or added != [
        "agent/dobi_v1_card.py", "agent/dobi_v1_card_weights.npz",
    ]:
        raise BuildError("unexpected frozen-package member diff")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    if partial.exists():
        raise BuildError("stale partial package exists")
    try:
        _write_deterministic(partial, infos, content)
        os.link(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    output_sha = sha256_file(output)
    payload: dict[str, Any] = {
        "schema": "ptcg.dobi-v1.elite-teacher-card-v1.package.v1",
        "created_at": evidence["field_result"]["created_at"],
        "parent": {"sha256": SOURCE.DOBI_ARCHIVE_SHA256},
        "candidate": {
            "archive_sha256": output_sha,
            "weights_sha256": weights_sha,
            "selected_arm": evidence["behavior"]["selected_arm"],
            "modified_members": changed,
            "added_members": added,
            "overlay_default": "on unless PTCG_DOBI_V1_CARD is not 1",
        },
        "evidence": {
            "source_lock_sha256": evidence["source"]["lock_sha256"],
            "behavior_result_sha256": evidence["behavior"]["result_sha256"],
            "direct_result_sha256": evidence["direct_result"]["result_sha256"],
            "field_result_sha256": evidence["field_result"]["result_sha256"],
        },
        "authorization": {
            "candidate_name_authorized": False,
            "upload_authorized": False,
        },
    }
    payload["manifest_sha256"] = canonical(payload)
    try:
        SOURCE.write_new(manifest, payload)
    except BaseException:
        if output.is_file() and sha256_file(output) == output_sha:
            output.unlink()
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        payload = build(args.output, args.manifest)
    except (BuildError, SOURCE.LockError, OSError, TypeError, ValueError,
            tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
