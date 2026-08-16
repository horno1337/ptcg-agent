"""Deterministically build the current-meta Grimmsnarl submission.

The frozen Dobi-v2 archive is the parent.  Exactly four bytes-level changes are
made and nothing else is touched:

  agent/md_v1_weights.npz        -> retrained MAIN head
  agent/dobi_v1_card_weights.npz -> retrained CARD head
  agent/md_v1.py                 -> its WEIGHTS_SHA256 constant
  agent/dobi_v1_card.py          -> its WEIGHTS_SHA256 constant

Those two modules verify their own artifact hash and fail CLOSED to the Qu-v2B
router, so shipping new weights without updating the constant would silently
deploy the parent policy.  Every other member -- engine, decks, data, main.py,
the Qu-v2B base weights, the md_v2 card overlay -- is copied through unchanged.

Determinism: members are emitted in sorted order with mtime 0, uid/gid 0, and
canonical 0644/0755 modes, so two builds from the same inputs are byte
identical.  Portable modes matter because the ladder extracts and executes as
different UIDs; a mode-0600 weights file is unreadable there and degrades
silently to the rules fallback.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
PARENT_SHA256 = "409dad4477e1ad36050c3240bfa11fcd3eea322c842eb6ccb7028ebe71afa8e4"
REPLACE = {
    "agent/md_v1_weights.npz": ("main", "agent/md_v1.py"),
    "agent/dobi_v1_card_weights.npz": ("card", "agent/dobi_v1_card.py"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--main-weights", type=Path)
    p.add_argument("--card-weights", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parent", type=Path, default=PARENT_ARCHIVE)
    args = p.parse_args()

    parent_bytes = args.parent.read_bytes()
    if sha256_bytes(parent_bytes) != PARENT_SHA256:
        raise SystemExit("parent archive is not frozen Dobi-v2")

    new_weights = {}
    if args.main_weights:
        new_weights["agent/md_v1_weights.npz"] = args.main_weights.read_bytes()
    if args.card_weights:
        new_weights["agent/dobi_v1_card_weights.npz"] = args.card_weights.read_bytes()
    if not new_weights:
        raise SystemExit("nothing to replace: pass --main-weights and/or --card-weights")
    new_sha = {k: sha256_bytes(v) for k, v in new_weights.items()}

    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(parent_bytes), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            members[m.name] = tar.extractfile(m).read()

    changed = []
    for weights_name, (_head, module_name) in REPLACE.items():
        if weights_name not in new_weights:
            continue
        if weights_name not in members or module_name not in members:
            raise SystemExit(f"parent archive missing {weights_name}/{module_name}")
        old_sha = sha256_bytes(members[weights_name])
        members[weights_name] = new_weights[weights_name]
        source = members[module_name].decode("utf-8")
        if old_sha not in source:
            raise SystemExit(f"{module_name} does not pin {old_sha[:16]}")
        members[module_name] = source.replace(
            old_sha, new_sha[weights_name]).encode("utf-8")
        changed.append({"weights": weights_name, "module": module_name,
                        "old_sha256": old_sha, "new_sha256": new_sha[weights_name]})

    # Deterministic archive: sorted names, zeroed metadata, canonical modes.
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size = len(members[name])
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(members[name]))
    packed = io.BytesIO()
    with gzip.GzipFile(fileobj=packed, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(raw.getvalue())
    blob = packed.getvalue()
    args.out.write_bytes(blob)

    manifest = {
        "schema": "ptcg.grim-current-meta-submission.v1",
        "parent_archive": str(args.parent), "parent_sha256": PARENT_SHA256,
        "archive": str(args.out), "archive_sha256": sha256_bytes(blob),
        "members": len(members), "changed": changed,
        "upload_authorized": False,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
