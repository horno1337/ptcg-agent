"""Deterministically build the exact-Alakazam August BC submission.

Base is the frozen Dobi-v2 archive, so the engine, data tables, main.py and the
Qu-v2B base weights are the exact bytes already proven on the ladder. Changes:

  decks/deck.csv                  -> the exact Field-v3 Alakazam registration
  agent/alakazam_bc.py            -> added (hash-bound overlay)
  agent/alakazam_main_weights.npz -> added
  agent/alakazam_card_weights.npz -> added
  agent/policy.py                 -> Alakazam overlay block, default-ON

The Grim overlays (md_v1, dobi_v1_card, md_v2_card) are left in place. They are
scoped to the Grimmsnarl registration by `supports_deck`, so under an Alakazam
registration they return None and are inert; removing them would change more
bytes than it protects.

The overlay block is inserted into the PACKAGED dispatcher rather than shipping
the worktree's policy.py, so every other routing decision stays byte-identical
to the version that has already run on the ladder.

Determinism: sorted members, mtime 0, uid/gid 0, canonical 0644/0755 modes.
Portable modes matter -- the ladder extracts and executes as different UIDs, and
a mode-0600 weights file is unreadable there and degrades silently to rules.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
PARENT_SHA256 = "409dad4477e1ad36050c3240bfa11fcd3eea322c842eb6ccb7028ebe71afa8e4"

ANCHOR = "            # Gated selective Dobi ST_CARD correction."
BLOCK = '''            # Exact-Alakazam August BC specialist.  Owns ST_MAIN and
            # ST_CARD only for its own registration, verifies both artifact
            # hashes itself, and returns None on any scope/load/decode miss.
            if (
                view.select_type in (ST_MAIN, ST_CARD)
                and os.environ.get("PTCG_ALAKAZAM_BC", "1") == "1"
            ):
                try:
                    from . import alakazam_bc as _alakazam_bc
                    alakazam_action = _alakazam_bc.decide(view, registration)
                    if alakazam_action is not None:
                        return alakazam_action
                except Exception:
                    pass
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--main-weights", type=Path, required=True)
    p.add_argument("--card-weights", type=Path, required=True)
    p.add_argument("--module", type=Path, default=ROOT / "agent" / "alakazam_bc.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parent", type=Path, default=PARENT_ARCHIVE)
    args = p.parse_args()

    parent_bytes = args.parent.read_bytes()
    if sha256_bytes(parent_bytes) != PARENT_SHA256:
        raise SystemExit("parent archive is not frozen Dobi-v2")

    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(parent_bytes), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile():
                members[m.name] = tar.extractfile(m).read()

    module_src = args.module.read_text(encoding="utf-8")
    import re
    want_main = re.search(r'MAIN_WEIGHTS_SHA256 = \(\s*"([0-9a-f]{64})"', module_src)
    want_card = re.search(r'CARD_WEIGHTS_SHA256 = \(\s*"([0-9a-f]{64})"', module_src)
    if not (want_main and want_card):
        raise SystemExit("alakazam_bc.py does not pin both artifact hashes")
    main_bytes = args.main_weights.read_bytes()
    card_bytes = args.card_weights.read_bytes()
    if sha256_bytes(main_bytes) != want_main.group(1):
        raise SystemExit("MAIN weights do not match the module's pinned hash")
    if sha256_bytes(card_bytes) != want_card.group(1):
        raise SystemExit("CARD weights do not match the module's pinned hash")

    # deck registration
    deck_ns: dict = {}
    exec(compile(re.search(r"TARGET_DECK = \((.|\n)*?\n\)", module_src).group(0),
                 "<deck>", "exec"), deck_ns)
    deck = deck_ns["TARGET_DECK"]
    if len(deck) != 60:
        raise SystemExit("Alakazam registration is not 60 cards")
    old_deck = members["decks/deck.csv"].decode().split()
    members["decks/deck.csv"] = ("\n".join(str(int(c)) for c in deck) + "\n").encode()

    # dispatcher
    policy = members["agent/policy.py"].decode("utf-8")
    if ANCHOR not in policy:
        raise SystemExit("packaged dispatcher anchor not found")
    if "alakazam_bc" in policy:
        raise SystemExit("packaged dispatcher already wired")
    members["agent/policy.py"] = policy.replace(
        ANCHOR, BLOCK + ANCHOR, 1).encode("utf-8")

    members["agent/alakazam_bc.py"] = module_src.encode("utf-8")
    members["agent/alakazam_main_weights.npz"] = main_bytes
    members["agent/alakazam_card_weights.npz"] = card_bytes

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
        "schema": "ptcg.alakazam-august-submission.v1",
        "parent_archive": str(args.parent), "parent_sha256": PARENT_SHA256,
        "archive": str(args.out), "archive_sha256": sha256_bytes(blob),
        "members": len(members),
        "registration_changed": {"from_first3": old_deck[:3],
                                 "to_sha256": hashlib.sha256(
                                     ",".join(str(int(c)) for c in sorted(deck)
                                              ).encode()).hexdigest()},
        "added": ["agent/alakazam_bc.py", "agent/alakazam_main_weights.npz",
                  "agent/alakazam_card_weights.npz"],
        "main_weights_sha256": sha256_bytes(main_bytes),
        "card_weights_sha256": sha256_bytes(card_bytes),
        "dispatcher": "PTCG_ALAKAZAM_BC defaults to 1 in the package",
        "upload_authorized": False,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
