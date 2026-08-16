"""Build the exact-4b090895 Alakazam + Battle Cage challenger.

This is a DECK change plus one deterministic rule, and no retraining. The
confirmed MAIN/CARD weights are reused byte-identically and only rebound to the
new registration, because 4b090895 first appears in the archive on 2026-08-13:
there are 132 games of it in total, which is two orders of magnitude below the
novelty gate's 400-game floor. A card that did not exist in the meta when the
corpus was recorded cannot be behaviour-cloned from that corpus, so the Battle
Cage decision is a rule and everything else stays the policy that already
passed its gate.

Changes against the frozen Dobi-v2 base:
  decks/deck.csv                  -> the exact 4b090895 registration
  agent/alakazam_bc.py            -> rebound TARGET_DECK / TARGET_DECK_SHA256
  agent/alakazam_battle_cage.py   -> added
  agent/alakazam_*_weights.npz    -> the CONFIRMED heads, unchanged bytes
  agent/policy.py                 -> guard then head, both default-ON
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
PARENT_SHA256 = "409dad4477e1ad36050c3240bfa11fcd3eea322c842eb6ccb7028ebe71afa8e4"
CAGE_ANCHOR = "            # Gated selective Dobi ST_CARD correction."
BLOCK = '''            # Battle Cage guard, then the exact-Alakazam BC specialist.
            # The guard runs first: it encodes a card the training corpus never
            # contained, so the head has no opinion worth deferring to.
            if view.select_type == ST_MAIN and os.environ.get(
                    "PTCG_ALAKAZAM_BATTLE_CAGE", "1") == "1":
                try:
                    from . import alakazam_battle_cage as _alakazam_cage
                    cage_action = _alakazam_cage.decide(view, registration)
                    if cage_action is not None:
                        return cage_action
                except Exception:
                    pass
            if view.select_type in (ST_MAIN, ST_CARD) and os.environ.get(
                    "PTCG_ALAKAZAM_BC", "1") == "1":
                try:
                    from . import alakazam_bc as _alakazam_bc
                    alakazam_action = _alakazam_bc.decide(view, registration)
                    if alakazam_action is not None:
                        return alakazam_action
                except Exception:
                    pass
'''


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def deck_sha(cards) -> str:
    return hashlib.sha256(
        ",".join(str(int(c)) for c in sorted(cards)).encode()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--main-weights", type=Path, required=True)
    p.add_argument("--card-weights", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True,
                   help="behaviour report carrying the added/removed deck diff")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    parent_bytes = PARENT_ARCHIVE.read_bytes()
    if sha_bytes(parent_bytes) != PARENT_SHA256:
        raise SystemExit("parent archive is not frozen Dobi-v2")

    import sys
    sys.path.insert(0, str(ROOT))
    from agent import alakazam_bc as BASE
    from agent import alakazam_battle_cage as GUARD

    diff = json.loads(args.report.read_text())["deck"]
    deck = collections.Counter(BASE.TARGET_DECK)
    for a in diff["added_vs_ours"]:
        deck[a["card_id"]] += a["count"]
    for r in diff["removed_vs_ours"]:
        deck[r["card_id"]] -= r["count"]
    cards = sorted(c for cid, n in deck.items() for c in [cid] * n if n > 0)
    if len(cards) != 60:
        raise SystemExit(f"reconstructed registration has {len(cards)} cards")
    target = deck_sha(cards)
    if target != diff["sha256"] or target != GUARD.TARGET_DECK_SHA256:
        raise SystemExit("reconstructed registration does not match the pins")

    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(parent_bytes), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile():
                members[m.name] = tar.extractfile(m).read()

    # rebind the specialist module to the new registration
    src = (ROOT / "agent" / "alakazam_bc.py").read_text(encoding="utf-8")
    body = "TARGET_DECK = (\n" + "".join(
        "    " + ", ".join(str(c) for c in cards[i:i + 12]) + ",\n"
        for i in range(0, 60, 12)) + ")"
    src = re.sub(r"TARGET_DECK = \((?:.|\n)*?\n\)", body, src, count=1)
    src = src.replace(BASE.TARGET_DECK_SHA256, target)
    main_b, card_b = args.main_weights.read_bytes(), args.card_weights.read_bytes()
    if sha_bytes(main_b) != BASE.MAIN_WEIGHTS_SHA256 or \
            sha_bytes(card_b) != BASE.CARD_WEIGHTS_SHA256:
        raise SystemExit("weights are not the CONFIRMED heads")

    members["decks/deck.csv"] = ("\n".join(str(c) for c in cards) + "\n").encode()
    members["agent/alakazam_bc.py"] = src.encode("utf-8")
    members["agent/alakazam_battle_cage.py"] = (
        ROOT / "agent" / "alakazam_battle_cage.py").read_bytes()
    members["agent/alakazam_main_weights.npz"] = main_b
    members["agent/alakazam_card_weights.npz"] = card_b
    policy = members["agent/policy.py"].decode("utf-8")
    if CAGE_ANCHOR not in policy or "alakazam" in policy:
        raise SystemExit("packaged dispatcher anchor missing or already wired")
    members["agent/policy.py"] = policy.replace(
        CAGE_ANCHOR, BLOCK + CAGE_ANCHOR, 1).encode("utf-8")

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size = len(members[name])
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(members[name]))
    packed = io.BytesIO()
    with gzip.GzipFile(fileobj=packed, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(raw.getvalue())
    blob = packed.getvalue()
    args.out.write_bytes(blob)

    manifest = {
        "schema": "ptcg.alakazam-cage-submission.v1",
        "parent_sha256": PARENT_SHA256,
        "archive": str(args.out), "archive_sha256": sha_bytes(blob),
        "registration_sha256": target,
        "registration_change": {"added": diff["added_vs_ours"],
                                "removed": diff["removed_vs_ours"]},
        "weights_unchanged": {"main": sha_bytes(main_b), "card": sha_bytes(card_b)},
        "retraining": False,
        "added_modules": ["agent/alakazam_battle_cage.py"],
        "members": len(members), "upload_authorized": False,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
