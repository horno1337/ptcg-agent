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
BLOCK = '''            # Board-safety guards, Battle Cage guard, then the exact-Alakazam
            # BC specialist. The guards run around the head because they encode
            # arithmetic and a card the training corpus never contained, so the
            # head has no opinion worth deferring to on either.
            _alakazam_guards = None
            if os.environ.get("PTCG_ALAKAZAM_GUARDS", "1") == "1":
                try:
                    from . import alakazam_lethal_guards as _alakazam_guards
                except Exception:
                    _alakazam_guards = None
            if _alakazam_guards is not None:
                try:
                    draw_action = _alakazam_guards.decide(view, registration)
                    if draw_action is not None:
                        return draw_action
                except Exception:
                    pass
            overlay_action = None
            if view.select_type == ST_MAIN and os.environ.get(
                    "PTCG_ALAKAZAM_BATTLE_CAGE", "1") == "1":
                try:
                    from . import alakazam_battle_cage as _alakazam_cage
                    overlay_action = _alakazam_cage.decide(view, registration)
                except Exception:
                    overlay_action = None
            if overlay_action is None and view.select_type in (
                    ST_MAIN, ST_CARD) and os.environ.get(
                        "PTCG_ALAKAZAM_BC", "1") == "1":
                try:
                    from . import alakazam_bc as _alakazam_bc
                    overlay_action = _alakazam_bc.decide(view, registration)
                except Exception:
                    overlay_action = None
            if overlay_action is not None:
                if _alakazam_guards is not None:
                    try:
                        from . import alakazam_bc as _alakazam_rerank

                        def _rerank(blocked, _v=view, _r=registration):
                            return _alakazam_rerank.decide(_v, _r, veto=blocked)

                        fixed = _alakazam_guards.correct(
                            view, registration, overlay_action, rerank=_rerank)
                        if fixed is not None:
                            return fixed
                    except Exception:
                        pass
                return overlay_action
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
    p.add_argument("--main-weights-sha256", default=None,
                   help="accept a RETRAINED MAIN head with this hash instead of "
                        "the confirmed one. Opt-in and recorded; omitting it "
                        "keeps the confirmed-weights build byte-identical.")
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
    expect_main = args.main_weights_sha256 or BASE.MAIN_WEIGHTS_SHA256
    if sha_bytes(main_b) != expect_main:
        raise SystemExit("MAIN weights do not match the expected hash")
    if sha_bytes(card_b) != BASE.CARD_WEIGHTS_SHA256:
        raise SystemExit("CARD weights are not the CONFIRMED head")
    if expect_main != BASE.MAIN_WEIGHTS_SHA256:
        # The runtime verifies its own weight hashes, so a retrained MAIN has to
        # be re-pinned in the PACKAGED source or the overlay fails soft to rules.
        src = src.replace(BASE.MAIN_WEIGHTS_SHA256, expect_main)
        if expect_main not in src:
            raise SystemExit("failed to re-pin MAIN weights hash in the package")

    members["decks/deck.csv"] = ("\n".join(str(c) for c in cards) + "\n").encode()
    members["agent/alakazam_bc.py"] = src.encode("utf-8")
    members["agent/alakazam_battle_cage.py"] = (
        ROOT / "agent" / "alakazam_battle_cage.py").read_bytes()
    members["agent/alakazam_lethal_guards.py"] = (
        ROOT / "agent" / "alakazam_lethal_guards.py").read_bytes()
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
        "weights": {"main": sha_bytes(main_b), "card": sha_bytes(card_b)},
        "retraining": {"main": sha_bytes(main_b) != BASE.MAIN_WEIGHTS_SHA256,
                       "card": False},
        "added_modules": ["agent/alakazam_battle_cage.py",
                          "agent/alakazam_lethal_guards.py"],
        "members": len(members), "upload_authorized": False,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
