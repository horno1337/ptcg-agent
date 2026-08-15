"""Rebuild the evaluation field from measured ladder shares.

The previous schedule was assembled from seven hand-picked lists. Measured
against the Aug-14 archive it covered only 34.7% of seat-decks and mis-weighted
what it did cover -- Mega Lucario and Mega Kangaskhan carried ~14.2% each
against a measured 3.2% and 3.5%, while Grimmsnarl carried 6.6% against 13.6%,
and Mega Froslass (17.2%) and Alakazam (13.9%) were absent entirely. A gate run
on that schedule answers a precise question about a field that does not exist.

This derives archetype weights and a representative exact list per archetype
from a named source archive, and freezes them with the source date, the
representative hashes, and the uncovered mass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from tools.il_dataset import decks_from_document  # noqa: E402

FAMILIES = [
    ("Mega Froslass", "Mega Froslass"), ("Dragapult", "Dragapult"),
    ("Grimmsnarl", "Grimmsnarl"), ("Alakazam", "Alakazam"),
    ("Hydrapple", "Hydrapple"), ("Crustle", "Crustle"),
    ("Mega Kangaskhan", "Mega Kangaskhan"), ("Mega Lucario", "Mega Lucario"),
    ("Annihilape", "Annihilape"), ("Teal Mask Ogerpon", "Teal Mask Ogerpon"),
    ("Mega Gardevoir", "Mega Gardevoir"), ("Mega Lopunny", "Mega Lopunny"),
    ("Mega Venusaur", "Mega Venusaur"),
]


def deck_sha(deck) -> str:
    return hashlib.sha256(
        ",".join(str(int(c)) for c in sorted(int(x) for x in deck)).encode()
    ).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--coverage", type=float, default=0.90,
                   help="archetype mass to cover before truncating")
    args = p.parse_args()

    cards = {c["cardId"]: c for c in json.load(open(ROOT / "data" / "cards.json"))}
    seat_counts: Counter = Counter()
    decks_by_hash: dict[str, list[int]] = {}
    games = unreadable = 0

    with zipfile.ZipFile(args.archive) as z:
        for member in z.namelist():
            try:
                decks = decks_from_document(json.loads(z.read(member))) or {}
            except Exception:
                unreadable += 1
                continue
            if not decks:
                continue
            games += 1
            for d in decks.values():
                h = deck_sha(d)
                seat_counts[h] += 1
                decks_by_hash.setdefault(h, sorted(int(x) for x in d))

    def archetype(deck: list[int]) -> str:
        counts = Counter(deck)
        mons = [(n, cards.get(cid, {}).get("name", "")) for cid, n in counts.items()
                if cards.get(cid, {}).get("hp", 0)]
        names = " ".join(nm for _n, nm in mons)
        for key, fam in FAMILIES:
            if key.lower() in names.lower():
                return fam
        mons.sort(key=lambda t: -t[0])
        return mons[0][1] if mons else "unknown"

    fam_seats: Counter = Counter()
    fam_lists: dict[str, Counter] = defaultdict(Counter)
    for h, n in seat_counts.items():
        fam = archetype(decks_by_hash[h])
        fam_seats[fam] += n
        fam_lists[fam][h] += n

    total = sum(fam_seats.values())
    ordered = fam_seats.most_common()
    chosen, acc = [], 0
    for fam, n in ordered:
        if acc / total >= args.coverage:
            break
        rep_hash, rep_seats = fam_lists[fam].most_common(1)[0]
        chosen.append({
            "archetype": fam,
            "measured_share": n / total,
            "measured_seats": n,
            "distinct_lists": len(fam_lists[fam]),
            "representative_sha256": rep_hash,
            "representative_seats": rep_seats,
            "representative_share_within_archetype": rep_seats / n,
            "deck": decks_by_hash[rep_hash],
        })
        acc += n

    covered = sum(c["measured_seats"] for c in chosen)
    for c in chosen:                      # renormalise across the covered set
        c["field_weight"] = c["measured_seats"] / covered

    payload = {
        "schema": "ptcg.current-field.v3",
        "source_archive": args.archive.name,
        "source_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "source_date": args.archive.stem.split("episodes-")[-1],
        "games_scanned": games,
        "unreadable_members": unreadable,
        "total_seat_decks": total,
        "distinct_registrations": len(seat_counts),
        "archetype_coverage_target": args.coverage,
        "covered_seats": covered,
        "covered_share": covered / total,
        "uncovered_share": 1.0 - covered / total,
        "uncovered_archetypes": [
            {"archetype": f, "share": n / total}
            for f, n in ordered[len(chosen):]
        ],
        "rows": chosen,
    }
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"{games} games, {total} seat-decks, {len(seat_counts)} registrations")
    print(f"covered {covered/total:.1%} with {len(chosen)} archetypes; "
          f"uncovered {1-covered/total:.1%}\n")
    for c in chosen:
        print(f"  {c['field_weight']:6.2%}  {c['archetype']:22s} "
              f"rep {c['representative_sha256'][:8]} "
              f"({c['representative_share_within_archetype']:.0%} of "
              f"{c['distinct_lists']} lists)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
