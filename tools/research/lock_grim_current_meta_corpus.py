"""Verify, weight and lock the current-meta Grimmsnarl BC corpus.

Runs every pre-training check and writes an immutable lock. Nothing is trained
until this passes: a weighting mistake that reaches training is indistinguishable
from a model improvement afterwards.

Weighting is two-stage, as specified:
  1. inside a matchup   ordinary win 1.0, weak-matchup win 1.5 or 2.0, loss 0.6
  2. across matchups    each matchup's total mass renormalised to its Field-v3
                        share; the out-of-field tail pinned to exactly 7.8%
Stage 2 makes an explicit mass cap unnecessary -- a matchup cannot become
overrepresented however aggressively its wins are boosted internally.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from tools.il_dataset import decks_from_document  # noqa: E402

GRIM_SHA256 = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
WEAK_MIN_GAMES = 30          # raw games floor for a weakness tier
WEAK_MIN_ESS = 30.0          # effective sample floor, binding
MAX_EPISODE_SHARE = 0.05     # no episode may exceed this of its matchup bucket

# The exact-Grim mirror is excluded from weakness-tier eligibility. Including
# both seats of every mirror forces its win rate to ~50% by construction, so it
# would land in the 45-55% band as a classification artifact rather than as a
# matchup we struggle against. It keeps baseline 1.0/0.6 weights, its Field-v3
# mass share, and its place in validation as a real field slice. Recorded here
# BEFORE any extraction outcome was inspected.
MIRROR_ARCHETYPE = "Grimmsnarl"
MIRROR_TIER_EXCLUDED = True
TIER_STRONG = 0.45           # train WR under this -> winning games 2.0
TIER_MILD = 0.55             # train WR under this -> winning games 1.5
LOSS_WEIGHT = 0.6
TAIL_SHARES = {"Mega Lucario": 0.032, "Annihilape": 0.0265, "_residual": 0.0195}
TAIL_TOTAL = 0.078
SPLITS = (("train", 0.80), ("validation", 0.10), ("test", 0.10))


def deck_sha(deck) -> str:
    return hashlib.sha256(
        ",".join(str(int(c)) for c in sorted(int(x) for x in deck)).encode()
    ).hexdigest()


def split_for(episode_id: int) -> str:
    """Deterministic, game-grouped, append-stable."""
    h = hashlib.sha256(f"grim-current-meta-v1:{episode_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2 ** 64
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if u < acc:
            return name
    return SPLITS[-1][0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--verify-sample", type=int, default=400)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    field = json.loads(args.field.read_text())
    field_share = {r["archetype"]: r["measured_share"] for r in field["rows"]}
    rep_hash = {r["representative_sha256"]: r["archetype"] for r in field["rows"]}
    cards = {c["cardId"]: c for c in json.load(open(ROOT / "data" / "cards.json"))}
    FAMILIES = [r["archetype"] for r in field["rows"]] + \
               ["Mega Lucario", "Annihilape"]

    def archetype_of(deck) -> str:
        h = deck_sha(deck)
        if h in rep_hash:
            return rep_hash[h]
        counts = Counter(int(x) for x in deck)
        names = " ".join(cards.get(cid, {}).get("name", "")
                         for cid in counts if cards.get(cid, {}).get("hp", 0))
        for fam in FAMILIES:
            if fam.lower() in names.lower():
                return fam
        return "_residual"

    # ---- 1. inventory + 2. hash verification --------------------------------
    rows = manifest["games"]
    files = sorted(args.episodes.glob("*.json.gz"))
    verified = mismatched = 0
    step = max(1, len(rows) // max(args.verify_sample, 1))
    games = []
    for i, row in enumerate(rows):
        path = args.episodes / row["stored"]
        if not path.is_file():
            continue
        raw = gzip.decompress(path.read_bytes())
        if i % step == 0:
            verified += 1
            if hashlib.sha256(raw).hexdigest() != row["content_sha256"]:
                mismatched += 1
        doc = json.loads(raw)
        decks = decks_from_document(doc) or {}
        grim_seats = [s for s, d in decks.items() if deck_sha(d) == GRIM_SHA256]
        if not grim_seats:
            continue
        mirror = len(grim_seats) == 2
        # Both sides of a mirror are valid exact-list behaviour and seat index
        # may correlate with turn order, so neither is discarded. The episode
        # stays in ONE split and the two seats share a single game's mass.
        split = split_for(row["episode_id"])
        for seat in grim_seats:
            opp = decks.get(1 - seat)
            arche = "Grimmsnarl" if mirror else (
                archetype_of(opp) if opp else "_residual")
            reward = next((s["reward"] for s in row["seats"]
                           if s["seat"] == seat), None)
            if reward is None:
                continue
            games.append({"episode_id": row["episode_id"], "seat": seat,
                          "archetype": arche, "won": reward > 0,
                          "mirror": mirror, "split": split})

    # ---- 3. split disjointness ---------------------------------------------
    by_split = defaultdict(set)
    for g in games:
        by_split[g["split"]].add(g["episode_id"])
    overlaps = {
        f"{a}&{b}": len(by_split[a] & by_split[b])
        for i, a in enumerate(by_split) for b in list(by_split)[i + 1:]
    }

    # ---- 4. matchup rates on TRAIN ONLY, then tiers -------------------------
    train = [g for g in games if g["split"] == "train"]
    per_matchup = defaultdict(lambda: {"n": 0, "w": 0})
    for g in train:
        per_matchup[g["archetype"]]["n"] += 1
        per_matchup[g["archetype"]]["w"] += int(g["won"])
    # Baseline-weight ESS decides tier ELIGIBILITY, computed before any boost
    # so the criterion cannot depend on the tier it is gating.
    base_ess = {}
    for arche in per_matchup:
        w = [1.0 if g["won"] else LOSS_WEIGHT
             for g in train if g["archetype"] == arche]
        base_ess[arche] = (sum(w) ** 2) / sum(x * x for x in w) if w else 0.0

    tiers = {}
    for arche, st in per_matchup.items():
        wr = st["w"] / st["n"] if st["n"] else 0.0
        boost = 1.0
        mirror_blocked = MIRROR_TIER_EXCLUDED and arche == MIRROR_ARCHETYPE
        if (not mirror_blocked and arche in field_share
                and st["n"] >= WEAK_MIN_GAMES
                and base_ess[arche] >= WEAK_MIN_ESS):
            if wr < TIER_STRONG:
                boost = 2.0
            elif wr < TIER_MILD:
                boost = 1.5
        tiers[arche] = {"train_games": st["n"], "train_wr": wr,
                        "win_weight": boost, "loss_weight": LOSS_WEIGHT,
                        "eligible": arche in field_share and not mirror_blocked,
                        "mirror_tier_excluded": mirror_blocked,
                        "baseline_ess": base_ess[arche],
                        "met_sample_floor": (st["n"] >= WEAK_MIN_GAMES
                                             and base_ess[arche] >= WEAK_MIN_ESS)}

    # ---- stage 1 then stage 2 ----------------------------------------------
    # Stage 1. A mirror episode contributes one game's mass split across its
    # two seats in proportion to the outcome weights: 1.0/(1.0+0.6) = 62.5% to
    # the winning seat, 37.5% to the losing seat.
    raw_w = defaultdict(float)
    per_episode = defaultdict(list)
    for g in train:
        per_episode[g["episode_id"]].append(g)
    for eid, seats in per_episode.items():
        if len(seats) == 2 and all(x["mirror"] for x in seats):
            base = {id(x): (1.0 if x["won"] else LOSS_WEIGHT) for x in seats}
            total = sum(base.values()) or 1.0
            for x in seats:
                x["w1"] = base[id(x)] / total          # sums to exactly 1 game
        else:
            for x in seats:
                t = tiers[x["archetype"]]
                x["w1"] = t["win_weight"] if x["won"] else LOSS_WEIGHT
    for g in train:
        raw_w[g["archetype"]] += g["w1"]

    # Sparse tail buckets are folded into _residual rather than being given
    # large per-episode weights.
    for g in train:
        a = g["archetype"]
        if a not in field_share and a != "_residual":
            if (per_matchup[a]["n"] < WEAK_MIN_GAMES
                    or base_ess[a] < WEAK_MIN_ESS):
                g["archetype"] = "_residual"
    raw_w = defaultdict(float)
    for g in train:
        raw_w[g["archetype"]] += g["w1"]

    target = {}
    for arche in raw_w:
        if arche in field_share:
            target[arche] = field_share[arche]
        elif arche in TAIL_SHARES:
            target[arche] = TAIL_SHARES[arche]
        else:
            target[arche] = None                # folded into residual below
    residual = [a for a, v in target.items() if v is None] + \
               (["_residual"] if "_residual" in raw_w else [])
    residual = [a for a in dict.fromkeys(residual) if a in raw_w]
    res_mass = sum(raw_w[a] for a in residual) or 1.0
    for a in residual:
        target[a] = TAIL_SHARES["_residual"] * raw_w[a] / res_mass

    total_target = sum(v for v in target.values() if v)
    for g in train:
        a = g["archetype"]
        g["weight"] = g["w1"] * (target[a] / total_target) / raw_w[a] * len(train)

    # ---- 5/6. effective sample size, max weight, mass check ----------------
    per = defaultdict(list)
    for g in train:
        per[g["archetype"]].append(g["weight"])
    mass = {a: sum(v) for a, v in per.items()}
    tot_mass = sum(mass.values())
    report_rows = []
    for a in sorted(per, key=lambda x: -mass[x]):
        v = per[a]
        ess = (sum(v) ** 2) / sum(x * x for x in v) if v else 0.0
        report_rows.append({
            "archetype": a, "raw_games": len(v),
            "effective_sample_size": ess,
            "mass_share": mass[a] / tot_mass,
            "target_share": target[a],
            "max_game_weight": max(v),
            "in_field": a in field_share,
            **{k: tiers[a][k] for k in ("train_wr", "win_weight", "met_sample_floor")},
        })
    tail_mass = sum(mass[a] for a in mass if a not in field_share)
    # Binding: no single episode may dominate its matchup bucket.
    episode_share = defaultdict(lambda: defaultdict(float))
    for g in train:
        episode_share[g["archetype"]][g["episode_id"]] += g["weight"]
    worst = {a: (max(v.values()) / mass[a] if mass[a] else 0.0)
             for a, v in episode_share.items()}
    episode_share_violations = {a: s for a, s in worst.items()
                                if s > MAX_EPISODE_SHARE}

    lock = {
        "schema": "ptcg.grim-current-meta-corpus-lock.v1",
        "grim_sha256": GRIM_SHA256,
        "field_source": {"file": str(args.field), "date": field["source_date"],
                         "archive_sha256": field["source_sha256"]},
        "inventory": {
            "manifest_games": len(rows),
            "stored_files": len(files),
            "usable_games": len(games),
            "unreadable_members": sum(a["unreadable"] for a in manifest["archives"]),
            "id_conflicts": len(manifest.get("id_conflicts", [])),
        },
        "hash_verification": {"sampled": verified, "mismatched": mismatched},
        "splits": {k: len(v) for k, v in by_split.items()},
        "split_overlaps": overlaps,
        "weighting": {
            "stage1": {"ordinary_win": 1.0, "weak_win_mild": 1.5,
                       "weak_win_strong": 2.0, "loss": LOSS_WEIGHT},
            "stage2": "matchup mass renormalised to Field-v3 share; tail pinned to 7.8%",
            "weak_min_games": WEAK_MIN_GAMES,
            "tier_thresholds": {"strong_below": TIER_STRONG, "mild_below": TIER_MILD},
            "tail_shares": TAIL_SHARES,
            "mirror_archetype": MIRROR_ARCHETYPE,
            "mirror_excluded_from_weakness_tier": MIRROR_TIER_EXCLUDED,
            "mirror_rationale": (
                "including both seats forces ~50% by construction; a 45-55% "
                "tier would be a classification artifact, not difficulty"),
            "mirror_seat_split": {"winning_seat": 0.625, "losing_seat": 0.375,
                                  "per_episode_mass": 1.0},
        },
        "matchups": report_rows,
        "checks": {
            "no_split_overlap": all(v == 0 for v in overlaps.values()),
            "no_hash_mismatch": mismatched == 0,
            "tail_mass_share": tail_mass / tot_mass,
            "tail_within_tolerance": abs(tail_mass / tot_mass - TAIL_TOTAL) < 0.005,
            "max_game_weight": max((r["max_game_weight"] for r in report_rows),
                                   default=0.0),
            "max_episode_share_by_matchup": worst,
            "episode_share_violations": episode_share_violations,
            "no_episode_share_violation": not episode_share_violations,
            "ess_floor": WEAK_MIN_ESS,
            "max_episode_share_limit": MAX_EPISODE_SHARE,
        },
    }
    body = json.dumps(lock, indent=1, sort_keys=True)
    lock_hash = hashlib.sha256(body.encode()).hexdigest()
    args.out.write_text(body[:-2] + f',\n "lock_sha256": "{lock_hash}"\n}}\n'
                        if body.endswith("\n}") else body)
    (args.out.parent / (args.out.stem + "-sha256.txt")).write_text(lock_hash + "\n")

    print(f"games usable {len(games)}  splits {lock['splits']}")
    print(f"hash verify {verified} sampled, {mismatched} mismatched")
    print(f"split overlaps {overlaps}")
    print(f"\n{'archetype':20s} {'raw':>6} {'ESS':>8} {'mass':>7} {'target':>7} "
          f"{'wr':>6} {'boost':>5} {'maxw':>7}")
    for r in report_rows:
        print(f"{r['archetype']:20s} {r['raw_games']:6d} {r['effective_sample_size']:8.1f} "
              f"{r['mass_share']:6.2%} {r['target_share']:6.2%} "
              f"{r['train_wr']:5.1%} {r['win_weight']:5.1f} {r['max_game_weight']:7.3f}")
    if episode_share_violations:
        print(f"EPISODE-SHARE VIOLATIONS: {episode_share_violations}")
    print(f"\ntail mass {lock['checks']['tail_mass_share']:.2%} "
          f"(target {TAIL_TOTAL:.1%})  max weight "
          f"{lock['checks']['max_game_weight']:.3f}")
    print(f"lock sha256 {lock_hash}")
    ok = (lock["checks"]["no_split_overlap"] and lock["checks"]["no_hash_mismatch"]
          and lock["checks"]["tail_within_tolerance"]
          and lock["checks"]["no_episode_share_violation"])
    print("LOCK OK" if ok else "LOCK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
