"""Verify, weight and lock the current-meta Grimmsnarl BC corpus.

Consumes the facts file produced by build_grim_corpus_facts.py, so a weighting
revision costs seconds rather than a 30-minute re-parse.  Nothing is trained
until this passes: a weighting mistake that reaches training is
indistinguishable from a model improvement afterwards.

Weighting is two-stage:
  1. inside a matchup   ordinary win 1.0, weak-matchup win 1.5 or 2.0, loss 0.6
  2. across matchups    each matchup's mass renormalised to its Field-v3 share

Stage 2 makes an explicit mass cap unnecessary: a matchup cannot become
overrepresented however aggressively its wins are boosted internally.

Tail handling.  The out-of-field tail is folded in rather than dropped, but its
internal composition follows the MEASURED field (field.json uncovered_archetypes),
never the raw corpus and never a hardcoded table.  An earlier revision hardcoded
"Annihilape" at 2.65%; Field-v3 contains no Annihilape at all, and the true
residual behind Mega Lucario is 4.76%, not 1.95%.  Reading the field file is the
only way to keep these aligned.

Weights are computed PER HEAD.  A seat with no MAIN decision cannot carry MAIN
mass, so renormalising once at game level would leave each head's realised
distribution slightly off Field-v3.  Per-head renormalisation makes the realised
training distribution exact for both heads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GRIM_SHA256 = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
WEAK_MIN_GAMES = 30          # raw games floor for a weakness tier, binding
WEAK_MIN_ESS = 30.0          # effective sample floor, binding
MAX_EPISODE_SHARE = 0.05     # no episode may exceed this of its matchup bucket
TIER_STRONG = 0.45           # train WR under this -> winning games 2.0
TIER_MILD = 0.55             # train WR under this -> winning games 1.5
LOSS_WEIGHT = 0.6
TAIL_TOLERANCE = 0.005

# The exact-Grim mirror is excluded from weakness-tier eligibility.  Including
# both seats of every mirror forces its win rate to ~50% by construction, so it
# would land in the 45-55% band as a classification artifact rather than as a
# matchup we struggle against.  It keeps baseline 1.0/0.6 weights, its Field-v3
# mass share, and its place in validation as a real field slice.  Recorded
# BEFORE any extraction outcome was inspected.
MIRROR_ARCHETYPE = "Grimmsnarl"
MIRROR_TIER_EXCLUDED = True
MIRROR_WIN_SHARE = 1.0 / (1.0 + LOSS_WEIGHT)     # 0.625


def _ess(values) -> float:
    s = sum(values)
    q = sum(v * v for v in values)
    return (s * s / q) if q > 0 else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights-out", type=Path, required=True)
    args = p.parse_args()

    field = json.loads(args.field.read_text())
    facts_meta = json.loads(
        args.facts.with_name(args.facts.stem + "-meta.json").read_text())

    # ---- field targets: in-field measured share + measured tail -------------
    in_field = {r["archetype"]: r["measured_share"] for r in field["rows"]}
    field_tail = {u["archetype"]: u["share"] for u in field["uncovered_archetypes"]}
    tail_total = field["uncovered_share"]

    seats = []
    episodes = {}
    for line in args.facts.open():
        row = json.loads(line)
        episodes[row["episode_id"]] = row["split"]
        for s in row["seats"]:
            seats.append({**s, "episode_id": row["episode_id"],
                          "split": row["split"], "mirror": row["mirror"]})

    # ---- split disjointness (episode-grouped by construction; verify) -------
    by_split = defaultdict(set)
    for s in seats:
        by_split[s["split"]].add(s["episode_id"])
    overlaps = {f"{a}&{b}": len(by_split[a] & by_split[b])
                for i, a in enumerate(sorted(by_split))
                for b in sorted(by_split)[i + 1:]}

    train = [s for s in seats if s["split"] == "train"]

    # ---- fold sparse tail archetypes into _residual -------------------------
    raw_n = defaultdict(int)
    for s in train:
        raw_n[s["archetype"]] += 1
    for s in seats:
        a = s["archetype"]
        if a not in in_field and raw_n[a] < WEAK_MIN_GAMES:
            s["archetype"] = "_residual"

    train = [s for s in seats if s["split"] == "train"]

    # ---- tier assignment from TRAIN ONLY ------------------------------------
    stat = defaultdict(lambda: {"n": 0, "w": 0})
    for s in train:
        stat[s["archetype"]]["n"] += 1
        stat[s["archetype"]]["w"] += int(s["won"])
    base_ess = {}
    for a in stat:
        vals = [1.0 if s["won"] else LOSS_WEIGHT
                for s in train if s["archetype"] == a]
        base_ess[a] = _ess(vals)

    tiers = {}
    for a, st in stat.items():
        wr = st["w"] / st["n"] if st["n"] else 0.0
        mirror_blocked = MIRROR_TIER_EXCLUDED and a == MIRROR_ARCHETYPE
        eligible = (a in in_field) and not mirror_blocked
        floor_ok = st["n"] >= WEAK_MIN_GAMES and base_ess[a] >= WEAK_MIN_ESS
        boost = 1.0
        if eligible and floor_ok:
            if wr < TIER_STRONG:
                boost = 2.0
            elif wr < TIER_MILD:
                boost = 1.5
        tiers[a] = {"train_games": st["n"], "train_wins": st["w"],
                    "train_wr": wr, "win_weight": boost,
                    "loss_weight": LOSS_WEIGHT,
                    "tier_eligible": eligible,
                    "mirror_tier_excluded": mirror_blocked,
                    "baseline_ess": base_ess[a],
                    "met_sample_floor": floor_ok,
                    "in_field": a in in_field}

    # ---- stage 1 -----------------------------------------------------------
    # Mirror: one episode contributes ONE game's mass, split 62.5/37.5 by
    # outcome across its two exact-Grim seats.  No artificial mirror boost.
    per_ep = defaultdict(list)
    for s in seats:
        per_ep[s["episode_id"]].append(s)
    for eid, group in per_ep.items():
        if len(group) == 2 and all(x["mirror"] for x in group):
            for x in group:
                x["stage1"] = MIRROR_WIN_SHARE if x["won"] else 1.0 - MIRROR_WIN_SHARE
                x["stage1_eval"] = x["stage1"]
        else:
            for x in group:
                t = tiers.get(x["archetype"], {"win_weight": 1.0})
                x["stage1"] = t["win_weight"] if x["won"] else LOSS_WEIGHT
                # Sealed readouts use BASELINE outcome weights: the tier boosts
                # are a training device, not an evaluation target.
                x["stage1_eval"] = 1.0 if x["won"] else LOSS_WEIGHT

    # ---- stage 2, per head -------------------------------------------------
    present_tail = sorted({s["archetype"] for s in seats
                           if s["archetype"] not in in_field})
    tail_weight = {}
    known = {a: field_tail[a] for a in present_tail if a in field_tail}
    unknown = [a for a in present_tail if a not in field_tail]
    # Field tail archetypes absent from the corpus cannot be represented; their
    # mass redistributes across the tail archetypes we actually hold, so the
    # tail still totals exactly the measured uncovered share.
    residual_field_mass = tail_total - sum(known.values())
    known_sum = sum(known.values()) or 1.0
    for a in present_tail:
        if a in known:
            tail_weight[a] = known[a]
        else:
            tail_weight[a] = 0.0
    if unknown:
        # "_residual" absorbs the measured mass of field tail archetypes that
        # are either absent or too sparse to stand alone.
        share = max(residual_field_mass, 0.0) / len(unknown)
        for a in unknown:
            tail_weight[a] = share
    else:
        scale = tail_total / known_sum
        for a in known:
            tail_weight[a] = known[a] * scale
    target = {**in_field, **tail_weight}

    def assign(head: str, stage1_key: str, weight_key: str,
               splits: tuple[str, ...]) -> dict:
        """Renormalise so this head's realised mass matches Field-v3 exactly."""
        rows = [s for s in seats
                if s["split"] in splits and s[f"{head}_decisions"] > 0]
        raw = defaultdict(float)
        for s in rows:
            raw[s["archetype"]] += s[stage1_key]
        present = [a for a in raw if raw[a] > 0]
        tot_target = sum(target.get(a, 0.0) for a in present) or 1.0
        for s in rows:
            a = s["archetype"]
            s[weight_key] = (s[stage1_key] * (target.get(a, 0.0) / tot_target)
                             / raw[a] * len(rows))
        for s in seats:
            if s["split"] in splits and s[f"{head}_decisions"] <= 0:
                s[weight_key] = 0.0
        return {"rows": len(rows), "present": present, "tot_target": tot_target}

    assign("main", "stage1", "main_weight", ("train",))
    assign("card", "stage1", "card_weight", ("train",))
    assign("main", "stage1_eval", "main_weight_eval", ("validation", "test"))
    assign("card", "stage1_eval", "card_weight_eval", ("validation", "test"))
    for s in seats:
        s.setdefault("main_weight", 0.0)
        s.setdefault("card_weight", 0.0)
        s.setdefault("main_weight_eval", 0.0)
        s.setdefault("card_weight_eval", 0.0)

    # ---- reporting + binding checks per head -------------------------------
    def report(head: str, weight_key: str) -> dict:
        rows = [s for s in train if s[weight_key] > 0]
        per = defaultdict(list)
        for s in rows:
            per[s["archetype"]].append(s[weight_key])
        mass = {a: sum(v) for a, v in per.items()}
        tot = sum(mass.values()) or 1.0
        ep = defaultdict(lambda: defaultdict(float))
        for s in rows:
            ep[s["archetype"]][s["episode_id"]] += s[weight_key]
        worst = {a: (max(v.values()) / mass[a] if mass[a] else 0.0)
                 for a, v in ep.items()}
        table = []
        for a in sorted(per, key=lambda x: -mass[x]):
            t = tiers.get(a, {})
            table.append({
                "archetype": a, "raw_games": len(per[a]),
                "effective_sample_size": _ess(per[a]),
                "mass_share": mass[a] / tot,
                "target_share": target.get(a, 0.0),
                "max_game_weight": max(per[a]),
                "max_episode_share": worst[a],
                "in_field": a in in_field,
                "train_wr": t.get("train_wr", 0.0),
                "win_weight": t.get("win_weight", 1.0),
                "tier_eligible": t.get("tier_eligible", False),
                "met_sample_floor": t.get("met_sample_floor", False),
                "baseline_ess": t.get("baseline_ess", 0.0),
                "decisions": sum(s[f"{head}_decisions"] for s in rows
                                 if s["archetype"] == a),
            })
        tail_mass = sum(mass[a] for a in mass if a not in in_field) / tot
        violations = {a: v for a, v in worst.items() if v > MAX_EPISODE_SHARE}
        # A boost may only be applied where the ESS floor is genuinely met.
        bad_boost = {a: tiers[a] for a in per
                     if tiers.get(a, {}).get("win_weight", 1.0) > 1.0
                     and not tiers[a]["met_sample_floor"]}
        return {
            "table": table, "total_mass": tot,
            "tail_mass_share": tail_mass,
            "tail_target": tail_total,
            "tail_within_tolerance": abs(tail_mass - tail_total) < TAIL_TOLERANCE,
            "max_game_weight": max((r["max_game_weight"] for r in table), default=0.0),
            "max_episode_share": max(worst.values()) if worst else 0.0,
            "episode_share_violations": violations,
            "no_episode_share_violation": not violations,
            "boost_without_ess_floor": bad_boost,
            "no_boost_without_ess_floor": not bad_boost,
            "missing_field_archetypes": [a for a in in_field if a not in per],
            "total_decisions": sum(r["decisions"] for r in table),
        }

    rep = {"main": report("main", "main_weight"),
           "card": report("card", "card_weight")}

    # ---- persist the per-seat weights the trainer consumes ------------------
    wtmp = args.weights_out.with_suffix(".part")
    with wtmp.open("w") as fh:
        for s in sorted(seats, key=lambda x: (x["episode_id"], x["seat"])):
            fh.write(json.dumps({
                "episode_id": s["episode_id"], "seat": s["seat"],
                "split": s["split"], "archetype": s["archetype"],
                "opp_sha256": s["opp_sha256"], "won": s["won"],
                "mirror": s["mirror"],
                "main_decisions": s["main_decisions"],
                "card_decisions": s["card_decisions"],
                "stage1": s["stage1"], "stage1_eval": s["stage1_eval"],
                "main_weight": s["main_weight"], "card_weight": s["card_weight"],
                "main_weight_eval": s["main_weight_eval"],
                "card_weight_eval": s["card_weight_eval"],
            }, sort_keys=True) + "\n")
    wtmp.replace(args.weights_out)
    weights_sha = hashlib.sha256(args.weights_out.read_bytes()).hexdigest()

    checks = {
        "no_split_overlap": all(v == 0 for v in overlaps.values()),
        "no_content_hash_mismatch": not facts_meta["content_hash_mismatched"],
        "content_hash_verified": facts_meta["content_hash_verified"],
        "main_tail_within_tolerance": rep["main"]["tail_within_tolerance"],
        "card_tail_within_tolerance": rep["card"]["tail_within_tolerance"],
        "main_no_episode_share_violation": rep["main"]["no_episode_share_violation"],
        "card_no_episode_share_violation": rep["card"]["no_episode_share_violation"],
        "main_no_boost_without_ess_floor": rep["main"]["no_boost_without_ess_floor"],
        "card_no_boost_without_ess_floor": rep["card"]["no_boost_without_ess_floor"],
        "all_field_archetypes_present": (not rep["main"]["missing_field_archetypes"]
                                         and not rep["card"]["missing_field_archetypes"]),
    }
    lock = {
        "schema": "ptcg.grim-current-meta-corpus-lock.v2",
        "grim_sha256": GRIM_SHA256,
        "facts": {"file": str(args.facts), "sha256": facts_meta["facts_sha256"],
                  "episodes_ok": facts_meta["episodes_ok"],
                  "episodes_error": facts_meta["episodes_error"],
                  "errors": facts_meta["errors"], "seats": facts_meta["seats"]},
        "field_source": {"file": str(args.field), "date": field["source_date"],
                         "archive_sha256": field["source_sha256"],
                         "covered_share": field["covered_share"],
                         "uncovered_share": tail_total},
        "weights_dataset": {"file": str(args.weights_out), "sha256": weights_sha,
                            "rows": len(seats)},
        "splits": {k: len(v) for k, v in sorted(by_split.items())},
        "split_overlaps": overlaps,
        "weighting": {
            "stage1": {"ordinary_win": 1.0, "weak_win_mild": 1.5,
                       "weak_win_strong": 2.0, "loss": LOSS_WEIGHT},
            "stage2": "per-head renormalisation to Field-v3 measured share",
            "weak_min_games": WEAK_MIN_GAMES, "weak_min_ess": WEAK_MIN_ESS,
            "tier_thresholds": {"strong_below": TIER_STRONG, "mild_below": TIER_MILD},
            "tail_source": "field.json uncovered_archetypes (measured, not hardcoded)",
            "tail_target_total": tail_total,
            "tail_composition": tail_weight,
            "eval_weights": "baseline 1.0/0.6, field-normalised, no tier boosts",
            "mirror_archetype": MIRROR_ARCHETYPE,
            "mirror_excluded_from_weakness_tier": MIRROR_TIER_EXCLUDED,
            "mirror_seat_split": {"winning_seat": MIRROR_WIN_SHARE,
                                  "losing_seat": 1.0 - MIRROR_WIN_SHARE,
                                  "per_episode_mass": 1.0},
        },
        "tiers": tiers,
        "heads": {h: {k: v for k, v in rep[h].items() if k != "table"}
                  for h in rep},
        "matchups": {h: rep[h]["table"] for h in rep},
        "checks": checks,
    }
    body = json.dumps(lock, indent=1, sort_keys=True)
    lock_hash = hashlib.sha256(body.encode()).hexdigest()
    args.out.write_text(body + "\n")
    args.out.with_name(args.out.stem + "-sha256.txt").write_text(lock_hash + "\n")

    # ---- human-readable verification table ---------------------------------
    print(f"episodes ok {facts_meta['episodes_ok']}  errors {facts_meta['episodes_error']} "
          f"{facts_meta['errors']}")
    print(f"seats {len(seats)}   splits {lock['splits']}   overlaps {overlaps}")
    print(f"content hashes verified {facts_meta['content_hash_verified']}, "
          f"mismatched {len(facts_meta['content_hash_mismatched'])}")
    for head in ("main", "card"):
        r = rep[head]
        print(f"\n===== {head.upper()} head  ({r['rows'] if 'rows' in r else ''}"
              f"{r['total_decisions']:,} decisions) =====")
        print(f"{'archetype':<24}{'raw':>6}{'ESS':>9}{'mass':>8}{'target':>8}"
              f"{'wr':>7}{'bst':>5}{'maxw':>7}{'maxep':>7}")
        for t in r["table"]:
            print(f"{t['archetype']:<24}{t['raw_games']:>6}"
                  f"{t['effective_sample_size']:>9.1f}{t['mass_share']:>8.2%}"
                  f"{t['target_share']:>8.2%}{t['train_wr']:>7.1%}"
                  f"{t['win_weight']:>5.1f}{t['max_game_weight']:>7.2f}"
                  f"{t['max_episode_share']:>7.2%}")
        print(f"  tail {r['tail_mass_share']:.2%} (target {tail_total:.2%})  "
              f"max weight {r['max_game_weight']:.3f}  "
              f"max episode share {r['max_episode_share']:.2%}")
        if r["episode_share_violations"]:
            print(f"  EPISODE-SHARE VIOLATIONS {r['episode_share_violations']}")
        if r["boost_without_ess_floor"]:
            print(f"  BOOST WITHOUT ESS FLOOR {list(r['boost_without_ess_floor'])}")

    print("\n---- binding checks ----")
    for k, v in checks.items():
        print(f"  {'PASS' if v is True else ('FAIL' if v is False else v)}  {k}")
    print(f"\nweights dataset {args.weights_out}  sha256 {weights_sha[:16]}  "
          f"rows {len(seats)}")
    print(f"lock sha256 {lock_hash}")
    ok = all(v is True for v in checks.values() if isinstance(v, bool))
    print("LOCK OK" if ok else "LOCK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
