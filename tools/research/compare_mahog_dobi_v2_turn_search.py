#!/usr/bin/env python3
"""Outcome-blind Mahog agreement diagnostic for corrected Dobi-v2 search.

Mahog's current top-30 Grim list differs from Dobi's list by two cards.  To
avoid pretending that Dobi can inhabit impossible private states, this keeps
only roots whose complete visible acting-seat state can be reconciled exactly
with Dobi's registered 60.  Opponent decks are inferred by turn_search's
public posterior and are never read from the replay registration.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
os.environ.setdefault("PTCG_TURN_SEARCH", "1")

from agent import search_policy as SP  # noqa: E402
from agent.obsview import ST_MAIN, ObsView  # noqa: E402
from agent.seat_policy import (  # noqa: E402
    FrozenDobiV2Policy, QuV2BasePolicy, SeatPolicyTable, load_frozen_runtime,
)
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.index_corpus import deck_sha256  # noqa: E402
from tools.research import compare_current_grim_to_dobi_v2 as COMPARE  # noqa: E402
from tools.research.compare_alphastarmie_dobi_v2_turn_search import (  # noqa: E402
    canonical, semantic, sha256, valid, write_new,
)


RUN = ROOT / "tools/checkpoints/mahog-turn-search-agreement-v2-20260813"
LOCK = RUN / "lock.json"
ROOTS = RUN / "roots.jsonl"
RESULT = RUN / "result.json"
REPLAYS = ROOT / "tools/checkpoints/grim-mahog-55475820/replays"
ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
DECK_PATH = ROOT / "decks/md_v1_grimmsnarl.csv"
PILOT = "Mahog"
SUBMISSION_ID = 55475820
DISPLAY_RANK = 26
DISPLAY_SCORE = 1093.4
PILOT_DECK_SHA256 = "e2e03fe8ef9592b1204c159a7725da3945675fa872b1e18baca550560503a78e"
BUDGET_S = 5.0
PARTICLES = 8
PER_STRATUM = 128
WORKERS = 8
THREADS = 2


class DiagnosticError(RuntimeError):
    pass


def read_deck() -> tuple[int, ...]:
    deck = tuple(int(line) for line in DECK_PATH.read_text().split() if line.strip())
    if len(deck) != 60 or deck_sha256(deck) != COMPARE.TARGET_DECK_SHA256:
        raise DiagnosticError("Dobi Grim deck identity drifted")
    return deck


def own_state_reachable(view: ObsView, deck: tuple[int, ...]) -> bool:
    """Exact public/private consistency test used by search reconstruction."""
    seen = SP._seen_ids(view.me or {}, with_hand=True)
    stadium = (view.current or {}).get("stadium")
    entries = stadium if isinstance(stadium, list) else [stadium]
    for entry in entries:
        if (isinstance(entry, dict)
                and entry.get("playerIndex") == view.my_index
                and isinstance(entry.get("id"), int)):
            seen.append(entry["id"])
    pool = SP._remainder(list(deck), seen)
    expected = len((view.me or {}).get("prize") or ()) + int(
        (view.me or {}).get("deckCount") or 0)
    return len(pool) == expected


def replay_files() -> list[Path]:
    paths = sorted(path for path in REPLAYS.glob("*.json") if path.stem.isdigit())
    if not paths:
        raise DiagnosticError("Mahog replay directory is empty")
    return paths


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ROOTS, RESULT)):
        raise DiagnosticError("diagnostic already locked or consumed")
    runtime = load_frozen_runtime(ARCHIVE)
    deck = read_deck()
    candidates = {"dobi_agrees": [], "dobi_disagrees": []}
    population = Counter()
    exclusions = Counter()
    selected_replays: dict[str, str] = {}
    eligible_seats = 0
    seen_roots: set[tuple[int, int, int]] = set()

    for path in replay_files():
        document = json.loads(path.read_text(encoding="utf-8"))
        names = (document.get("info") or {}).get("TeamNames") or ["?", "?"]
        decks = COMPARE.decks_from_document(document)
        episode = int((document.get("info") or {}).get("EpisodeId") or path.stem)
        for seat, registered in decks.items():
            if names[seat] != PILOT or deck_sha256(registered) != PILOT_DECK_SHA256:
                continue
            eligible_seats += 1
            selected_replays.setdefault(str(path.resolve()), sha256(path))
            ours = FrozenDobiV2Policy(runtime)
            for step, obs, logged_raw in COMPARE.decisions(document, seat):
                key = (episode, seat, step)
                if key in seen_roots:
                    continue
                seen_roots.add(key)
                view = ObsView(obs)
                logged = valid(view, logged_raw)
                if (logged is None or view.select_type != ST_MAIN
                        or view.min_count != 1 or view.max_count != 1
                        or not 2 <= len(view.options) <= 24
                        or not isinstance(obs.get("search_begin_input"), str)
                        or not obs.get("search_begin_input")
                        or not isinstance(obs.get("remainingOverageTime"), (int, float))
                        or float(obs["remainingOverageTime"]) < 161.0):
                    exclusions["not_search_capable_main"] += 1
                    continue
                if not own_state_reachable(view, deck):
                    exclusions["not_reachable_with_dobi_list"] += 1
                    continue
                decision = ours.decide(obs, deck, seat)
                dobi = valid(view, decision.action)
                if dobi is None:
                    raise DiagnosticError("frozen Dobi emitted an invalid action")
                stratum = ("dobi_agrees" if semantic(view, logged) == semantic(view, dobi)
                           else "dobi_disagrees")
                population[stratum] += 1
                root_id = canonical({
                    "episode": episode, "seat": seat, "step": step,
                    "fingerprint": CFO.public_root_fingerprint(obs),
                })
                candidates[stratum].append({
                    "root_id": root_id, "episode_id": episode, "seat": seat,
                    "step": step, "stratum": stratum,
                    "pilot_action": logged, "dobi_action": dobi,
                    "dobi_route": decision.route, "observation": obs,
                })
            if ours.overlay_faults:
                raise DiagnosticError(f"Dobi overlay faults: {dict(ours.overlay_faults)}")

    selected: list[dict[str, Any]] = []
    for stratum in ("dobi_agrees", "dobi_disagrees"):
        rows = sorted(candidates[stratum], key=lambda row: row["root_id"])
        if len(rows) < PER_STRATUM:
            raise DiagnosticError(f"only {len(rows)} roots in {stratum}; need {PER_STRATUM}")
        selected.extend(rows[:PER_STRATUM])
    selected.sort(key=lambda row: row["root_id"])
    ROOTS.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOTS, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    artifacts = {
        "archive": {"path": str(ARCHIVE.resolve()), "sha256": sha256(ARCHIVE)},
        "deck": {"path": str(DECK_PATH.resolve()), "sha256": sha256(DECK_PATH)},
        "turn_search": {"path": str((ROOT / "agent/turn_search.py").resolve()),
                        "sha256": sha256(ROOT / "agent/turn_search.py")},
        "seat_policy": {"path": str((ROOT / "agent/seat_policy.py").resolve()),
                        "sha256": sha256(ROOT / "agent/seat_policy.py")},
        "diagnostic": {"path": str(Path(__file__).resolve()),
                       "sha256": sha256(Path(__file__))},
        "roots": {"path": str(ROOTS.resolve()), "sha256": sha256(ROOTS)},
        "selected_replays": selected_replays,
    }
    lock: dict[str, Any] = {
        "schema": "ptcg.mahog-turn-search-agreement-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_search_results": True,
        "outcomes_read_for_selection": False,
        "pilot": {"name": PILOT, "submission_id": SUBMISSION_ID,
                  "display_rank": DISPLAY_RANK, "display_score": DISPLAY_SCORE,
                  "deck_sha256": PILOT_DECK_SHA256},
        "scope": ("Mahog MAIN roots exactly reachable under Dobi's 60; "
                  "opponent inferred from public posterior"),
        "deck_difference": "58/60 cards shared; two-card list confound remains",
        "panel": {"selection": "ascending SHA-256 within agreement stratum",
                  "per_stratum": PER_STRATUM, "total": len(selected),
                  "population": dict(population), "eligible_seats": eligible_seats,
                  "exclusions": dict(exclusions)},
        "search": {"budget_s": BUDGET_S, "particles": PARTICLES,
                   "workers": WORKERS, "threads_requested": THREADS,
                   "opponent_binding": "per-particle public posterior"},
        "artifacts": artifacts,
        "interpretation": ("descriptive agreement only; different registered lists "
                           "and pilot agreement is not gameplay value"),
        "promotion_authority": False, "upload_authority": False,
    }
    lock["lock_sha256"] = canonical(lock)
    write_new(LOCK, lock)
    return lock


def load_roots() -> list[dict[str, Any]]:
    return [json.loads(line) for line in ROOTS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def worker(spec: dict[str, int]) -> list[dict[str, Any]]:
    from agent import model as RepoModel
    from agent import turn_search as TS
    runtime = load_frozen_runtime(ARCHIVE)
    deck = read_deck()
    net = RepoModel.load()
    output = []
    for row in load_roots()[spec["index"]::spec["workers"]]:
        obs = row["observation"]
        view = ObsView(obs)
        seat = int(row["seat"])
        ours, theirs = FrozenDobiV2Policy(runtime), QuV2BasePolicy(runtime)
        # The opponent's bound deck is only a fail-closed fallback. Corrected
        # turn_search overrides it with each public-posterior particle deck.
        table = (SeatPolicyTable().bind(seat, ours, deck)
                                  .bind(1 - seat, theirs, deck))
        started = time.monotonic()
        with TS.seat_context(table, seat):
            result = TS.analyze(view, net, list(deck), budget_s=BUDGET_S,
                                max_particles=PARTICLES)
        elapsed = time.monotonic() - started
        baseline = list(row["dobi_action"])
        chosen = (list(result.chosen_action)
                  if result is not None and result.chosen_action is not None
                  else baseline)
        if valid(view, chosen) is None:
            raise DiagnosticError("search emitted an invalid action")
        pilot_sem = semantic(view, row["pilot_action"])
        dobi_sem, search_sem = semantic(view, baseline), semantic(view, chosen)
        override = search_sem != dobi_sem
        output.append({
            "root_id": row["root_id"], "stratum": row["stratum"],
            "reason": None if result is None else result.reason,
            "valid_particles": int(TS.last_stats.get("valid_particles", 0) or 0),
            "elapsed_s": elapsed, "override": override,
            "pilot_dobi_agree": pilot_sem == dobi_sem,
            "pilot_search_agree": pilot_sem == search_sem,
            "fixed": pilot_sem != dobi_sem and pilot_sem == search_sem,
            "broken": pilot_sem == dobi_sem and pilot_sem != search_sem,
            "changed_but_neither": override and pilot_sem != dobi_sem and pilot_sem != search_sem,
            "overlay_faults": {"ours": dict(ours.overlay_faults),
                               "theirs": dict(theirs.overlay_faults)},
        })
    return output


def run() -> dict[str, Any]:
    if RESULT.exists():
        raise DiagnosticError("diagnostic result already exists")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if canonical({k: v for k, v in lock.items() if k != "lock_sha256"}) != lock["lock_sha256"]:
        raise DiagnosticError("lock self-hash mismatch")
    for name, row in lock["artifacts"].items():
        if name == "selected_replays":
            for path, digest in row.items():
                if sha256(Path(path)) != digest:
                    raise DiagnosticError(f"selected replay drifted: {path}")
        elif sha256(Path(row["path"])) != row["sha256"]:
            raise DiagnosticError(f"locked artifact drifted: {name}")

    specs = [{"index": i, "workers": WORKERS} for i in range(WORKERS)]
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for part in pool.map(worker, specs):
            rows.extend(part)
    wall = time.monotonic() - started
    rows.sort(key=lambda row: row["root_id"])
    if len(rows) != 2 * PER_STRATUM or len({r["root_id"] for r in rows}) != len(rows):
        raise DiagnosticError("root result coverage incomplete or duplicated")

    by_stratum = {}
    for stratum in ("dobi_agrees", "dobi_disagrees"):
        sample = [row for row in rows if row["stratum"] == stratum]
        by_stratum[stratum] = {
            "n": len(sample),
            "search_agrees_with_pilot": sum(r["pilot_search_agree"] for r in sample),
            "search_agreement_rate": statistics.fmean(
                float(r["pilot_search_agree"]) for r in sample),
            "overrides": sum(r["override"] for r in sample),
            "fixed": sum(r["fixed"] for r in sample),
            "broken": sum(r["broken"] for r in sample),
            "changed_but_neither": sum(r["changed_but_neither"] for r in sample),
        }
    population = lock["panel"]["population"]
    total = sum(population.values())
    p_agree = population["dobi_agrees"] / total
    standardized = (p_agree * by_stratum["dobi_agrees"]["search_agreement_rate"]
                    + (1 - p_agree)
                    * by_stratum["dobi_disagrees"]["search_agreement_rate"])
    faults = sum(sum(r["overlay_faults"][side].values())
                 for r in rows for side in ("ours", "theirs"))
    elapsed = sorted(r["elapsed_s"] for r in rows)
    reasons = Counter(r["reason"] if r["reason"] is not None else "no_result"
                      for r in rows)
    payload: dict[str, Any] = {
        "schema": "ptcg.mahog-turn-search-agreement-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"], "wall_seconds": wall,
        "panel_roots": len(rows),
        "population": {**population, "eligible": total,
                       "frozen_dobi_pilot_agreement": p_agree,
                       "standardized_search_pilot_agreement": standardized,
                       "standardized_search_minus_dobi": standardized - p_agree},
        "by_stratum": by_stratum,
        "search": {"reasons": dict(reasons),
                   "reached_evidence_floor": sum(r["valid_particles"] >= 5 for r in rows),
                   "coverage": statistics.fmean(float(r["valid_particles"] >= 5) for r in rows),
                   "overrides": sum(r["override"] for r in rows),
                   "override_rate": statistics.fmean(float(r["override"]) for r in rows),
                   "elapsed_mean_s": statistics.fmean(elapsed),
                   "elapsed_p95_s": elapsed[math.floor(.95 * (len(elapsed) - 1))]},
        "faults": {"overlay": faults}, "rows": rows,
        "verdict_limit": lock["interpretation"],
        "promotion_authority": False, "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    args = parser.parse_args()
    value = build_lock() if args.stage == "lock" else run()
    if args.stage == "lock":
        print(json.dumps({"lock_sha256": value["lock_sha256"],
                          "pilot": value["pilot"], "panel": value["panel"],
                          "outcomes_read_for_selection": value["outcomes_read_for_selection"]},
                         indent=2, sort_keys=True))
    else:
        print(json.dumps({k: value[k] for k in
                          ("population", "by_stratum", "search", "faults",
                           "result_sha256")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        print(f"DIAGNOSTIC ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
