#!/usr/bin/env python3
"""Outcome-blind AlphaStarmie agreement diagnostic for Dobi-v2 turn search.

Only exact-list Grim mirrors are eligible.  This matters because the current
planner binds one registered deck per simulated seat for the whole analysis;
outside a mirror, its opponent policy is not yet rebound to each belief
particle's hypothesized deck.  The mirror restriction makes that registration
exact rather than silently confounded.

The locked panel balances roots where frozen Dobi agrees/disagrees with the
logged Alpha action.  Results are standardized back to the full eligible
population, so the balanced panel is informative without distorting overall
agreement.  No game outcome is copied into the lock or root panel.
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

from agent.obsview import ST_MAIN, ObsView  # noqa: E402
from agent.seat_policy import (  # noqa: E402
    FrozenDobiV2Policy, QuV2BasePolicy, SeatPolicyTable, load_frozen_runtime,
)
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools.index_corpus import deck_sha256  # noqa: E402
from tools.research import compare_current_grim_to_dobi_v2 as COMPARE  # noqa: E402


RUN = ROOT / "tools/checkpoints/alphastarmie-turn-search-agreement-20260813"
LOCK = RUN / "lock.json"
ROOTS = RUN / "roots.jsonl"
RESULT = RUN / "result.json"
SOURCE = ROOT / "tools/checkpoints/alphatcg-grim-scout-20260813/result-mirror.json"
ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
DECK_PATH = ROOT / "decks/md_v1_grimmsnarl.csv"
BUDGET_S = 5.0
PARTICLES = 8
PER_STRATUM = 128
WORKERS = 8
THREADS = 2
EXPECTED_FREEZE = (
    "5f3c6131"  # human-facing prefix; full frozen artifact is bound below
)


class DiagnosticError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def read_deck() -> tuple[int, ...]:
    deck = tuple(int(line) for line in DECK_PATH.read_text().split() if line.strip())
    if len(deck) != 60:
        raise DiagnosticError("frozen Grim deck is not 60 cards")
    if deck_sha256(deck) != COMPARE.TARGET_DECK_SHA256:
        raise DiagnosticError("frozen Grim deck identity drifted")
    return deck


def write_new(path: Path, value: Mapping[str, Any], mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
        handle.write("\n")


def semantic(view: ObsView, action: list[int]) -> tuple[str, ...]:
    return COMPARE.semantic_set(view, action)


def valid(view: ObsView, action: Any) -> list[int] | None:
    row = COMPARE.valid_action(view, action)
    return list(row) if row is not None else None


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ROOTS, RESULT)):
        raise DiagnosticError("diagnostic already locked or consumed")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    runtime = load_frozen_runtime(ARCHIVE)
    deck = read_deck()
    candidates: dict[str, list[dict[str, Any]]] = {
        "dobi_agrees": [], "dobi_disagrees": [],
    }
    replay_hashes: dict[str, str] = {}
    seen: set[tuple[int, int, int]] = set()
    population = Counter()
    excluded_nonmirror = 0

    # Read only identity/path/seat. Outcome fields in the source result are
    # deliberately never accessed and never copied.
    for game in source["cohort"]["games"]:
        episode = int(game["episode_id"])
        seat = int(game["seat"])
        path = Path(game["path"])
        document = json.loads(path.read_text(encoding="utf-8"))
        registered = COMPARE.decks_from_document(document)
        if tuple(sorted(registered.get(seat, ()))) != tuple(sorted(deck)):
            raise DiagnosticError("Alpha seat deck drifted")
        if tuple(sorted(registered.get(1 - seat, ()))) != tuple(sorted(deck)):
            excluded_nonmirror += 1
            continue
        replay_hashes.setdefault(str(path.resolve()), sha256(path))
        ours = FrozenDobiV2Policy(runtime)
        for step, obs, logged_raw in COMPARE.decisions(document, seat):
            key = (episode, seat, step)
            if key in seen:
                continue
            seen.add(key)
            view = ObsView(obs)
            logged = valid(view, logged_raw)
            if (
                logged is None or view.select_type != ST_MAIN
                or view.min_count != 1 or view.max_count != 1
                or not 2 <= len(view.options) <= 24
                or not isinstance(obs.get("search_begin_input"), str)
                or not obs.get("search_begin_input")
                or not isinstance(obs.get("remainingOverageTime"), (int, float))
                or float(obs["remainingOverageTime"]) < 161.0
            ):
                continue
            decision = ours.decide(obs, deck, seat)
            dobi = valid(view, decision.action)
            if dobi is None:
                raise DiagnosticError("frozen Dobi emitted an invalid action")
            agrees = semantic(view, logged) == semantic(view, dobi)
            stratum = "dobi_agrees" if agrees else "dobi_disagrees"
            population[stratum] += 1
            root_id = canonical({
                "episode": episode, "seat": seat, "step": step,
                "fingerprint": CFO.public_root_fingerprint(obs),
            })
            candidates[stratum].append({
                "root_id": root_id,
                "episode_id": episode,
                "seat": seat,
                "step": step,
                "stratum": stratum,
                "alpha_action": logged,
                "dobi_action": dobi,
                "dobi_route": decision.route,
                "observation": obs,
            })
        if ours.overlay_faults:
            raise DiagnosticError(f"Dobi overlay faults: {dict(ours.overlay_faults)}")

    selected = []
    for stratum in ("dobi_agrees", "dobi_disagrees"):
        rows = sorted(candidates[stratum], key=lambda row: row["root_id"])
        if len(rows) < PER_STRATUM:
            raise DiagnosticError(
                f"only {len(rows)} roots in {stratum}; need {PER_STRATUM}"
            )
        selected.extend(rows[:PER_STRATUM])
    selected.sort(key=lambda row: row["root_id"])
    ROOTS.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOTS, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    freeze_path = ROOT / "tools/checkpoints/dobi-v2-turn-search-budget-sweep-20260813/freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_hash = freeze.get("freeze_sha256") or sha256(freeze_path)
    if not str(freeze_hash).startswith(EXPECTED_FREEZE):
        raise DiagnosticError(f"turn-search freeze drifted: {freeze_hash}")
    lock: dict[str, Any] = {
        "schema": "ptcg.alphastarmie-turn-search-agreement-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_search_results": True,
        "outcomes_read_for_selection": False,
        "scope": "exact byte-identical Grim mirror seats only",
        "reason_for_mirror_scope": (
            "the current seat table binds one opponent registration per analysis; "
            "only an exact mirror makes that registration unconfounded"
        ),
        "panel": {
            "selection": "ascending SHA-256 root_id within each Dobi-agreement stratum",
            "per_stratum": PER_STRATUM,
            "total": len(selected),
            "population": dict(population),
            "eligible_mirror_games": len(replay_hashes),
            "excluded_nonmirror_games": excluded_nonmirror,
        },
        "search": {
            "budget_s": BUDGET_S, "particles": PARTICLES,
            "workers": WORKERS, "threads_requested": THREADS,
            "freeze_sha256": freeze_hash,
        },
        "metrics": [
            "population-standardized semantic agreement",
            "override-only fixed/broken/neither",
            "reasons, coverage, faults, latency",
        ],
        "interpretation": (
            "diagnostic alignment only; Alpha agreement is not gameplay value "
            "and confers no promotion authority"
        ),
        "artifacts": {
            "source": {"path": str(SOURCE.resolve()), "sha256": sha256(SOURCE)},
            "archive": {"path": str(ARCHIVE.resolve()), "sha256": sha256(ARCHIVE)},
            "deck": {"path": str(DECK_PATH.resolve()), "sha256": sha256(DECK_PATH)},
            "turn_search": {"path": str((ROOT / "agent/turn_search.py").resolve()),
                            "sha256": sha256(ROOT / "agent/turn_search.py")},
            "seat_policy": {"path": str((ROOT / "agent/seat_policy.py").resolve()),
                            "sha256": sha256(ROOT / "agent/seat_policy.py")},
            "freeze": {"path": str(freeze_path.resolve()), "sha256": sha256(freeze_path)},
            "roots": {"path": str(ROOTS.resolve()), "sha256": sha256(ROOTS)},
            "selected_replays": replay_hashes,
        },
        "promotion_authority": False,
        "upload_authority": False,
    }
    lock["lock_sha256"] = canonical(lock)
    write_new(LOCK, lock)
    return lock


def load_roots() -> list[dict[str, Any]]:
    return [json.loads(line) for line in ROOTS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def worker(spec: dict[str, Any]) -> list[dict[str, Any]]:
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
        ours = FrozenDobiV2Policy(runtime)
        theirs = QuV2BasePolicy(runtime)
        table = (SeatPolicyTable().bind(seat, ours, deck)
                                  .bind(1 - seat, theirs, TS.field_prior_deck()))
        started = time.monotonic()
        with TS.seat_context(table, seat):
            result = TS.analyze(
                view, net, list(deck), budget_s=BUDGET_S,
                max_particles=PARTICLES,
            )
        elapsed = time.monotonic() - started
        baseline = list(row["dobi_action"])
        chosen = (
            list(result.chosen_action)
            if result is not None and result.chosen_action is not None
            else baseline
        )
        if valid(view, chosen) is None:
            raise DiagnosticError("search emitted an invalid action")
        alpha_sem = semantic(view, row["alpha_action"])
        dobi_sem = semantic(view, baseline)
        search_sem = semantic(view, chosen)
        override = search_sem != dobi_sem
        output.append({
            "root_id": row["root_id"],
            "stratum": row["stratum"],
            "dobi_route": row["dobi_route"],
            "reason": None if result is None else result.reason,
            "valid_particles": int(TS.last_stats.get("valid_particles", 0) or 0),
            "elapsed_s": elapsed,
            "override": override,
            "alpha_dobi_agree": alpha_sem == dobi_sem,
            "alpha_search_agree": alpha_sem == search_sem,
            "fixed": bool(alpha_sem != dobi_sem and alpha_sem == search_sem),
            "broken": bool(alpha_sem == dobi_sem and alpha_sem != search_sem),
            "changed_but_neither": bool(
                override and alpha_sem != dobi_sem and alpha_sem != search_sem
            ),
            "overlay_faults": {
                "ours": dict(ours.overlay_faults),
                "theirs": dict(theirs.overlay_faults),
            },
        })
    return output


def spawn(spec: dict[str, Any]) -> list[dict[str, Any]]:
    # This outer worker is already a separate process. BLAS configuration must
    # therefore be inherited before numpy/runtime import in production runs;
    # the parent command sets it explicitly for the full diagnostic.
    return worker(spec)


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
        for part in pool.map(spawn, specs):
            rows.extend(part)
    wall = time.monotonic() - started
    rows.sort(key=lambda row: row["root_id"])
    if len(rows) != 2 * PER_STRATUM or len({r["root_id"] for r in rows}) != len(rows):
        raise DiagnosticError("root result coverage is incomplete or duplicated")

    reasons = Counter(row["reason"] for row in rows)
    faults = sum(
        sum(row["overlay_faults"][side].values())
        for row in rows for side in ("ours", "theirs")
    )
    by_stratum = {}
    for stratum in ("dobi_agrees", "dobi_disagrees"):
        sample = [row for row in rows if row["stratum"] == stratum]
        by_stratum[stratum] = {
            "n": len(sample),
            "search_agrees_with_alpha": sum(r["alpha_search_agree"] for r in sample),
            "search_agreement_rate": statistics.fmean(
                float(r["alpha_search_agree"]) for r in sample
            ),
            "overrides": sum(r["override"] for r in sample),
            "fixed": sum(r["fixed"] for r in sample),
            "broken": sum(r["broken"] for r in sample),
            "changed_but_neither": sum(r["changed_but_neither"] for r in sample),
        }
    population = lock["panel"]["population"]
    total_population = sum(population.values())
    p_agree = population["dobi_agrees"] / total_population
    standardized = (
        p_agree * by_stratum["dobi_agrees"]["search_agreement_rate"]
        + (1.0 - p_agree)
        * by_stratum["dobi_disagrees"]["search_agreement_rate"]
    )
    payload: dict[str, Any] = {
        "schema": "ptcg.alphastarmie-turn-search-agreement-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "wall_seconds": wall,
        "panel_roots": len(rows),
        "population": {
            **population,
            "eligible": total_population,
            "frozen_dobi_alpha_agreement": p_agree,
            "standardized_search_alpha_agreement": standardized,
            "standardized_search_minus_dobi": standardized - p_agree,
        },
        "by_stratum": by_stratum,
        "search": {
            "reasons": dict(reasons),
            "reached_evidence_floor": sum(r["valid_particles"] >= 5 for r in rows),
            "coverage": statistics.fmean(float(r["valid_particles"] >= 5) for r in rows),
            "overrides": sum(r["override"] for r in rows),
            "override_rate": statistics.fmean(float(r["override"]) for r in rows),
            "elapsed_mean_s": statistics.fmean(r["elapsed_s"] for r in rows),
            "elapsed_p95_s": sorted(r["elapsed_s"] for r in rows)[math.floor(.95*(len(rows)-1))],
        },
        "faults": {"overlay": faults},
        "rows": rows,
        "verdict_limit": (
            "agreement is descriptive, exact-mirror-only, and not promotion evidence"
        ),
        "promotion_authority": False,
        "upload_authority": False,
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
        print(json.dumps({
            "lock_sha256": value["lock_sha256"],
            "panel": value["panel"],
            "outcomes_read_for_selection": value["outcomes_read_for_selection"],
        }, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            "population": value["population"],
            "by_stratum": value["by_stratum"],
            "search": value["search"],
            "faults": value["faults"],
            "result_sha256": value["result_sha256"],
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        print(f"DIAGNOSTIC ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
