#!/usr/bin/env python3
"""Outcome-blind Dobi/search agreement diagnostic for exact-list Grim pilots."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any


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
from tools.research.compare_alphastarmie_dobi_v2_turn_search import (  # noqa: E402
    canonical, semantic, sha256, valid, write_new,
)


REPLAYS = ROOT / "tools/checkpoints/grim-grimmsnarl-phil-20260813/replays"
ARCHIVE = ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz"
DECK_PATH = ROOT / "decks/md_v1_grimmsnarl.csv"
BUDGET_S = 5.0
PARTICLES = 8
PER_STRATUM = 128
WORKERS = 8
THREADS = 2
PILOTS = {
    "grimmsnarl": {
        "name": "GrimmsnaRL", "submission_id": 55490217,
        "display_rank": 40, "display_score": 1068.8,
        "run": "grimmsnarl-turn-search-agreement-20260813",
    },
    "phil": {
        "name": "Phil_Hellmuth", "submission_id": 55454605,
        "display_rank": 62, "display_score": 1033.1,
        "run": "phil-hellmuth-turn-search-agreement-20260813",
    },
}


class DiagnosticError(RuntimeError):
    pass


def paths(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    run = ROOT / "tools/checkpoints" / config["run"]
    return run, run / "lock.json", run / "roots.jsonl", run / "result.json"


def read_deck() -> tuple[int, ...]:
    deck = tuple(int(x) for x in DECK_PATH.read_text().split())
    if len(deck) != 60 or deck_sha256(deck) != COMPARE.TARGET_DECK_SHA256:
        raise DiagnosticError("Dobi deck identity drifted")
    return deck


def replay_files() -> list[Path]:
    found = sorted(path for path in REPLAYS.glob("*.json") if path.stem.isdigit())
    if not found:
        raise DiagnosticError("replay directory is empty")
    return found


def eligible(obs: dict[str, Any], view: ObsView, logged: list[int] | None) -> bool:
    return bool(
        logged is not None and view.select_type == ST_MAIN
        and view.min_count == 1 and view.max_count == 1
        and 2 <= len(view.options) <= 24
        and isinstance(obs.get("search_begin_input"), str)
        and obs.get("search_begin_input")
        and isinstance(obs.get("remainingOverageTime"), (int, float))
        and float(obs["remainingOverageTime"]) >= 161.0
    )


def build_lock(config: dict[str, Any]) -> dict[str, Any]:
    run, lock_path, roots_path, result_path = paths(config)
    if any(path.exists() for path in (lock_path, roots_path, result_path)):
        raise DiagnosticError("diagnostic already locked or consumed")
    runtime, deck = load_frozen_runtime(ARCHIVE), read_deck()
    candidates = {"dobi_agrees": [], "dobi_disagrees": []}
    population = Counter()
    selected_replays: dict[str, str] = {}
    eligible_seats = 0
    seen: set[tuple[int, int, int]] = set()

    for path in replay_files():
        document = json.loads(path.read_text(encoding="utf-8"))
        names = (document.get("info") or {}).get("TeamNames") or ["?", "?"]
        registered = COMPARE.decks_from_document(document)
        episode = int((document.get("info") or {}).get("EpisodeId") or path.stem)
        for seat, seat_deck in registered.items():
            if names[seat] != config["name"]:
                continue
            if deck_sha256(seat_deck) != COMPARE.TARGET_DECK_SHA256:
                raise DiagnosticError(f"{config['name']} deck registration drifted")
            eligible_seats += 1
            selected_replays.setdefault(str(path.resolve()), sha256(path))
            policy = FrozenDobiV2Policy(runtime)
            for step, obs, raw in COMPARE.decisions(document, seat):
                key = (episode, seat, step)
                if key in seen:
                    continue
                seen.add(key)
                view = ObsView(obs)
                logged = valid(view, raw)
                if not eligible(obs, view, logged):
                    continue
                decision = policy.decide(obs, deck, seat)
                dobi = valid(view, decision.action)
                if dobi is None:
                    raise DiagnosticError("Dobi emitted invalid action")
                stratum = ("dobi_agrees" if semantic(view, logged) == semantic(view, dobi)
                           else "dobi_disagrees")
                population[stratum] += 1
                root_id = canonical({"episode": episode, "seat": seat, "step": step,
                                     "fingerprint": CFO.public_root_fingerprint(obs)})
                candidates[stratum].append({
                    "root_id": root_id, "episode_id": episode, "seat": seat,
                    "step": step, "stratum": stratum, "pilot_action": logged,
                    "dobi_action": dobi, "dobi_route": decision.route,
                    "observation": obs,
                })
            if policy.overlay_faults:
                raise DiagnosticError(f"overlay faults: {dict(policy.overlay_faults)}")

    selected = []
    for stratum in ("dobi_agrees", "dobi_disagrees"):
        rows = sorted(candidates[stratum], key=lambda row: row["root_id"])
        if len(rows) < PER_STRATUM:
            raise DiagnosticError(f"only {len(rows)} {stratum} roots")
        selected.extend(rows[:PER_STRATUM])
    selected.sort(key=lambda row: row["root_id"])
    run.mkdir(parents=True, exist_ok=True)
    fd = os.open(roots_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    artifact_paths = {
        "archive": ARCHIVE, "deck": DECK_PATH,
        "turn_search": ROOT / "agent/turn_search.py",
        "seat_policy": ROOT / "agent/seat_policy.py",
        "diagnostic": Path(__file__), "roots": roots_path,
    }
    lock: dict[str, Any] = {
        "schema": "ptcg.exact-grim-pilot-turn-search-agreement-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_search_results": True,
        "outcomes_read_for_selection": False,
        "pilot": {**{k: config[k] for k in
                       ("name", "submission_id", "display_rank", "display_score")},
                  "deck_sha256": COMPARE.TARGET_DECK_SHA256},
        "panel": {"selection": "ascending SHA-256 within agreement stratum",
                  "per_stratum": PER_STRATUM, "total": len(selected),
                  "population": dict(population), "eligible_seats": eligible_seats},
        "search": {"budget_s": BUDGET_S, "particles": PARTICLES,
                   "workers": WORKERS, "threads_requested": THREADS,
                   "opponent_binding": "per-particle public posterior"},
        "artifacts": {
            **{name: {"path": str(path.resolve()), "sha256": sha256(path)}
               for name, path in artifact_paths.items()},
            "selected_replays": selected_replays,
        },
        "interpretation": "agreement diagnostic only; no promotion authority",
        "promotion_authority": False, "upload_authority": False,
    }
    lock["lock_sha256"] = canonical(lock)
    write_new(lock_path, lock)
    return lock


def worker(spec: dict[str, Any]) -> list[dict[str, Any]]:
    from agent import model as RepoModel
    from agent import turn_search as TS
    config = PILOTS[spec["pilot"]]
    roots_path = paths(config)[2]
    roots = [json.loads(line) for line in roots_path.read_text().splitlines() if line]
    runtime, deck, net = load_frozen_runtime(ARCHIVE), read_deck(), RepoModel.load()
    output = []
    for row in roots[spec["index"]::spec["workers"]]:
        obs, seat = row["observation"], int(row["seat"])
        view = ObsView(obs)
        ours, theirs = FrozenDobiV2Policy(runtime), QuV2BasePolicy(runtime)
        table = (SeatPolicyTable().bind(seat, ours, deck)
                                  .bind(1 - seat, theirs, deck))
        started = time.monotonic()
        with TS.seat_context(table, seat):
            result = TS.analyze(view, net, list(deck), budget_s=BUDGET_S,
                                max_particles=PARTICLES)
        baseline = list(row["dobi_action"])
        chosen = (list(result.chosen_action)
                  if result is not None and result.chosen_action is not None
                  else baseline)
        if valid(view, chosen) is None:
            raise DiagnosticError("search emitted invalid action")
        pilot_sem = semantic(view, row["pilot_action"])
        dobi_sem, search_sem = semantic(view, baseline), semantic(view, chosen)
        override = search_sem != dobi_sem
        output.append({
            "root_id": row["root_id"], "stratum": row["stratum"],
            "reason": None if result is None else result.reason,
            "valid_particles": int(TS.last_stats.get("valid_particles", 0) or 0),
            "elapsed_s": time.monotonic() - started, "override": override,
            "pilot_search_agree": pilot_sem == search_sem,
            "fixed": pilot_sem != dobi_sem and pilot_sem == search_sem,
            "broken": pilot_sem == dobi_sem and pilot_sem != search_sem,
            "changed_but_neither": override and pilot_sem != dobi_sem and pilot_sem != search_sem,
            "overlay_faults": sum(ours.overlay_faults.values())
                              + sum(theirs.overlay_faults.values()),
        })
    return output


def run(pilot_key: str, config: dict[str, Any]) -> dict[str, Any]:
    _, lock_path, _, result_path = paths(config)
    if result_path.exists():
        raise DiagnosticError("result already exists")
    lock = json.loads(lock_path.read_text())
    if canonical({k: v for k, v in lock.items() if k != "lock_sha256"}) != lock["lock_sha256"]:
        raise DiagnosticError("lock self-hash mismatch")
    for name, row in lock["artifacts"].items():
        if name == "selected_replays":
            for path, digest in row.items():
                if sha256(Path(path)) != digest:
                    raise DiagnosticError(f"replay drifted: {path}")
        elif sha256(Path(row["path"])) != row["sha256"]:
            raise DiagnosticError(f"artifact drifted: {name}")

    started = time.monotonic()
    specs = [{"pilot": pilot_key, "index": i, "workers": WORKERS}
             for i in range(WORKERS)]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for part in pool.map(worker, specs):
            rows.extend(part)
    rows.sort(key=lambda row: row["root_id"])
    if len(rows) != 2 * PER_STRATUM or len({r["root_id"] for r in rows}) != len(rows):
        raise DiagnosticError("result coverage incomplete or duplicated")

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
    weight = population["dobi_agrees"] / total
    standardized = (weight * by_stratum["dobi_agrees"]["search_agreement_rate"]
                    + (1 - weight) * by_stratum["dobi_disagrees"]["search_agreement_rate"])
    se = math.sqrt(
        weight ** 2 * by_stratum["dobi_agrees"]["search_agreement_rate"]
        * (1 - by_stratum["dobi_agrees"]["search_agreement_rate"]) / PER_STRATUM
        + (1 - weight) ** 2 * by_stratum["dobi_disagrees"]["search_agreement_rate"]
        * (1 - by_stratum["dobi_disagrees"]["search_agreement_rate"]) / PER_STRATUM)
    delta = standardized - weight
    elapsed = sorted(r["elapsed_s"] for r in rows)
    reasons = Counter(r["reason"] if r["reason"] is not None else "no_result"
                      for r in rows)
    payload: dict[str, Any] = {
        "schema": "ptcg.exact-grim-pilot-turn-search-agreement-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"], "wall_seconds": time.monotonic() - started,
        "panel_roots": len(rows),
        "population": {**population, "eligible": total,
                       "frozen_dobi_pilot_agreement": weight,
                       "standardized_search_pilot_agreement": standardized,
                       "standardized_search_minus_dobi": delta,
                       "approx_delta_ci95": [delta - 1.96 * se, delta + 1.96 * se]},
        "by_stratum": by_stratum,
        "search": {"reasons": dict(reasons),
                   "coverage": statistics.fmean(float(r["valid_particles"] >= 5) for r in rows),
                   "overrides": sum(r["override"] for r in rows),
                   "override_rate": statistics.fmean(float(r["override"]) for r in rows),
                   "elapsed_mean_s": statistics.fmean(elapsed),
                   "elapsed_p95_s": elapsed[math.floor(.95 * (len(elapsed) - 1))]},
        "faults": {"overlay": sum(r["overlay_faults"] for r in rows)},
        "rows": rows, "promotion_authority": False, "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    write_new(result_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", choices=tuple(PILOTS), required=True)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    args = parser.parse_args()
    config = PILOTS[args.pilot]
    value = build_lock(config) if args.stage == "lock" else run(args.pilot, config)
    keys = (("lock_sha256", "pilot", "panel", "outcomes_read_for_selection")
            if args.stage == "lock"
            else ("population", "by_stratum", "search", "faults", "result_sha256"))
    print(json.dumps({key: value[key] for key in keys}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        print(f"DIAGNOSTIC ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
