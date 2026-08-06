"""Locked four-package direct screens for scoped Lucario/Grim ST_MAIN."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import md_v2_card, model, policy, qu_v2_features as QF, safety  # noqa: E402
from agent.obsview import ObsView, ST_CARD, ST_MAIN  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_lucario_majkel_bc_v1_same_deck as BASE  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import lock_lucario_grim_main_v1 as LOCKER  # noqa: E402
from tools.rl_env import OpponentSpec, build_paired_schedule, environment_manifest, schedule_manifest  # noqa: E402


RUN = LOCKER.RUN / "direct-discovery"
LOCK = RUN / "lock.json"
PACKAGES = (
    "conservative-card-off", "conservative-card-on",
    "focused-card-off", "focused-card-on",
)
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "training_lock": LOCKER.OUTPUT,
    "behavioral_report": LOCKER.RUN / "behavioral-report.json",
    "conservative": LOCKER.RUN / "conservative/model/candidate-qu-v2a-weights.npz",
    "focused": LOCKER.RUN / "focused/model/candidate-qu-v2a-weights.npz",
    "lucario_card": LOCKER.LUCARIO / "card/model/candidate-qu-v2a-weights.npz",
    "generic": ROOT / "agent/weights.npz",
    "lucario_deck": LOCKER.DECK,
    "dobi_main": ROOT / (
        "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
        "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
    ),
    "dobi_card": ROOT / "agent/md_v2_card_weights.npz",
    "dobi_deck": ROOT / "decks/md_v1_grimmsnarl.csv",
    "prior_direct": LOCKER.DIRECT_FAILURE,
}
GAMES = 2_560
SEED = 2_026_080_581
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


class ScopedController:
    def __init__(self, main_net, card_net, generic_net, name: str, deck: Sequence[int]):
        self.main_net = main_net; self.card_net = card_net; self.generic_net = generic_net
        self.name = name; self.deck = tuple(map(int, deck)); self.target = tuple(sorted(self.deck))
        self.calls = self.scoped_main_routes = self.card_routes = self.generic_routes = 0
        self.off_deck_routes = self.fallbacks = self.repairs = 0
        self.exceptions = Counter(); self.select_types = Counter(); self.latency_ms = []

    def act(self, obs: dict, registered_deck: Sequence[int] | None = None) -> list[int]:
        started = time.monotonic(); self.calls += 1
        registration = self.deck if registered_deck is None else tuple(map(int, registered_deck))
        try:
            view = ObsView(obs); self.select_types[str(view.select_type)] += 1
            exact = tuple(sorted(registration)) == self.target
            if not view.options: raise ValueError("empty option menu")
            scoped_main = (
                exact and view.select_type == ST_MAIN
                and md_v2_card.opponent_has_public_grim_signature(view)
            )
            if scoped_main:
                net = self.main_net; self.scoped_main_routes += 1
            elif exact and view.select_type == ST_CARD and self.card_net is not None:
                net = self.card_net; self.card_routes += 1
            else:
                net = self.generic_net; self.generic_routes += 1
                self.off_deck_routes += int(not exact and view.select_type in (ST_MAIN, ST_CARD))
            sample = QF.encode_public_observation(obs, registration)
            logits, _ = net.forward(sample)
            action = model.decode_qu_v2(logits, len(view.options), view.min_count, view.max_count)
        except Exception as error:
            self.exceptions[type(error).__name__] += 1; self.fallbacks += 1
            try: action = policy.decide_rules(obs)
            except Exception: action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        self.repairs += int(not BASE.same_action(repaired, action))
        self.latency_ms.append((time.monotonic() - started) * 1000.0)
        return repaired

    def diagnostics(self) -> dict:
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        return {
            "name": self.name, "calls": self.calls,
            "scoped_main_routes": self.scoped_main_routes,
            "card_routes": self.card_routes, "generic_routes": self.generic_routes,
            "off_deck_routes": self.off_deck_routes, "fallbacks": self.fallbacks,
            "repairs": self.repairs, "exceptions": dict(self.exceptions),
            "select_types": dict(self.select_types),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
            },
        }


def locations(package: str) -> tuple[Path, Path]:
    directory = RUN / package
    return directory / "attempt.json", directory / "result.json"


def opponents(deck, move) -> list[OpponentSpec]:
    return [OpponentSpec(
        key="grimmsnarl/frozen-dobi-v1", deck=tuple(deck), move=move,
        policy_id=(f"main:{BASE.sha256(PATHS['dobi_main'])}+card:{BASE.sha256(PATHS['dobi_card'])}+qu:{BASE.sha256(PATHS['generic'])}"),
        schedule_group="direct/frozen-dobi-v1-grimmsnarl",
    )]


def noop(_obs: dict, _rng) -> list[int]: return [0]


def build_lock() -> dict:
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing: raise BASE.GateError(f"bound artifacts missing: {missing}")
    behavior = json.loads(PATHS["behavioral_report"].read_text(encoding="utf-8"))
    prior = json.loads(PATHS["prior_direct"].read_text(encoding="utf-8"))
    if behavior.get("gameplay_authority") is not True or prior.get("decision", {}).get("valid") is not True:
        raise BASE.GateError("prior evidence contract failed")
    lucario_deck = BASE.read_deck(PATHS["lucario_deck"]); grim_deck = COMMON.read_deck(PATHS["dobi_deck"])
    opponent_specs = opponents(grim_deck, noop)
    schedule = build_paired_schedule(opponent_specs, GAMES, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES // 2, 1: GAMES // 2}: raise BASE.GateError("schedule is not seat balanced")
    payload = {
        "schema": "ptcg.lucario-grim-main-v1.direct-discovery-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_any_discovery_outcomes": True,
        "packages": list(PACKAGES),
        "artifacts": {name: {"path": str(path.resolve()), "sha256": BASE.sha256(path)} for name, path in PATHS.items()},
        "protocol": {
            "games_per_package": GAMES, "pairs_per_package": GAMES // 2, "seed": SEED,
            "same_schedule_for_all_packages": True,
            "candidate_seat_counts": {str(key): value for key, value in sorted(seats.items())},
            "opponent": "complete frozen Dobi-v1/Grimmsnarl",
            "main_scope": "exact Lucario ST_MAIN only after public Grimmsnarl signature",
            "card_factor": "confirmed Lucario ST_CARD on versus generic Qu-v2B off",
            "selection": "highest valid score; ties prefer conservative then card-off",
            "confirmation_eligible": "selected point estimate at least 0.45 with CI95 lower bound above 0.40",
            "authority": "discovery only; any selected package requires a fresh 5120-game confirmation",
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(schedule_manifest(schedule, opponent_specs)),
        "promotion_authority": False, "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload); return payload


def load_lock() -> dict:
    payload = json.loads(LOCK.read_text(encoding="utf-8")); claimed = payload.pop("lock_sha256", None)
    if payload.get("schema") != "ptcg.lucario-grim-main-v1.direct-discovery-lock.v1" or claimed != COMMON.canonical_sha256(payload):
        raise BASE.GateError("discovery lock identity drifted")
    payload["lock_sha256"] = claimed
    for name, record in payload["artifacts"].items():
        path = PATHS[name]
        if Path(record["path"]).resolve() != path.resolve() or record["sha256"] != BASE.sha256(path):
            raise BASE.GateError(f"bound artifact drift: {name}")
    return payload


def clean_dobi(value: dict) -> bool:
    return value.get("fallbacks") == 0 and value.get("repairs") == 0 and value.get("exceptions") == {} and value.get("off_deck_main_routes") == 0 and value.get("off_deck_card_routes") == 0


def run(package: str, quiet: bool) -> dict:
    lock = load_lock(); attempt, result = locations(package)
    if attempt.exists() or result.exists(): raise BASE.GateError("package attempt already consumed")
    arm, card_state = package.rsplit("-card-", 1)
    lucario_deck = BASE.read_deck(PATHS["lucario_deck"]); grim_deck = COMMON.read_deck(PATHS["dobi_deck"])
    generic = COMMON._load_net(PATHS["generic"], "Qu-v2B")
    candidate = ScopedController(
        COMMON._load_net(PATHS[arm], f"{arm} ST_MAIN"),
        COMMON._load_net(PATHS["lucario_card"], "Lucario ST_CARD") if card_state == "on" else None,
        generic, package, lucario_deck,
    )
    control = LAYERED.LayeredMirrorCardController(
        COMMON._load_net(PATHS["dobi_main"], "Dobi ST_MAIN"),
        COMMON._load_net(PATHS["dobi_card"], "Dobi ST_CARD"), generic,
        "frozen-dobi-v1", grim_deck,
    )
    opponent_specs = opponents(grim_deck, control.opponent_move)
    schedule = build_paired_schedule(opponent_specs, GAMES, seed=SEED)
    if COMMON.canonical_sha256(schedule_manifest(schedule, opponent_specs)) != lock["schedule_manifest_sha256"]:
        raise BASE.GateError("runtime schedule drifted")
    BASE.write_new(attempt, {"schema": f"ptcg.lucario-grim-main-v1.{package}-attempt.v1", "created_at": datetime.now(timezone.utc).isoformat(), "written_before_first_outcome": True, "lock_sha256": lock["lock_sha256"]})
    series = EVAL.run_series(package, candidate, lucario_deck, opponent_specs, schedule, max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet)
    cdiag = candidate.diagnostics(); odiag = control.diagnostics(); low, high = series.ci95
    valid = len(series.records) == GAMES and series.gate_valid and BASE.clean(cdiag) and clean_dobi(odiag)
    payload = {
        "schema": f"ptcg.lucario-grim-main-v1.{package}-result.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"], "package": package,
        "decision": {"valid": valid, "confirmation_eligible": bool(valid and series.score >= 0.45 and low > 0.40), "score": series.score, "wilson_ci95": [low, high]},
        "summary": series.summary(), "records": [asdict(row) for row in series.records],
        "controllers": {"lucario": cdiag, "dobi_grim": odiag},
        "environment": environment_manifest(lucario_deck, opponent_specs, str(PATHS["lucario_deck"])),
        "promotion_authority": False, "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload); BASE.write_new(result, payload); return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--lock-only", action="store_true"); parser.add_argument("--package", choices=PACKAGES); parser.add_argument("--quiet", action="store_true"); args = parser.parse_args()
    try:
        if args.lock_only:
            payload = build_lock(); BASE.write_new(LOCK, payload); print(json.dumps({"lock": str(LOCK), "lock_sha256": payload["lock_sha256"], "protocol": payload["protocol"], "packages": payload["packages"]}, indent=2, sort_keys=True)); return 0
        if args.package is None: raise BASE.GateError("--package required")
        payload = run(args.package, args.quiet)
    except (BASE.GateError, OSError, KeyError, TypeError, ValueError) as error: parser.error(str(error))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
