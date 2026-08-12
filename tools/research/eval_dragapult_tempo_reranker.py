"""Paired current-field screen for the Dragapult tempo family residual."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent import dragapult_bc as D, dragapult_tempo as T  # noqa: E402
from agent import model, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import ST_CARD, ST_MAIN, ObsView  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.research import train_dragapult_tempo_reranker as TRAIN  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = TRAIN.RUN / "gameplay-screen-v1"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
GAMES_PER_ARM = 512
SEED = 2_026_081_242
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
KEY_SLICES = ("Dragapult", "Mega Lucario", "Grimmsnarl")
PATHS = {
    "qu": RG.PATHS["qu"], "deck": RG.PATHS["deck"],
    "main": RG.PATHS["main"], "card": RG.PATHS["card"],
    "weights": TRAIN.WEIGHTS, "training_lock": TRAIN.LOCK,
    "training_result": TRAIN.RESULT, "tempo": Path(T.__file__).resolve(),
    "policy": Path(D.__file__).resolve(), "evaluator": Path(__file__).resolve(),
}


class GateError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return BASE.canonical_sha256(value)


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


class TempoController:
    def __init__(self, main, card, qu, deck, weights, name, enabled):
        self.main, self.card, self.qu = main, card, qu
        self.deck, self.weights = tuple(deck), weights
        self.name, self.enabled = name, enabled
        self.counts: Counter[str] = Counter()
        self.exceptions: Counter[str] = Counter()
        self.latency_ms: list[float] = []

    def act(self, obs: dict) -> list[int]:
        started = time.monotonic()
        self.counts["calls"] += 1
        try:
            view = ObsView(obs)
            if not view.options:
                raise ValueError("empty option menu")
            sample = FEATURES.encode_public_observation(obs, self.deck)
            if view.select_type == ST_MAIN:
                self.counts["main_routes"] += 1
                logits, _ = self.main.forward(sample)
                base = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
                action = T.rerank_main_family(
                    view, logits, base, self.weights,
                ) if self.enabled else base
                self.counts["tempo_roots"] += int(T.high_impact_main_root(view))
                self.counts["tempo_reranks"] += int(action != base)
                completed = D._guard_phantom_completion(view, action)
                self.counts["completion_guards"] += int(completed != action)
                action = completed
            elif view.select_type == ST_CARD:
                self.counts["card_routes"] += 1
                logits, _ = self.card.forward(sample)
                action = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
            else:
                self.counts["qu_routes"] += 1
                logits, _ = self.qu.forward(sample)
                action = model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                )
        except Exception as error:
            self.counts["fallbacks"] += 1
            self.exceptions[type(error).__name__] += 1
            action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        self.counts["repairs"] += int(list(repaired) != list(action))
        self.latency_ms.append((time.monotonic() - started) * 1_000.0)
        return repaired

    def diagnostics(self):
        latency = np.asarray(self.latency_ms, dtype=np.float64)
        counts = dict(self.counts)
        counts.setdefault("fallbacks", 0); counts.setdefault("repairs", 0)
        counts.setdefault("tempo_reranks", 0); counts.setdefault("completion_guards", 0)
        return {
            "name": self.name, **counts, "exceptions": dict(self.exceptions),
            "latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
            },
        }


def load_deck():
    deck = tuple(int(value) for value in PATHS["deck"].read_text().split())
    if len(deck) != 60 or not D.supports_deck(deck):
        raise GateError("Dragapult deck identity drifted")
    return deck


def build_lock():
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("tempo gameplay screen already locked or consumed")
    missing = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing:
        raise GateError(f"missing artifacts: {missing}")
    training = json.loads(TRAIN.RESULT.read_text(encoding="utf-8"))
    if not training.get("behavior_eligible"):
        raise GateError("tempo residual failed its behavior gate")
    deck = load_deck()
    qu = COMMON._load_net(PATHS["qu"], "frozen Qu-v2B")
    field = FIELD.current_field()
    opponents, _ = BASE.make_opponents(field, qu, "dragapult-tempo-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    payload = {
        "schema": "ptcg.dragapult-tempo-reranker.gameplay-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "candidate": "current Dragapult-v2 plus tempo family residual",
        "control": "exact current Dragapult-v2",
        "protocol": {
            "games_per_arm": GAMES_PER_ARM, "seed": SEED,
            "identical_schedule": True,
            "screen": "delta > 0; CI95 lower > -0.03; key slices >= -0.05",
            "key_slices": list(KEY_SLICES), "zero_faults": True,
            "fresh_confirmation_required": True,
        },
        "schedule_manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
        "deck": list(deck),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}
            for name, path in PATHS.items()
        },
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock():
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if value.get("schema") != "ptcg.dragapult-tempo-reranker.gameplay-lock.v1" or claimed != canonical(value):
        raise GateError("tempo gameplay lock self-hash failed")
    value["lock_sha256"] = claimed
    for row in value["artifacts"].values():
        path = Path(row["path"])
        if not path.is_file() or BASE.file_sha256(path) != row["sha256"]:
            raise GateError(f"artifact drifted: {path}")
    return value


def slice_records(records, archetype):
    return [row for row in records if row.opponent_key.split("/", 1)[0] == archetype]


def run(quiet: bool):
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("tempo gameplay attempt already consumed")
    write_new(ATTEMPT, {
        "schema": "ptcg.dragapult-tempo-reranker.gameplay-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    deck = tuple(lock["deck"])
    qu = COMMON._load_net(Path(lock["artifacts"]["qu"]["path"]), "Qu-v2B")
    main = COMMON._load_net(Path(lock["artifacts"]["main"]["path"]), "elite MAIN")
    card = COMMON._load_net(Path(lock["artifacts"]["card"]["path"]), "elite CARD")
    weights = load_weights(Path(lock["artifacts"]["weights"]["path"]))
    field = FIELD.current_field()
    series, diagnostics = {}, {}
    for arm in ("candidate", "control"):
        opponents, field_controller = BASE.make_opponents(field, qu, f"tempo-{arm}")
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise GateError(f"schedule drifted: {arm}")
        controller = TempoController(
            main, card, qu, deck, weights, f"tempo/{arm}", arm == "candidate",
        )
        value = EVAL.run_series(
            f"tempo/{arm}", controller, deck, opponents, schedule,
            max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
        )
        learner = controller.diagnostics(); field_diag = field_controller.diagnostics()
        clean = bool(
            len(value.records) == GAMES_PER_ARM and value.gate_valid
            and learner["fallbacks"] == 0 and learner["repairs"] == 0
            and learner["exceptions"] == {} and field_diag.get("fallbacks") == 0
            and field_diag.get("exceptions") == {}
        )
        series[arm] = value
        diagnostics[arm] = {"valid": clean, "learner": learner, "field": field_diag}
    candidate, control = series["candidate"], series["control"]
    overall = STATS.paired_delta_ci(candidate.records, control.records)
    slices = {
        name: STATS.paired_delta_ci(
            slice_records(candidate.records, name), slice_records(control.records, name),
        ) for name in KEY_SLICES
    }
    valid = all(row["valid"] for row in diagnostics.values()) and diagnostics["candidate"]["learner"]["tempo_reranks"] > 0
    passed = bool(
        valid and overall["mean_delta"] > 0 and overall["ci95"][0] > -0.03
        and all(row["mean_delta"] >= -0.05 for row in slices.values())
    )
    payload = {
        "schema": "ptcg.dragapult-tempo-reranker.gameplay-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {"valid": valid, "passed": passed, "overall": overall, "key_slices": slices},
        "summaries": {arm: value.summary() for arm, value in series.items()},
        "diagnostics": diagnostics,
        "records": {arm: [asdict(row) for row in value.records] for arm, value in series.items()},
        "promotion_authority": False, "package_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    write_new(RESULT, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock(); write_new(LOCK, value)
            print(json.dumps({"lock_sha256": value["lock_sha256"], "protocol": value["protocol"]}, sort_keys=True))
            return 0
        value = run(args.quiet)
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"decision": value["decision"], "summaries": value["summaries"], "result_sha256": value["result_sha256"]}, sort_keys=True))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
