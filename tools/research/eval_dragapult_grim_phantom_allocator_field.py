"""Current-field confirmation for the public-Grim scoped Phantom allocator."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from agent.obsview import ObsView  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_dragapult_route_guards as RG  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-grim-phantom-field-v1-20260812"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
DIRECT_RESULT = (
    ROOT / "tools/checkpoints/dragapult-grim-phantom-allocator-v1-20260812/result.json"
)
PACKAGE = ROOT / "submission-dragapult-completion-1-unsigned.tar.gz"
GAMES_PER_ARM = 2_048
SEED = 2_026_081_228
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
PUBLIC_GRIM_SIGNATURE = frozenset((646, 647, 648))


class GateError(RuntimeError):
    """The artifact, schedule, or runtime integrity contract failed."""


def canonical(value: Any) -> str:
    return RG.canonical(value)


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    RG.write_new(path, value)


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise GateError(f"missing artifact: {path}")
    return {"path": str(path.resolve()), "sha256": RG.BASE.file_sha256(path)}


def load_direct_result() -> dict[str, Any]:
    value = json.loads(DIRECT_RESULT.read_text(encoding="utf-8"))
    claimed = value.pop("result_sha256", None)
    if (
        value.get("schema")
        != "ptcg.dragapult-grim-phantom-allocator-result.v1"
        or claimed != canonical(value)
        or value.get("decision", {}).get("valid") is not True
        or value.get("decision", {}).get("passed") is not True
    ):
        raise GateError("passing direct Grim allocator result is absent")
    value["result_sha256"] = claimed
    return value


def _public_id(entry: object) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("id")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def opponent_has_public_grim_signature(view: ObsView) -> bool:
    opponent = view.opp
    if not isinstance(opponent, Mapping):
        return False
    return any(
        _public_id(entry) in PUBLIC_GRIM_SIGNATURE
        for zone in ("active", "bench")
        for entry in (opponent.get(zone) or ())
    )


class ScopedController:
    """Enable the validated allocator only after a public Grim reveal."""

    def __init__(self, main, card, qu, deck, name: str, *, candidate: bool):
        self.inner = RG.GuardedController(
            main, card, qu, deck, name,
            use_energy_guard=False,
            use_boss_guard=False,
            use_phantom_guard=False,
            use_completion_guard=True,
        )
        self.candidate = candidate
        self.scope: Counter[str] = Counter()

    def act(self, obs: dict) -> list[int]:
        eligible = opponent_has_public_grim_signature(ObsView(obs))
        self.scope["calls"] += 1
        self.scope["eligible_calls"] += int(eligible)
        self.inner.use_phantom_guard = bool(self.candidate and eligible)
        return self.inner.act(obs)

    def diagnostics(self) -> dict[str, Any]:
        return {**self.inner.diagnostics(), "scope": dict(self.scope)}


def clean(value: dict[str, Any], *, candidate: bool) -> bool:
    scope = value.get("scope", {})
    return bool(
        value.get("calls")
        == value.get("main_routes", 0)
        + value.get("card_routes", 0)
        + value.get("qu_routes", 0)
        and value.get("main_routes", 0) > 0
        and value.get("card_routes", 0) > 0
        and value.get("completion_guards", 0) > 0
        and scope.get("calls") == value.get("calls")
        and scope.get("eligible_calls", 0) > 0
        and (value.get("phantom_guards", 0) > 0 if candidate else True)
        and (value.get("phantom_guards", 0) == 0 if not candidate else True)
        and value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("field gate already locked or consumed")
    direct = load_direct_result()
    deck = RG.load_deck()
    qu = RG.COMMON._load_net(RG.PATHS["qu"], "frozen Qu-v2B")
    field = RG.FIELD.current_field()
    opponents, _ = RG.BASE.make_opponents(field, qu, "grim-allocator-field-lock")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    paths = {
        "evaluator": Path(__file__).resolve(),
        "controller": Path(RG.__file__).resolve(),
        "policy": RG.PATHS["guards"], "tests": RG.PATHS["tests"],
        "main": RG.PATHS["main"], "card": RG.PATHS["card"],
        "qu": RG.PATHS["qu"], "deck": RG.PATHS["deck"],
        "package": PACKAGE, "direct_result": DIRECT_RESULT,
    }
    payload = {
        "schema": "ptcg.dragapult-grim-phantom-field-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "candidate": (
            "exact dragapult-v2 plus Phantom dead-target allocator only after "
            "a public opposing Impidimp/Morgrem/Grimmsnarl reveal"
        ),
        "control": "exact dragapult-v2",
        "direct_result_sha256": direct["result_sha256"],
        "public_grim_signature": sorted(PUBLIC_GRIM_SIGNATURE),
        "protocol": {
            "games_per_arm": GAMES_PER_ARM, "seed": SEED,
            "identical_schedule": True,
            "overall": "point delta >= 0 and CI95 lower > -0.01",
            "grim_slice": "point delta >= 0 and CI95 lower > -0.075",
            "dragapult_lucario_guards": "each point delta >= -0.025",
            "zero_faults": True, "one_schedule_one_attempt": True,
        },
        "schedule_manifest_sha256": canonical(schedule_manifest(schedule, opponents)),
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "deck": list(deck),
        "promotion_authority": False, "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if (
        value.get("schema") != "ptcg.dragapult-grim-phantom-field-lock.v1"
        or claimed != canonical(value)
    ):
        raise GateError("field lock schema or self-hash failed")
    value["lock_sha256"] = claimed
    for name, row in value["artifacts"].items():
        path = Path(row["path"])
        if not path.is_file() or RG.BASE.file_sha256(path) != row["sha256"]:
            raise GateError(f"locked artifact drifted: {name}")
    return value


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("field attempt already consumed")
    attempt = {
        "schema": "ptcg.dragapult-grim-phantom-field-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = canonical(attempt)
    write_new(ATTEMPT, attempt)

    deck = tuple(lock["deck"])
    qu = RG.COMMON._load_net(RG.PATHS["qu"], "frozen Qu-v2B")
    main = RG.COMMON._load_net(RG.PATHS["main"], "Dragapult elite MAIN")
    card = RG.COMMON._load_net(RG.PATHS["card"], "Dragapult elite CARD")
    field = RG.FIELD.current_field()
    series, diagnostics = {}, {}
    for arm in ("candidate", "control"):
        opponents, field_controller = RG.BASE.make_opponents(
            field, qu, f"grim-allocator-{arm}-field",
        )
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if canonical(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise GateError(f"runtime schedule drifted: {arm}")
        policy = ScopedController(
            main, card, qu, deck, f"dragapult/grim-allocator/{arm}",
            candidate=arm == "candidate",
        )
        value = EVAL.run_series(
            f"dragapult-grim-allocator-field/{arm}", policy, deck,
            opponents, schedule, max_selects=MAX_SELECTS,
            time_bank_s=TIME_BANK_S, verbose=not quiet,
        )
        learner, field_diag = policy.diagnostics(), field_controller.diagnostics()
        valid = bool(
            len(value.records) == GAMES_PER_ARM and value.gate_valid
            and clean(learner, candidate=arm == "candidate")
            and field_diag.get("fallbacks", 0) == 0
            and field_diag.get("exceptions") == {}
        )
        series[arm] = value
        diagnostics[arm] = {"valid": valid, "learner": learner, "field": field_diag}

    candidate, control = series["candidate"], series["control"]
    overall = RG.STATS.paired_delta_ci(candidate.records, control.records)
    slices = {
        name: RG.STATS.paired_delta_ci(
            RG.slice_records(candidate.records, name),
            RG.slice_records(control.records, name),
        ) for name in ("Grimmsnarl", "Dragapult", "Mega Lucario")
    }
    valid = all(row["valid"] for row in diagnostics.values())
    passed = bool(
        valid and overall["mean_delta"] >= 0.0 and overall["ci95"][0] > -0.01
        and slices["Grimmsnarl"]["mean_delta"] >= 0.0
        and slices["Grimmsnarl"]["ci95"][0] > -0.075
        and all(slices[name]["mean_delta"] >= -0.025 for name in ("Dragapult", "Mega Lucario"))
    )
    payload = {
        "schema": "ptcg.dragapult-grim-phantom-field-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid, "passed": passed,
            "candidate_minus_control": overall, "slices": slices,
            "phantom_guard_fires": diagnostics["candidate"]["learner"].get(
                "phantom_guards", 0
            ),
        },
        "summaries": {name: value.summary() for name, value in series.items()},
        "diagnostics": diagnostics,
        "records": {
            name: [asdict(row) for row in value.records] for name, value in series.items()
        },
        "promotion_authority": passed, "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = canonical(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock(); write_new(LOCK, value)
            print(json.dumps({
                "lock_sha256": value["lock_sha256"], "protocol": value["protocol"],
            }, indent=2, sort_keys=True))
            return 0
        value = run(args.quiet)
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": value["decision"], "summaries": value["summaries"],
        "result_sha256": value["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if value["decision"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
