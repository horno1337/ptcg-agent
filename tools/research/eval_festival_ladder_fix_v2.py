"""Prospectively lock and evaluate the three Festival ladder fixes against v1."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import festival_lead as V2_RULES, model, qu_v2_features as FEATURES, safety  # noqa: E402
from agent.obsview import CTX_TO_HAND, ST_CARD, ST_MAIN  # noqa: E402
from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_dobi_v1_elite_teacher_card_v1_field as FIELD  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as FEST  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule  # noqa: E402


RUN = ROOT / "tools/checkpoints/festival-ladder-fix-v2-safe"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
V1_ARCHIVE = ROOT / "submission-festival-lead-bc-v1-experimental-unsigned.tar.gz"
GAMES_PER_ARM = 1_024
SEED = 2_026_080_88
NONINFERIORITY_MARGIN = -0.03
V1_ARCHIVE_SHA256 = "03f3f7cc1bd03f29e29c332cd518d37277dbe8bca3dbaa6c606f5df3c702aadd"


class GateError(RuntimeError):
    pass


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


class Controller:
    def __init__(self, rules, view_type, main, card, deck, name: str, *, guards: bool):
        self.rules, self.view_type = rules, view_type
        self.main, self.card, self.deck, self.name = main, card, tuple(deck), name
        self.guards = guards
        self.counts = Counter()
        self.exceptions = Counter()

    def act(self, obs):
        self.counts["calls"] += 1
        try:
            view = self.view_type(obs)
            action = None
            if self.guards:
                action = self.rules.hybrid_main_override(view)
                if action is not None:
                    selected = view.options[int(action[0])]
                    card_id = view.semantic_option_card_id(selected)
                    self.counts["boss_main_overrides" if card_id == self.rules.BOSS
                                else "backup_energy_overrides"] += 1
                if action is None:
                    action = self.rules.boss_target_override(view)
                    self.counts["boss_target_overrides"] += int(action is not None)
            thwackey = (
                view.select_type == ST_CARD and view.context == CTX_TO_HAND
                and view.effect_card_id == self.rules.THWACKEY
            )
            if action is None:
                if view.select_type == ST_MAIN:
                    net = self.main
                    self.counts["main_routes"] += 1
                elif view.select_type == ST_CARD and not thwackey:
                    net = self.card
                    self.counts["card_routes"] += 1
                else:
                    net = None
                    self.counts["rule_routes"] += 1
                    self.counts["thwackey_routes"] += int(thwackey)
                if net is None:
                    action = self.rules.decide(view, self.deck)
                else:
                    sample = FEATURES.encode_public_observation(obs, self.deck)
                    logits, _ = net.forward(sample)
                    action = model.decode_qu_v2(
                        logits, len(view.options), view.min_count, view.max_count,
                    )
            if action is None:
                raise ValueError("Festival route returned None")
        except Exception as error:
            self.counts["fallbacks"] += 1
            self.exceptions[type(error).__name__] += 1
            action = safety._fallback(obs)
        repaired = safety._repair(action, obs)
        self.counts["repairs"] += int(list(repaired) != list(action))
        return repaired

    def diagnostics(self) -> dict[str, Any]:
        return {"name": self.name, **dict(self.counts), "exceptions": dict(self.exceptions)}


def clean(value: Mapping[str, Any]) -> bool:
    return (
        value.get("calls", 0) > 0
        and value.get("fallbacks", 0) == 0
        and value.get("repairs", 0) == 0
        and value.get("exceptions") == {}
        and value.get("main_routes", 0) > 0
        and value.get("card_routes", 0) > 0
        and value.get("rule_routes", 0) > 0
        and value.get("thwackey_routes", 0) > 0
    )


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise GateError("v2 evaluation is already locked or consumed")
    if COMMON.file_sha256(V1_ARCHIVE) != V1_ARCHIVE_SHA256:
        raise GateError("Festival v1 archive drifted")
    deck = FEST.read_festival_deck(FEST.DECK)
    qu = COMMON._load_net(FEST.PARENT_WEIGHTS, "Qu-v2B")
    _snapshot, rows = FIELD.load_snapshot()
    expanded = FIELD.expand_field_rows(rows)
    opponents, _ = FEST.make_opponents(expanded, qu)
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise GateError("schedule is not seat balanced")
    artifacts = {
        "v1_archive": V1_ARCHIVE,
        "rules": ROOT / "agent/festival_lead.py",
        "hybrid": ROOT / "agent/festival_lead_bc.py",
        "tests": ROOT / "tests/test_festival_lead.py",
        "main_weights": FEST.MAIN_WEIGHTS,
        "card_weights": FEST.CARD_WEIGHTS,
        "parent_weights": FEST.PARENT_WEIGHTS,
        "deck": FEST.DECK,
        "field_snapshot": FIELD.FIELD_SNAPSHOT,
        "field_inventory": FIELD.FIELD_INVENTORY,
        "grim_variants": FIELD.GRIM_VARIANT_SNAPSHOT,
        "evaluator": Path(__file__).resolve(),
    }
    payload: dict[str, Any] = {
        "schema": "ptcg.festival-lead.ladder-fix-v2.lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_outcomes": True,
        "changes": [
            "do not search a second Festival Grounds already held",
            "power a benched Dipplin after the active Festival attacker is ready",
            "play and target Boss only for a visible one-hit Prize improvement",
        ],
        "control": "exact festival-test-1 archive behavior",
        "candidate": "same frozen BC weights plus the three ladder guards",
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "schedule_seed": SEED,
            "paired_identical_schedule": True,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "pass": "zero faults, all routes and a v2 main guard exercised, mean delta >= 0 and CI95 lower >= margin",
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256([
            asdict(row)
            for row in schedule
        ]),
        "artifacts": {
            key: {"path": str(path.resolve()), "sha256": COMMON.file_sha256(path)}
            for key, path in artifacts.items()
        },
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = value.pop("lock_sha256", None)
    if claimed != COMMON.canonical_sha256(value):
        raise GateError("lock self-hash failed")
    value["lock_sha256"] = claimed
    for descriptor in value["artifacts"].values():
        path = Path(descriptor["path"])
        if COMMON.file_sha256(path) != descriptor["sha256"]:
            raise GateError(f"bound artifact drifted: {path}")
    return value


def run(lock: Mapping[str, Any]) -> dict[str, Any]:
    if ATTEMPT.exists() or RESULT.exists():
        raise GateError("v2 evaluation attempt already consumed")
    attempt = {
        "schema": "ptcg.festival-lead.ladder-fix-v2.attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = COMMON.canonical_sha256(attempt)
    write_new(ATTEMPT, attempt)
    with tempfile.TemporaryDirectory(prefix="festival-ladder-fix-v2-") as temporary:
        root = Path(temporary)
        with tarfile.open(V1_ARCHIVE, "r:gz") as archive:
            archive.extractall(root, filter="data")
        (root / "agent").rename(root / "v1agent")
        sys.path.insert(0, str(root))
        try:
            v1_rules = importlib.import_module("v1agent.festival_lead")
            v1_view = importlib.import_module("v1agent.obsview").ObsView
            main = COMMON._load_net(FEST.MAIN_WEIGHTS, "Festival main")
            card = COMMON._load_net(FEST.CARD_WEIGHTS, "Festival card")
            qu = COMMON._load_net(FEST.PARENT_WEIGHTS, "Qu-v2B")
            deck = FEST.read_festival_deck(FEST.DECK)
            _snapshot, rows = FIELD.load_snapshot()
            expanded = FIELD.expand_field_rows(rows)
            left_opponents, left_field = FEST.make_opponents(expanded, qu)
            right_opponents, right_field = FEST.make_opponents(expanded, qu)
            left_schedule = build_paired_schedule(left_opponents, GAMES_PER_ARM, seed=SEED)
            right_schedule = build_paired_schedule(right_opponents, GAMES_PER_ARM, seed=SEED)
            manifest = COMMON.canonical_sha256([
                asdict(row)
                for row in left_schedule
            ])
            if manifest != lock["schedule_manifest_sha256"]:
                raise GateError("runtime schedule drifted")
            v1 = Controller(v1_rules, v1_view, main, card, deck, "festival-v1", guards=False)
            v2 = Controller(V2_RULES, V2_RULES.ObsView, main, card, deck,
                            "festival-ladder-fix-v2", guards=True)
            control = EVAL.run_series(
                "festival-v1", v1, deck, left_opponents, left_schedule,
                max_selects=5000, time_bank_s=600.0, verbose=False,
            )
            candidate = EVAL.run_series(
                "festival-ladder-fix-v2", v2, deck, right_opponents, right_schedule,
                max_selects=5000, time_bank_s=600.0, verbose=False,
            )
        finally:
            sys.path.pop(0)
    comparison = FEST.paired_delta_ci(candidate.records, control.records)
    control_diag, candidate_diag = v1.diagnostics(), v2.diagnostics()
    valid = bool(
        control.gate_valid and candidate.gate_valid
        and len(control.records) == len(candidate.records) == GAMES_PER_ARM
        and clean(control_diag) and clean(candidate_diag)
        and candidate_diag.get("backup_energy_overrides", 0)
             + candidate_diag.get("boss_main_overrides", 0) > 0
        and left_field.diagnostics().get("fallbacks") == 0
        and right_field.diagnostics().get("fallbacks") == 0
    )
    passed = bool(
        valid and comparison["mean_delta"] >= 0
        and comparison["ci95"][0] >= NONINFERIORITY_MARGIN
    )
    payload: dict[str, Any] = {
        "schema": "ptcg.festival-lead.ladder-fix-v2.result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {"valid": valid, "passed": passed,
                     "noninferiority_margin": NONINFERIORITY_MARGIN},
        "control": control.summary(), "candidate": candidate.summary(),
        "candidate_minus_control": comparison,
        "controllers": {"control": control_diag, "candidate": candidate_diag,
                        "control_field": left_field.diagnostics(),
                        "candidate_field": right_field.diagnostics()},
        "records": {"control": [asdict(row) for row in control.records],
                    "candidate": [asdict(row) for row in candidate.records]},
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    write_new(RESULT, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.stage == "lock":
            value = build_lock()
            write_new(LOCK, value)
            print(json.dumps({"lock_sha256": value["lock_sha256"]}, sort_keys=True))
        else:
            value = run(load_lock())
            print(json.dumps({"decision": value["decision"],
                              "control_score": value["control"]["score"],
                              "candidate_score": value["candidate"]["score"],
                              "candidate_minus_control": value["candidate_minus_control"],
                              "controller": value["controllers"]["candidate"],
                              "result_sha256": value["result_sha256"]}, sort_keys=True))
            return 0 if value["decision"]["passed"] else 2
    except (GateError, OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
