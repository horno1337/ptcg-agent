"""Locked transfer screen for the 49/60-overlap Dragapult/Froslass guide list."""

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

from tools import eval_ab as EVAL, index_corpus  # noqa: E402
from tools.research import eval_day1_multideck_bc_gameplay as BASE  # noqa: E402
from tools.research import eval_festival_lead_bc_v1_field as STATS  # noqa: E402
from tools.research import eval_lucario_day1_day2_factorial as FIELD  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import build_paired_schedule, schedule_manifest  # noqa: E402


RUN = ROOT / "tools/checkpoints/dragapult-froslass-transfer-20260811"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
ORIGINAL_DECK = ROOT / "decks/dragapult_07bed.csv"
GUIDE_DECK = ROOT / "decks/dragapult_froslass_metafy.csv"
D1 = ROOT / "tools/checkpoints/dragapult-bc-20260810/candidates"
D2 = ROOT / "tools/checkpoints/day2-expanded-bc-20260811/candidates/dragapult"
PATHS = {
    "original_deck": ORIGINAL_DECK,
    "guide_deck": GUIDE_DECK,
    "d1_main": D1 / "main/model/candidate-qu-v2a-weights.npz",
    "d1_card": D1 / "card/model/candidate-qu-v2a-weights.npz",
    "d2_main": D2 / "main/model/candidate-qu-v2a-weights.npz",
    "d2_card": D2 / "card/model/candidate-qu-v2a-weights.npz",
    "parent": BASE.PARENT_WEIGHTS,
    "coverage": ROOT / "tools/checkpoints/dragapult-froslass-metafy/coverage.json",
    "evaluator": Path(__file__).resolve(),
}
ARMS = {
    "original_d1": ("original", "d1_main", "d1_card"),
    "guide_qu": ("guide", None, None),
    "guide_d1": ("guide", "d1_main", "d1_card"),
    "guide_d2_main+d1_card": ("guide", "d2_main", "d1_card"),
    "guide_d1_main+d2_card": ("guide", "d1_main", "d2_card"),
    "guide_d2": ("guide", "d2_main", "d2_card"),
}
GAMES_PER_ARM = 256
SEED = 2026081124
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0


class TransferError(RuntimeError):
    """The guide-deck transfer screen failed closed."""


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_deck(path: Path) -> tuple[int, ...]:
    deck = tuple(
        int(value) for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    )
    if len(deck) != 60:
        raise TransferError(f"deck is not 60 cards: {path}")
    return deck


def load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop(key, None)
    if value.get("schema") != schema or claimed != BASE.canonical_sha256(value):
        raise TransferError(f"self-hash failed: {path}")
    value[key] = claimed
    return value


def build_lock() -> dict[str, Any]:
    if any(path.exists() for path in (LOCK, ATTEMPT, RESULT)):
        raise TransferError("transfer screen already locked or consumed")
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    if missing:
        raise TransferError(f"artifacts missing: {missing}")
    original, guide = read_deck(ORIGINAL_DECK), read_deck(GUIDE_DECK)
    if index_corpus.deck_sha256(original) != "07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725":
        raise TransferError("original Dragapult identity drifted")
    if index_corpus.deck_sha256(guide) != "4ffe6aeee43a40083a1c19855bc7de7eb6bdc1dfb7e7bb603d06f930d7e40f2b":
        raise TransferError("guide Dragapult identity drifted")
    coverage = load_self(
        PATHS["coverage"], "ptcg.dragapult-froslass-metafy.coverage.v1",
        "result_sha256",
    )
    if (
        coverage["target"]["deck_sha256"] != index_corpus.deck_sha256(guide)
        or coverage["summary"]["exact_unique_games"] != 0
    ):
        raise TransferError("guide coverage premise drifted")
    field = FIELD.current_field()
    qu = COMMON._load_net(PATHS["parent"], "parent")
    opponents, _controller = BASE.make_opponents(field, qu, "transfer-lock-field")
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    seats = Counter(row.learner_seat for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise TransferError("schedule is not seat balanced")
    payload = {
        "schema": "ptcg.dragapult-froslass.transfer-lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "arms": ARMS,
        "decks": {
            "original": {"cards": list(original), "sha256": index_corpus.deck_sha256(original)},
            "guide": {"cards": list(guide), "sha256": index_corpus.deck_sha256(guide)},
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "total_games": GAMES_PER_ARM * len(ARMS),
            "seed": SEED,
            "identical_opponent_and_seat_schedule": True,
            "screen_only": True,
            "guide_eligibility": (
                "valid best guide BC arm must have positive paired point estimates "
                "versus both guide Qu and original-list Day-1 BC"
            ),
            "no_interim_stopping": True,
        },
        "schedule_manifest_sha256": BASE.canonical_sha256(
            schedule_manifest(schedule, opponents)
        ),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": BASE.file_sha256(path)}
            for name, path in PATHS.items()
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = BASE.canonical_sha256(payload)
    return payload


def load_lock() -> dict[str, Any]:
    value = load_self(LOCK, "ptcg.dragapult-froslass.transfer-lock.v1", "lock_sha256")
    for name, row in value["artifacts"].items():
        path = PATHS[name]
        if Path(row["path"]).resolve() != path.resolve() or row["sha256"] != BASE.file_sha256(path):
            raise TransferError(f"artifact drifted: {name}")
    return value


def run(quiet: bool) -> dict[str, Any]:
    lock = load_lock()
    if ATTEMPT.exists() or RESULT.exists():
        raise TransferError("transfer attempt already consumed")
    attempt = {
        "schema": "ptcg.dragapult-froslass.transfer-attempt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    }
    attempt["attempt_sha256"] = BASE.canonical_sha256(attempt)
    write_new(ATTEMPT, attempt)
    nets = {
        name: COMMON._load_net(PATHS[name], name)
        for name in ("d1_main", "d1_card", "d2_main", "d2_card", "parent")
    }
    decks = {name: tuple(row["cards"]) for name, row in lock["decks"].items()}
    field = FIELD.current_field()
    series = {}
    diagnostics = {}
    for arm, (deck_name, main_name, card_name) in ARMS.items():
        deck = decks[deck_name]
        controller = BASE.DualHeadController(
            nets[main_name] if main_name else None,
            nets[card_name] if card_name else None,
            nets["parent"], deck, arm,
        )
        opponents, field_controller = BASE.make_opponents(field, nets["parent"], f"{arm}-field")
        schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
        if BASE.canonical_sha256(schedule_manifest(schedule, opponents)) != lock["schedule_manifest_sha256"]:
            raise TransferError(f"schedule drifted: {arm}")
        value = EVAL.run_series(
            f"dragapult-transfer/{arm}", controller, deck, opponents, schedule,
            max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S, verbose=not quiet,
        )
        learner_diag = controller.diagnostics()
        field_diag = field_controller.diagnostics()
        clean = BASE._clean_control(learner_diag) if arm == "guide_qu" else BASE._clean_candidate(learner_diag)
        valid = bool(
            len(value.records) == GAMES_PER_ARM and value.gate_valid and clean
            and field_diag.get("fallbacks") == 0 and field_diag.get("exceptions") == {}
        )
        series[arm] = value
        diagnostics[arm] = {"valid": valid, "learner": learner_diag, "field": field_diag}
    comparisons = {}
    for arm, value in series.items():
        comparisons[arm] = {
            "versus_original_d1": STATS.paired_delta_ci(
                value.records, series["original_d1"].records,
            ),
            "versus_guide_qu": STATS.paired_delta_ci(
                value.records, series["guide_qu"].records,
            ),
        }
    guide_arms = [arm for arm in ARMS if arm.startswith("guide_") and arm != "guide_qu"]
    best = max(guide_arms, key=lambda arm: series[arm].score)
    eligible = bool(
        diagnostics[best]["valid"]
        and comparisons[best]["versus_original_d1"]["mean_delta"] > 0
        and comparisons[best]["versus_guide_qu"]["mean_delta"] > 0
    )
    payload = {
        "schema": "ptcg.dragapult-froslass.transfer-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "all_valid": all(row["valid"] for row in diagnostics.values()),
            "best_guide_arm": best,
            "guide_opens_rule_development": eligible,
            "requires_fresh_confirmation_before_package": True,
        },
        "summaries": {arm: value.summary() for arm, value in series.items()},
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "records": {
            arm: [asdict(row) for row in value.records] for arm, value in series.items()
        },
        "promotion_authority": False,
        "package_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = BASE.canonical_sha256(payload)
    write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage == "lock":
            value = build_lock()
            write_new(LOCK, value)
            print(json.dumps({
                "lock_sha256": value["lock_sha256"],
                "protocol": value["protocol"],
                "decks": {name: row["sha256"] for name, row in value["decks"].items()},
            }, indent=2, sort_keys=True))
            return 0
        value = run(args.quiet)
    except (TransferError, BASE.GameplayError, OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "decision": value["decision"],
        "scores": {arm: row["score"] for arm, row in value["summaries"].items()},
        "comparisons": value["comparisons"],
        "result_sha256": value["result_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if value["decision"]["all_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
