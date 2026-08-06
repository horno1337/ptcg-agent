"""Locked recent-frequency non-mirror field gate for the 100k mirror candidate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL  # noqa: E402
from tools.research import eval_md_v2_card_v1_gameplay as LAYERED  # noqa: E402
from tools.research import eval_md_v2_scaled_gameplay as COMMON  # noqa: E402
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/field-gate"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
LOCK_SCHEMA = "ptcg.md-v3.exact-mirror-league-100k.field-lock.v1"
ATTEMPT_SCHEMA = "ptcg.md-v3.exact-mirror-league-100k.field-attempt.v1"
RESULT_SCHEMA = "ptcg.md-v3.exact-mirror-league-100k.field-result.v1"
GAMES_PER_ARM = 5_120
TOTAL_GAMES = 10_240
SEED = 2_026_080_241
NONINFERIORITY_MARGIN = -0.015
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0
PATHS = {
    "evaluator": Path(__file__).resolve(),
    "snapshotter": ROOT / "tools/research/snapshot_recent_weighted_field_from_archives.py",
    "field_snapshot": RUN / "recent-field-20260729-31.json",
    "source_inventory": ROOT / "tools/checkpoints/md-next-recent-20260729-31/inventory.json",
    "candidate": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/training/terminal-update-16-candidate/candidate-qu-v2a-weights.npz",
    "candidate_checkpoint": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/training/terminal-update-16-candidate/candidate-qu-v2a-ppo-v2-checkpoint.pt",
    "training_lock": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/training-lock.json",
    "mirror_replication_lock": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/gameplay-replication-10240/lock.json",
    "mirror_replication_result": ROOT / "tools/checkpoints/md-v3-mirror-league-100k-v2/gameplay-replication-10240/result.json",
    "frozen_main": ROOT / "tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz",
    "card": ROOT / "agent/md_v2_card_weights.npz",
    "qu": ROOT / "agent/weights.npz",
    "deck": ROOT / "decks/md_v1_grimmsnarl.csv",
}


class FieldError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FieldError(f"refusing to overwrite {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_snapshot() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(PATHS["field_snapshot"].read_text(encoding="utf-8"))
    if payload.get("schema") != "ptcg.recent-frequency-weighted-field.v2":
        raise FieldError("recent field schema mismatch")
    claimed = payload.get("snapshot_sha256")
    check = dict(payload)
    check.pop("snapshot_sha256", None)
    if claimed != COMMON.canonical_sha256(check):
        raise FieldError("recent field self-hash mismatch")
    rows = [dict(row) for row in payload.get("field", [])]
    if not rows or sum(row.get("archetype") == "Grimmsnarl" for row in rows) != 1:
        raise FieldError("field must contain exactly one Grimmsnarl row")
    primary = [row for row in rows if row["archetype"] != "Grimmsnarl"]
    total = sum(int(row["registered_seats"]) for row in primary)
    if total <= 0:
        raise FieldError("non-mirror field is empty")
    for row in primary:
        row["field_weight"] = int(row["registered_seats"]) / total
    if not math.isclose(sum(row["field_weight"] for row in primary), 1.0,
                        rel_tol=0.0, abs_tol=1e-12):
        raise FieldError("renormalized non-mirror weights do not sum to one")
    return payload, primary


def _make_opponents(
    rows: Sequence[Mapping[str, Any]], qu_net: Any,
) -> tuple[list[OpponentSpec], EVAL.DeployableReflex]:
    qu_sha = _sha256(PATHS["qu"])
    controller = EVAL.DeployableReflex(qu_net, f"field-qu-v2b:{qu_sha}")
    opponents = []
    for row in rows:
        registration = tuple(int(card) for card in row["deck"])

        def move(obs, rng, deck=registration):
            del rng
            return controller.act(obs, deck)

        opponents.append(OpponentSpec(
            key=f"{row['archetype']}/qu-v2b",
            deck=registration,
            move=move,
            weight=float(row["field_weight"]),
            policy_id=f"qu-v2b:{qu_sha}",
            schedule_group="recent-nonmirror-field/qu-v2b",
        ))
    return opponents, controller


def _build_schedule(rows: Sequence[Mapping[str, Any]]):
    qu = COMMON._load_net(PATHS["qu"], "schedule-only Qu-v2B")
    opponents, _ = _make_opponents(rows, qu)
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    return opponents, schedule


def _score_values(records: Sequence[Any]) -> list[float]:
    return [
        1.0 if row.result == "win" else 0.5 if row.result == "draw" else 0.0
        for row in records
    ]


def _delta_ci(left_records: Sequence[Any], right_records: Sequence[Any]):
    left = _score_values(left_records)
    right = _score_values(right_records)
    if len(left) < 2 or len(right) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((value - left_mean) ** 2 for value in left) / (len(left) - 1)
    right_var = sum((value - right_mean) ** 2 for value in right) / (len(right) - 1)
    delta = left_mean - right_mean
    se = math.sqrt(left_var / len(left) + right_var / len(right))
    z = 1.959963984540054
    return {"delta": delta, "ci95": [delta - z * se, delta + z * se]}


def _clean_layered(value: Mapping[str, Any]) -> bool:
    return (
        value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
        and value.get("off_deck_main_routes") == 0
        and value.get("off_deck_card_routes") == 0
    )


def _clean_field(value: Mapping[str, Any]) -> bool:
    return value.get("fallbacks") == 0 and value.get("exceptions") == {}


def build_lock() -> dict[str, Any]:
    if any(not path.is_file() for path in PATHS.values()):
        missing = [name for name, path in PATHS.items() if not path.is_file()]
        raise FieldError(f"bound field artifacts missing: {missing}")
    snapshot, rows = _load_snapshot()
    opponents, schedule = _build_schedule(rows)
    manifest = schedule_manifest(schedule, opponents)
    seats = Counter(row.learner_seat for row in schedule)
    matchups = Counter(opponents[row.opponent_index].key for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise FieldError("field schedule is not exactly seat balanced")
    excluded = {
        row["archetype"]: {
            "registered_seats": row["registered_seats"],
            "observed_share": row["observed_share"],
        }
        for row in snapshot.get("excluded_tail", [])
    }
    mirror = json.loads(PATHS["mirror_replication_result"].read_text())
    if not mirror.get("decision", {}).get("positive_evidence"):
        raise FieldError("bound mirror replication is not positive")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in PATHS.items()
        },
        "field": {
            "source_dates": ["2026-07-29", "2026-07-30", "2026-07-31"],
            "source_registered_seats": snapshot["source"]["registered_seats"],
            "source_included_share": snapshot["selection"]["included_share"],
            "primary_definition": "all >=0.5% archetypes except Grimmsnarl, renormalized by registered seats",
            "primary_registered_seats": sum(row["registered_seats"] for row in rows),
            "primary": [{
                "archetype": row["archetype"],
                "registered_seats": row["registered_seats"],
                "weight": row["field_weight"],
                "scheduled_games_per_arm": matchups[f"{row['archetype']}/qu-v2b"],
                "deck_sha256": row["representative_deck_sha256"],
            } for row in rows],
            "excluded_mirror": "Grimmsnarl; direct mirror improvement already established in a fixed 10,240-game replication",
            "cinderace_diagnostic": excluded.get("Cinderace", {"registered_seats": 0, "observed_share": 0.0}),
            "cinderace_rule": "reported as below the prospectively fixed 0.5% inclusion floor; no gameplay stratum allocated",
            "no_posthoc_stratum_dropping": True,
        },
        "protocol": {
            "total_engine_games": TOTAL_GAMES,
            "games_per_arm": GAMES_PER_ARM,
            "pairs_per_arm": GAMES_PER_ARM // 2,
            "schedule_seed": SEED,
            "arm_order": ["candidate", "control"],
            "learner_deck": "exact MD-v3 Grimmsnarl list in both arms",
            "candidate": "fixed update-130 100k exact-mirror-league ST_MAIN + frozen MD-v3 ST_CARD + Qu-v2B residual",
            "control": "complete frozen ladder-proven MD-v3",
            "opponents": "frozen Qu-v2B piloting each recent representative exact list",
            "identical_schedule_per_arm": True,
            "score": "(wins + 0.5 * official draws) / scheduled games",
            "primary_comparison": "candidate score minus control score on the complete non-mirror schedule",
            "interval": "ordinary unpooled two-sided normal CI95 for score difference",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "pass": "valid zero-fault arms and CI95 lower bound strictly greater than -0.015",
            "report_all_primary_strata": True,
            "one_schedule_one_attempt": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(manifest),
        "prior_mirror_evidence": {
            "result_sha256": mirror["result_sha256"],
            "games": mirror["summary"]["scheduled_games"],
            "score": mirror["decision"]["score"],
            "wilson_ci95": mirror["decision"]["wilson_ci95"],
        },
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def _load_lock() -> dict[str, Any]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    claimed = payload.pop("lock_sha256", None)
    if payload.get("schema") != LOCK_SCHEMA or claimed != COMMON.canonical_sha256(payload):
        raise FieldError("field lock identity drifted")
    payload["lock_sha256"] = claimed
    for name, record in payload["artifacts"].items():
        path = PATHS[name]
        if Path(record["path"]).resolve() != path.resolve() or _sha256(path) != record["sha256"]:
            raise FieldError(f"bound artifact drift: {name}")
    return payload


def run(quiet: bool) -> dict[str, Any]:
    lock = _load_lock()
    _snapshot, rows = _load_snapshot()
    deck = COMMON.read_deck(PATHS["deck"])
    candidate_net = COMMON._load_net(PATHS["candidate"], "100k mirror candidate")
    frozen_main = COMMON._load_net(PATHS["frozen_main"], "frozen MD-v3 main")
    card = COMMON._load_net(PATHS["card"], "frozen MD-v3 card")
    qu = COMMON._load_net(PATHS["qu"], "frozen Qu-v2B")
    candidate = LAYERED.LayeredMirrorCardController(
        candidate_net, card, qu, "100k-mirror-terminal", deck,
    )
    control = LAYERED.LayeredMirrorCardController(
        frozen_main, card, qu, "complete-frozen-md-v3", deck,
    )
    candidate_opponents, candidate_field = _make_opponents(rows, qu)
    control_opponents, control_field = _make_opponents(rows, qu)
    candidate_schedule = build_paired_schedule(
        candidate_opponents, GAMES_PER_ARM, seed=SEED,
    )
    control_schedule = build_paired_schedule(
        control_opponents, GAMES_PER_ARM, seed=SEED,
    )
    candidate_manifest = schedule_manifest(candidate_schedule, candidate_opponents)
    control_manifest = schedule_manifest(control_schedule, control_opponents)
    expected = lock["schedule_manifest_sha256"]
    if (
        COMMON.canonical_sha256(candidate_manifest) != expected
        or COMMON.canonical_sha256(control_manifest) != expected
        or candidate_manifest != control_manifest
    ):
        raise FieldError("runtime schedules differ from the prospective lock")
    if ATTEMPT.exists() or RESULT.exists():
        raise FieldError("field-gate attempt has already been consumed")
    _write_new(ATTEMPT, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_outcome": True,
        "lock_sha256": lock["lock_sha256"],
    })
    candidate_result = EVAL.run_series(
        "100k-mirror-candidate/recent-nonmirror-field",
        candidate, deck, candidate_opponents, candidate_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
        verbose=not quiet,
    )
    control_result = EVAL.run_series(
        "frozen-md-v3/recent-nonmirror-field",
        control, deck, control_opponents, control_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
        verbose=not quiet,
    )
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    candidate_field_diag = candidate_field.diagnostics()
    control_field_diag = control_field.diagnostics()
    comparison = _delta_ci(candidate_result.records, control_result.records)
    if comparison is None:
        raise FieldError("primary comparison is empty")
    by_key: dict[str, dict[str, Any]] = {}
    keys = [opponent.key for opponent in candidate_opponents]
    for key in keys:
        left = [row for row in candidate_result.records if row.opponent_key == key]
        right = [row for row in control_result.records if row.opponent_key == key]
        delta = _delta_ci(left, right)
        by_key[key] = {
            "candidate": dict(Counter(row.result for row in left)),
            "control": dict(Counter(row.result for row in right)),
            "games_per_arm": len(left),
            "candidate_score": sum(_score_values(left)) / len(left),
            "control_score": sum(_score_values(right)) / len(right),
            "candidate_minus_control": delta,
        }
    valid = (
        len(candidate_result.records) == GAMES_PER_ARM
        and len(control_result.records) == GAMES_PER_ARM
        and candidate_result.gate_valid and control_result.gate_valid
        and _clean_layered(candidate_diag) and _clean_layered(control_diag)
        and _clean_field(candidate_field_diag) and _clean_field(control_field_diag)
    )
    passed = bool(valid and comparison["ci95"][0] > NONINFERIORITY_MARGIN)
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed_noninferiority": passed,
            "margin": NONINFERIORITY_MARGIN,
            "candidate_minus_control": comparison,
        },
        "summaries": {
            "candidate": candidate_result.summary(),
            "control": control_result.summary(),
        },
        "by_matchup": by_key,
        "records": {
            "candidate": [asdict(row) for row in candidate_result.records],
            "control": [asdict(row) for row in control_result.records],
        },
        "controllers": {
            "candidate": candidate_diag,
            "control": control_diag,
            "candidate_field": candidate_field_diag,
            "control_field": control_field_diag,
        },
        "environments": {
            "candidate": environment_manifest(deck, candidate_opponents, str(PATHS["field_snapshot"])),
            "control": environment_manifest(deck, control_opponents, str(PATHS["field_snapshot"])),
        },
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    _write_new(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.lock_only:
            payload = build_lock()
            _write_new(LOCK, payload)
            print(json.dumps({
                "lock": str(LOCK),
                "lock_sha256": payload["lock_sha256"],
                "protocol": payload["protocol"],
                "field": payload["field"],
            }, indent=2, sort_keys=True))
            return 0
        payload = run(args.quiet)
    except (OSError, KeyError, TypeError, ValueError, FieldError) as error:
        parser.error(str(error))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0 if payload["decision"]["passed_noninferiority"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
