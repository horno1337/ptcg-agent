"""Prospectively lock and run the selective ST_CARD recent-field gate.

The field includes Grimmsnarl rather than removing it: deployment can observe
only the opposing public Grimmsnarl signature, not the opponent's hidden exact
registration.  This gate therefore measures the complete Aug 1--5 deployment
mixture, with the fixed top-eight exact Grimmsnarl variants covering >99% of
recent Grim registrations and renormalized within the existing Grim mass.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools import eval_ab as EVAL, rl_env  # noqa: E402
from tools.research import (  # noqa: E402
    eval_dobi_v1_elite_teacher_card_v1_gameplay as MIRROR,
    eval_md_v2_card_v1_gameplay as CARD_GAME,
    eval_md_v2_scaled_gameplay as COMMON,
    lock_dobi_v1_elite_teacher_card_v1 as SOURCE,
)
from tools.rl_env import (  # noqa: E402
    OpponentSpec,
    build_paired_schedule,
    environment_manifest,
    schedule_manifest,
)


RUN = SOURCE.RUN / "field"
LOCK = RUN / "lock.json"
ATTEMPT = RUN / "attempt.json"
RESULT = RUN / "result.json"
LOCK_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.field-lock.v1"
ATTEMPT_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.field-attempt.v1"
RESULT_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.field-result.v1"

GAMES_PER_ARM = 5_120
TOTAL_ENGINE_GAMES = 10_240
WITHIN_ARM_SEAT_PAIRS = 2_560
CROSS_ARM_PAIRED_UNITS = 5_120
SEED = 2_026_080_73
NONINFERIORITY_MARGIN = -0.015
Z_95 = 1.959963984540054
MAX_SELECTS = 5_000
TIME_BANK_S = 600.0

FIELD_SNAPSHOT = ROOT / (
    "tools/checkpoints/recent-field-20260801-05/"
    "recent-field-20260801-05.json"
)
FIELD_INVENTORY = ROOT / (
    "tools/checkpoints/recent-field-20260801-05/inventory.json"
)
SNAPSHOTTER = ROOT / (
    "tools/research/snapshot_recent_weighted_field_from_archives.py"
)
GRIM_VARIANT_SNAPSHOT = ROOT / (
    "tools/checkpoints/recent-field-20260801-05/grim-variants.json"
)
GRIM_VARIANT_SNAPSHOTTER = ROOT / (
    "tools/research/snapshot_grim_variants_from_archives.py"
)
SNAPSHOT_FILE_SHA256 = (
    "f558119122a1fe63b289b032e3bef9a91851c105e5cd670ae85e58ee7398dfbd"
)
SNAPSHOT_SELF_SHA256 = (
    "4c2b142f2b115ad5164a3ec9277bbe2724168e1a65f7e6e21c91b18a803c27b0"
)
INVENTORY_FILE_SHA256 = (
    "7ad2a6f45f2199df79c404b00a4898ae44b213089d038208d60125119eafee32"
)
INVENTORY_SELF_SHA256 = (
    "42d68579ac87b7ff7469e1822c3b158458df447886faa5185b73788316f8ff8b"
)
INVENTORY_HASH_DOMAIN = b"ptcg.md-next.recent-archive-inventory.v1\0"
GRIM_VARIANT_FILE_SHA256 = (
    "c27b337495ce2430e83bb9ec5f408e9150824e6b6c8ad256708c1f082bf2cd29"
)
GRIM_VARIANT_SELF_SHA256 = (
    "673b9a120a6c253a3d749d82b7305af1e742c5366871f5dce0b7e02756416d71"
)
SOURCE_DATES = (
    "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05",
)
SOURCE_REGISTERED_SEATS = 46_750
INCLUDED_REGISTERED_SEATS = 46_429
MINIMUM_OBSERVED_SHARE = 0.005

# Exact inventory and representative-list identity, ordered as frozen in the
# Aug 1--5 snapshot.  Field weights are registered_seats / 46,429.
EXPECTED_FIELD = (
    ("Grimmsnarl", 17_973, "cafa7652a6349be806d8ac2b9abfdb6c72ca3821f368e0d912e2d989f3b54cdd"),
    ("Teal Mask Ogerpon ex", 5_447, "c2f97a7cf840ea0dfa8e926e1d30c2105be0550e0b9181c86c90c66b746f4c54"),
    ("Alakazam", 5_159, "606a775392ffe25e058b19c17801d58a4bf30f7cd8c62782388d3de7e7eb5283"),
    ("Mega Lopunny ex", 3_760, "0246cdc0d97432d77e97ecac8ce4f272d858e9b3c86795aa37eef4286d965820"),
    ("Crustle", 3_631, "89057065bdc7a1c8b9066665401cf1e0446a57156f6a3e2e3754ae4533bd5308"),
    ("Dragapult", 2_453, "89e6155f25310ee695c0761c85d3ae8e44f376456ff0539231820f8e803f2d5e"),
    ("Mega Froslass", 2_042, "ba51a134262bb1c2fbb14fd1357602734dd1abb5b6b08aa68e6a168efcd455eb"),
    ("Cynthia's Garchomp ex", 1_725, "eff68cb08be178b9c7f06c409b61e88ae9200ab6dc26e05f4bf29eed86040455"),
    ("Fezandipiti ex", 1_318, "f4af3f88f120f86d14f7da1b2e99e3530cf769be695b886b74d4925e7c8fb893"),
    ("Mega Lucario", 1_161, "dc8571d0bc2e546a1f85b938696cfc40a1451c68a4ccc1f695e7c3e1c74f1278"),
    ("Team Rocket's Mewtwo ex", 910, "9ee28285f80b60f173776bd71552f7205f1f8594ac509ad7f7ded264da45e560"),
    ("Mega Kangaskhan ex", 590, "d3f092b737c0990149541576444e767d34a7367576d15a8f1b39f7b15460645d"),
    ("Applin", 260, "3722e7ea2641042c208e00d6b753d97d065e31a388b534d5d87104565174d878"),
)
EXPECTED_GRIM_VARIANTS = (
    (1, 14_923, "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"),
    (2, 1_883, "e2e03fe8ef9592b1204c159a7725da3945675fa872b1e18baca550560503a78e"),
    (3, 457, "a9cf3d228c6a32c3c2cfdbf4b42f570fc7d00f081e4e39e395c3a6793271d26c"),
    (4, 207, "978ab31c50aa7fdaf11e38ba108cf0eda7fab6ca3597bfa3b45a7f74cd6a9327"),
    (5, 153, "22a88f292df485391b7927bdcc4d801945e1fbc31236e0d6eb113e3235e9878f"),
    (6, 91, "de5b9940c5be5708af75d38f0bbb3fd85600e90fea9349a8ab330fb1f4638ffe"),
    (7, 52, "f0b7412ebea0d10e9ac4f9491c6977f5af31f7858a5c405dd2c5f573eb420248"),
    (8, 47, "ecc0d1237736ea60dbaf4bd2ece0314a6d25117ff5485d7b1e2d432431dbd8dc"),
)
GRIM_SELECTED_SEATS = 17_813
GRIM_OMITTED_TAIL_SEATS = 160
GRIM_ALLOCATIONS = (1_660, 210, 50, 24, 16, 10, 6, 6)
NON_GRIM_ALLOCATIONS = {
    "Teal Mask Ogerpon ex": 600,
    "Alakazam": 570,
    "Mega Lopunny ex": 414,
    "Crustle": 400,
    "Dragapult": 270,
    "Mega Froslass": 226,
    "Cynthia's Garchomp ex": 190,
    "Fezandipiti ex": 146,
    "Mega Lucario": 128,
    "Team Rocket's Mewtwo ex": 100,
    "Mega Kangaskhan ex": 66,
    "Applin": 28,
}

FROZEN_MAIN = ROOT / (
    "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
    "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
)
FROZEN_CARD = ROOT / "agent/md_v2_card_weights.npz"
FROZEN_QU = ROOT / "agent/weights.npz"
DECK = ROOT / "decks/md_v1_grimmsnarl.csv"


class FieldError(RuntimeError):
    """A prospective field artifact, schedule, or outcome drifted."""


class AuditedSelectiveCardController(MIRROR.SelectiveCardController):
    """Selective candidate plus exact frozen-Dobi off-route shadow audit."""

    def __init__(self, main_net, candidate_card, parent_card, qu_net,
                 name: str, registered_deck: Sequence[int]):
        super().__init__(
            main_net, candidate_card, parent_card, qu_net,
            name, registered_deck,
        )
        self.shadow = CARD_GAME.LayeredMirrorCardController(
            main_net, parent_card, qu_net,
            f"{name}/frozen-dobi-shadow", registered_deck,
        )
        self.off_route_prompts = 0
        self.off_route_identity_mismatches = 0
        self.off_route_audit_faults: Counter[str] = Counter()

    def act(self, obs: dict, registered_deck=None) -> list[int]:
        before = self.candidate_family_routes
        action = super().act(obs, registered_deck)
        if self.candidate_family_routes == before:
            self.off_route_prompts += 1
            try:
                expected = self.shadow.act(obs, registered_deck)
                if list(action) != list(expected):
                    self.off_route_identity_mismatches += 1
            except Exception as error:  # pragma: no cover - fail-closed audit
                self.off_route_audit_faults[type(error).__name__] += 1
        return action

    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics()
        result.update({
            "off_route_prompts": self.off_route_prompts,
            "off_route_identity_mismatches": self.off_route_identity_mismatches,
            "off_route_audit_faults": dict(self.off_route_audit_faults),
            "off_route_shadow": self.shadow.diagnostics(),
        })
        return result


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FieldError(f"missing field artifact: {path}")
    return {"path": str(path.resolve()), "sha256": COMMON.file_sha256(path)}


def _load_self(path: Path, schema: str, key: str) -> dict[str, Any]:
    try:
        return COMMON.load_self_hashed_json(path, schema=schema, hash_key=key)
    except COMMON.EvaluationError as error:
        raise FieldError(str(error)) from error


def load_snapshot() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if COMMON.file_sha256(FIELD_SNAPSHOT) != SNAPSHOT_FILE_SHA256:
        raise FieldError("Aug 1-5 field snapshot file identity drifted")
    if COMMON.file_sha256(FIELD_INVENTORY) != INVENTORY_FILE_SHA256:
        raise FieldError("Aug 1-5 source inventory file identity drifted")
    snapshot = _load_self(
        FIELD_SNAPSHOT, "ptcg.recent-frequency-weighted-field.v2",
        "snapshot_sha256",
    )
    inventory = json.loads(FIELD_INVENTORY.read_text(encoding="utf-8"))
    inventory_body = {
        key: value for key, value in inventory.items()
        if key != "inventory_sha256"
    }
    calculated_inventory_hash = hashlib.sha256(
        INVENTORY_HASH_DOMAIN + json.dumps(
            inventory_body, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        snapshot["snapshot_sha256"] != SNAPSHOT_SELF_SHA256
        or inventory.get("schema") != "ptcg.md-next.recent-archive-inventory.v1"
        or inventory.get("inventory_sha256") != INVENTORY_SELF_SHA256
        or calculated_inventory_hash != INVENTORY_SELF_SHA256
        or snapshot.get("source", {}).get("inventory_sha256")
            != INVENTORY_SELF_SHA256
        or snapshot.get("source", {}).get("inventory_file_sha256")
            != INVENTORY_FILE_SHA256
    ):
        raise FieldError("Aug 1-5 snapshot/inventory lineage drifted")
    source = snapshot.get("source", {})
    selection = snapshot.get("selection", {})
    dates = tuple(row.get("date") for row in source.get("archives", ()))
    if (
        dates != SOURCE_DATES
        or source.get("registered_seats") != SOURCE_REGISTERED_SEATS
        or selection.get("included_seats") != INCLUDED_REGISTERED_SEATS
        or not math.isclose(
            float(selection.get("included_share", -1.0)),
            INCLUDED_REGISTERED_SEATS / SOURCE_REGISTERED_SEATS,
            rel_tol=0.0, abs_tol=1e-15,
        )
        or not math.isclose(
            float(selection.get("minimum_observed_seat_share", -1.0)),
            MINIMUM_OBSERVED_SHARE, rel_tol=0.0, abs_tol=0.0,
        )
    ):
        raise FieldError("Aug 1-5 field source/selection contract drifted")
    rows = [dict(row) for row in snapshot.get("field", ())]
    actual = tuple((
        row.get("archetype"), row.get("registered_seats"),
        row.get("representative_deck_sha256"),
    ) for row in rows)
    if actual != EXPECTED_FIELD:
        raise FieldError("Aug 1-5 field representative inventory drifted")
    for row in rows:
        deck = row.get("deck")
        weight = row.get("field_weight")
        expected = int(row["registered_seats"]) / INCLUDED_REGISTERED_SEATS
        if (
            not isinstance(deck, list) or len(deck) != 60
            or any(not isinstance(card, int) or isinstance(card, bool)
                   for card in deck)
            or not isinstance(weight, (int, float)) or isinstance(weight, bool)
            or not math.isclose(float(weight), expected,
                                rel_tol=0.0, abs_tol=1e-15)
        ):
            raise FieldError(f"invalid field row: {row.get('archetype')}")
    if not math.isclose(sum(float(row["field_weight"]) for row in rows), 1.0,
                        rel_tol=0.0, abs_tol=1e-12):
        raise FieldError("Aug 1-5 field weights do not sum to one")
    return snapshot, rows


def load_grim_variants() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if COMMON.file_sha256(GRIM_VARIANT_SNAPSHOT) != GRIM_VARIANT_FILE_SHA256:
        raise FieldError("Grim variant snapshot file identity drifted")
    snapshot = _load_self(
        GRIM_VARIANT_SNAPSHOT,
        "ptcg.dobi-v1.grim-variants-20260801-05.v1",
        "snapshot_sha256",
    )
    selection = snapshot.get("selection", {})
    variants = [dict(row) for row in snapshot.get("variants", ())]
    actual = tuple((
        row.get("rank"), row.get("registered_seats"),
        row.get("canonical_deck_sha256"),
    ) for row in variants[:8])
    if (
        snapshot.get("snapshot_sha256") != GRIM_VARIANT_SELF_SHA256
        or snapshot.get("registration_only") is not True
        or snapshot.get("source", {}).get("inventory_sha256")
            != INVENTORY_SELF_SHA256
        or selection.get("grim_registered_seats") != EXPECTED_FIELD[0][1]
        or selection.get("exact_variants") != 24
        or selection.get("selected_variants") != 8
        or selection.get("selected_registered_seats") != GRIM_SELECTED_SEATS
        or selection.get("omitted_tail_seats") != GRIM_OMITTED_TAIL_SEATS
        or actual != EXPECTED_GRIM_VARIANTS
    ):
        raise FieldError("Grim variant snapshot contract drifted")
    if any(
        not isinstance(row.get("deck"), list) or len(row["deck"]) != 60
        for row in variants
    ):
        raise FieldError("invalid Grim variant deck")
    return snapshot, variants


def expand_field_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replace the representative Grim row by its locked top-eight mixture."""
    _snapshot, variants = load_grim_variants()
    grim = [row for row in rows if row.get("archetype") == "Grimmsnarl"]
    if len(grim) != 1:
        raise FieldError("expected exactly one aggregate Grim field row")
    grim_mass = float(grim[0]["field_weight"])
    expanded: list[dict[str, Any]] = []
    for row in rows:
        if row.get("archetype") != "Grimmsnarl":
            item = dict(row)
            item["opponent_key"] = f"{row['archetype']}/qu-v2b"
            item["scheduled_games_per_arm"] = NON_GRIM_ALLOCATIONS[
                str(row["archetype"])
            ]
            expanded.append(item)
    for variant in variants[:8]:
        rank = int(variant["rank"])
        expanded.append({
            "archetype": "Grimmsnarl",
            "variant_rank": rank,
            "registered_seats": int(variant["registered_seats"]),
            "field_weight": (
                grim_mass * int(variant["registered_seats"]) / GRIM_SELECTED_SEATS
            ),
            "representative_deck_sha256": variant["deck_value_sha256"],
            "canonical_deck_sha256": variant["canonical_deck_sha256"],
            "deck": list(variant["deck"]),
            "opponent_key": f"Grimmsnarl-v{rank}/qu-v2b",
            "scheduled_games_per_arm": GRIM_ALLOCATIONS[rank - 1],
        })
    if not math.isclose(sum(float(row["field_weight"]) for row in expanded), 1.0,
                        rel_tol=0.0, abs_tol=1e-12):
        raise FieldError("expanded field weights do not sum to one")
    if sum(int(row["scheduled_games_per_arm"]) for row in expanded) != GAMES_PER_ARM:
        raise FieldError("fixed field allocations do not sum to games per arm")
    return expanded


def _make_opponents(rows: Sequence[Mapping[str, Any]], qu_net: Any):
    qu_sha = COMMON.file_sha256(FROZEN_QU)
    controller = EVAL.DeployableReflex(
        qu_net, f"field-qu-v2b:{qu_sha}",
    )
    opponents: list[OpponentSpec] = []
    for row in rows:
        registration = tuple(int(card) for card in row["deck"])

        def move(obs, rng, deck=registration):
            del rng
            return controller.act(obs, deck)

        opponents.append(OpponentSpec(
            key=str(row.get("opponent_key", f"{row['archetype']}/qu-v2b")),
            deck=registration,
            move=move,
            weight=int(row["scheduled_games_per_arm"]) / GAMES_PER_ARM,
            policy_id=f"qu-v2b:{qu_sha}",
            schedule_group="recent-field-20260801-05/qu-v2b",
        ))
    return opponents, controller


def build_schedule(rows: Sequence[Mapping[str, Any]]):
    qu = COMMON._load_net(FROZEN_QU, "schedule-only frozen Qu-v2B")
    opponents, _controller = _make_opponents(expand_field_rows(rows), qu)
    schedule = build_paired_schedule(opponents, GAMES_PER_ARM, seed=SEED)
    return opponents, schedule


def _score(result: str) -> float:
    # Invalid scheduled outcomes remain in the denominator and separately make
    # the gate invalid; assigning zero keeps the diagnostic finite/reportable.
    return 1.0 if result == "win" else 0.5 if result == "draw" else 0.0


def paired_delta_ci(
    candidate_records: Sequence[Any], control_records: Sequence[Any],
) -> dict[str, Any]:
    if len(candidate_records) != len(control_records) or len(candidate_records) < 2:
        raise FieldError("paired field records are empty or unequal")
    deltas: list[float] = []
    for candidate, control in zip(candidate_records, control_records, strict=True):
        left_key = (
            candidate.episode_id, candidate.pair_id, candidate.learner_seat,
            candidate.opponent_key,
        )
        right_key = (
            control.episode_id, control.pair_id, control.learner_seat,
            control.opponent_key,
        )
        if left_key != right_key:
            raise FieldError("cross-arm schedule-assignment pairing drifted")
        deltas.append(_score(candidate.result) - _score(control.result))
    mean = sum(deltas) / len(deltas)
    variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    standard_error = math.sqrt(variance / len(deltas))
    return {
        "paired_units": len(deltas),
        "pairing_unit": (
            "same episode_id/pair_id/learner_seat/opponent assignment across arms"
        ),
        "mean_delta": mean,
        "sample_standard_deviation": math.sqrt(variance),
        "standard_error": standard_error,
        "ci95": [mean - Z_95 * standard_error, mean + Z_95 * standard_error],
        "estimator": "two-sided paired normal CI95 over per-assignment score deltas",
    }


def _clean_layered(value: Mapping[str, Any]) -> bool:
    return (
        value.get("fallbacks") == 0
        and value.get("repairs") == 0
        and value.get("exceptions") == {}
        and value.get("off_deck_main_routes") == 0
        and value.get("off_deck_card_routes") == 0
        and value.get("calls") == value.get("main_routes", 0)
            + value.get("card_routes", 0) + value.get("qu_routes", 0)
    )


def _clean_field(value: Mapping[str, Any]) -> bool:
    return value.get("fallbacks") == 0 and value.get("exceptions") == {}


def _clean_candidate(value: Mapping[str, Any]) -> bool:
    shadow = value.get("off_route_shadow", {})
    return (
        _clean_layered(value)
        and value.get("candidate_family_routes", 0) > 0
        and value.get("parent_card_routes", 0) > 0
        and value.get("family_classification_faults") == 0
        and value.get("candidate_runtime_fallbacks") == 0
        and value.get("parent_card_runtime_faults") == 0
        and value.get("off_route_prompts", 0) > 0
        and value.get("off_route_identity_mismatches") == 0
        and value.get("off_route_audit_faults") == {}
        and _clean_layered(shadow)
        and shadow.get("calls") == value.get("off_route_prompts")
    )


def build_lock() -> dict[str, Any]:
    existing = [
        str(path) for path in (LOCK, ATTEMPT, RESULT) if path.exists()
    ]
    if existing:
        raise FieldError(
            "field gate must be locked before any field outcome/attempt: "
            + ", ".join(existing)
        )
    source = MIRROR.PREP.load_and_verify_lock(SOURCE.OUTPUT)
    MIRROR.PREP.verify_bound_artifacts(source)
    try:
        mirror_lock, mirror_paths = MIRROR.load_lock()
    except MIRROR.GameplayError as error:
        raise FieldError(str(error)) from error
    mirror_result = _load_self(
        MIRROR.RESULT, MIRROR.RESULT_SCHEMA, "result_sha256",
    )
    if (
        mirror_result.get("gameplay_lock_sha256") != mirror_lock["lock_sha256"]
        or mirror_result.get("decision", {}).get("valid") is not True
        or mirror_result.get("decision", {}).get("passed") is not True
        or mirror_lock.get("source_lock_sha256") != source["lock_sha256"]
    ):
        raise FieldError("direct mirror gate did not authorize the field gate")
    candidate = mirror_lock.get("artifacts", {}).get("candidate_weights", {})
    candidate_path = mirror_paths["candidate_weights"]
    if (
        not candidate_path.is_file()
        or COMMON.file_sha256(candidate_path) != candidate.get("sha256")
        or candidate.get("sha256")
            != mirror_lock.get("candidate", {}).get("weights_sha256")
    ):
        raise FieldError("direct-gate candidate identity drifted")
    snapshot, rows = load_snapshot()
    grim_snapshot, _grim_variants = load_grim_variants()
    expanded_rows = expand_field_rows(rows)
    opponents, schedule = build_schedule(rows)
    manifest = schedule_manifest(schedule, opponents)
    seats = Counter(row.learner_seat for row in schedule)
    matchups = Counter(opponents[row.opponent_index].key for row in schedule)
    if seats != {0: GAMES_PER_ARM // 2, 1: GAMES_PER_ARM // 2}:
        raise FieldError("field schedule is not exactly seat balanced")
    fixed_paths = {
        "source_lock": SOURCE.OUTPUT,
        "direct_mirror_lock": MIRROR.LOCK,
        "direct_mirror_result": MIRROR.RESULT,
        "candidate_weights": candidate_path,
        "frozen_main": FROZEN_MAIN,
        "frozen_card": FROZEN_CARD,
        "frozen_qu": FROZEN_QU,
        "deck": DECK,
        "field_snapshot": FIELD_SNAPSHOT,
        "field_source_inventory": FIELD_INVENTORY,
        "snapshotter": SNAPSHOTTER,
        "grim_variant_snapshot": GRIM_VARIANT_SNAPSHOT,
        "grim_variant_snapshotter": GRIM_VARIANT_SNAPSHOTTER,
        "evaluator": Path(__file__).resolve(),
        "direct_gameplay_evaluator": Path(MIRROR.__file__).resolve(),
        "layered_controller": Path(CARD_GAME.__file__).resolve(),
        "eval_ab": Path(EVAL.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
        "common_gameplay": Path(COMMON.__file__).resolve(),
    }
    artifacts = {name: _record(path) for name, path in fixed_paths.items()}
    if (
        artifacts["field_snapshot"]["sha256"] != SNAPSHOT_FILE_SHA256
        or artifacts["field_source_inventory"]["sha256"]
            != INVENTORY_FILE_SHA256
        or artifacts["grim_variant_snapshot"]["sha256"]
            != GRIM_VARIANT_FILE_SHA256
        or artifacts["frozen_card"]["sha256"] != SOURCE.PARENT_NPZ_SHA256
    ):
        raise FieldError("fixed field artifact identity drifted")
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_engine_outcomes": True,
        "source_lock_sha256": source["lock_sha256"],
        "direct_mirror_result_sha256": mirror_result["result_sha256"],
        "candidate": {
            "selected_arm": mirror_lock["candidate"]["selected_arm"],
            "weights_sha256": candidate["sha256"],
            "scope": MIRROR.CANDIDATE_SCOPE,
            "outside_scope": "complete frozen Dobi-v1 action",
        },
        "control": {
            "name": "complete frozen ladder-proven Dobi-v1",
            "policy_id": mirror_lock["control"]["policy_id"],
        },
        "field": {
            "snapshot_file_sha256": SNAPSHOT_FILE_SHA256,
            "snapshot_self_sha256": snapshot["snapshot_sha256"],
            "inventory_file_sha256": INVENTORY_FILE_SHA256,
            "inventory_self_sha256": INVENTORY_SELF_SHA256,
            "source_dates": list(SOURCE_DATES),
            "source_registered_seats": SOURCE_REGISTERED_SEATS,
            "included_registered_seats": INCLUDED_REGISTERED_SEATS,
            "included_share": INCLUDED_REGISTERED_SEATS / SOURCE_REGISTERED_SEATS,
            "minimum_observed_share": MINIMUM_OBSERVED_SHARE,
            "grim_variant_snapshot_file_sha256": GRIM_VARIANT_FILE_SHA256,
            "grim_variant_snapshot_self_sha256": grim_snapshot["snapshot_sha256"],
            "definition": (
                "all archetypes at >=0.5% recent registered-seat share, "
                "with the top-eight exact Grim variants renormalized inside "
                "the unchanged aggregate Grim mass"
            ),
            "grim_variant_selection": {
                "rule": grim_snapshot["selection"]["rule"],
                "selected_variants": 8,
                "selected_registered_seats": GRIM_SELECTED_SEATS,
                "selected_share_of_grim": GRIM_SELECTED_SEATS / 17_973,
                "omitted_tail_seats": GRIM_OMITTED_TAIL_SEATS,
                "omitted_tail_share_of_grim": GRIM_OMITTED_TAIL_SEATS / 17_973,
                "omitted_tail_share_of_included_field": (
                    GRIM_OMITTED_TAIL_SEATS / INCLUDED_REGISTERED_SEATS
                ),
                "scheduled_games_per_arm": list(GRIM_ALLOCATIONS),
            },
            "representatives": [{
                "archetype": row["archetype"],
                "variant_rank": row.get("variant_rank"),
                "registered_seats": row["registered_seats"],
                "weight": row["field_weight"],
                "scheduled_games_per_arm": matchups[row["opponent_key"]],
                "deck_sha256": row["representative_deck_sha256"],
                "canonical_deck_sha256": row.get("canonical_deck_sha256"),
            } for row in expanded_rows],
            "no_posthoc_stratum_dropping": True,
        },
        "protocol": {
            "games_per_arm": GAMES_PER_ARM,
            "total_engine_games": TOTAL_ENGINE_GAMES,
            "within_arm_seat_pairs": WITHIN_ARM_SEAT_PAIRS,
            "cross_arm_paired_schedule_units": CROSS_ARM_PAIRED_UNITS,
            "schedule_seed": SEED,
            "seat_counts_per_arm": {"0": GAMES_PER_ARM // 2,
                                      "1": GAMES_PER_ARM // 2},
            "arm_order": ["candidate", "control"],
            "learner_deck": "exact Dobi-v1 Grimmsnarl list in both arms",
            "opponents": (
                "frozen Qu-v2B piloting each Aug 1-5 representative exact list"
            ),
            "identical_schedule_per_arm": True,
            "native_engine_rng": (
                "unseedable; pairing binds schedule assignments, not common "
                "engine random numbers"
            ),
            "score": "(wins + 0.5 * official draws) / scheduled games",
            "pairing_unit": (
                "same episode_id/pair_id/learner_seat/opponent assignment "
                "across candidate and control arms"
            ),
            "interval": (
                "two-sided paired normal CI95 over 5,120 per-assignment "
                "candidate-minus-control score deltas; z=1.959963984540054"
            ),
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "pass": (
                "valid zero-fault arms, exact off-route identity, and paired "
                "CI95 lower bound >= -0.015"
            ),
            "one_schedule_one_attempt": True,
            "no_interim_stopping": True,
            "report_all_included_strata": True,
        },
        "schedule_manifest_sha256": COMMON.canonical_sha256(manifest),
        "artifacts": artifacts,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = COMMON.canonical_sha256(payload)
    return payload


def load_lock() -> tuple[dict[str, Any], dict[str, Path]]:
    lock = _load_self(LOCK, LOCK_SCHEMA, "lock_sha256")
    paths: dict[str, Path] = {}
    for name, descriptor in lock.get("artifacts", {}).items():
        path = Path(str(descriptor.get("path", "")))
        if (
            not path.is_file()
            or COMMON.file_sha256(path) != descriptor.get("sha256")
        ):
            raise FieldError(f"field artifact drifted: {name}")
        paths[name] = path
    required = {
        "source_lock", "direct_mirror_lock", "direct_mirror_result",
        "candidate_weights", "frozen_main", "frozen_card", "frozen_qu",
        "deck", "field_snapshot", "field_source_inventory", "snapshotter",
        "grim_variant_snapshot", "grim_variant_snapshotter",
        "evaluator", "direct_gameplay_evaluator", "layered_controller",
        "eval_ab", "rl_env", "common_gameplay",
    }
    if set(paths) != required:
        raise FieldError("field artifact inventory is incomplete or expanded")
    try:
        source = MIRROR.PREP.load_and_verify_lock(SOURCE.OUTPUT)
        MIRROR.PREP.verify_bound_artifacts(source)
        direct_lock, direct_paths = MIRROR.load_lock()
    except (MIRROR.GameplayError, MIRROR.PREP.ExtractionError) as error:
        raise FieldError(str(error)) from error
    direct_result = _load_self(
        MIRROR.RESULT, MIRROR.RESULT_SCHEMA, "result_sha256",
    )
    expected_paths = {
        "source_lock": SOURCE.OUTPUT,
        "direct_mirror_lock": MIRROR.LOCK,
        "direct_mirror_result": MIRROR.RESULT,
        "candidate_weights": direct_paths["candidate_weights"],
        "frozen_main": FROZEN_MAIN,
        "frozen_card": FROZEN_CARD,
        "frozen_qu": FROZEN_QU,
        "deck": DECK,
        "field_snapshot": FIELD_SNAPSHOT,
        "field_source_inventory": FIELD_INVENTORY,
        "snapshotter": SNAPSHOTTER,
        "grim_variant_snapshot": GRIM_VARIANT_SNAPSHOT,
        "grim_variant_snapshotter": GRIM_VARIANT_SNAPSHOTTER,
        "evaluator": Path(__file__).resolve(),
        "direct_gameplay_evaluator": Path(MIRROR.__file__).resolve(),
        "layered_controller": Path(CARD_GAME.__file__).resolve(),
        "eval_ab": Path(EVAL.__file__).resolve(),
        "rl_env": Path(rl_env.__file__).resolve(),
        "common_gameplay": Path(COMMON.__file__).resolve(),
    }
    if any(paths[name].resolve() != path.resolve()
           for name, path in expected_paths.items()):
        raise FieldError("field artifact path binding drifted")
    load_snapshot()
    load_grim_variants()
    protocol = lock.get("protocol")
    if (
        lock.get("written_before_engine_outcomes") is not True
        or lock.get("source_lock_sha256") != source["lock_sha256"]
        or lock.get("direct_mirror_result_sha256")
            != direct_result["result_sha256"]
        or direct_result.get("gameplay_lock_sha256")
            != direct_lock["lock_sha256"]
        or direct_result.get("decision", {}).get("valid") is not True
        or direct_result.get("decision", {}).get("passed") is not True
        or lock.get("candidate") != {
            "selected_arm": direct_lock["candidate"]["selected_arm"],
            "weights_sha256": direct_lock["candidate"]["weights_sha256"],
            "scope": MIRROR.CANDIDATE_SCOPE,
            "outside_scope": "complete frozen Dobi-v1 action",
        }
        or lock.get("control") != {
            "name": "complete frozen ladder-proven Dobi-v1",
            "policy_id": direct_lock["control"]["policy_id"],
        }
        or not isinstance(protocol, Mapping)
        or protocol.get("games_per_arm") != GAMES_PER_ARM
        or protocol.get("total_engine_games")
            != TOTAL_ENGINE_GAMES
        or protocol.get("within_arm_seat_pairs") != WITHIN_ARM_SEAT_PAIRS
        or protocol.get("cross_arm_paired_schedule_units")
            != CROSS_ARM_PAIRED_UNITS
        or protocol.get("schedule_seed") != SEED
        or protocol.get("seat_counts_per_arm") != {
            "0": GAMES_PER_ARM // 2, "1": GAMES_PER_ARM // 2,
        }
        or protocol.get("noninferiority_margin") != NONINFERIORITY_MARGIN
        or protocol.get("one_schedule_one_attempt") is not True
        or protocol.get("no_interim_stopping") is not True
        or lock.get("promotion_authority") is not False
        or lock.get("upload_authority") is not False
    ):
        raise FieldError("field protocol/candidate contract drifted")
    return lock, paths


def run(lock: Mapping[str, Any], paths: Mapping[str, Path], *, quiet: bool):
    if ATTEMPT.exists() or RESULT.exists():
        raise FieldError("field-gate attempt has already been consumed")
    _snapshot, rows = load_snapshot()
    expanded_rows = expand_field_rows(rows)
    deck = COMMON.read_deck(paths["deck"])
    main = COMMON._load_net(paths["frozen_main"], "frozen Dobi ST_MAIN")
    candidate_card = COMMON._load_net(
        paths["candidate_weights"], "selected candidate ST_CARD",
    )
    frozen_card = COMMON._load_net(paths["frozen_card"], "frozen Dobi ST_CARD")
    qu = COMMON._load_net(paths["frozen_qu"], "frozen Qu-v2B")
    candidate = AuditedSelectiveCardController(
        main, candidate_card, frozen_card, qu,
        "selective-card+frozen-dobi/field", deck,
    )
    control = CARD_GAME.LayeredMirrorCardController(
        main, frozen_card, qu, "complete-frozen-dobi/field", deck,
    )
    candidate_opponents, candidate_field = _make_opponents(expanded_rows, qu)
    control_opponents, control_field = _make_opponents(expanded_rows, qu)
    candidate_schedule = build_paired_schedule(
        candidate_opponents, GAMES_PER_ARM, seed=SEED,
    )
    control_schedule = build_paired_schedule(
        control_opponents, GAMES_PER_ARM, seed=SEED,
    )
    left_manifest = schedule_manifest(candidate_schedule, candidate_opponents)
    right_manifest = schedule_manifest(control_schedule, control_opponents)
    expected = lock["schedule_manifest_sha256"]
    if (
        left_manifest != right_manifest
        or COMMON.canonical_sha256(left_manifest) != expected
        or COMMON.canonical_sha256(right_manifest) != expected
    ):
        raise FieldError("runtime schedules differ from the prospective lock")
    SOURCE.write_new(ATTEMPT, {
        "schema": ATTEMPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_first_engine_outcome": True,
        "field_lock_sha256": lock["lock_sha256"],
    })
    candidate_result = EVAL.run_series(
        "selective-card-v1/recent-field-20260801-05",
        candidate, deck, candidate_opponents, candidate_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
        verbose=not quiet,
    )
    control_result = EVAL.run_series(
        "frozen-dobi-v1/recent-field-20260801-05",
        control, deck, control_opponents, control_schedule,
        max_selects=MAX_SELECTS, time_bank_s=TIME_BANK_S,
        verbose=not quiet,
    )
    comparison = paired_delta_ci(
        candidate_result.records, control_result.records,
    )
    candidate_diag = candidate.diagnostics()
    control_diag = control.diagnostics()
    candidate_field_diag = candidate_field.diagnostics()
    control_field_diag = control_field.diagnostics()
    valid = (
        len(candidate_result.records) == GAMES_PER_ARM
        and len(control_result.records) == GAMES_PER_ARM
        and candidate_result.gate_valid and control_result.gate_valid
        and _clean_candidate(candidate_diag)
        and _clean_layered(control_diag)
        and _clean_field(candidate_field_diag)
        and _clean_field(control_field_diag)
    )
    passed = bool(
        valid and comparison["ci95"][0] >= NONINFERIORITY_MARGIN
    )
    by_matchup: dict[str, Any] = {}
    for opponent in candidate_opponents:
        left = [row for row in candidate_result.records
                if row.opponent_key == opponent.key]
        right = [row for row in control_result.records
                 if row.opponent_key == opponent.key]
        by_matchup[opponent.key] = {
            "games_per_arm": len(left),
            "candidate": dict(Counter(row.result for row in left)),
            "control": dict(Counter(row.result for row in right)),
            "paired_candidate_minus_control": paired_delta_ci(left, right),
        }
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "field_lock_sha256": lock["lock_sha256"],
        "decision": {
            "valid": valid,
            "passed_noninferiority": passed,
            "margin": NONINFERIORITY_MARGIN,
            "candidate_minus_control": comparison,
            "off_route_identity": {
                "audited_prompts": candidate_diag["off_route_prompts"],
                "mismatches": candidate_diag["off_route_identity_mismatches"],
                "faults": candidate_diag["off_route_audit_faults"],
                "passed": (
                    candidate_diag["off_route_prompts"] > 0
                    and candidate_diag["off_route_identity_mismatches"] == 0
                    and candidate_diag["off_route_audit_faults"] == {}
                ),
            },
        },
        "summaries": {
            "candidate": candidate_result.summary(),
            "control": control_result.summary(),
        },
        "by_matchup": by_matchup,
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
            "candidate": environment_manifest(
                deck, candidate_opponents, str(paths["field_snapshot"]),
            ),
            "control": environment_manifest(
                deck, control_opponents, str(paths["field_snapshot"]),
            ),
        },
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["result_sha256"] = COMMON.canonical_sha256(payload)
    SOURCE.write_new(RESULT, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.lock_only == args.run:
        parser.error("choose exactly one of --lock-only or --run")
    try:
        if args.lock_only:
            if LOCK.exists():
                raise FieldError(f"refusing to overwrite {LOCK}")
            payload = build_lock()
            SOURCE.write_new(LOCK, payload)
            print(json.dumps({
                "lock_sha256": payload["lock_sha256"],
                "protocol": payload["protocol"],
                "field": payload["field"],
            }, sort_keys=True))
        else:
            lock, paths = load_lock()
            payload = run(lock, paths, quiet=args.quiet)
            print(json.dumps(payload["decision"], sort_keys=True))
    except (OSError, TypeError, ValueError, FieldError,
            COMMON.EvaluationError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
