"""Freeze the selective Sixth Sense -> Dobi-v1 ST_CARD experiment."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import il_dataset, index_corpus  # noqa: E402
from tools.research import (  # noqa: E402
    lock_dobi_v1_elite_teacher_main_v1 as MAIN_LOCK,
)


RUN = ROOT / "tools/checkpoints/dobi-v1-elite-teacher-card-v1"
OUTPUT = RUN / "cohort-lock.json"
PREFERENCES = RUN / "preferences.jsonl.gz"
PRESERVATION = RUN / "dobi-preservation.jsonl.gz"
EXTRACTION_RESULT = RUN / "extraction-result.json"
TRAINING_RESULT = RUN / "training-result.json"
BEHAVIOR_METRICS = RUN / "behavior-metrics.json"
SCREEN_RESULT = RUN / "behavior-screen-result.json"

TEACHER_DIR = ROOT / (
    "tools/checkpoints/dobi-v1-top-grim-comparison-v1/replays/sixth-sense"
)
DOBI_DIR = ROOT / (
    "tools/checkpoints/dobi-v1-top-grim-comparison-v1/replays/dobi-v1"
)
TEACHER_TEAM = "Sixth Sense"
DOBI_TEAM = "増殖するG"
TEACHER_SUBMISSION_ID = 55_138_264
DOBI_SUBMISSION_IDS = [
    55_193_892, 55_194_498, 55_195_586,
    55_195_620, 55_220_532, 55_305_666,
]
TEACHER_RECEIPT_SHA256 = (
    "8013bfd769b9979fc2e8cb25d684bca636c0de395907d32f611cf38b0b635ceb"
)
DOBI_RECEIPT_SHA256 = (
    "ad8d6e04e3e8dcae6a958dfd4ea646d54db4d6b0af39c4c4369dede5bd4f00d1"
)

TARGET_DECK_PATH = ROOT / "decks/md_v1_grimmsnarl.csv"
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
PARENT_NPZ = ROOT / "agent/md_v2_card_weights.npz"
PARENT_NPZ_SHA256 = (
    "1aef7068130072dd00afe0e85d6485d9749c6326351597339c1bd33d6d239cf7"
)
PARENT_CHECKPOINT = ROOT / (
    "tools/checkpoints/md-v2-card-v1/model/candidate-qu-v2a-checkpoint.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "cd843d9564e9e8eb899cab43a9fb1f7dc3d7263800fb1380fd276428ee001161"
)
DOBI_ARCHIVE = ROOT / "submission-dobi-v1-unsigned.tar.gz"
DOBI_ARCHIVE_SHA256 = (
    "fdd50192ab1bf4fdb2097e4f2bd3c015a841ec1099aa6fbe806ea760db5c9c3f"
)
DOBI_PACKAGE_MANIFEST = ROOT / "tools/checkpoints/dobi-v1/package-manifest.json"
DOBI_PACKAGE_MANIFEST_FILE_SHA256 = (
    "c8c5c353dac1cc9dbf62d07e24054224337f7548fe4609e87503ac0dc227f9df"
)
DOBI_PACKAGE_MANIFEST_SHA256 = (
    "8be4aee761d0a09f73fbe8ff623aa4ce6600048f632fef2b5db23beec2e77ad6"
)
GRIM_VARIANT_SNAPSHOT_SHA256 = (
    "c27b337495ce2430e83bb9ec5f408e9150824e6b6c8ad256708c1f082bf2cd29"
)
DOBI_MEMBER_SHA256 = {
    "agent/model.py": "238a21d2830ae067178d01913e7fa5092bd33aa38cb6d34ffc936fef769556ef",
    "agent/qu_v2_features.py": "b2225e0075fa7597c1df6afbfc1a5e18427d6e6d64e5bcc1a16ae92ba34594d7",
    "agent/md_v1.py": "061a4f043a4725f60e54703ef4dcb1ad935b159ad51ae04cc38c4172c301bf28",
    "agent/md_v1_weights.npz": "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c",
    "agent/md_v2_card.py": "4957b3adc5748b1b5468d750eb4a92effabe81a96fcb4b38b5d8edda7eb6858a",
    "agent/md_v2_card_weights.npz": PARENT_NPZ_SHA256,
    "agent/weights.npz": "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447",
    "agent/obsview.py": "d04119a25f3f569e22276a0a8d48824c23fa0cbbb7b080f265651a5ee712dc72",
    "agent/safety.py": "3d1b6bc84ade3bc8bd1e996e40bf25f06788e82cd67412b5cee389aa42bb417e",
    "agent/policy.py": "4dbbd18e30d1678cf509fa98537747b152406bcd023e635b69bddbcd38b535ba",
    "decks/deck.csv": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
}

TEACHER_SPLIT_DOMAIN = (
    "ptcg.dobi-v1.elite-teacher-card-v1.teacher-split.v1"
)
PRESERVATION_SPLIT_DOMAIN = (
    "ptcg.dobi-v1.elite-teacher-card-v1.preservation-split.v1"
)
LOCK_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.cohort-lock.v1"
EXPECTED_TEACHER_FILES = 300
EXPECTED_TEACHER_MIRROR_WINS = 33
EXPECTED_DOBI_NUMERIC_FILES = 313
EXPECTED_DOBI_VALID_FILES = 307
EXPECTED_DOBI_MIRRORS = 68

FAMILY_WEIGHTS = {
    "munkidori_damage_source": 2.0,
    "munkidori_damage_destination": 3.0,
    "spikemuth": 1.0,
    "poke_pad": 1.0,
    "petrel": 1.5,
    "poffin": 1.5,
    "night_stretcher": 1.0,
    "boss": 1.0,
}


class LockError(RuntimeError):
    """The prospective cohort or artifact contract is invalid."""


sha256_file = MAIN_LOCK.sha256_file
canonical_sha256 = MAIN_LOCK.canonical_sha256
runtime_environment = MAIN_LOCK.runtime_environment
write_new = MAIN_LOCK.write_new


def target_deck() -> tuple[int, ...]:
    try:
        deck = tuple(sorted(
            int(line.strip())
            for line in TARGET_DECK_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ))
    except (OSError, ValueError) as error:
        raise LockError(f"cannot read target deck: {error}") from error
    if len(deck) != 60 or index_corpus.deck_sha256(deck) != TARGET_DECK_SHA256:
        raise LockError("target deck identity drifted")
    return deck


def _explicit_seat(document: Mapping[str, Any], team: str) -> int:
    info = document.get("info")
    names = info.get("TeamNames") if isinstance(info, Mapping) else None
    if not isinstance(names, list) or len(names) != 2:
        raise LockError("replay does not expose exactly two team names")
    matches = [index for index, name in enumerate(names) if name == team]
    if len(matches) != 1:
        raise LockError(f"expected exactly one {team!r} seat")
    return matches[0]


def _inspect(path: Path, *, team: str, deck: Sequence[int]) -> dict[str, Any]:
    if not path.stem.isdigit():
        raise LockError(f"non-numeric replay filename: {path}")
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise LockError(f"malformed replay: {path}") from error
    if not isinstance(document, dict):
        raise LockError(f"replay root is not an object: {path}")
    inspected = index_corpus.inspect_document(raw)
    episode_id = int(path.stem)
    if (
        inspected.get("valid_for_bc") is not True
        or inspected.get("document_episode_id") != episode_id
        or document.get("statuses") != ["DONE", "DONE"]
    ):
        raise LockError(f"replay is not a complete BC episode: {path}")
    seat = _explicit_seat(document, team)
    decks = il_dataset.decks_from_document(document)
    actor = tuple(sorted(int(card) for card in decks.get(seat, ())))
    opponent = tuple(sorted(int(card) for card in decks.get(1 - seat, ())))
    if actor != tuple(deck) or len(opponent) != 60:
        raise LockError(f"actor/opponent deck identity drifted: {path}")
    rewards = document.get("rewards")
    if (
        not isinstance(rewards, list)
        or len(rewards) != 2
        or float(rewards[seat]) not in (-1.0, 0.0, 1.0)
        or float(rewards[seat]) != -float(rewards[1 - seat])
    ):
        raise LockError(f"invalid replay reward: {path}")
    reward = float(rewards[seat])
    return {
        "episode_id": episode_id,
        "path": str(path.resolve()),
        "content_sha256": hashlib.sha256(raw).hexdigest(),
        "actor_seat": seat,
        "team": team,
        "outcome": "win" if reward > 0 else "loss" if reward < 0 else "draw",
        "reward": reward,
        "exact_mirror": opponent == tuple(deck),
        "opponent_deck_sha256": index_corpus.deck_sha256(opponent),
    }


def _numeric_files(directory: Path, expected: int) -> list[Path]:
    paths = sorted(
        (path for path in directory.glob("[0-9]*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if len(paths) != expected:
        raise LockError(f"expected {expected} files in {directory}, got {len(paths)}")
    return paths


def _split_rank(domain: str, episode_id: int) -> int:
    material = f"{domain}\0{episode_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _assign(rows: Sequence[Mapping[str, Any]], validation: int,
            domain: str) -> list[dict[str, Any]]:
    if not 0 < validation < len(rows):
        raise LockError("invalid game split size")
    ordered = sorted(rows, key=lambda row: (
        _split_rank(domain, int(row["episode_id"])), int(row["episode_id"]),
    ))
    validation_ids = {
        int(row["episode_id"]) for row in ordered[:validation]
    }
    result = []
    for row in sorted(rows, key=lambda item: int(item["episode_id"])):
        enriched = dict(row)
        episode_id = int(row["episode_id"])
        enriched["supervision_split"] = (
            "validation" if episode_id in validation_ids else "train"
        )
        enriched["split_rank"] = _split_rank(domain, episode_id)
        result.append(enriched)
    return result


def _artifact(path: Path, expected: str | None = None) -> dict[str, str]:
    if not path.is_file():
        raise LockError(f"missing bound artifact: {path}")
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise LockError(f"bound artifact drifted: {path}")
    return {"path": str(path.resolve()), "sha256": digest}


def verify_frozen_dobi_archive() -> dict[str, Any]:
    """Bind the control to the actual ladder-proven Dobi-v1 tarball."""
    if sha256_file(DOBI_ARCHIVE) != DOBI_ARCHIVE_SHA256:
        raise LockError("frozen Dobi-v1 archive identity drifted")
    if sha256_file(DOBI_PACKAGE_MANIFEST) \
            != DOBI_PACKAGE_MANIFEST_FILE_SHA256:
        raise LockError("frozen Dobi-v1 package manifest file drifted")
    try:
        manifest = json.loads(DOBI_PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise LockError("cannot read frozen Dobi-v1 package manifest") from error
    body = {key: value for key, value in manifest.items()
            if key != "manifest_sha256"}
    if (
        manifest.get("schema") != "ptcg.dobi-v1.package.v1"
        or manifest.get("name") != "dobi-v1"
        or manifest.get("manifest_sha256") != DOBI_PACKAGE_MANIFEST_SHA256
        or canonical_sha256(body) != DOBI_PACKAGE_MANIFEST_SHA256
        or manifest.get("candidate", {}).get("sha256") != DOBI_ARCHIVE_SHA256
        or manifest.get("candidate", {}).get("st_main_weights_sha256")
            != DOBI_MEMBER_SHA256["agent/md_v1_weights.npz"]
    ):
        raise LockError("frozen Dobi-v1 package manifest contract drifted")
    try:
        with tarfile.open(DOBI_ARCHIVE, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise LockError("frozen Dobi-v1 archive has duplicate members")
            by_name = {member.name: member for member in members}
            for name, expected in DOBI_MEMBER_SHA256.items():
                member = by_name.get(name)
                if member is None or not member.isfile():
                    raise LockError(f"frozen Dobi-v1 member missing: {name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise LockError(f"cannot read frozen Dobi-v1 member: {name}")
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                if digest.hexdigest() != expected:
                    raise LockError(f"frozen Dobi-v1 member drifted: {name}")
    except (OSError, tarfile.TarError) as error:
        raise LockError("cannot audit frozen Dobi-v1 archive") from error
    if (
        sha256_file(PARENT_NPZ) != DOBI_MEMBER_SHA256[
            "agent/md_v2_card_weights.npz"
        ]
        or sha256_file(TARGET_DECK_PATH) != DOBI_MEMBER_SHA256["decks/deck.csv"]
    ):
        raise LockError("worktree frozen Dobi component differs from archive")
    return manifest


def verify_lock(payload: Mapping[str, Any]) -> bool:
    if payload.get("schema") != LOCK_SCHEMA:
        return False
    body = {key: value for key, value in payload.items() if key != "lock_sha256"}
    return payload.get("lock_sha256") == canonical_sha256(body)


def build_lock() -> dict[str, Any]:
    downstream = (
        OUTPUT, PREFERENCES, PRESERVATION, EXTRACTION_RESULT,
        TRAINING_RESULT, BEHAVIOR_METRICS, SCREEN_RESULT,
        RUN / "arms",
        RUN / "gameplay/lock.json",
        RUN / "gameplay/result.json",
        RUN / "gameplay/result.json.attempt.json",
        RUN / "field/lock.json",
        RUN / "field/attempt.json",
        RUN / "field/result.json",
        RUN / "package-manifest.json",
        ROOT / "submission-dobi-v1-elite-teacher-card-v1-unsigned.tar.gz",
    )
    existing = [str(path) for path in downstream if path.exists()]
    if existing:
        raise LockError(
            "prospective source lock requires an outcome-free run directory: "
            + ", ".join(existing)
        )
    dobi_manifest = verify_frozen_dobi_archive()
    deck = target_deck()
    teacher_receipt = TEACHER_DIR / ".done_subs.json"
    dobi_receipt = DOBI_DIR / ".done_subs.json"
    if json.loads(teacher_receipt.read_text(encoding="utf-8")) != [
        TEACHER_SUBMISSION_ID
    ]:
        raise LockError("teacher acquisition receipt drifted")
    if json.loads(dobi_receipt.read_text(encoding="utf-8")) != DOBI_SUBMISSION_IDS:
        raise LockError("Dobi acquisition receipt drifted")

    teacher_all = [
        _inspect(path, team=TEACHER_TEAM, deck=deck)
        for path in _numeric_files(TEACHER_DIR, EXPECTED_TEACHER_FILES)
    ]
    teacher = [
        row for row in teacher_all
        if row["exact_mirror"] and row["outcome"] == "win"
    ]
    # Six byte-valid exact mirrors have the Dobi team in both seats and cannot
    # identify which submission instance is the actor.  Reuse only the 307
    # independently resolved rows from the bound comparison result; never pick
    # a seat from those six ambiguous twins.
    comparison_path = ROOT / (
        "tools/checkpoints/dobi-v1-top-grim-comparison-v1/result.json"
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison_games = comparison.get("cohorts", {}).get("dobi-v1", {}).get(
        "games"
    )
    if not isinstance(comparison_games, list) or len(comparison_games) \
            != EXPECTED_DOBI_VALID_FILES:
        raise LockError("bound comparison Dobi inventory drifted")
    valid_ids = {int(row["episode_id"]) for row in comparison_games}
    dobi_paths = _numeric_files(DOBI_DIR, EXPECTED_DOBI_NUMERIC_FILES)
    selected_paths = [path for path in dobi_paths if int(path.stem) in valid_ids]
    if len(selected_paths) != EXPECTED_DOBI_VALID_FILES:
        raise LockError("resolved Dobi replay inventory is incomplete")
    dobi_all = [
        _inspect(path, team=DOBI_TEAM, deck=deck) for path in selected_paths
    ]
    dobi = [row for row in dobi_all if row["exact_mirror"]]
    if len(teacher) != EXPECTED_TEACHER_MIRROR_WINS:
        raise LockError(f"teacher mirror-win count drifted: {len(teacher)}")
    if len(dobi) != EXPECTED_DOBI_MIRRORS:
        raise LockError(f"Dobi mirror count drifted: {len(dobi)}")
    if {row["episode_id"] for row in teacher} & {
        row["episode_id"] for row in dobi
    }:
        raise LockError("teacher and preservation episode sets overlap")
    if len({row["content_sha256"] for row in teacher + dobi}) != len(teacher) + len(dobi):
        raise LockError("selected cohorts contain duplicate content")

    teacher = _assign(teacher, 7, TEACHER_SPLIT_DOMAIN)
    dobi = _assign(dobi, 14, PRESERVATION_SPLIT_DOMAIN)
    if Counter(row["supervision_split"] for row in teacher) != {
        "train": 26, "validation": 7,
    }:
        raise LockError("teacher split count drifted")
    if Counter(row["supervision_split"] for row in dobi) != {
        "train": 54, "validation": 14,
    }:
        raise LockError("preservation split count drifted")

    source_paths = {
        "preregistration": ROOT / (
            "tools/research/dobi-v1-elite-teacher-card-v1-preregistration.md"
        ),
        "lock_builder": Path(__file__).resolve(),
        "extractor": ROOT / (
            "tools/research/prepare_dobi_v1_elite_teacher_card_v1.py"
        ),
        "trainer": ROOT / (
            "tools/research/train_dobi_v1_elite_teacher_card_v1.py"
        ),
        "behavior_evaluator": ROOT / (
            "tools/research/eval_dobi_v1_elite_teacher_card_v1_behavior.py"
        ),
        "gameplay_evaluator": ROOT / (
            "tools/research/eval_dobi_v1_elite_teacher_card_v1_gameplay.py"
        ),
        "field_evaluator": ROOT / (
            "tools/research/eval_dobi_v1_elite_teacher_card_v1_field.py"
        ),
        "package_builder": ROOT / (
            "tools/build_dobi_v1_elite_teacher_card_v1_submission.py"
        ),
        "tests": ROOT / "tests/test_dobi_v1_elite_teacher_card_v1.py",
        "field_tests": ROOT / (
            "tests/test_dobi_v1_elite_teacher_card_v1_field.py"
        ),
        "deployment_card_router": ROOT / "agent/dobi_v1_card.py",
        "deployment_card_router_tests": ROOT / (
            "tests/test_dobi_v1_card_runtime.py"
        ),
        "lineage_tests": ROOT / (
            "tests/test_dobi_v1_elite_teacher_card_lineage.py"
        ),
        "field_snapshotter": ROOT / (
            "tools/research/snapshot_recent_weighted_field_from_archives.py"
        ),
        "grim_variant_snapshotter": ROOT / (
            "tools/research/snapshot_grim_variants_from_archives.py"
        ),
        "grim_variant_snapshot": ROOT / (
            "tools/checkpoints/recent-field-20260801-05/grim-variants.json"
        ),
        "field_snapshot": ROOT / (
            "tools/checkpoints/recent-field-20260801-05/"
            "recent-field-20260801-05.json"
        ),
        "field_source_inventory": ROOT / (
            "tools/checkpoints/recent-field-20260801-05/inventory.json"
        ),
        "comparison_result": comparison_path,
        "card_disagreement_result": ROOT / (
            "tools/checkpoints/dobi-v1-top-grim-comparison-v1/"
            "st-card-disagreement-v1.json"
        ),
        "prior_card_validation": ROOT / (
            "tools/checkpoints/md-v2-card-v1/validation-result.json"
        ),
        "prior_card_gameplay": ROOT / (
            "tools/checkpoints/md-v2-card-v1/gameplay-result.json"
        ),
        "parent_npz": PARENT_NPZ,
        "parent_checkpoint": PARENT_CHECKPOINT,
        "frozen_dobi_archive": DOBI_ARCHIVE,
        "frozen_dobi_package_manifest": DOBI_PACKAGE_MANIFEST,
        "target_deck": TARGET_DECK_PATH,
        "teacher_receipt": teacher_receipt,
        "dobi_receipt": dobi_receipt,
        "production_model": ROOT / "agent/model.py",
        "production_features": ROOT / "agent/qu_v2_features.py",
        "production_policy": ROOT / "agent/policy.py",
        "production_safety": ROOT / "agent/safety.py",
        "card_router": ROOT / "agent/md_v2_card.py",
        "observation_view": ROOT / "agent/obsview.py",
        "action_parser": ROOT / "tools/analyze_ladder_replays.py",
        "dataset_parser": ROOT / "tools/il_dataset.py",
        "family_semantics": ROOT / (
            "tools/research/analyze_dobi_v1_elite_card_disagreement.py"
        ),
        "research_model": ROOT / "tools/research/qu_v2a_model.py",
        "research_features": ROOT / "tools/research/qu_v2a_features.py",
        "training_utilities": ROOT / (
            "tools/research/train_dobi_v1_elite_teacher_main_v1.py"
        ),
        "main_lock_utilities": ROOT / (
            "tools/research/lock_dobi_v1_elite_teacher_main_v1.py"
        ),
        "main_extraction_utilities": ROOT / (
            "tools/research/prepare_dobi_v1_elite_teacher_main_v1.py"
        ),
        "base_training_utilities": ROOT / "tools/research/train_qu_v2a.py",
        "main_behavior_utilities": ROOT / (
            "tools/research/eval_dobi_v1_elite_teacher_main_v1_behavior.py"
        ),
        "corpus_indexer": ROOT / "tools/index_corpus.py",
    }
    artifacts = {
        name: _artifact(path) for name, path in source_paths.items()
    }
    for name, expected in (
        ("parent_npz", PARENT_NPZ_SHA256),
        ("parent_checkpoint", PARENT_CHECKPOINT_SHA256),
        ("frozen_dobi_archive", DOBI_ARCHIVE_SHA256),
        ("frozen_dobi_package_manifest", DOBI_PACKAGE_MANIFEST_FILE_SHA256),
        ("grim_variant_snapshot", GRIM_VARIANT_SNAPSHOT_SHA256),
        ("teacher_receipt", TEACHER_RECEIPT_SHA256),
        ("dobi_receipt", DOBI_RECEIPT_SHA256),
    ):
        if artifacts[name]["sha256"] != expected:
            raise LockError(f"fixed artifact hash drifted: {name}")

    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_optimizer_or_candidate_outcomes": True,
        "candidate_only": True,
        "runtime_environment": runtime_environment(),
        "target": {
            "deck_sha256": TARGET_DECK_SHA256,
            "select_type": 1,
            "runtime_scope": (
                "exact own deck + ST_CARD + public opposing Grim signature + "
                "one of eight fixed semantic families"
            ),
            "teacher_team": TEACHER_TEAM,
            "teacher_submission_id": TEACHER_SUBMISSION_ID,
            "parent_npz_sha256": PARENT_NPZ_SHA256,
            "frozen_dobi_archive_sha256": DOBI_ARCHIVE_SHA256,
            "frozen_dobi_manifest_sha256": dobi_manifest["manifest_sha256"],
        },
        "disclosed_exploratory_evidence": {
            "leader_exact_mirror_record": {"win": 33, "loss": 20},
            "dobi_exact_mirror_record": {"win": 26, "loss": 42},
            "deployed_winner_seat_semantic_disagreement": [459, 1581],
            "prior_st_card_gameplay_score": 0.63828125,
            "prior_st_main_teacher_experiment": (
                "failed locked behavior screen; no candidate reused"
            ),
        },
        "cohorts": {
            "teacher": {
                "role": "explicit winning Sixth Sense seat only",
                "split_domain": TEACHER_SPLIT_DOMAIN,
                "games": teacher,
                "counts": {"train": 26, "validation": 7},
            },
            "preservation": {
                "role": "explicit Dobi-v1 seat, wins and losses",
                "split_domain": PRESERVATION_SPLIT_DOMAIN,
                "games": dobi,
                "counts": {"train": 54, "validation": 14},
            },
            "episode_overlap": 0,
        },
        "preference_extraction": {
            "families": list(FAMILY_WEIGHTS),
            "exclude": ["other_st_card", "semantic order-only", "raw index-only"],
            "family_weights": FAMILY_WEIGHTS,
            "normalization": "each touched teacher game has total mass 1.0",
            "public_signature_required": True,
            "deployment_family_gate": True,
            "other_st_card_runtime": "frozen parent",
        },
        "training": {
            "device": "cpu",
            "epochs": 5,
            "batch_size": 64,
            "learning_rate": 1e-5,
            "weight_decay": 1e-5,
            "gradient_clip": 1.0,
            "seed": 202608071,
            "arms": [
                {"name": "kl1", "kl_coefficient": 1.0},
                {"name": "kl3", "kl_coefficient": 3.0},
            ],
            "trainable_prefixes": ["option1.", "context1.", "policy."],
            "validation_in_optimizer": False,
            "production_numpy_parent_authoritative": True,
            "same_rows_and_orders_across_arms": True,
        },
        "behavior_screen": {
            "minimum_overall_teacher_capture": 0.20,
            "minimum_destination_captures": 2,
            "minimum_destination_capture_rate": 0.20,
            "minimum_teacher_game_touch_rate": 0.50,
            "maximum_preservation_change": 0.08,
            "maximum_mean_parent_kl": 0.04,
            "required_faults": 0,
            "arm_selection": [
                "higher destination capture rate",
                "higher overall capture rate",
                "lower preservation change rate",
                "kl3 conservative tie break",
            ],
            "failure": "stop without alternate arms, epochs, families, or thresholds",
        },
        "gameplay_gates": {
            "direct_mirror": {
                "games": 10240,
                "seat_balanced": True,
                "one_schedule_one_attempt": True,
                "no_interim_stopping": True,
                "pass": "zero faults and Wilson CI95 lower bound > 0.50",
            },
            "field": {
                "games_per_arm": 5120,
                "total_engine_games": 10240,
                "recent_frequency_snapshot": "2026-08-01 through 2026-08-05",
                "snapshot_file_sha256": (
                    "f558119122a1fe63b289b032e3bef9a91851c105e5cd670ae"
                    "85e58ee7398dfbd"
                ),
                "included_registered_seats": 46429,
                "source_registered_seats": 46750,
                "includes_grimmsnarl": True,
                "grim_variant_selection": {
                    "rule": "smallest descending prefix reaching >=99%",
                    "selected_variants": 8,
                    "selected_registered_seats": 17813,
                    "omitted_tail_seats": 160,
                    "scheduled_games_per_arm": [1660, 210, 50, 24, 16, 10, 6, 6],
                },
                "seat_balanced": True,
                "cross_arm_paired_schedule_units": 5120,
                "pairing_unit": (
                    "same episode/pair/learner-seat/opponent assignment"
                ),
                "interval": (
                    "two-sided paired normal CI95 over per-assignment score "
                    "deltas; native engine RNG is unseedable"
                ),
                "one_schedule_one_attempt": True,
                "pass": "zero faults and paired delta CI95 lower >= -0.015",
            },
            "identity": (
                "exact frozen-Dobi action outside public signature plus fixed "
                "family route"
            ),
            "deployment": [
                "tests/test_safety.py", "200-game random smoke",
                "exact-extracted-tarball non-owner UID audit",
            ],
        },
        "artifacts": artifacts,
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    if OUTPUT.exists():
        print(f"error: refusing to overwrite {OUTPUT}", file=sys.stderr)
        return 2
    try:
        payload = build_lock()
        write_new(OUTPUT, payload)
    except (OSError, TypeError, ValueError, LockError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "path": str(OUTPUT.resolve()),
        "lock_sha256": payload["lock_sha256"],
        "teacher_games": 33,
        "preservation_games": 68,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
