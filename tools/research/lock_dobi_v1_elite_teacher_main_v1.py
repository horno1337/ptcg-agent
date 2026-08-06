"""Freeze the Sixth Sense -> Dobi-v1 elite-teacher ST_MAIN experiment.

This is the second-stage lock authorized by the observational leader screen.
It binds every replay byte, the uniquely identified teacher seat, the fixed
game-grouped split, the parent policy, the two training arms, and the offline
behavioral gates.  It intentionally grants neither promotion nor upload
authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import analyze_ladder_replays as LADDER  # noqa: E402
from tools import il_dataset, index_corpus  # noqa: E402


RUN = ROOT / "tools/checkpoints/dobi-v1-elite-teacher-main-v1b"
OUTPUT = RUN / "cohort-lock.json"
DEFAULT_COHORT = ROOT / (
    "tools/checkpoints/dobi-v1-top-grim-comparison-v1/replays/sixth-sense"
)
COHORT_RECEIPT_SHA256 = (
    "8013bfd769b9979fc2e8cb25d684bca636c0de395907d32f611cf38b0b635ceb"
)
PREFERENCES = RUN / "preferences.jsonl.gz"
PRESERVATION = RUN / "teacher-preservation.jsonl.gz"
EXTRACTION_RESULT = RUN / "extraction-result.json"
TRAINING_RESULT = RUN / "training-result.json"
BEHAVIOR_METRICS = RUN / "behavior-metrics.json"
SCREEN_RESULT = RUN / "behavior-screen-result.json"
ABORTED_V1_RUN = ROOT / (
    "tools/checkpoints/dobi-v1-elite-teacher-main-v1"
)
ABORTED_V1_RECEIPT = ROOT / (
    "tools/research/dobi-v1-elite-teacher-main-v1-abort.json"
)
ABORTED_V1_RECEIPT_SHA256 = (
    "7d2e2ebb2989ce0dda7fe69dd1dcd2037579e6790ce9df38d37928fd5def3ebf"
)
ABORTED_V1_FILE_SHA256 = {
    "cohort_lock": "b5deb50a715246799337e67ea4acc13605d9601b7656a5ba226028e65018a08f",
    "extraction_result": "1dd29911c747b5f1d1819d08ef418f4eddf1cff5756ba1c8951b87cc4a31cb21",
    "preferences": "144b3245dc49c1672670fdbc9d7171e15cce2f3bf21489a56c4edd1087b1774f",
    "preservation": "be23159f518e9bf6acc0777de843c95628d3696df6d11f8a134aba04039d6e1e",
}

TEAM = "Sixth Sense"
SUBMISSION_ID = 55_138_264
TARGET_DECK_PATH = ROOT / "decks/md_v1_grimmsnarl.csv"
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
PARENT_NPZ = ROOT / (
    "tools/checkpoints/md-v3-mirror-league-100k-v2/training/"
    "terminal-update-16-candidate/candidate-qu-v2a-weights.npz"
)
PARENT_NPZ_SHA256 = (
    "bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c"
)
PARENT_CHECKPOINT = (
    ROOT / "tools/checkpoints/dobi-v1-bc-recent-v1/dobi-v1-bc-adapter.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "1e00476c2500b97aea710c8bd654742191dfdcb2633491296e01898ce42d7f42"
)
OBSERVATIONAL_PREREGISTRATION = ROOT / (
    "tools/research/dobi-v1-top-grim-comparison-v1-preregistration.md"
)
ELITE_PREREGISTRATION = ROOT / (
    "tools/research/dobi-v1-elite-teacher-main-v1-preregistration.md"
)
OBSERVATIONAL_RESULT = ROOT / (
    "tools/checkpoints/dobi-v1-top-grim-comparison-v1/result.json"
)
TRANSITIVE_DEPENDENCIES = {
    "production_model": ROOT / "agent/model.py",
    "production_qu_v2_features": ROOT / "agent/qu_v2_features.py",
    "observation_view": ROOT / "agent/obsview.py",
    "base_features": ROOT / "agent/features.py",
    "card_database_loader": ROOT / "agent/cards.py",
    "card_database": ROOT / "data/cards.json",
    "attack_database": ROOT / "data/attacks.json",
    "ladder_action_rows": ROOT / "tools/analyze_ladder_replays.py",
    "imitation_dataset_parser": ROOT / "tools/il_dataset.py",
    "corpus_indexer": ROOT / "tools/index_corpus.py",
    "episode_downloader": ROOT / "tools/download_episodes.py",
    "research_qu_v2a_features": ROOT / "tools/research/qu_v2a_features.py",
    "research_qu_v2a_model": ROOT / "tools/research/qu_v2a_model.py",
    "research_qu_v2a_trainer": ROOT / "tools/research/train_qu_v2a.py",
}

EXPECTED_FILES = 300
EXPECTED_WINS = 171
EXPECTED_LOSSES = 129
EXPECTED_DRAWS = 0
EXPECTED_EXACT_MIRRORS = 53
EXPECTED_EXACT_MIRROR_WINS = 33
SPLIT_DOMAIN = "ptcg.dobi-v1.elite-teacher-main-v1.split.v1"
LOCK_SCHEMA = "ptcg.dobi-v1.elite-teacher-main-v1b.cohort-lock.v1"


class LockError(RuntimeError):
    """Fail-closed cohort or preregistration error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def runtime_environment() -> dict[str, Any]:
    """Bind the numerical stack used to decode the frozen NumPy parent."""
    config = getattr(np.__config__, "CONFIG", {})
    dependencies = (
        config.get("Build Dependencies", {})
        if isinstance(config, Mapping) else {}
    )
    blas = dependencies.get("blas", {}) if isinstance(
        dependencies, Mapping) else {}
    blas_identity = {
        key: blas.get(key)
        for key in ("name", "version", "openblas configuration")
        if isinstance(blas, Mapping) and blas.get(key) is not None
    }
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "numpy_blas": blas_identity,
        "numpy_blas_sha256": canonical_sha256(blas_identity),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def write_new(
    path: Path, payload: Mapping[str, Any],
    ownership_ledger: list[Path] | None = None,
) -> None:
    """Publish one JSON artifact without overwriting prior evidence."""
    if path.exists():
        raise LockError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        if ownership_ledger is not None:
            ownership_ledger.append(path)
    except BaseException:
        # If an asynchronous interruption lands immediately after link(2),
        # remove only the target that shares our temporary inode.  Never
        # disturb an independently created path.
        try:
            if path.exists() and os.path.samefile(temporary, path):
                path.unlink()
        except OSError:
            pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    else:
        # Once the no-clobber hard link exists, the JSON artifact is complete.
        # A failure to remove only the private temporary name must not turn a
        # successful publication into an apparent failed transaction.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def target_deck() -> tuple[int, ...]:
    deck = tuple(sorted(
        int(line.strip())
        for line in TARGET_DECK_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ))
    if len(deck) != 60 or index_corpus.deck_sha256(deck) != TARGET_DECK_SHA256:
        raise LockError("target Grimmsnarl deck identity drifted")
    return deck


def explicit_teacher_seat(replay: Mapping[str, Any]) -> int:
    """Resolve only the named teacher, never an exact-deck mirror opponent."""
    info = replay.get("info")
    teams = info.get("TeamNames") if isinstance(info, Mapping) else None
    if not isinstance(teams, list) or len(teams) != 2:
        raise LockError("replay does not expose exactly two TeamNames")
    matches = [index for index, name in enumerate(teams) if name == TEAM]
    if len(matches) != 1:
        raise LockError(
            f"expected exactly one {TEAM!r} seat, found {len(matches)}")
    return matches[0]


def inspect_teacher_replay(path: Path, deck: Sequence[int]) -> dict[str, Any]:
    if not path.stem.isdigit():
        raise LockError(f"teacher replay has non-numeric filename: {path}")
    raw = path.read_bytes()
    inspected = index_corpus.inspect_document(raw)
    if inspected.get("valid_for_bc") is not True:
        raise LockError(f"teacher replay is invalid for BC: {path}")
    try:
        replay = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise LockError(f"teacher replay is malformed: {path}: {error}") from error
    seat = explicit_teacher_seat(replay)
    decks = il_dataset.decks_from_document(replay)
    teacher_deck = tuple(sorted(int(card) for card in decks.get(seat, ())))
    if teacher_deck != tuple(deck):
        raise LockError(
            f"named teacher seat does not use the exact target deck: {path}")
    opponent_deck = tuple(sorted(int(card) for card in decks.get(1 - seat, ())))
    if len(opponent_deck) != 60:
        raise LockError(f"opponent registration is incomplete: {path}")
    rewards = replay.get("rewards")
    if (
        not isinstance(rewards, list)
        or len(rewards) != 2
        or float(rewards[seat]) not in (-1.0, 0.0, 1.0)
    ):
        raise LockError(f"teacher reward is invalid: {path}")
    reward = float(rewards[seat])
    outcome = "win" if reward > 0.0 else "loss" if reward < 0.0 else "draw"
    exact_mirror = opponent_deck == tuple(deck)
    matchup = (
        "exact Grimmsnarl mirror"
        if exact_mirror else LADDER.archetype(opponent_deck)
    )
    episode_id = int(path.stem)
    if inspected.get("document_episode_id") != episode_id:
        raise LockError(f"episode ID disagrees with filename: {path}")
    return {
        "episode_id": episode_id,
        "path": str(path.resolve()),
        "content_sha256": hashlib.sha256(raw).hexdigest(),
        "teacher_seat": seat,
        "outcome": outcome,
        "reward": reward,
        "matchup": matchup,
        "exact_mirror": exact_mirror,
        "opponent_deck_sha256": index_corpus.deck_sha256(opponent_deck),
    }


def split_stratum(row: Mapping[str, Any]) -> str:
    return "exact_mirror" if bool(row.get("exact_mirror")) else "nonmirror"


def _split_rank(episode_id: int, stratum: str) -> int:
    material = f"{SPLIT_DOMAIN}\0{stratum}\0{episode_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def assign_game_splits(rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    """Make the locked 80/20 game split in mirror/non-mirror strata."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[split_stratum(row)].append(row)
    assignments: dict[int, str] = {}
    for stratum, group in groups.items():
        ordered = sorted(
            group,
            key=lambda row: (
                _split_rank(int(row["episode_id"]), stratum),
                int(row["episode_id"]),
            ),
        )
        validation_count = (
            min(len(ordered) - 1, max(1, (len(ordered) + 2) // 5))
            if len(ordered) > 1 else 0
        )
        validation = {
            int(row["episode_id"]) for row in ordered[:validation_count]
        }
        for row in ordered:
            episode_id = int(row["episode_id"])
            assignments[episode_id] = (
                "validation" if episode_id in validation else "train"
            )
    return assignments


def _artifact(path: Path, expected: str | None = None) -> dict[str, str]:
    if not path.is_file():
        raise LockError(f"bound artifact is missing: {path}")
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise LockError(f"bound artifact drifted: {path}")
    return {"path": str(path.resolve()), "sha256": digest}


def _aborted_v1_artifacts() -> dict[str, dict[str, str]]:
    """Verify and bind the immutable pre-optimizer v1 abort lineage."""
    paths = {
        "cohort_lock": ABORTED_V1_RUN / "cohort-lock.json",
        "extraction_result": ABORTED_V1_RUN / "extraction-result.json",
        "preferences": ABORTED_V1_RUN / "preferences.jsonl.gz",
        "preservation": ABORTED_V1_RUN / "teacher-preservation.jsonl.gz",
    }
    artifacts = {
        name: _artifact(path, ABORTED_V1_FILE_SHA256[name])
        for name, path in paths.items()
    }
    receipt_artifact = _artifact(
        ABORTED_V1_RECEIPT, ABORTED_V1_RECEIPT_SHA256,
    )
    try:
        receipt = json.loads(ABORTED_V1_RECEIPT.read_text(encoding="utf-8"))
        old_lock = json.loads(paths["cohort_lock"].read_text(encoding="utf-8"))
        old_extraction = json.loads(
            paths["extraction_result"].read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError) as error:
        raise LockError("aborted v1 lineage is unreadable") from error
    old_lock_body = {
        key: value for key, value in old_lock.items() if key != "lock_sha256"
    }
    old_extraction_body = {
        key: value for key, value in old_extraction.items()
        if key != "result_sha256"
    }
    if (
        receipt.get("schema")
        != "ptcg.dobi-v1.elite-teacher-main-v1.abort-receipt.v1"
        or receipt.get("source_commit")
        != "ecea398837608d51de571ae0b3212ebbfd45b093"
        or receipt.get("optimizer_steps") != 0
        or receipt.get("candidate_artifacts_created") != 0
        or receipt.get("cohort_lock_file_sha256")
        != ABORTED_V1_FILE_SHA256["cohort_lock"]
        or receipt.get("extraction_result_file_sha256")
        != ABORTED_V1_FILE_SHA256["extraction_result"]
        or receipt.get("preferences_file_sha256")
        != ABORTED_V1_FILE_SHA256["preferences"]
        or receipt.get("preservation_file_sha256")
        != ABORTED_V1_FILE_SHA256["preservation"]
        or old_lock.get("schema")
        != "ptcg.dobi-v1.elite-teacher-main-v1.cohort-lock.v1"
        or old_lock.get("lock_sha256")
        != canonical_sha256(old_lock_body)
        or old_lock.get("lock_sha256") != receipt.get("cohort_lock_sha256")
        or old_extraction.get("result_sha256")
        != canonical_sha256(old_extraction_body)
        or old_extraction.get("result_sha256")
        != receipt.get("extraction_result_sha256")
        or old_extraction.get("cohort_lock_sha256")
        != old_lock.get("lock_sha256")
        or old_extraction.get("preferences", {}).get("compressed_sha256")
        != ABORTED_V1_FILE_SHA256["preferences"]
        or old_extraction.get("teacher_preservation", {}).get(
            "compressed_sha256"
        ) != ABORTED_V1_FILE_SHA256["preservation"]
    ):
        raise LockError("aborted v1 lineage contract failed")
    absent = receipt.get("required_absent_relative_paths")
    if not isinstance(absent, list) or any(
        (ABORTED_V1_RUN / str(relative)).exists() for relative in absent
    ):
        raise LockError("aborted v1 unexpectedly contains candidate outcomes")
    return {"abort_receipt": receipt_artifact, **artifacts}


def build_lock(cohort_dir: Path = DEFAULT_COHORT) -> dict[str, Any]:
    deck = target_deck()
    aborted_v1 = _aborted_v1_artifacts()
    cohort_dir = cohort_dir.expanduser().resolve()
    receipt = cohort_dir / ".done_subs.json"
    try:
        receipt_submission_ids = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise LockError("leader cohort acquisition receipt is unavailable") from error
    if receipt_submission_ids != [SUBMISSION_ID]:
        raise LockError(
            "leader cohort acquisition receipt does not name the locked "
            "submission"
        )
    replay_paths = sorted(
        (path for path in cohort_dir.glob("[0-9]*.json")
         if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if len(replay_paths) != EXPECTED_FILES:
        raise LockError(
            f"expected {EXPECTED_FILES} leader replays, found {len(replay_paths)}")
    rows = [inspect_teacher_replay(path, deck) for path in replay_paths]
    episode_ids = [int(row["episode_id"]) for row in rows]
    contents = [str(row["content_sha256"]) for row in rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise LockError("leader cohort contains duplicate episode IDs")
    if len(contents) != len(set(contents)):
        raise LockError("leader cohort contains duplicate replay content")

    counts = Counter(str(row["outcome"]) for row in rows)
    mirrors = [row for row in rows if row["exact_mirror"]]
    mirror_wins = [row for row in mirrors if row["outcome"] == "win"]
    expected = {
        "win": EXPECTED_WINS,
        "loss": EXPECTED_LOSSES,
        "draw": EXPECTED_DRAWS,
        "exact_mirror": EXPECTED_EXACT_MIRRORS,
        "exact_mirror_win": EXPECTED_EXACT_MIRROR_WINS,
    }
    actual = {
        "win": counts["win"],
        "loss": counts["loss"],
        "draw": counts["draw"],
        "exact_mirror": len(mirrors),
        "exact_mirror_win": len(mirror_wins),
    }
    if actual != expected:
        raise LockError(f"leader cohort summary drifted: {actual} != {expected}")

    splits = assign_game_splits(rows)
    frozen_rows = []
    for row in rows:
        item = dict(row)
        stratum = split_stratum(row)
        item["supervision_split"] = splits[int(row["episode_id"])]
        item["split_stratum"] = stratum
        item["split_rank"] = _split_rank(int(row["episode_id"]), stratum)
        frozen_rows.append(item)
    split_counts = Counter(
        row["supervision_split"] for row in frozen_rows)

    script_paths = {
        "observational_preregistration": OBSERVATIONAL_PREREGISTRATION,
        "elite_teacher_preregistration": ELITE_PREREGISTRATION,
        "observational_result": OBSERVATIONAL_RESULT,
        "lock_builder": Path(__file__).resolve(),
        "extractor": ROOT / (
            "tools/research/prepare_dobi_v1_elite_teacher_main_v1.py"
        ),
        "trainer": ROOT / (
            "tools/research/train_dobi_v1_elite_teacher_main_v1.py"
        ),
        "behavior_screen": ROOT / (
            "tools/research/eval_dobi_v1_elite_teacher_main_v1_behavior.py"
        ),
    }
    payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "written_before_training_outcomes": True,
        "candidate_only": True,
        "runtime_environment": runtime_environment(),
        "screen_disclosed": {
            "leader_submission_id": SUBMISSION_ID,
            "leader_team": TEAM,
            "leader_record": actual,
            "leader_mirror_win_rate": EXPECTED_EXACT_MIRROR_WINS / EXPECTED_EXACT_MIRRORS,
            "dobi_historical_mirror_record": {"win": 26, "loss": 42},
            "observational_not_causal": True,
        },
        "target": {
            "deck_path": str(TARGET_DECK_PATH.resolve()),
            "deck_sha256": TARGET_DECK_SHA256,
            "select_type": 0,
            "teacher_team": TEAM,
            "teacher_submission_id": SUBMISSION_ID,
            "explicit_teacher_seat_required": True,
            "exact_mirror_opponent_never_supervised": True,
        },
        "artifacts": {
            **{
                f"aborted_v1_{name}": descriptor
                for name, descriptor in aborted_v1.items()
            },
            "cohort_acquisition_receipt": _artifact(
                receipt, COHORT_RECEIPT_SHA256,
            ),
            "parent_npz": _artifact(PARENT_NPZ, PARENT_NPZ_SHA256),
            "parent_checkpoint": _artifact(
                PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256),
            **{
                f"dependency_{name}": _artifact(path)
                for name, path in TRANSITIVE_DEPENDENCIES.items()
            },
            **{name: _artifact(path) for name, path in script_paths.items()},
        },
        "cohort": {
            "directory": str(cohort_dir.expanduser().resolve()),
            "files": len(rows),
            "counts": actual,
            "supervision_split_counts": dict(sorted(split_counts.items())),
            "split_domain": SPLIT_DOMAIN,
            "split_rule": (
                "within exact-mirror/non-mirror strata, lowest deterministic "
                "20% hash rank validates; all games grouped; no prompt crosses"
            ),
            "games": frozen_rows,
        },
        "preference_extraction": {
            "schema": "ptcg.dobi-v1.elite-teacher-main-v1b.preference.v1",
            "supervised_seat": "the explicit Sixth Sense seat only",
            "supervised_outcome": "win only",
            "select_type": 0,
            "parent_disagreement_required": True,
            "disagreement_unit": (
                "ordered public semantic action; interchangeable physical "
                "copies of the same card are agreements"
            ),
            "families": {
                "early_setup": "own ST_MAIN turns 1 through 3 inclusive",
                "mirror_boss_legal": (
                    "exact registered mirror and Boss's Orders card 1182 is a "
                    "legal semantic ST_MAIN option"
                ),
            },
            "preference": "logged teacher action over frozen Dobi-v1 greedy action",
            "per_game_normalized": True,
            "exact_mirror_game_multiplier": 2.5,
            "nonmirror_game_multiplier": 1.0,
            "st_card_changed": False,
        },
        "training": {
            "objective": (
                "parent-relative pairwise logistic preference plus independent "
                "frozen-parent sequential KL"
            ),
            "initial_checkpoint": "frozen Dobi-v1",
            "parent": "same frozen Dobi-v1",
            "parent_logit_authority": (
                "bound production agent.qu_v2_features + "
                "agent.model.QuV2Net NumPy runtime; Torch checkpoint only "
                "initializes candidate parameters; research/production "
                "feature arrays must match exactly on every row"
            ),
            "parent_kl_states": (
                "every valid ST_MAIN prompt from the explicit teacher seat in "
                "the 240 training-split locked games, including wins and "
                "losses; validation-game prompts are audit/evaluation only; "
                "independent of preference weights"
            ),
            "trainable_prefixes": ["option1.", "context1.", "policy."],
            "frozen": ["public backbone", "value heads", "ST_CARD", "Qu-v2B"],
            "arms": [
                {"name": "kl1", "kl_coefficient": 1.0},
                {"name": "kl3", "kl_coefficient": 3.0},
            ],
            "epochs": 3,
            "batch_size": 64,
            "learning_rate": 5e-6,
            "weight_decay": 1e-5,
            "gradient_clip": 1.0,
            "seed": 202608061,
            "device": "cpu",
            "one_preservation_batch_per_preference_batch": True,
            "selection": (
                "highest held-out teacher adoption among behavior-screen "
                "qualifiers; exact tie chooses kl3"
            ),
            "no_alternate_epoch_after_gameplay": True,
        },
        "behavior_screen": {
            "authoritative_inference": (
                "bound deployable NumPy artifacts through production "
                "agent.qu_v2_features encoder, agent.model.QuV2Net, and "
                "agent.model.decode_qu_v2; exact full checkpoint-to-NumPy "
                "export parity required before scoring; actions and KL use "
                "production runtime logits"
            ),
            "external_metrics_authorized": False,
            "teacher_validation_min_adoption": 0.20,
            "minimum_signature_progress": 0.25,
            "signature": {
                "cohort": (
                    "held-out validation preference disagreements whose legal "
                    "ST_MAIN option semantics expose card 112, 104, or 1182"
                ),
                "card_ids": {
                    "positive": [112, 1182],
                    "negative": [104],
                },
                "action_score": (
                    "number of selected Munkidori(112) plus selected Boss's "
                    "Orders(1182) minus selected Froslass(104)"
                ),
                "aggregation": "unweighted sum over eligible validation prompts",
                "required_direction": "aggregate teacher score > parent score",
                "progress_formula": (
                    "(candidate_score-parent_score)/"
                    "(teacher_score-parent_score)"
                ),
                "required_interval": [0.25, 1.0],
            },
            "preservation_source": (
                "all ST_MAIN prompts from validation-split locked teacher games, "
                "regardless of outcome"
            ),
            "maximum_overall_preservation_change": 0.05,
            "maximum_matchup_preservation_change": 0.07,
            "minimum_matchup_prompts_for_gate": 100,
            "maximum_mean_parent_kl": 0.03,
            "offline_nll_is_strength_evidence": False,
        },
        "authorized_runtime_scope_if_all_gates_pass": {
            "exact_registered_deck": True,
            "select_type": 0,
            "opponent_public_board_requires_any_card_id": [
                646, 647, 648,
            ],
            "card_names": [
                "Marnie's Impidimp", "Marnie's Morgrem",
                "Marnie's Grimmsnarl ex",
            ],
            "scope_miss_or_exception": "fall through to frozen Dobi-v1",
            "standalone_candidate_npz_authorized": False,
            "gameplay_controller_requirement": (
                "candidate only at exact-deck ST_MAIN with public Grim line; "
                "frozen Dobi ST_CARD and Qu-v2B elsewhere; route identity and "
                "exception fallback tested before gameplay"
            ),
        },
        "gameplay_gates": {
            "order": ["direct_exact_mirror", "recent_frequency_field"],
            "direct_exact_mirror": {
                "games": 10240,
                "control": "frozen Dobi-v1",
                "pass": "valid zero-fault Wilson CI95 lower bound > 0.50",
            },
            "recent_frequency_field": {
                "games_per_arm": 5120,
                "control": "frozen Dobi-v1",
                "pass": (
                    "valid zero-fault candidate-control point delta > 0 and "
                    "independent CI95 lower bound > -0.01"
                ),
                "required_reported_strata": [
                    "exact mirror", "Alakazam", "Crustle",
                    "Teal Mask Ogerpon ex", "Mega Froslass", "Dragapult",
                    "other",
                ],
            },
            "temporal_corroboration": {
                "source": (
                    "first complete official replay day on or after 2026-08-06"
                ),
                "untouched_by_training_and_arm_selection": True,
                "promotion_requirement": "corroboration only; separately reported",
            },
        },
        "promotion_authority": False,
        "upload_authority": False,
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    return payload


def verify_lock(payload: Mapping[str, Any]) -> bool:
    body = dict(payload)
    claimed = body.pop("lock_sha256", None)
    return isinstance(claimed, str) and canonical_sha256(body) == claimed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        payload = build_lock(args.cohort_dir)
        write_new(args.output.expanduser().resolve(), payload)
    except (LockError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "lock": str(args.output.expanduser().resolve()),
        "lock_sha256": payload["lock_sha256"],
        "cohort": {
            "files": payload["cohort"]["files"],
            "counts": payload["cohort"]["counts"],
            "supervision_split_counts": payload["cohort"]["supervision_split_counts"],
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
