"""Extract locked selective ST_CARD preferences and Dobi preservation states."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import md_v2_card, model, qu_v2_features as QF  # noqa: E402
from tools import analyze_ladder_replays as LADDER, il_dataset, index_corpus  # noqa: E402
from tools.research import (  # noqa: E402
    analyze_dobi_v1_elite_card_disagreement as SEMANTICS,
    lock_dobi_v1_elite_teacher_card_v1 as LOCK,
    prepare_dobi_v1_elite_teacher_main_v1 as MAIN_PREP,
)


RESULT_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.extraction-result.v1"
PREFERENCE_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.preference.v1"
PRESERVATION_SCHEMA = "ptcg.dobi-v1.elite-teacher-card-v1.preservation.v1"


class ExtractionError(RuntimeError):
    """The locked extraction contract failed closed."""


def load_and_verify_lock(path: Path = LOCK.OUTPUT) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ExtractionError(f"cannot read lock: {error}") from error
    if not isinstance(payload, dict) or not LOCK.verify_lock(payload):
        raise ExtractionError("cohort-lock schema/self-hash failed")
    if payload.get("runtime_environment") != LOCK.runtime_environment():
        raise ExtractionError("locked numerical runtime changed")
    return payload


def verify_bound_artifacts(lock: Mapping[str, Any]) -> None:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ExtractionError("lock has no artifact inventory")
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping):
            raise ExtractionError(f"invalid artifact descriptor: {name}")
        path = Path(str(descriptor.get("path", "")))
        if (
            not path.is_file()
            or LOCK.sha256_file(path) != descriptor.get("sha256")
        ):
            raise ExtractionError(f"bound artifact drifted: {name}")


def _verify_replay(
    game: Mapping[str, Any], *, team: str, target_deck: Sequence[int],
) -> tuple[dict[str, Any], int]:
    path = Path(str(game.get("path", "")))
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, TypeError, ValueError) as error:
        raise ExtractionError(f"cannot read locked replay: {path}") from error
    if not isinstance(document, dict) or hashlib.sha256(raw).hexdigest() \
            != game.get("content_sha256"):
        raise ExtractionError(f"locked replay content drifted: {path}")
    try:
        seat = LOCK._explicit_seat(document, team)
    except LOCK.LockError as error:
        raise ExtractionError(str(error)) from error
    decks = il_dataset.decks_from_document(document)
    actor = tuple(sorted(int(card) for card in decks.get(seat, ())))
    opponent = tuple(sorted(int(card) for card in decks.get(1 - seat, ())))
    rewards = document.get("rewards")
    reward = float(rewards[seat]) if isinstance(rewards, list) and len(rewards) == 2 else 9.0
    outcome = "win" if reward > 0 else "loss" if reward < 0 else "draw"
    if (
        seat != int(game.get("actor_seat", -1))
        or actor != tuple(target_deck)
        or opponent != tuple(target_deck)
        or outcome != game.get("outcome")
        or reward != float(game.get("reward"))
        or index_corpus.deck_sha256(opponent)
            != game.get("opponent_deck_sha256")
    ):
        raise ExtractionError(f"locked replay metadata drifted: {path}")
    return document, seat


def _games(lock: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    cohorts = lock.get("cohorts")
    cohort = cohorts.get(name) if isinstance(cohorts, Mapping) else None
    games = cohort.get("games") if isinstance(cohort, Mapping) else None
    if not isinstance(games, list):
        raise ExtractionError(f"missing {name} game inventory")
    return sorted(games, key=lambda row: int(row["episode_id"]))


def _semantic_set(view, action: Sequence[int]) -> tuple[str, ...]:
    MAIN_PREP.validate_action_sequence(
        action, len(view.options), view.min_count, view.max_count,
    )
    return SEMANTICS.semantic_action_signature(view, action)[1]


def _preference_weights(
    rows: Sequence[Mapping[str, Any]], family_weights: Mapping[str, Any],
) -> list[float]:
    if not rows:
        raise ExtractionError("no preference rows were extracted")
    raw_by_game: dict[int, float] = defaultdict(float)
    raw: list[float] = []
    for row in rows:
        family = str(row["family"])
        value = float(family_weights.get(family, 0.0))
        if value <= 0.0:
            raise ExtractionError(f"invalid family weight: {family}")
        raw.append(value)
        raw_by_game[int(row["episode_id"])] += value
    weights = [
        value / raw_by_game[int(row["episode_id"])]
        for row, value in zip(rows, raw, strict=True)
    ]
    mass: dict[int, float] = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        mass[int(row["episode_id"])] += weight
    if any(abs(value - 1.0) > 1e-9 for value in mass.values()):
        raise ExtractionError("per-game preference normalization failed")
    return weights


def extract(lock_path: Path = LOCK.OUTPUT) -> dict[str, Any]:
    lock = load_and_verify_lock(lock_path)
    verify_bound_artifacts(lock)
    if any(path.exists() for path in (
        LOCK.PREFERENCES, LOCK.PRESERVATION, LOCK.EXTRACTION_RESULT,
    )):
        raise ExtractionError("refusing to overwrite extraction artifacts")

    target_deck = LOCK.target_deck()
    parent = model.load(str(LOCK.PARENT_NPZ))
    if not isinstance(parent, model.QuV2Net):
        raise ExtractionError("deployed ST_CARD parent did not load")

    preferences: list[dict[str, Any]] = []
    preference_counts: Counter[str] = Counter()
    preservation_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    family_games: dict[str, set[int]] = defaultdict(set)
    touched_games: set[int] = set()

    for game in _games(lock, "teacher"):
        document, seat = _verify_replay(
            game, team=LOCK.TEACHER_TEAM, target_deck=target_deck,
        )
        if game.get("outcome") != "win":
            raise ExtractionError("non-winning teacher game entered supervision")
        episode_id = int(game["episode_id"])
        split = str(game["supervision_split"])
        preference_counts["games_verified"] += 1
        preference_counts[f"games_{split}"] += 1
        for prompt_index, (view, logged) in enumerate(
            LADDER.action_rows(document, seat)
        ):
            if not md_v2_card.supports_view(view, target_deck):
                continue
            family = SEMANTICS.classify_family(view)
            if family not in LOCK.FAMILY_WEIGHTS:
                preference_counts["excluded_other_st_card"] += 1
                continue
            logged_action = MAIN_PREP.validate_action_sequence(
                logged, len(view.options), view.min_count, view.max_count,
            )
            features = QF.encode_public_observation(view.obs, target_deck)
            logits, _ = parent.forward(features)
            parent_action = tuple(model.decode_qu_v2(
                logits, len(view.options), view.min_count, view.max_count,
            ))
            MAIN_PREP.validate_action_sequence(
                parent_action, len(view.options), view.min_count, view.max_count,
            )
            if _semantic_set(view, logged_action) == _semantic_set(
                view, parent_action,
            ):
                preference_counts["semantic_parent_agreements"] += 1
                continue
            # ST_CARD actions are order-free sets.  Use one deterministic path
            # for pairwise likelihood while retaining the production decode
            # order separately for exact parent/KL verification.
            preferred = tuple(sorted(logged_action))
            rejected = tuple(sorted(parent_action))
            row = {
                "schema": PREFERENCE_SCHEMA,
                "episode_id": episode_id,
                "teacher_seat": seat,
                "teacher_outcome": "win",
                "supervision_split": split,
                "exact_mirror": True,
                "matchup": "exact Grimmsnarl mirror",
                "prompt_index": prompt_index,
                "family": family,
                "observation": view.obs,
                "preferred": list(preferred),
                "rejected": list(rejected),
                "parent_action": list(parent_action),
            }
            preferences.append(row)
            touched_games.add(episode_id)
            preference_counts["semantic_disagreements"] += 1
            preference_counts[f"split_{split}"] += 1
            family_counts[family] += 1
            family_counts[f"{split}:{family}"] += 1
            family_games[family].add(episode_id)

    family_weights = lock["preference_extraction"]["family_weights"]
    weights = _preference_weights(preferences, family_weights)
    per_game_mass: dict[int, float] = defaultdict(float)
    for row, weight in zip(preferences, weights, strict=True):
        per_game_mass[int(row["episode_id"])] += weight

    preference_writer = MAIN_PREP._GzipJsonl(LOCK.RUN, "card-preferences")
    preservation_writer = MAIN_PREP._GzipJsonl(LOCK.RUN, "dobi-preservation")
    published: list[Path] = []
    try:
        for row, weight in zip(preferences, weights, strict=True):
            enriched = dict(row)
            enriched["raw_family_weight"] = float(family_weights[row["family"]])
            enriched["preference_weight"] = weight
            enriched["game_preference_mass"] = per_game_mass[int(row["episode_id"])]
            preference_writer.write(enriched)
        preference_writer.close()

        for game in _games(lock, "preservation"):
            document, seat = _verify_replay(
                game, team=LOCK.DOBI_TEAM, target_deck=target_deck,
            )
            episode_id = int(game["episode_id"])
            split = str(game["supervision_split"])
            preservation_counts["games_verified"] += 1
            preservation_counts[f"games_{split}"] += 1
            for prompt_index, (view, _logged) in enumerate(
                LADDER.action_rows(document, seat)
            ):
                if not md_v2_card.supports_view(view, target_deck):
                    continue
                features = QF.encode_public_observation(view.obs, target_deck)
                logits, _ = parent.forward(features)
                parent_action = tuple(model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                ))
                MAIN_PREP.validate_action_sequence(
                    parent_action, len(view.options), view.min_count,
                    view.max_count,
                )
                preservation_writer.write({
                    "schema": PRESERVATION_SCHEMA,
                    "episode_id": episode_id,
                    "dobi_seat": seat,
                    "dobi_outcome": str(game["outcome"]),
                    "supervision_split": split,
                    "exact_mirror": True,
                    "matchup": "exact Grimmsnarl mirror",
                    "prompt_index": prompt_index,
                    "family": SEMANTICS.classify_family(view),
                    "observation": view.obs,
                    "parent_action": list(parent_action),
                    "training_kl_state": True,
                })
                preservation_counts["prompts"] += 1
                preservation_counts[f"split_{split}"] += 1
                preservation_counts[f"outcome_{game['outcome']}"] += 1
        preservation_writer.close()

        MAIN_PREP._publish_new(
            preference_writer.path, LOCK.PREFERENCES, published,
        )
        MAIN_PREP._publish_new(
            preservation_writer.path, LOCK.PRESERVATION, published,
        )
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "cohort_lock_file_sha256": LOCK.sha256_file(lock_path),
            "cohort_lock_sha256": lock["lock_sha256"],
            "parent_npz_sha256": LOCK.sha256_file(LOCK.PARENT_NPZ),
            "select_type": 1,
            "explicit_teacher_seat_only": True,
            "winning_exact_mirror_teacher_only": True,
            "explicit_dobi_preservation_seat_only": True,
            "preference_counts": dict(sorted(preference_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "games_by_family": {
                family: len(games)
                for family, games in sorted(family_games.items())
            },
            "preferences": {
                "path": str(LOCK.PREFERENCES.resolve()),
                "lines": preference_writer.lines,
                "compressed_sha256": LOCK.sha256_file(LOCK.PREFERENCES),
                "uncompressed_jsonl_sha256": (
                    preference_writer.uncompressed_sha256.hexdigest()
                ),
                "touched_games": len(touched_games),
                "per_game_mass_assertion": "exactly 1.0",
                "total_effective_mass": sum(weights),
            },
            "preservation": {
                "path": str(LOCK.PRESERVATION.resolve()),
                "lines": preservation_writer.lines,
                "compressed_sha256": LOCK.sha256_file(LOCK.PRESERVATION),
                "uncompressed_jsonl_sha256": (
                    preservation_writer.uncompressed_sha256.hexdigest()
                ),
                "counts": dict(sorted(preservation_counts.items())),
            },
            "training_authority": False,
            "promotion_authority": False,
            "upload_authority": False,
        }
        result["result_sha256"] = LOCK.canonical_sha256(result)
        LOCK.write_new(LOCK.EXTRACTION_RESULT, result, published)
        return result
    except BaseException:
        for writer in (preference_writer, preservation_writer):
            writer.discard()
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        preference_writer.path.unlink(missing_ok=True)
        preservation_writer.path.unlink(missing_ok=True)


def main() -> int:
    try:
        result = extract()
    except (OSError, TypeError, ValueError, ExtractionError,
            MAIN_PREP.ExtractionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "preferences": result["preferences"]["lines"],
        "preservation": result["preservation"]["lines"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
