"""Extract the locked Sixth Sense -> Dobi-v1 ST_MAIN preferences.

This script is intentionally narrower than generic behavioral cloning.  It
uses only the explicitly named Sixth Sense seat, only winning-seat ST_MAIN
decisions, and only two preregistered prompt families:

* the teacher's first three own turns; and
* exact Grimmsnarl mirrors where Boss's Orders is a legal main-phase action.

The logged teacher action becomes preferred only when frozen Dobi-v1 chooses
something else.  Every replay, seat, split, input model, and implementation
file is verified against ``cohort-lock.json`` before an output is published.
No artifact produced here has training, promotion, or upload authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model, qu_v2_features as QF  # noqa: E402
from agent.obsview import (  # noqa: E402
    AREA_DECK,
    AREA_DISCARD,
    AREA_HAND,
    AREA_LOOKING,
    OT_ATTACH,
    OT_EVOLVE,
    OT_PLAY,
    ObsView,
    ST_MAIN,
)
from tools import analyze_ladder_replays as LADDER, il_dataset, index_corpus  # noqa: E402
from tools.research import lock_dobi_v1_elite_teacher_main_v1 as LOCK  # noqa: E402


RESULT_SCHEMA = "ptcg.dobi-v1.elite-teacher-main-v1.extraction-result.v1"
PREFERENCE_SCHEMA = "ptcg.dobi-v1.elite-teacher-main-v1.preference.v1"
PRESERVATION_SCHEMA = (
    "ptcg.dobi-v1.elite-teacher-main-v1.teacher-preservation.v1"
)
BOSS_ORDERS_CARD_ID = 1182
FUNGIBLE_COPY_AREAS = frozenset((
    AREA_DECK, AREA_HAND, AREA_DISCARD, AREA_LOOKING,
))


class ExtractionError(RuntimeError):
    """A fail-closed lock, replay, or action-contract failure."""


def load_and_verify_lock(path: Path = LOCK.OUTPUT) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ExtractionError(f"cannot read cohort lock {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != LOCK.LOCK_SCHEMA:
        raise ExtractionError("unexpected cohort-lock schema")
    if not LOCK.verify_lock(payload):
        raise ExtractionError("cohort-lock self hash failed")
    expected_environment = payload.get("runtime_environment")
    actual_environment = LOCK.runtime_environment()
    if expected_environment != actual_environment:
        raise ExtractionError(
            "locked numerical runtime changed: "
            f"{actual_environment!r} != {expected_environment!r}"
        )
    return payload


def verify_bound_artifacts(lock: Mapping[str, Any]) -> None:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ExtractionError("cohort lock has no artifact bindings")
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping):
            raise ExtractionError(f"invalid artifact binding: {name}")
        path = Path(str(descriptor.get("path", "")))
        expected = descriptor.get("sha256")
        if not path.is_file() or not isinstance(expected, str):
            raise ExtractionError(f"bound artifact is unavailable: {name}")
        if LOCK.sha256_file(path) != expected:
            raise ExtractionError(f"bound artifact drifted: {name}")


def validate_action_sequence(
    action: Sequence[int], n_options: int, min_count: int, max_count: int,
) -> tuple[int, ...]:
    """Return a legal no-replacement action or fail closed.

    ``maxCount == 0`` follows the runtime contract and means all options may be
    selected.  Engine edge cases with fewer options than ``minCount`` are
    bounded by the number of options, just like ``decode_qu_v2``.
    """
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (n_options, min_count, max_count)
    ):
        raise ExtractionError("selection counts must be integers")
    if n_options < 0 or min_count < 0 or max_count < 0:
        raise ExtractionError("selection counts cannot be negative")
    if max_count > 0 and min_count > max_count:
        raise ExtractionError("selection minCount exceeds maxCount")
    if not isinstance(action, (list, tuple)):
        raise ExtractionError("action is not an index sequence")
    normalized: list[int] = []
    for index in action:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ExtractionError("action index is not an integer")
        if not 0 <= index < n_options:
            raise ExtractionError("action index is outside the option menu")
        normalized.append(index)
    if len(normalized) != len(set(normalized)):
        raise ExtractionError("action repeats an option")
    effective_min = min(min_count, n_options)
    effective_max = min(max_count, n_options) if max_count > 0 else n_options
    if not effective_min <= len(normalized) <= effective_max:
        raise ExtractionError("action violates the selection cardinality")
    return tuple(normalized)


def classify_families(
    view: ObsView, own_turn: int, exact_mirror: bool,
) -> tuple[str, ...]:
    """Classify only the two locked ST_MAIN families."""
    if view.select_type != ST_MAIN:
        return ()
    families: list[str] = []
    if 1 <= own_turn <= 3:
        families.append("early_setup")
    if exact_mirror and any(
        view.semantic_option_card_id(option) == BOSS_ORDERS_CARD_ID
        for option in view.options
    ):
        families.append("mirror_boss_legal")
    return tuple(families)


def option_semantic_token(view: ObsView, option: Mapping[str, Any]) -> str:
    """Canonical public meaning, collapsing interchangeable card copies."""
    if not isinstance(option, Mapping):
        raise ExtractionError("option is not a mapping")
    area = option.get("area")
    card_id = view.semantic_option_card_id(dict(option))
    bare_main_hand_card = (
        view.select_type == ST_MAIN
        and option.get("type") in (OT_PLAY, OT_ATTACH, OT_EVOLVE)
        and card_id is not None
    )
    token: dict[str, Any] = {
        "type": option.get("type"),
        "area": area,
        "player_index": option.get("playerIndex", view.my_index),
        "card_id": card_id,
    }
    if card_id is None or (
        area not in FUNGIBLE_COPY_AREAS and not bare_main_hand_card
    ):
        token["index"] = option.get("index")
    token["extra"] = {
        str(key): value
        for key, value in option.items()
        if key not in {"type", "area", "playerIndex", "cardId", "index"}
    }
    return json.dumps(
        token, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def semantic_action_signature(
    view: ObsView, action: Sequence[int],
) -> tuple[str, ...]:
    """Ordered semantic action; order stays meaningful, copy index does not."""
    validate_action_sequence(
        action, len(view.options), view.min_count, view.max_count,
    )
    return tuple(
        option_semantic_token(view, view.options[index]) for index in action
    )


def normalized_preference_weights(
    rows: Sequence[Mapping[str, Any]], mirror_multiplier: float = 2.5,
) -> list[float]:
    """Give each touched game total mass 1.0, or 2.5 for exact mirrors."""
    if not rows:
        return []
    counts = Counter(int(row["episode_id"]) for row in rows)
    mirror_by_game: dict[int, bool] = {}
    for row in rows:
        episode_id = int(row["episode_id"])
        exact_mirror = bool(row["exact_mirror"])
        previous = mirror_by_game.setdefault(episode_id, exact_mirror)
        if previous != exact_mirror:
            raise ExtractionError("one episode has inconsistent matchup labels")
    return [
        (mirror_multiplier if mirror_by_game[int(row["episode_id"])] else 1.0)
        / counts[int(row["episode_id"])]
        for row in rows
    ]


class _GzipJsonl:
    """Deterministic gzip JSONL written to a temporary path."""

    def __init__(self, directory: Path, label: str):
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{label}.", dir=directory)
        os.close(descriptor)
        self.path = Path(name)
        self._raw = self.path.open("wb")
        self._gzip = gzip.GzipFile(
            filename="", mode="wb", fileobj=self._raw, mtime=0,
        )
        self._text = io.TextIOWrapper(self._gzip, encoding="utf-8", newline="\n")
        self.uncompressed_sha256 = hashlib.sha256()
        self.lines = 0
        self.closed = False

    def write(self, row: Mapping[str, Any]) -> None:
        raw = (
            json.dumps(
                row, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
        self._text.write(raw.decode("utf-8"))
        self.uncompressed_sha256.update(raw)
        self.lines += 1

    def close(self) -> None:
        if self.closed:
            return
        self._text.flush()
        self._text.detach()
        self._gzip.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        self.closed = True

    def discard(self) -> None:
        try:
            self.close()
        finally:
            self.path.unlink(missing_ok=True)


def _publish_new(temporary: Path, output: Path) -> None:
    if output.exists():
        raise ExtractionError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise ExtractionError(f"refusing to overwrite {output}") from error


def _locked_game_map(lock: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cohort = lock.get("cohort")
    games = cohort.get("games") if isinstance(cohort, Mapping) else None
    if not isinstance(games, list) or len(games) != LOCK.EXPECTED_FILES:
        raise ExtractionError("cohort-lock game inventory is incomplete")
    return sorted(games, key=lambda game: int(game["episode_id"]))


def _verify_replay(
    game: Mapping[str, Any], target_deck: tuple[int, ...],
) -> tuple[dict[str, Any], int]:
    path = Path(str(game.get("path", "")))
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ExtractionError(f"cannot read locked replay {path}: {error}") from error
    if hashlib.sha256(raw).hexdigest() != game.get("content_sha256"):
        raise ExtractionError(f"locked replay content drifted: {path}")
    try:
        replay = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ExtractionError(f"locked replay is malformed: {path}") from error
    if not isinstance(replay, dict):
        raise ExtractionError(f"locked replay root is not an object: {path}")
    try:
        seat = LOCK.explicit_teacher_seat(replay)
    except LOCK.LockError as error:
        raise ExtractionError(str(error)) from error
    if seat != int(game.get("teacher_seat", -1)):
        raise ExtractionError(f"locked teacher seat drifted: {path}")
    decks = il_dataset.decks_from_document(replay)
    teacher_deck = tuple(sorted(int(card) for card in decks.get(seat, ())))
    opponent_deck = tuple(sorted(int(card) for card in decks.get(1 - seat, ())))
    if teacher_deck != target_deck or len(opponent_deck) != 60:
        raise ExtractionError(f"locked replay deck identity drifted: {path}")
    rewards = replay.get("rewards")
    if not isinstance(rewards, list) or len(rewards) != 2:
        raise ExtractionError(f"locked replay reward is missing: {path}")
    reward = float(rewards[seat])
    outcome = "win" if reward > 0.0 else "loss" if reward < 0.0 else "draw"
    exact_mirror = opponent_deck == target_deck
    matchup = (
        "exact Grimmsnarl mirror"
        if exact_mirror else LADDER.archetype(opponent_deck)
    )
    if (
        outcome != game.get("outcome")
        or reward != float(game.get("reward"))
        or exact_mirror is not bool(game.get("exact_mirror"))
        or matchup != game.get("matchup")
        or index_corpus.deck_sha256(opponent_deck)
        != game.get("opponent_deck_sha256")
    ):
        raise ExtractionError(f"locked replay scientific metadata drifted: {path}")
    return replay, seat


def _own_turn_ordinals(
    action_rows: Iterable[tuple[ObsView, list[int]]],
) -> Iterable[tuple[ObsView, list[int], int]]:
    turn_ordinals: dict[int, int] = {}
    for view, action in action_rows:
        if view.select_type != ST_MAIN:
            continue
        raw_turn = int(view.turn)
        if raw_turn not in turn_ordinals:
            turn_ordinals[raw_turn] = len(turn_ordinals) + 1
        yield view, action, turn_ordinals[raw_turn]


def extract(lock_path: Path = LOCK.OUTPUT) -> dict[str, Any]:
    lock = load_and_verify_lock(lock_path)
    verify_bound_artifacts(lock)
    if LOCK.PREFERENCES.exists() or LOCK.PRESERVATION.exists() \
            or LOCK.EXTRACTION_RESULT.exists():
        raise ExtractionError("refusing to overwrite extraction artifacts")

    target_deck = LOCK.target_deck()
    parent = model.load(str(LOCK.PARENT_NPZ))
    if not isinstance(parent, model.QuV2Net):
        raise ExtractionError("frozen Dobi-v1 NumPy parent did not load")

    preferences: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    preservation_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    family_games: dict[str, set[int]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    matchup_counts: Counter[str] = Counter()
    preference_games: set[int] = set()

    preservation_writer = _GzipJsonl(LOCK.RUN, "teacher-preservation")
    preference_writer: _GzipJsonl | None = None
    published: list[Path] = []
    try:
        for game in _locked_game_map(lock):
            replay, seat = _verify_replay(game, target_deck)
            episode_id = int(game["episode_id"])
            outcome = str(game["outcome"])
            split = str(game["supervision_split"])
            matchup = str(game["matchup"])
            exact_mirror = bool(game["exact_mirror"])
            counts["games_verified"] += 1
            counts[f"games_{outcome}"] += 1
            counts[f"games_split_{split}"] += 1

            for prompt_index, (view, logged, own_turn) in enumerate(
                _own_turn_ordinals(LADDER.action_rows(replay, seat))
            ):
                preferred = validate_action_sequence(
                    logged, len(view.options), view.min_count, view.max_count,
                )
                encoded = QF.encode_public_observation(view.obs, target_deck)
                logits, _ = parent.forward(encoded)
                parent_action = tuple(model.decode_qu_v2(
                    logits, len(view.options), view.min_count, view.max_count,
                ))
                validate_action_sequence(
                    parent_action, len(view.options), view.min_count,
                    view.max_count,
                )
                preservation_writer.write({
                    "schema": PRESERVATION_SCHEMA,
                    "episode_id": episode_id,
                    "teacher_seat": seat,
                    "teacher_outcome": outcome,
                    "supervision_split": split,
                    "matchup": matchup,
                    "exact_mirror": exact_mirror,
                    "own_turn": own_turn,
                    "prompt_index": prompt_index,
                    "observation": view.obs,
                    "parent_action": list(parent_action),
                    "training_kl_state": True,
                })
                preservation_counts["prompts"] += 1
                preservation_counts[f"outcome_{outcome}"] += 1
                preservation_counts[f"split_{split}"] += 1
                preservation_counts[f"matchup_{matchup}"] += 1

                if outcome != "win":
                    continue
                families = classify_families(view, own_turn, exact_mirror)
                if not families:
                    continue
                counts["eligible_family_prompts"] += 1
                rejected = parent_action
                if rejected == preferred:
                    counts["parent_agreements"] += 1
                    continue
                if semantic_action_signature(view, rejected) \
                        == semantic_action_signature(view, preferred):
                    counts["semantic_parent_agreements_raw_index_only"] += 1
                    continue
                row = {
                    "schema": PREFERENCE_SCHEMA,
                    "episode_id": episode_id,
                    "teacher_seat": seat,
                    "teacher_outcome": outcome,
                    "supervision_split": split,
                    "matchup": matchup,
                    "exact_mirror": exact_mirror,
                    "own_turn": own_turn,
                    "prompt_index": prompt_index,
                    "families": list(families),
                    "observation": view.obs,
                    "preferred": list(preferred),
                    "rejected": list(rejected),
                }
                preferences.append(row)
                preference_games.add(episode_id)
                split_counts[split] += 1
                matchup_counts[matchup] += 1
                for family in families:
                    family_counts[family] += 1
                    family_games[family].add(episode_id)

        preservation_writer.close()
        weights = normalized_preference_weights(
            preferences,
            float(lock["preference_extraction"]["exact_mirror_game_multiplier"]),
        )
        per_game_counts = Counter(int(row["episode_id"]) for row in preferences)
        mass_by_game: dict[int, float] = defaultdict(float)
        mass_by_split: dict[str, float] = defaultdict(float)
        mass_by_mirror: dict[str, float] = defaultdict(float)
        mass_by_matchup: dict[str, float] = defaultdict(float)
        game_metadata: dict[int, Mapping[str, Any]] = {}
        for row, weight in zip(preferences, weights, strict=True):
            episode_id = int(row["episode_id"])
            mass_by_game[episode_id] += weight
            mass_by_split[str(row["supervision_split"])] += weight
            mass_by_mirror[
                "exact_mirror" if row["exact_mirror"] else "nonmirror"
            ] += weight
            mass_by_matchup[str(row["matchup"])] += weight
            game_metadata[episode_id] = row
        per_game_mass: dict[str, Any] = {}
        for episode_id, realized in sorted(mass_by_game.items()):
            row = game_metadata[episode_id]
            expected = (
                float(lock["preference_extraction"][
                    "exact_mirror_game_multiplier"
                ]) if row["exact_mirror"] else 1.0
            )
            if abs(realized - expected) > 1e-9:
                raise ExtractionError(
                    f"preference mass drifted in episode {episode_id}: "
                    f"{realized} != {expected}"
                )
            per_game_mass[str(episode_id)] = {
                "decisions": per_game_counts[episode_id],
                "expected": expected,
                "realized": realized,
                "split": row["supervision_split"],
                "matchup": row["matchup"],
                "exact_mirror": row["exact_mirror"],
            }
        preference_writer = _GzipJsonl(LOCK.RUN, "preferences")
        for row, weight in zip(preferences, weights, strict=True):
            episode_id = int(row["episode_id"])
            enriched = dict(row)
            enriched["game_preference_count"] = per_game_counts[episode_id]
            enriched["game_multiplier"] = (
                float(lock["preference_extraction"][
                    "exact_mirror_game_multiplier"
                ]) if row["exact_mirror"] else 1.0
            )
            enriched["preference_weight"] = weight
            preference_writer.write(enriched)
        preference_writer.close()

        _publish_new(preservation_writer.path, LOCK.PRESERVATION)
        published.append(LOCK.PRESERVATION)
        _publish_new(preference_writer.path, LOCK.PREFERENCES)
        published.append(LOCK.PREFERENCES)

        counts["preferences"] = len(preferences)
        counts["preference_games"] = len(preference_games)
        counts["train_preferences"] = split_counts["train"]
        counts["validation_preferences"] = split_counts["validation"]
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "cohort_lock_file_sha256": LOCK.sha256_file(lock_path),
            "cohort_lock_sha256": lock["lock_sha256"],
            "parent_npz_sha256": LOCK.sha256_file(LOCK.PARENT_NPZ),
            "explicit_teacher_seat_only": True,
            "winning_teacher_seat_only": True,
            "select_type": ST_MAIN,
            "st_card_changed": False,
            "counts": dict(sorted(counts.items())),
            "preferences": {
                "path": str(LOCK.PREFERENCES.resolve()),
                "lines": preference_writer.lines,
                "compressed_sha256": LOCK.sha256_file(LOCK.PREFERENCES),
                "uncompressed_jsonl_sha256": (
                    preference_writer.uncompressed_sha256.hexdigest()
                ),
                "by_split": dict(sorted(split_counts.items())),
                "by_matchup": dict(sorted(matchup_counts.items())),
                "by_family": dict(sorted(family_counts.items())),
                "games_by_family": {
                    name: len(episode_ids)
                    for name, episode_ids in sorted(family_games.items())
                },
                "realized_effective_mass": {
                    "total": sum(weights),
                    "by_split": dict(sorted(mass_by_split.items())),
                    "by_mirror_class": dict(sorted(mass_by_mirror.items())),
                    "by_matchup": dict(sorted(mass_by_matchup.items())),
                    "per_game_assertion": "passed",
                    "per_game": per_game_mass,
                },
            },
            "teacher_preservation": {
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
        LOCK.write_new(LOCK.EXTRACTION_RESULT, result)
        published.append(LOCK.EXTRACTION_RESULT)
    except BaseException:
        for path in reversed(published):
            # These files were created in this invocation and the result was
            # not published.  Removing them restores the all-or-none contract.
            path.unlink(missing_ok=True)
        raise
    finally:
        preservation_writer.discard()
        if preference_writer is not None:
            preference_writer.discard()
    # Console delivery is not part of the artifact transaction.  A closed
    # pipe after successful publication cannot invalidate or partially remove
    # the three mutually bound extraction outputs.
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK.OUTPUT)
    args = parser.parse_args()
    try:
        extract(args.lock.expanduser().resolve())
    except (ExtractionError, LOCK.LockError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
