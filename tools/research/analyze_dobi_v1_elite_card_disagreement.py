"""Descriptive same-state ST_CARD screen against the Sixth Sense pilot.

This script replays the frozen Dobi-v1 ST_CARD network on the exact public
observations where Sixth Sense made an ST_CARD choice in exact-list
Grimmsnarl mirrors.  It compares the greedy network action with the logged
action without changing the observation, option order, or registered deck.

The comparison is deliberately descriptive.  The leader's logged action is
not a counterfactual outcome label, and this already-inspected cohort cannot
authorize training, gameplay promotion, packaging, or upload.  Two scopes are
reported:

* ``all_st_card_counterfactual`` applies the frozen ST_CARD network to every
  ST_CARD prompt so the model itself can be inventoried; and
* ``deployed_public_signature`` retains only prompts where the shipped router
  would actually hand control to that network (an opposing public Impidimp,
  Morgrem, or Grimmsnarl ex is visible).

The Kaggle record convention is important: the action at step ``t`` answers
the observation at step ``t - 1``.  Terminal observations are therefore not
treated as unpaired policy prompts.

Example::

    python tools/research/analyze_dobi_v1_elite_card_disagreement.py \
      tools/checkpoints/dobi-v1-top-grim-comparison-v1/replays/sixth-sense \
      --json-out /tmp/dobi-v1-elite-card-disagreement.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for directory in (str(ROOT), str(ROOT / "tools")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import md_v2_card, model, qu_v2_features as QF  # noqa: E402
from agent.obsview import (  # noqa: E402
    AREA_DECK,
    AREA_DISCARD,
    AREA_HAND,
    AREA_LOOKING,
    CTX_DAMAGE_COUNTER,
    CTX_REMOVE_DAMAGE_COUNTER,
    ST_CARD,
    ObsView,
)
from tools import index_corpus  # noqa: E402


SCHEMA = "ptcg.dobi-v1.elite-card-disagreement.v1"
DEFAULT_REPLAYS = (
    ROOT
    / "tools/checkpoints/dobi-v1-top-grim-comparison-v1/replays/sixth-sense"
)
DEFAULT_WEIGHTS = ROOT / "agent/md_v2_card_weights.npz"
EXPECTED_EXACT_MIRRORS = 53
EXPECTED_WEIGHTS_SHA256 = (
    "1aef7068130072dd00afe0e85d6485d9749c6326351597339c1bd33d6d239cf7"
)
TARGET_DECK_SHA256 = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)
NUMERIC_REPLAY = re.compile(r"^(?P<episode>[0-9]+)\.json$")

# Fixed family inventory requested before this implementation.  The two
# Munkidori phases share an effect card and are separated by SelectContext.
MUNKIDORI = 112
FAMILY_EFFECT_IDS = {
    "spikemuth": 1259,
    "poke_pad": 1152,
    "petrel": 1219,
    "poffin": 1086,
    "night_stretcher": 1097,
    "boss": 1182,
}
FIXED_FAMILIES = (
    "munkidori_damage_source",
    "munkidori_damage_destination",
    "spikemuth",
    "poke_pad",
    "petrel",
    "poffin",
    "night_stretcher",
    "boss",
)
ALL_FAMILIES = (*FIXED_FAMILIES, "other_st_card")
OUTCOMES = ("win", "draw", "loss")
FUNGIBLE_COPY_AREAS = frozenset((AREA_DECK, AREA_HAND, AREA_DISCARD, AREA_LOOKING))


class ScreenError(RuntimeError):
    """The frozen artifact or replay cohort cannot be screened safely."""


@dataclass(frozen=True)
class CohortGame:
    path: Path
    episode_id: int
    seat: int
    outcome: str
    deck: tuple[int, ...]
    content_sha256: str


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _outcome(reward: float) -> str:
    return "win" if reward > 0 else "loss" if reward < 0 else "draw"


def _valid_deck(action: Any) -> tuple[int, ...] | None:
    if not isinstance(action, list) or len(action) != 60:
        return None
    if not all(
        isinstance(card_id, int)
        and not isinstance(card_id, bool)
        and card_id > 0
        for card_id in action
    ):
        return None
    return tuple(int(card_id) for card_id in action)


def _registered_decks(document: Mapping[str, Any]) -> tuple[
    dict[int, tuple[int, ...]] | None, str | None
]:
    steps = document.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, "invalid_steps"
    by_seat: dict[int, set[tuple[int, ...]]] = {0: set(), 1: set()}
    raw_by_canonical: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {}
    for step in steps:
        if not isinstance(step, list) or len(step) != 2:
            return None, "invalid_step_shape"
        for seat in (0, 1):
            row = step[seat]
            if not isinstance(row, Mapping):
                return None, "invalid_step_row"
            deck = _valid_deck(row.get("action"))
            if deck is None:
                continue
            canonical = tuple(sorted(deck))
            by_seat[seat].add(canonical)
            raw_by_canonical[(seat, canonical)] = deck
    if any(not by_seat[seat] for seat in (0, 1)):
        return None, "missing_deck_registration"
    if any(len(by_seat[seat]) != 1 for seat in (0, 1)):
        return None, "conflicting_deck_registrations"
    decks = {}
    for seat in (0, 1):
        canonical = next(iter(by_seat[seat]))
        decks[seat] = raw_by_canonical[(seat, canonical)]
    return decks, None


def _team_names(document: Mapping[str, Any]) -> list[str] | None:
    info = document.get("info")
    if not isinstance(info, Mapping):
        return None
    names = info.get("TeamNames")
    if (
        not isinstance(names, list)
        or len(names) != 2
        or not all(isinstance(name, str) and name for name in names)
    ):
        return None
    return list(names)


def audit_replay(
    document: Any,
    *,
    filename_episode_id: int,
    team: str,
) -> tuple[str, dict[str, Any]]:
    """Classify one file as filtered, eligible, or excluded with one reason."""
    if not isinstance(document, Mapping):
        return "exclude:non_mapping_json", {}
    info = document.get("info")
    episode_id = info.get("EpisodeId") if isinstance(info, Mapping) else None
    if (
        not isinstance(episode_id, int)
        or isinstance(episode_id, bool)
        or episode_id != filename_episode_id
    ):
        return "exclude:episode_id_mismatch", {"document_episode_id": episode_id}

    decks, deck_error = _registered_decks(document)
    if deck_error is not None or decks is None:
        return f"exclude:{deck_error}", {}
    target = tuple(md_v2_card.TARGET_DECK)
    if not all(tuple(sorted(decks[seat])) == target for seat in (0, 1)):
        return "filter:not_exact_mirror", {}

    names = _team_names(document)
    if names is None:
        return "exclude:invalid_team_names", {}
    matching = [seat for seat, name in enumerate(names) if name == team]
    if len(matching) != 1:
        return "exclude:team_seat_not_unique", {"matching_team_seats": matching}
    seat = matching[0]

    statuses = document.get("statuses")
    if statuses != ["DONE", "DONE"]:
        return "exclude:non_done_status", {"statuses": statuses}
    rewards = document.get("rewards")
    if (
        not isinstance(rewards, list)
        or len(rewards) != 2
        or any(
            not isinstance(reward, (int, float))
            or isinstance(reward, bool)
            or not math.isfinite(float(reward))
            or float(reward) not in (-1.0, 0.0, 1.0)
            for reward in rewards
        )
        or float(rewards[0]) != -float(rewards[1])
    ):
        return "exclude:invalid_rewards", {"rewards": rewards}
    return "eligible", {
        "episode_id": episode_id,
        "seat": seat,
        "outcome": _outcome(float(rewards[seat])),
        "deck": decks[seat],
    }


def classify_family(view: ObsView) -> str:
    """Return one mutually exclusive member of the fixed family inventory."""
    if view.select_type != ST_CARD:
        raise ValueError("family classification requires ST_CARD")
    effect = view.effect_card_id
    if effect == MUNKIDORI and view.context == CTX_REMOVE_DAMAGE_COUNTER:
        return "munkidori_damage_source"
    if effect == MUNKIDORI and view.context == CTX_DAMAGE_COUNTER:
        return "munkidori_damage_destination"
    for family, card_id in FAMILY_EFFECT_IDS.items():
        if effect == card_id:
            return family
    return "other_st_card"


def canonical_action(
    action: Any,
    *,
    option_count: int,
    min_count: int,
    max_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate an engine action and return (sequence, order-free canonical)."""
    for name, value in (
        ("option_count", option_count),
        ("min_count", min_count),
        ("max_count", max_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if option_count <= 0 or min_count < 0 or max_count < 0:
        raise ValueError("invalid selection sizes")
    if max_count > 0 and min_count > max_count:
        raise ValueError("min_count exceeds max_count")
    if not isinstance(action, (list, tuple)):
        raise ValueError("action must be a list or tuple")
    if not all(
        isinstance(index, int)
        and not isinstance(index, bool)
        and 0 <= index < option_count
        for index in action
    ):
        raise ValueError("action contains an invalid option index")
    sequence = tuple(int(index) for index in action)
    if len(set(sequence)) != len(sequence):
        raise ValueError("action contains duplicate option indices")
    effective_min = min(min_count, option_count)
    effective_max = min(max_count, option_count) if max_count > 0 else option_count
    if not effective_min <= len(sequence) <= effective_max:
        raise ValueError("action violates selection cardinality")
    return sequence, tuple(sorted(sequence))


def option_semantic_token(view: ObsView, option: Mapping[str, Any]) -> str:
    """Canonical public meaning of one ST_CARD option.

    Indices identify engine options, not always distinct behavior.  Copies of
    the same card in deck, hand, discard, or the public ``looking`` buffer are
    interchangeable, so their raw index is omitted when the card identity can
    be resolved.  Board and hidden-zone targets retain their exact slot: two
    Munkidori with different damage/energy or two face-down prizes are not
    assumed equivalent.
    """
    if not isinstance(option, Mapping):
        raise ValueError("option must be a mapping")
    area = option.get("area")
    card_id = view.option_card_id(dict(option))
    token: dict[str, Any] = {
        "type": option.get("type"),
        "area": area,
        "player_index": option.get("playerIndex", view.my_index),
        "card_id": card_id,
    }
    if card_id is None or area not in FUNGIBLE_COPY_AREAS:
        token["index"] = option.get("index")
    # Preserve any effect-specific selector data beyond the ordinary address.
    token["extra"] = {
        str(key): value
        for key, value in option.items()
        if key not in {"type", "area", "playerIndex", "cardId", "index"}
    }
    return json.dumps(
        token,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def semantic_action_signature(
    view: ObsView,
    action: Sequence[int],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ordered and order-free public semantic action signatures."""
    tokens = tuple(option_semantic_token(view, view.options[index]) for index in action)
    return tokens, tuple(sorted(tokens))


def _base_summary(
    records: Sequence[Mapping[str, Any]],
    cohort_games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    decisions = len(records)
    semantic_set_agreements = sum(
        bool(record["semantic_set_agreement"]) for record in records
    )
    semantic_set_disagreements = decisions - semantic_set_agreements
    semantic_sequence_agreements = sum(
        bool(record["semantic_sequence_agreement"]) for record in records
    )
    raw_sequence_agreements = sum(
        bool(record["raw_index_sequence_agreement"]) for record in records
    )
    raw_set_agreements = sum(
        bool(record["raw_index_set_agreement"]) for record in records
    )
    semantic_order_only = sum(
        bool(record["semantic_set_agreement"])
        and not bool(record["semantic_sequence_agreement"])
        for record in records
    )
    raw_order_only = sum(
        bool(record["raw_index_set_agreement"])
        and not bool(record["raw_index_sequence_agreement"])
        for record in records
    )
    raw_index_only = sum(
        bool(record["semantic_set_agreement"])
        and not bool(record["raw_index_set_agreement"])
        for record in records
    )
    games_with_decision = {int(record["episode_id"]) for record in records}
    games_touched = {
        int(record["episode_id"])
        for record in records
        if not bool(record["semantic_set_agreement"])
    }
    cohort_ids = {int(game["episode_id"]) for game in cohort_games}
    return {
        "cohort_seat_games": len(cohort_ids),
        "decisions": decisions,
        "primary_comparison_unit": "public_semantic_action_set",
        "logged_action_agreements": semantic_set_agreements,
        "logged_action_disagreements": semantic_set_disagreements,
        "decision_disagreement_rate": _rate(semantic_set_disagreements, decisions),
        "semantic_action_set_agreements": semantic_set_agreements,
        "semantic_action_set_disagreements": semantic_set_disagreements,
        "semantic_action_set_disagreement_rate": _rate(
            semantic_set_disagreements, decisions
        ),
        "semantic_action_sequence_agreements": semantic_sequence_agreements,
        "semantic_action_sequence_disagreements": (
            decisions - semantic_sequence_agreements
        ),
        "semantic_order_only_differences": semantic_order_only,
        "raw_index_sequence_agreements": raw_sequence_agreements,
        "raw_index_sequence_disagreements": decisions - raw_sequence_agreements,
        "raw_index_set_agreements": raw_set_agreements,
        "raw_index_set_disagreements": decisions - raw_set_agreements,
        "raw_index_order_only_differences": raw_order_only,
        "interchangeable_copy_index_only_differences": raw_index_only,
        "seat_games_with_decision": len(games_with_decision),
        "game_coverage_rate": _rate(len(games_with_decision), len(cohort_ids)),
        "seat_games_touched": len(games_touched),
        "game_touch_rate_among_games_with_decision": _rate(
            len(games_touched), len(games_with_decision)
        ),
        "cohort_game_touch_rate": _rate(len(games_touched), len(cohort_ids)),
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    cohort_games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize decisions and action-changing game touch by seat outcome."""
    summary = _base_summary(records, cohort_games)
    summary["by_outcome"] = {}
    for label in OUTCOMES:
        outcome_records = [record for record in records if record["outcome"] == label]
        outcome_games = [game for game in cohort_games if game["outcome"] == label]
        summary["by_outcome"][label] = _base_summary(outcome_records, outcome_games)
    return summary


def _record_example(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "episode_id",
            "step",
            "outcome",
            "family",
            "public_signature",
            "effect_card_id",
            "context",
            "logged_action",
            "model_action",
            "option_card_ids",
        )
    }


def _scope_report(
    records: Sequence[Mapping[str, Any]],
    cohort_games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fixed = [record for record in records if record["family"] in FIXED_FAMILIES]
    report = {
        "overall": summarize_records(records, cohort_games),
        "fixed_family_union": summarize_records(fixed, cohort_games),
        "families": {
            family: summarize_records(
                [record for record in records if record["family"] == family],
                cohort_games,
            )
            for family in ALL_FAMILIES
        },
    }
    report["disagreement_examples"] = {
        family: [
            _record_example(record)
            for record in records
            if record["family"] == family and not record["semantic_set_agreement"]
        ][:3]
        for family in ALL_FAMILIES
    }
    return report


def _add_diagnostic(
    counts: Counter[str],
    examples: list[dict[str, Any]],
    reason: str,
    detail: Mapping[str, Any],
) -> None:
    counts[reason] += 1
    if len(examples) < 20:
        examples.append({"reason": reason, **dict(detail)})


def discover_cohort(
    replay_dir: Path,
    *,
    team: str,
) -> tuple[list[CohortGame], dict[str, Any]]:
    """Discover and content-lock the exact-mirror leader-seat cohort."""
    if not replay_dir.is_dir():
        raise ScreenError(f"replay directory does not exist: {replay_dir}")
    files = sorted(
        (path for path in replay_dir.iterdir() if path.is_file()),
        key=lambda path: path.name,
    )
    numeric = [path for path in files if NUMERIC_REPLAY.fullmatch(path.name)]
    ignored = [path.name for path in files if path not in numeric]
    if not numeric:
        raise ScreenError(f"no numeric Kaggle replay JSONs in {replay_dir}")

    games: list[CohortGame] = []
    manifest: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    exclusion_examples: list[dict[str, Any]] = []
    filtered = 0
    seen_episode_ids: set[int] = set()
    for path in numeric:
        match = NUMERIC_REPLAY.fullmatch(path.name)
        assert match is not None
        filename_episode_id = int(match.group("episode"))
        raw = path.read_bytes()
        content_sha = hashlib.sha256(raw).hexdigest()
        manifest.append({"file": path.name, "sha256": content_sha})
        try:
            document = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            _add_diagnostic(
                exclusions,
                exclusion_examples,
                "invalid_json",
                {"file": path.name, "error": type(error).__name__},
            )
            continue
        status, detail = audit_replay(
            document,
            filename_episode_id=filename_episode_id,
            team=team,
        )
        if status == "filter:not_exact_mirror":
            filtered += 1
            continue
        if status != "eligible":
            _add_diagnostic(
                exclusions,
                exclusion_examples,
                status.removeprefix("exclude:"),
                {"file": path.name, **detail},
            )
            continue
        episode_id = int(detail["episode_id"])
        if episode_id in seen_episode_ids:
            _add_diagnostic(
                exclusions,
                exclusion_examples,
                "duplicate_episode_id",
                {"file": path.name, "episode_id": episode_id},
            )
            continue
        seen_episode_ids.add(episode_id)
        games.append(CohortGame(
            path=path,
            episode_id=episode_id,
            seat=int(detail["seat"]),
            outcome=str(detail["outcome"]),
            deck=tuple(int(card_id) for card_id in detail["deck"]),
            content_sha256=content_sha,
        ))
    games.sort(key=lambda game: game.episode_id)
    diagnostics = {
        "numeric_replay_files": len(numeric),
        "control_or_non_numeric_files_ignored": ignored,
        "non_exact_mirror_files_filtered": filtered,
        "exact_mirror_games_eligible": len(games),
        "game_audit_exclusions": dict(sorted(exclusions.items())),
        "game_audit_exclusion_examples": exclusion_examples,
        "input_file_manifest_sha256": canonical_sha256(manifest),
        "eligible_cohort_manifest_sha256": canonical_sha256([
            {
                "episode_id": game.episode_id,
                "seat": game.seat,
                "outcome": game.outcome,
                "sha256": game.content_sha256,
            }
            for game in games
        ]),
    }
    return games, diagnostics


def _selection_contract(select: Mapping[str, Any]) -> tuple[int, int, int]:
    options = select.get("option")
    if not isinstance(options, list) or not options:
        raise ValueError("missing_options")
    minimum = select.get("minCount", 1)
    maximum = select.get("maxCount", 1)
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
    ):
        raise ValueError("invalid_cardinality")
    return len(options), int(minimum), int(maximum)


def predict_action(
    net: model.QuV2Net,
    observation: Mapping[str, Any],
    deck: Sequence[int],
) -> list[int]:
    encoded = QF.encode_public_observation(observation, deck)
    logits, _ = net.forward(encoded)
    select = observation["select"]
    return model.decode_qu_v2(
        logits,
        len(select["option"]),
        int(select.get("minCount", 1)),
        int(select.get("maxCount", 1)),
    )


def collect_records(
    games: Sequence[CohortGame],
    net: model.QuV2Net,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the frozen network on each valid same-state ST_CARD prompt."""
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    exclusion_examples: list[dict[str, Any]] = []
    inactive_rows_ignored = 0
    for game in games:
        raw = game.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != game.content_sha256:
            raise ScreenError(f"replay drifted during analysis: {game.path}")
        document = json.loads(raw)
        steps = document["steps"]
        for step_index in range(1, len(steps)):
            source_row = steps[step_index - 1][game.seat]
            if source_row.get("status") == "INACTIVE":
                inactive_rows_ignored += 1
                continue
            observation = source_row.get("observation")
            if not isinstance(observation, Mapping):
                continue
            select = observation.get("select")
            if not isinstance(select, Mapping) or select.get("type") != ST_CARD:
                continue
            detail = {"episode_id": game.episode_id, "step": step_index - 1}
            current = observation.get("current")
            if (
                not isinstance(current, Mapping)
                or current.get("yourIndex") != game.seat
            ):
                _add_diagnostic(
                    exclusions, exclusion_examples, "wrong_or_missing_actor_view", detail
                )
                continue
            try:
                option_count, minimum, maximum = _selection_contract(select)
            except ValueError as error:
                _add_diagnostic(
                    exclusions, exclusion_examples, str(error), detail
                )
                continue
            paired_row = steps[step_index][game.seat]
            logged = paired_row.get("action")
            try:
                logged_sequence, logged_canonical = canonical_action(
                    logged,
                    option_count=option_count,
                    min_count=minimum,
                    max_count=maximum,
                )
            except ValueError as error:
                _add_diagnostic(
                    exclusions,
                    exclusion_examples,
                    "invalid_logged_action",
                    {**detail, "error": str(error)},
                )
                continue
            try:
                predicted = predict_action(net, observation, game.deck)
                model_sequence, model_canonical = canonical_action(
                    predicted,
                    option_count=option_count,
                    min_count=minimum,
                    max_count=maximum,
                )
            except Exception as error:  # audit, do not silently reinterpret a prompt
                _add_diagnostic(
                    exclusions,
                    exclusion_examples,
                    "model_inference_failure",
                    {**detail, "error": type(error).__name__},
                )
                continue
            view = ObsView(dict(observation))
            family = classify_family(view)
            option_card_ids = [view.option_card_id(option) for option in view.options]
            logged_semantic_sequence, logged_semantic_set = semantic_action_signature(
                view, logged_sequence
            )
            model_semantic_sequence, model_semantic_set = semantic_action_signature(
                view, model_sequence
            )
            records.append({
                "episode_id": game.episode_id,
                "step": step_index - 1,
                "outcome": game.outcome,
                "family": family,
                "public_signature": md_v2_card.opponent_has_public_grim_signature(view),
                "effect_card_id": view.effect_card_id,
                "context": view.context,
                "logged_action": list(logged_sequence),
                "model_action": list(model_sequence),
                "option_card_ids": option_card_ids,
                "raw_index_sequence_agreement": logged_sequence == model_sequence,
                "raw_index_set_agreement": logged_canonical == model_canonical,
                "semantic_sequence_agreement": (
                    logged_semantic_sequence == model_semantic_sequence
                ),
                "semantic_set_agreement": logged_semantic_set == model_semantic_set,
            })
    return records, {
        "inactive_source_rows_ignored": inactive_rows_ignored,
        "decision_audit_exclusions": dict(sorted(exclusions.items())),
        "decision_audit_exclusion_examples": exclusion_examples,
    }


def load_frozen_net(weights: Path) -> model.QuV2Net:
    digest = file_sha256(weights)
    if digest != EXPECTED_WEIGHTS_SHA256 or digest != md_v2_card.WEIGHTS_SHA256:
        raise ScreenError(
            f"ST_CARD weight identity mismatch: got {digest}, "
            f"expected {EXPECTED_WEIGHTS_SHA256}"
        )
    try:
        with np.load(weights, allow_pickle=False) as archive:
            return model.QuV2Net(archive)
    except (OSError, ValueError) as error:
        raise ScreenError(f"cannot load frozen ST_CARD weights: {error}") from error


def analyze(
    replay_dir: Path,
    *,
    team: str,
    weights: Path,
    expected_exact_mirrors: int = EXPECTED_EXACT_MIRRORS,
) -> dict[str, Any]:
    target = tuple(md_v2_card.TARGET_DECK)
    if index_corpus.deck_sha256(target) != TARGET_DECK_SHA256:
        raise ScreenError("target deck identity drifted")
    if md_v2_card.TARGET_DECK_SHA256 != TARGET_DECK_SHA256:
        raise ScreenError("runtime target deck hash drifted")
    if not weights.is_file():
        raise ScreenError(f"missing frozen ST_CARD weights: {weights}")
    net = load_frozen_net(weights)
    games, game_audit = discover_cohort(replay_dir, team=team)
    if len(games) != expected_exact_mirrors:
        raise ScreenError(
            f"expected {expected_exact_mirrors} eligible exact mirrors, "
            f"found {len(games)}"
        )
    records, decision_audit = collect_records(games, net)
    game_rows = [
        {"episode_id": game.episode_id, "outcome": game.outcome}
        for game in games
    ]
    public_records = [record for record in records if record["public_signature"]]
    all_report = _scope_report(records, game_rows)
    public_report = _scope_report(public_records, game_rows)
    game_exclusions = game_audit["game_audit_exclusions"]
    decision_exclusions = decision_audit["decision_audit_exclusions"]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "analysis_design": {
            "same_state": True,
            "prospective": False,
            "prior_exploratory_sizing_seen": True,
            "thresholds_or_go_no_go_rules": None,
            "logged_action_role": (
                "descriptive elite-pilot comparison, not a causal value label"
            ),
        },
        "cohort": {
            "team": team,
            "deck_sha256": TARGET_DECK_SHA256,
            "exact_mirror_games": len(games),
            "by_outcome": dict(sorted(Counter(game.outcome for game in games).items())),
        },
        "artifacts": {
            "weights": {
                "path": str(weights.resolve()),
                "sha256": file_sha256(weights),
            },
            "analysis_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__)),
            },
            "feature_schema": QF.SCHEMA,
            "feature_dependency_fingerprint": QF.assert_feature_dependency_lock(),
            "execution_environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        },
        "audit": {
            **game_audit,
            **decision_audit,
            "complete": not game_exclusions and not decision_exclusions,
        },
        "scopes": {
            "all_st_card_counterfactual": {
                "runtime_scope": False,
                "note": (
                    "Frozen ST_CARD model applied to every ST_CARD observation; "
                    "early prompts without a public opposing Grimmsnarl-line "
                    "signature are outside the shipped router."
                ),
                **all_report,
            },
            "deployed_public_signature": {
                "runtime_scope": True,
                "note": (
                    "Only observations where frozen Dobi-v1 actually routes "
                    "ST_CARD to this specialist."
                ),
                "decisions_excluded_as_out_of_route": len(records) - len(public_records),
                **public_report,
            },
        },
        "authority": {
            "training": False,
            "gameplay_gate": False,
            "promotion": False,
            "packaging": False,
            "upload": False,
        },
        "interpretation": (
            "Post-hoc contingency evidence only. Agreement with a winning-seat "
            "logged action can nominate a supervised target family, but cannot "
            "establish that action as better or authorize a model change."
        ),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replay_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_REPLAYS,
        help="directory containing numeric Sixth Sense Kaggle replay JSONs",
    )
    parser.add_argument("--team", default="Sixth Sense")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--expected-exact-mirrors",
        type=int,
        default=EXPECTED_EXACT_MIRRORS,
    )
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_exact_mirrors <= 0:
        raise SystemExit("--expected-exact-mirrors must be positive")
    if args.json_out is not None and args.json_out.exists():
        raise SystemExit(f"refusing to overwrite {args.json_out}")
    try:
        result = analyze(
            args.replay_dir.resolve(),
            team=args.team,
            weights=args.weights.resolve(),
            expected_exact_mirrors=args.expected_exact_mirrors,
        )
    except ScreenError as error:
        raise SystemExit(str(error)) from error
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    compact = {
        "schema": result["schema"],
        "result_sha256": result["result_sha256"],
        "cohort": result["cohort"],
        "audit_complete": result["audit"]["complete"],
        "all_st_card": result["scopes"]["all_st_card_counterfactual"]["overall"],
        "deployed_scope": result["scopes"]["deployed_public_signature"]["overall"],
        "json_out": str(args.json_out.resolve()) if args.json_out else None,
    }
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
