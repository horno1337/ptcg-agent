"""Select the locked development roots for Qu-v2C label reliability.

This experiment asks whether two independent exact-terminal panel runs assign
stable signs and action rankings to the same public roots.  It is deliberately
separate from model fitting and promotion:

* input must be the provenance-locked ``factual-critic`` root corpus;
* exactly one stable Qu-v2B root is selected from each of 30 unique games;
* outcome, learner seat, and B/parent agreement are exactly balanced;
* a deterministic diversity-first rule covers turns, option counts, and
  opponent archetypes; and
* the derived public/private artifacts preserve the original physical
  privilege boundary.

The output is development-only.  Neither selecting these roots nor measuring
their label repeatability can authorize a teacher, actor, or submission.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tools.research import mine_qu_v2c_roots as MINE  # noqa: E402
from tools.research import validate_qu_v2c_roots as VALIDATE  # noqa: E402


SCHEMA = "ptcg.qu-v2c.reliability-root-selection.v1"
SELECTION_MODE = "label-reliability-development"
SELECTION_SEED = 230723
ROOT_COUNT = 30
MIN_STABLE_REFLEX_MARGIN = 1e-6
SELECTION_POLICY = (
    "development-only deterministic diversity-first selection of exactly one "
    "frozen-Qu-v2B stable-margin root from each of 30 unique factual-critic "
    "games; exact 15/15 win/loss, 15/15 learner-seat, and 15/15 "
    "B-parent agreement/disagreement balance; fixed eight-stratum quotas; "
    "maximize new/balanced opponent archetype, turn bucket, option count, "
    "and exact turn with sha256(seed,root_id) tie-break"
)
GENERALIZATION_SCHEMA = "ptcg.qu-v2c.generalization-root-selection.v1"
GENERALIZATION_SELECTION_MODE = "label-generalization-heldout"
GENERALIZATION_SELECTION_SEED = 240724
GENERALIZATION_SELECTION_POLICY = (
    "held-out deterministic diversity-first selection of exactly one "
    "frozen-Qu-v2B stable-margin root from each of 30 unique factual-critic "
    "games after explicit game-level cohort exclusions; balance outcome, "
    "learner seat, B-parent relation, opponent archetype, turn bucket, option "
    "count, and exact turn without claiming infeasible exact marginal quotas"
)
REPLICATION_CANDIDATE_SCHEMA = (
    "ptcg.qu-v2c.replication-candidate-root-selection.v1")
REPLICATION_CANDIDATE_SELECTION_MODE = (
    "label-generalization-replication-candidates")
REPLICATION_CANDIDATE_SELECTION_SEED = GENERALIZATION_SELECTION_SEED
REPLICATION_CANDIDATE_ROOT_COUNT = 40
REPLICATION_CANDIDATE_SELECTION_POLICY = (
    "predeclared ordered pool of exactly 40 fresh frozen-Qu-v2B "
    "stable-margin roots from unique factual-critic games after explicit "
    "game-level cohort exclusions; run both actual 16-rollout panels on all "
    "40 and mechanically retain the first 30 in this artifact order that "
    "complete cleanly in both, without consulting outcomes or label signs"
)
REPLICATION_CANDIDATE_V2_SCHEMA = (
    "ptcg.qu-v2c.replication-candidate-root-selection.v2")
REPLICATION_CANDIDATE_V2_SELECTION_MODE = (
    "label-generalization-replication-candidates-v2")
REPLICATION_CANDIDATE_V2_SELECTION_POLICY = (
    "predeclared diversity-priority pool of exactly 40 fresh frozen-Qu-v2B "
    "stable-margin roots from unique factual-critic games after explicit "
    "game-level exclusions; preserve deterministic greedy traversal so the "
    "first 30 roots exactly equal the standalone diversity-balanced 30-root "
    "selection, followed by ten ordered alternates; run both actual panels "
    "on all 40 and mechanically retain the first 30 in this artifact order "
    "that complete both, without consulting outcomes or label signs"
)
REPLICATION_FINAL_SCHEMA = (
    "ptcg.qu-v2c.replication-final-root-selection.v1")
REPLICATION_FINAL_SELECTION_MODE = (
    "label-generalization-replication-finalized")
REPLICATION_FINAL_SELECTION_POLICY = (
    "from one predeclared ordered 40-root replication candidate pool, retain "
    "the first 30 root identities mechanically complete in both actual "
    "discovery and confirmation runs; selection uses no terminal outcomes, "
    "action values, label signs, critic scores, or confirmation results"
)
CANARY_OVERRIDE_SCHEMA = "ptcg.qu-v2c.canary-override-root-selection.v1"
CANARY_OVERRIDE_SELECTION_MODE = "qu-v2c-canary-override-candidates"
CANARY_OVERRIDE_SELECTION_SEED = 260726
CANARY_OVERRIDE_ROOT_COUNT = 45
CANARY_OVERRIDE_SELECTION_POLICY = (
    "development-only ordered pool of exactly 45 public-only guarded "
    "Qu-v2C overrides from unique factual-corpus games; membership may use "
    "the frozen canary decision but never terminal outcomes, rollout values, "
    "label signs, or confirmation results; run both actual discovery and "
    "independent confirmation panels on all roots"
)

# These eight fixed cells make every requested binary marginal exactly 15/15.
# ``False`` means B and its parent agree; ``True`` means they disagree.
STRATUM_TARGETS: dict[tuple[str, int, bool], int] = {
    ("loss", 0, False): 4,
    ("loss", 0, True): 4,
    ("loss", 1, False): 4,
    ("loss", 1, True): 3,
    ("win", 0, False): 3,
    ("win", 0, True): 4,
    ("win", 1, False): 4,
    ("win", 1, True): 4,
}

DEFAULT_ROOT_DIR = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-roots-v1"
    / "factual-critic-pilot"
)
DEFAULT_OUT = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-roots-v1"
    / "label-reliability-development-30"
)


class SelectionError(RuntimeError):
    """The parent corpus or deterministic selection violated its contract."""


@dataclass(frozen=True)
class Candidate:
    """Validated selection metadata for one positionally paired root."""

    public: dict[str, Any]
    privileged: dict[str, Any]
    root_id: str
    game_key: str
    episode_id: str
    outcome: str
    seat: int
    disagreement: bool
    turn: int
    turn_bucket: str
    option_count: int
    archetype: str
    margin: float

    @property
    def stratum(self) -> tuple[str, int, bool]:
        return (self.outcome, self.seat, self.disagreement)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(record) + b"\n" for record in records)


def _atomic_bytes(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise SelectionError(f"stale partial output exists: {temporary}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _turn_bucket(turn: int) -> str:
    if turn <= 3:
        return "early_1_3"
    if turn <= 6:
        return "middle_4_6"
    return "late_7_plus"


def _validate_parent_manifest(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema") != MINE.SCHEMA
        or manifest.get("selection_mode") != "factual-critic"
        or manifest.get("selection_policy")
        != MINE.FACTUAL_CRITIC_SELECTION_POLICY
    ):
        raise SelectionError(
            "reliability roots require the locked factual-critic parent")
    if not _is_sha256(manifest.get("manifest_sha256")):
        raise SelectionError("parent root manifest hash is malformed")
    weights = manifest.get("weights")
    b_record = weights.get("qu_v2b") if isinstance(weights, Mapping) else None
    parent_record = (
        weights.get("parent") if isinstance(weights, Mapping) else None)
    if (
        not isinstance(b_record, Mapping)
        or b_record.get("sha256") != MINE.FROZEN_QU_V2B_SHA256
        or not isinstance(parent_record, Mapping)
        or parent_record.get("sha256") != MINE.FROZEN_QU_V2A_PARENT_SHA256
    ):
        raise SelectionError("parent corpus does not bind both frozen policies")
    mining = manifest.get("mining")
    if (
        not isinstance(mining, Mapping)
        or not isinstance(mining.get("inputs"), list)
    ):
        raise SelectionError("parent corpus has no replay-input provenance")


def _candidate(
    public: Mapping[str, Any],
    privileged: Mapping[str, Any],
) -> Candidate | None:
    """Validate one parent row and return it when margin-stable."""
    root_id = public.get("root_id")
    if (
        public.get("schema") != MINE.PUBLIC_SCHEMA
        or privileged.get("schema") != MINE.PRIVILEGED_SCHEMA
        or not _is_sha256(root_id)
        or privileged.get("root_id") != root_id
    ):
        raise SelectionError("parent public/private root identity is malformed")
    public_source = public.get("source")
    private_source = privileged.get("source")
    selection = public.get("selection")
    prompt = public.get("prompt")
    b_record = public.get("qu_v2b")
    parent_record = public.get("parent")
    if not all(isinstance(value, Mapping) for value in (
        public_source, private_source, selection, prompt, b_record,
        parent_record,
    )):
        raise SelectionError(f"root {root_id} lacks selection metadata")
    assert isinstance(public_source, Mapping)
    assert isinstance(private_source, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(prompt, Mapping)
    assert isinstance(b_record, Mapping)

    if (
        selection.get("mode") != "factual-critic"
        or selection.get("policy") != MINE.FACTUAL_CRITIC_SELECTION_POLICY
        or selection.get("factual_terminal_return_label") is not True
        or selection.get("supported_exact_root") is not True
    ):
        raise SelectionError(f"root {root_id} is not a factual-critic root")
    episode_raw = public_source.get("episode_id")
    if (
        isinstance(episode_raw, bool)
        or not isinstance(episode_raw, (str, int))
    ):
        raise SelectionError(f"root {root_id} has invalid episode identity")
    episode_id = str(episode_raw)
    replay_sha = public_source.get("replay_sha256")
    if not _is_sha256(replay_sha):
        raise SelectionError(f"root {root_id} has invalid replay hash")
    for key in ("episode_id", "source_step", "learner_seat", "replay_sha256"):
        if str(private_source.get(key)) != str(public_source.get(key)):
            raise SelectionError(
                f"root {root_id} public/private source {key} diverged")

    outcome = public_source.get("outcome")
    reward = public_source.get("learner_reward")
    if (
        outcome not in ("win", "loss")
        or not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or not math.isfinite(float(reward))
        or (outcome == "win" and float(reward) <= 0.0)
        or (outcome == "loss" and float(reward) >= 0.0)
    ):
        raise SelectionError(f"root {root_id} has invalid resolved outcome")
    seat = public_source.get("learner_seat")
    if seat not in (0, 1) or prompt.get("selecting_seat") != seat:
        raise SelectionError(f"root {root_id} has invalid learner seat")
    disagreement = selection.get("frozen_b_parent_disagreement")
    if not isinstance(disagreement, bool):
        raise SelectionError(f"root {root_id} lacks B/parent agreement state")
    expected_disagreement = (
        b_record.get("semantic_action")
        != parent_record.get("semantic_action")
    )
    if disagreement != expected_disagreement:
        raise SelectionError(
            f"root {root_id} B/parent disagreement flag drifted")

    margin = b_record.get("margin")
    if (
        not isinstance(margin, (int, float))
        or isinstance(margin, bool)
        or not math.isfinite(float(margin))
        or float(margin) < 0.0
    ):
        raise SelectionError(f"root {root_id} has invalid Qu-v2B margin")
    if float(margin) <= MIN_STABLE_REFLEX_MARGIN:
        return None

    turn = prompt.get("turn")
    options = prompt.get("semantic_options")
    if (
        not isinstance(turn, int)
        or isinstance(turn, bool)
        or turn < 1
        or not isinstance(options, list)
        or not 2 <= len(options) <= 12
    ):
        raise SelectionError(f"root {root_id} has invalid prompt diversity data")
    archetype = public_source.get("opponent_archetype")
    if not isinstance(archetype, str) or not archetype.strip():
        raise SelectionError(f"root {root_id} has no opponent archetype")

    game_key = _value_sha256({
        "episode_id": episode_id,
        "replay_sha256": replay_sha,
    })
    return Candidate(
        public=copy.deepcopy(dict(public)),
        privileged=copy.deepcopy(dict(privileged)),
        root_id=root_id,
        game_key=game_key,
        episode_id=episode_id,
        outcome=outcome,
        seat=int(seat),
        disagreement=disagreement,
        turn=turn,
        turn_bucket=_turn_bucket(turn),
        option_count=len(options),
        archetype=archetype.strip(),
        margin=float(margin),
    )


def _cell_can_fill(
    candidates: Sequence[Candidate],
    used_games: set[str],
    remaining: Mapping[tuple[str, int, bool], int],
    outcome: str,
    seat: int,
) -> bool:
    """Return whether distinct remaining games can fill one binary cell."""
    slots: list[tuple[bool, int]] = []
    for disagreement in (False, True):
        count = remaining[(outcome, seat, disagreement)]
        slots.extend((disagreement, index) for index in range(count))
    if not slots:
        return True
    games_by_label = {
        disagreement: sorted({
            candidate.game_key
            for candidate in candidates
            if (
                candidate.outcome == outcome
                and candidate.seat == seat
                and candidate.disagreement is disagreement
                and candidate.game_key not in used_games
            )
        })
        for disagreement in (False, True)
    }
    # Scarcer labels are matched first.  A standard augmenting-path matching
    # proves feasibility even when one game has both agreement root types.
    slots.sort(key=lambda item: (len(games_by_label[item[0]]), item))
    matched_game: dict[str, tuple[bool, int]] = {}

    def assign(slot: tuple[bool, int], seen: set[str]) -> bool:
        for game in games_by_label[slot[0]]:
            if game in seen:
                continue
            seen.add(game)
            previous = matched_game.get(game)
            if previous is None or assign(previous, seen):
                matched_game[game] = slot
                return True
        return False

    return all(assign(slot, set()) for slot in slots)


def _can_complete(
    candidates: Sequence[Candidate],
    used_games: set[str],
    remaining: Mapping[tuple[str, int, bool], int],
) -> bool:
    if any(count < 0 for count in remaining.values()):
        return False
    return all(
        _cell_can_fill(candidates, used_games, remaining, outcome, seat)
        for outcome in ("loss", "win")
        for seat in (0, 1)
    )


def _tie_rank(root_id: str) -> int:
    material = f"{SELECTION_SEED}:{root_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _generalization_tie_rank(root_id: str) -> int:
    material = f"{GENERALIZATION_SELECTION_SEED}:{root_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _diversity_score(
    candidate: Candidate,
    archetypes: Counter[str],
    turn_buckets: Counter[str],
    option_counts: Counter[int],
    exact_turns: Counter[int],
) -> tuple[int, ...]:
    """Lexicographic, deterministic diversity-first candidate score."""
    return (
        int(not archetypes[candidate.archetype]),
        int(not turn_buckets[candidate.turn_bucket]),
        int(not option_counts[candidate.option_count]),
        int(not exact_turns[candidate.turn]),
        -archetypes[candidate.archetype],
        -turn_buckets[candidate.turn_bucket],
        -option_counts[candidate.option_count],
        -exact_turns[candidate.turn],
        # After diversity, prefer a larger numerical margin, quantized so that
        # tiny platform float differences cannot change the selected root.
        int(min(candidate.margin, 1_000.0) * 1_000_000),
        -_tie_rank(candidate.root_id),
    )


def select_roots(
    public_records: Sequence[Mapping[str, Any]],
    privileged_records: Sequence[Mapping[str, Any]],
    *,
    excluded_game_keys: frozenset[str] = frozenset(),
) -> tuple[list[Candidate], dict[str, Any]]:
    """Return the exact deterministic 30-game reliability root panel."""
    if len(public_records) != len(privileged_records):
        raise SelectionError("parent public/private root counts diverged")
    eligible: list[Candidate] = []
    unstable = 0
    for public, privileged in zip(public_records, privileged_records):
        candidate = _candidate(public, privileged)
        if candidate is None:
            unstable += 1
        elif candidate.game_key in excluded_game_keys:
            continue
        else:
            eligible.append(candidate)
    if len({candidate.root_id for candidate in eligible}) != len(eligible):
        raise SelectionError("eligible parent roots are duplicated")
    if not _can_complete(eligible, set(), STRATUM_TARGETS):
        raise SelectionError(
            "factual parent cannot satisfy locked reliability strata")

    remaining = dict(STRATUM_TARGETS)
    selected: list[Candidate] = []
    used_games: set[str] = set()
    archetypes: Counter[str] = Counter()
    turn_buckets: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    exact_turns: Counter[int] = Counter()

    while len(selected) < ROOT_COUNT:
        feasible: list[Candidate] = []
        for candidate in eligible:
            if (
                candidate.game_key in used_games
                or remaining[candidate.stratum] <= 0
            ):
                continue
            next_remaining = dict(remaining)
            next_remaining[candidate.stratum] -= 1
            if _can_complete(
                eligible, used_games | {candidate.game_key}, next_remaining,
            ):
                feasible.append(candidate)
        if not feasible:
            raise SelectionError(
                "deterministic reliability selection became infeasible")
        chosen = max(
            feasible,
            key=lambda candidate: _diversity_score(
                candidate, archetypes, turn_buckets, option_counts,
                exact_turns,
            ),
        )
        selected.append(chosen)
        used_games.add(chosen.game_key)
        remaining[chosen.stratum] -= 1
        archetypes[chosen.archetype] += 1
        turn_buckets[chosen.turn_bucket] += 1
        option_counts[chosen.option_count] += 1
        exact_turns[chosen.turn] += 1

    if (
        any(remaining.values())
        or len(selected) != ROOT_COUNT
        or len(used_games) != ROOT_COUNT
    ):
        raise SelectionError("locked reliability selection is incomplete")

    # Artifact order is independent of greedy traversal and is itself stable.
    selected.sort(key=lambda candidate: (
        candidate.outcome,
        candidate.seat,
        candidate.disagreement,
        candidate.game_key,
        candidate.root_id,
    ))
    outcome_counts = Counter(candidate.outcome for candidate in selected)
    seat_counts = Counter(str(candidate.seat) for candidate in selected)
    disagreement_counts = Counter(
        "disagree" if candidate.disagreement else "agree"
        for candidate in selected
    )
    if (
        outcome_counts != Counter({"loss": 15, "win": 15})
        or seat_counts != Counter({"0": 15, "1": 15})
        or disagreement_counts != Counter({"agree": 15, "disagree": 15})
    ):
        raise SelectionError("locked reliability marginal balance drifted")

    diagnostics = {
        "parent_records": len(public_records),
        "excluded_parent_games": len(excluded_game_keys),
        "eligible_stable_margin_roots": len(eligible),
        "rejected_unstable_margin_roots": unstable,
        "eligible_unique_games": len({
            candidate.game_key for candidate in eligible
        }),
        "selected_roots": len(selected),
        "selected_unique_games": len(used_games),
        "outcomes": dict(sorted(outcome_counts.items())),
        "learner_seats": dict(sorted(seat_counts.items())),
        "b_parent_relation": dict(sorted(disagreement_counts.items())),
        "strata": {
            f"{outcome}/seat-{seat}/"
            f"{'disagree' if disagreement else 'agree'}": sum(
                candidate.stratum == (outcome, seat, disagreement)
                for candidate in selected
            )
            for outcome, seat, disagreement in STRATUM_TARGETS
        },
        "opponent_archetypes": dict(sorted(Counter(
            candidate.archetype for candidate in selected).items())),
        "turn_buckets": dict(sorted(Counter(
            candidate.turn_bucket for candidate in selected).items())),
        "exact_turns": dict(sorted(Counter(
            str(candidate.turn) for candidate in selected).items())),
        "option_counts": dict(sorted(Counter(
            str(candidate.option_count) for candidate in selected).items())),
        "selected_root_ids_sha256": _value_sha256([
            candidate.root_id for candidate in selected
        ]),
        "selected_game_keys_sha256": _value_sha256([
            candidate.game_key for candidate in selected
        ]),
    }
    return selected, diagnostics


def select_generalization_roots(
    public_records: Sequence[Mapping[str, Any]],
    privileged_records: Sequence[Mapping[str, Any]],
    *,
    excluded_game_keys: frozenset[str] = frozenset(),
    root_count: int = ROOT_COUNT,
    preserve_selection_order: bool = False,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Select a disjoint held-out panel when exact development quotas are spent."""
    if len(public_records) != len(privileged_records):
        raise SelectionError("parent public/private root counts diverged")
    if (
        not isinstance(root_count, int)
        or isinstance(root_count, bool)
        or root_count < ROOT_COUNT
    ):
        raise SelectionError(
            f"held-out generalization requires at least {ROOT_COUNT} roots")
    eligible: list[Candidate] = []
    unstable = 0
    for public, privileged in zip(public_records, privileged_records):
        candidate = _candidate(public, privileged)
        if candidate is None:
            unstable += 1
        elif candidate.game_key not in excluded_game_keys:
            eligible.append(candidate)
    if len({candidate.root_id for candidate in eligible}) != len(eligible):
        raise SelectionError("eligible parent roots are duplicated")
    eligible_games = {candidate.game_key for candidate in eligible}
    if len(eligible_games) < root_count:
        raise SelectionError(
            f"fewer than {root_count} unused stable-margin games remain for "
            "held-out generalization"
        )

    selected: list[Candidate] = []
    used_games: set[str] = set()
    outcomes: Counter[str] = Counter()
    seats: Counter[int] = Counter()
    relations: Counter[bool] = Counter()
    strata: Counter[tuple[str, int, bool]] = Counter()
    archetypes: Counter[str] = Counter()
    turn_buckets: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    exact_turns: Counter[int] = Counter()

    while len(selected) < root_count:
        available = [
            candidate for candidate in eligible
            if candidate.game_key not in used_games
        ]
        if not available:
            raise SelectionError(
                "held-out generalization selection became incomplete")

        def score(candidate: Candidate) -> tuple[int, ...]:
            return (
                int(not strata[candidate.stratum]),
                int(not archetypes[candidate.archetype]),
                int(not turn_buckets[candidate.turn_bucket]),
                int(not option_counts[candidate.option_count]),
                int(not exact_turns[candidate.turn]),
                -outcomes[candidate.outcome],
                -seats[candidate.seat],
                -relations[candidate.disagreement],
                -strata[candidate.stratum],
                -archetypes[candidate.archetype],
                -turn_buckets[candidate.turn_bucket],
                -option_counts[candidate.option_count],
                -exact_turns[candidate.turn],
                candidate.option_count,
                int(min(candidate.margin, 1_000.0) * 1_000_000),
                -_generalization_tie_rank(candidate.root_id),
            )

        chosen = max(available, key=score)
        selected.append(chosen)
        used_games.add(chosen.game_key)
        outcomes[chosen.outcome] += 1
        seats[chosen.seat] += 1
        relations[chosen.disagreement] += 1
        strata[chosen.stratum] += 1
        archetypes[chosen.archetype] += 1
        turn_buckets[chosen.turn_bucket] += 1
        option_counts[chosen.option_count] += 1
        exact_turns[chosen.turn] += 1

    if not preserve_selection_order:
        selected.sort(key=lambda candidate: (
            candidate.game_key, candidate.root_id))
    diagnostics = {
        "parent_records": len(public_records),
        "excluded_parent_games": len(excluded_game_keys),
        "eligible_stable_margin_roots": len(eligible),
        "rejected_unstable_margin_roots": unstable,
        "eligible_unique_games": len(eligible_games),
        "selected_roots": len(selected),
        "selected_unique_games": len(used_games),
        "outcomes": dict(sorted(outcomes.items())),
        "learner_seats": {
            str(key): value for key, value in sorted(seats.items())
        },
        "b_parent_relation": {
            ("disagree" if key else "agree"): value
            for key, value in sorted(relations.items())
        },
        "strata": {
            f"{outcome}/seat-{seat}/"
            f"{'disagree' if disagreement else 'agree'}": count
            for (outcome, seat, disagreement), count in sorted(strata.items())
        },
        "opponent_archetypes": dict(sorted(archetypes.items())),
        "turn_buckets": dict(sorted(turn_buckets.items())),
        "exact_turns": {
            str(key): value for key, value in sorted(exact_turns.items())
        },
        "option_counts": {
            str(key): value for key, value in sorted(option_counts.items())
        },
        "selected_root_ids_sha256": _value_sha256([
            candidate.root_id for candidate in selected
        ]),
        "selected_game_keys_sha256": _value_sha256([
            candidate.game_key for candidate in selected
        ]),
    }
    return selected, diagnostics


def _prepare_output(path: Path) -> None:
    resolved = path.resolve()
    protected = tuple(
        (ROOT / name).resolve() for name in ("agent", "data", "decks"))
    if any(resolved == tree or tree in resolved.parents for tree in protected):
        raise SelectionError(
            "refusing to write reliability data in a production tree")
    if resolved.exists():
        if not resolved.is_dir():
            raise SelectionError(f"output exists and is not a directory: {path}")
        if any(resolved.iterdir()):
            raise SelectionError(f"output directory is not empty: {path}")
    else:
        resolved.mkdir(parents=True, mode=0o755)


def write_selection_artifacts(
    output: Path,
    parent_dir: Path,
    parent_manifest: Mapping[str, Any],
    selected: Sequence[Candidate],
    diagnostics: Mapping[str, Any],
    *,
    exclusions: Sequence[Mapping[str, Any]] = (),
    generalization: bool = False,
    mechanical_preflight: Mapping[str, Any] | None = None,
    replication_candidate_pool: bool = False,
    replication_candidate_pool_v2: bool = False,
    canary_override_pool: bool = False,
    replication_finalization: Mapping[str, Any] | None = None,
    expected_root_count: int | None = None,
) -> dict[str, Any]:
    """Write a privilege-separated derived root corpus and return its manifest."""
    if sum((
        bool(generalization),
        bool(replication_candidate_pool),
        bool(replication_candidate_pool_v2),
        bool(canary_override_pool),
    )) > 1:
        raise SelectionError(
            "selection route flags are mutually exclusive")
    if expected_root_count is None:
        expected_root_count = (
            REPLICATION_CANDIDATE_ROOT_COUNT
            if (
                replication_candidate_pool
                or replication_candidate_pool_v2
            )
            else CANARY_OVERRIDE_ROOT_COUNT
            if canary_override_pool
            else ROOT_COUNT
        )
    if expected_root_count < 1:
        raise SelectionError("expected root count must be positive")
    if len(selected) != expected_root_count:
        raise SelectionError("refusing to write a partial reliability selection")
    _prepare_output(output)
    public = [candidate.public for candidate in selected]
    privileged = [candidate.privileged for candidate in selected]
    public_bytes = _jsonl(public)
    privileged_bytes = _jsonl(privileged)
    public_path = output / "public-roots.jsonl"
    privileged_path = output / "privileged-roots.jsonl"
    _atomic_bytes(public_path, public_bytes, 0o644)
    _atomic_bytes(privileged_path, privileged_bytes, 0o600)

    parent_artifacts = parent_manifest.get("artifacts")
    parent_mining = parent_manifest.get("mining")
    if (
        not isinstance(parent_artifacts, Mapping)
        or not isinstance(parent_mining, Mapping)
        or not isinstance(parent_mining.get("inputs"), list)
    ):
        raise SelectionError("parent provenance cannot be inherited")
    inherited_artifacts = {
        label: {
            key: record.get(key)
            for key in ("sha256", "records", "mode", "public_only")
        }
        for label, record in parent_artifacts.items()
        if isinstance(record, Mapping)
    }
    parent_source_hashes = parent_manifest.get("source_files_sha256")
    if not isinstance(parent_source_hashes, Mapping):
        raise SelectionError("parent source provenance cannot be inherited")
    source_hashes = copy.deepcopy(dict(parent_source_hashes))
    source_hashes.update({
        "reliability_selector": _sha256_file(Path(__file__).resolve()),
        "root_miner": _sha256_file(Path(MINE.__file__).resolve()),
        "root_validator": _sha256_file(Path(VALIDATE.__file__).resolve()),
    })
    if canary_override_pool:
        selection_mode = CANARY_OVERRIDE_SELECTION_MODE
        selection_policy = CANARY_OVERRIDE_SELECTION_POLICY
        derivation_schema = CANARY_OVERRIDE_SCHEMA
        selection_seed = CANARY_OVERRIDE_SELECTION_SEED
    elif replication_candidate_pool_v2:
        selection_mode = REPLICATION_CANDIDATE_V2_SELECTION_MODE
        selection_policy = REPLICATION_CANDIDATE_V2_SELECTION_POLICY
        derivation_schema = REPLICATION_CANDIDATE_V2_SCHEMA
        selection_seed = REPLICATION_CANDIDATE_SELECTION_SEED
    elif replication_candidate_pool:
        selection_mode = REPLICATION_CANDIDATE_SELECTION_MODE
        selection_policy = REPLICATION_CANDIDATE_SELECTION_POLICY
        derivation_schema = REPLICATION_CANDIDATE_SCHEMA
        selection_seed = REPLICATION_CANDIDATE_SELECTION_SEED
    elif generalization and replication_finalization is not None:
        selection_mode = REPLICATION_FINAL_SELECTION_MODE
        selection_policy = REPLICATION_FINAL_SELECTION_POLICY
        derivation_schema = REPLICATION_FINAL_SCHEMA
        selection_seed = REPLICATION_CANDIDATE_SELECTION_SEED
    elif generalization:
        selection_mode = GENERALIZATION_SELECTION_MODE
        selection_policy = GENERALIZATION_SELECTION_POLICY
        derivation_schema = GENERALIZATION_SCHEMA
        selection_seed = GENERALIZATION_SELECTION_SEED
    else:
        selection_mode = SELECTION_MODE
        selection_policy = SELECTION_POLICY
        derivation_schema = SCHEMA
        selection_seed = SELECTION_SEED
    manifest: dict[str, Any] = {
        # Keep the root-corpus envelope compatible with the strict shared
        # loader.  ``derivation.schema`` identifies this derived experiment.
        "schema": MINE.SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "development_only": True,
        "sealed_test": False,
        "strength_question_answered": False,
        "teacher_actor_authorization": False,
        "contains_privileged_exact_hidden_state": True,
        "privileged_artifact_must_never_enter_actor_training": True,
        "selection_mode": selection_mode,
        "selection_policy": selection_policy,
        "semantic_identity": parent_manifest.get("semantic_identity"),
        "engine_rng_seedable": parent_manifest.get("engine_rng_seedable"),
        "native_branch_validation": (
            "pending; exact panel generation revalidates every native root"
        ),
        "weights": copy.deepcopy(parent_manifest.get("weights")),
        "registered_learner_deck": copy.deepcopy(
            parent_manifest.get("registered_learner_deck")),
        "artifacts": {
            "public_roots": {
                "path": public_path.name,
                "sha256": _sha256_bytes(public_bytes),
                "records": len(public),
                "mode": "0644",
                "public_only": True,
            },
            "privileged_roots": {
                "path": privileged_path.name,
                "sha256": _sha256_bytes(privileged_bytes),
                "records": len(privileged),
                "mode": "0600",
                "public_only": False,
            },
        },
        # The panel labeler resolves source registrations through this exact
        # inherited input table.  It remains bound by the parent manifest hash.
        "mining": {
            "inputs": copy.deepcopy(parent_mining["inputs"]),
            "derived_from_parent": True,
        },
        "parent": {
            "root_dir": str(parent_dir.resolve()),
            "manifest_sha256": parent_manifest.get("manifest_sha256"),
            "artifacts": inherited_artifacts,
            "selection_mode": parent_manifest.get("selection_mode"),
            "selection_policy": parent_manifest.get("selection_policy"),
        },
        "derivation": {
            "schema": derivation_schema,
            "selection_seed": selection_seed,
            "selection_policy": selection_policy,
            "root_count": expected_root_count,
            "unique_game_requirement": expected_root_count,
            "minimum_stable_qu_v2b_margin": MIN_STABLE_REFLEX_MARGIN,
            "exact_marginal_quotas": (
                not generalization
                and not replication_candidate_pool
                and not replication_candidate_pool_v2
                and not canary_override_pool),
            "stratum_targets": (
                None
                if (
                    generalization
                    or replication_candidate_pool
                    or replication_candidate_pool_v2
                    or canary_override_pool
                )
                else {
                    f"{outcome}/seat-{seat}/"
                    f"{'disagree' if disagreement else 'agree'}": target
                    for (outcome, seat, disagreement), target
                    in STRATUM_TARGETS.items()
                }
            ),
            "diagnostics": copy.deepcopy(dict(diagnostics)),
            "exclusions": copy.deepcopy(list(exclusions)),
            "mechanical_preflight": (
                None if mechanical_preflight is None
                else copy.deepcopy(dict(mechanical_preflight))
            ),
            "replication_finalization": (
                None if replication_finalization is None
                else copy.deepcopy(dict(replication_finalization))
            ),
        },
        "source_files_sha256": source_hashes,
    }
    manifest["manifest_sha256"] = _value_sha256(manifest)
    _atomic_bytes(
        output / "manifest.json",
        json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
        0o644,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--exclude-root-dir",
        action="append",
        default=[],
        help=(
            "previous derived reliability root directory whose source games "
            "must be excluded; repeat for multiple locked cohorts"
        ),
    )
    parser.add_argument(
        "--heldout-generalization",
        action="store_true",
        help=(
            "select a disjoint diversity-balanced held-out cohort without "
            "claiming the development selector's exact marginal quotas"
        ),
    )
    return parser


def load_excluded_games(
    paths: Sequence[str | Path],
    *,
    factual_parent_manifest_sha256: str,
    factual_parent_weights: Mapping[str, Any],
    available_game_keys: frozenset[str],
    allow_overlap: bool = False,
) -> tuple[frozenset[str], list[dict[str, Any]]]:
    excluded: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        manifest, public, privileged = VALIDATE.load_root_artifacts(path)
        parent = manifest.get("parent")
        valid_selection_route = (
            (
                manifest.get("selection_mode") == SELECTION_MODE
                and manifest.get("selection_policy") == SELECTION_POLICY
            )
            or (
                manifest.get("selection_mode")
                == GENERALIZATION_SELECTION_MODE
                and manifest.get("selection_policy")
                == GENERALIZATION_SELECTION_POLICY
            )
            or (
                manifest.get("selection_mode")
                == REPLICATION_CANDIDATE_SELECTION_MODE
                and manifest.get("selection_policy")
                == REPLICATION_CANDIDATE_SELECTION_POLICY
            )
            or (
                manifest.get("selection_mode")
                == REPLICATION_CANDIDATE_V2_SELECTION_MODE
                and manifest.get("selection_policy")
                == REPLICATION_CANDIDATE_V2_SELECTION_POLICY
            )
            or (
                manifest.get("selection_mode")
                == REPLICATION_FINAL_SELECTION_MODE
                and manifest.get("selection_policy")
                == REPLICATION_FINAL_SELECTION_POLICY
            )
        )
        if (
            not valid_selection_route
            or not isinstance(parent, Mapping)
            or not _is_sha256(parent.get("manifest_sha256"))
            or manifest.get("weights") != factual_parent_weights
        ):
            raise SelectionError(
                f"exclusion is not a sibling reliability cohort: {path}")
        before = len(excluded)
        for public_record, privileged_record in zip(public, privileged):
            candidate = _candidate(public_record, privileged_record)
            if candidate is None:
                raise SelectionError(
                    f"exclusion contains an unstable root: {path}")
            if candidate.game_key not in available_game_keys:
                raise SelectionError(
                    "excluded game is absent from the refreshed factual "
                    f"parent: {path}")
            excluded.add(candidate.game_key)
        if not allow_overlap and len(excluded) - before != len(public):
            raise SelectionError(
                f"exclusion cohorts overlap or duplicate games: {path}")
        provenance.append({
            "root_dir": str(path),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "factual_parent_manifest_sha256":
                parent.get("manifest_sha256"),
            "current_factual_parent_manifest_sha256":
                factual_parent_manifest_sha256,
            "matched_by": (
                "exact_parent_manifest"
                if parent.get("manifest_sha256")
                == factual_parent_manifest_sha256
                else "append-stable game identity and frozen weights"
            ),
            "games": len(public),
            "new_unique_games": len(excluded) - before,
            "game_keys_sha256": _value_sha256(sorted(
                _candidate(public_record, privileged_record).game_key
                for public_record, privileged_record
                in zip(public, privileged)
            )),
        })
    return frozenset(excluded), provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    parent_dir = Path(args.root_dir).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    try:
        manifest, public, privileged = VALIDATE.load_root_artifacts(parent_dir)
        _validate_parent_manifest(manifest)
        current_candidates = [
            candidate
            for public_record, privileged_record in zip(public, privileged)
            if (
                candidate := _candidate(
                    public_record, privileged_record)
            ) is not None
        ]
        excluded, exclusion_provenance = load_excluded_games(
            args.exclude_root_dir,
            factual_parent_manifest_sha256=manifest["manifest_sha256"],
            factual_parent_weights=manifest["weights"],
            available_game_keys=frozenset(
                candidate.game_key for candidate in current_candidates),
        )
        select = (
            select_generalization_roots
            if args.heldout_generalization else select_roots
        )
        selected, diagnostics = select(
            public, privileged, excluded_game_keys=excluded)
        derived = write_selection_artifacts(
            output,
            parent_dir,
            manifest,
            selected,
            diagnostics,
            exclusions=exclusion_provenance,
            generalization=args.heldout_generalization,
        )
    except (OSError, ValueError, VALIDATE.ValidationError, SelectionError) as exc:
        parser.error(str(exc))
    print(
        "Qu-v2C "
        f"{'held-out generalization' if args.heldout_generalization else 'label-reliability'} "
        "roots: "
        f"{diagnostics['selected_roots']} roots from "
        f"{diagnostics['selected_unique_games']} games; "
        f"{len(diagnostics['opponent_archetypes'])} archetypes",
        flush=True,
    )
    print(
        f"Manifest: {output / 'manifest.json'} "
        f"({derived['manifest_sha256']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
