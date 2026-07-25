"""Mine provenance-locked Qu-v2C critic roots from Qu-v2B ladder replays.

Kaggle replay records retain two different views of a game:

* ``steps[t][seat].observation`` is the legitimate public actor view; and
* ``steps[0][0].visualize`` is the privileged visualization trajectory.

This tool aligns those views with the strict exact-hidden projection checks
from :mod:`tools.counterfactual_oracle`.  It writes two physically separate
JSONL artifacts:

``public-roots.jsonl``
    Public observations, semantic actions, and frozen-policy diagnostics.  This
    file is safe input to a future public actor/teacher.

``privileged-roots.jsonl``
    Exact hidden zones and native ``SearchBegin`` state.  This file is
    research-only input to the asymmetric critic/oracle and is mode ``0600``.

The first pre-registered shard is deliberately narrow: resolved Qu-v2B losses,
supported one-pick MAIN roots, and an occurrence-aware semantic disagreement
between frozen Qu-v2B and its frozen Qu-v2A predecessor.  A losing game is a
query distribution, not a hard action label; no logged action is promoted as
correct by this tool.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for directory in (str(ROOT), str(TOOLS)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from agent import model, policy  # noqa: E402
from agent import qu_v2_features as QF  # noqa: E402
from agent import turn_search as TS  # noqa: E402
from agent.obsview import ST_MAIN  # noqa: E402
from tools import analyze_ladder_replays as ALR  # noqa: E402
from tools import cabt as CABT  # noqa: E402
from tools import counterfactual_oracle as CFO  # noqa: E402
from tools import il_dataset  # noqa: E402
from tools.research import qu_v2c_privileged_features as PF  # noqa: E402


SCHEMA = "ptcg.qu-v2c.root-mining.v1"
PUBLIC_SCHEMA = "ptcg.qu-v2c.public-roots.v1"
PRIVILEGED_SCHEMA = "ptcg.qu-v2c.privileged-roots.v1"
SEMANTIC_IDENTITY = "turn_search.semantic_action.occurrence-aware.v1"
CRITICAL_SELECTION_POLICY = (
    "loss + frozen-B/frozen-parent occurrence-aware semantic disagreement + "
    "ST_MAIN + min=max=1 + 2..12 options"
)
FACTUAL_CRITIC_SELECTION_POLICY = (
    "all resolved terminal games + ST_MAIN + min=max=1 + 2..12 options; "
    "logged action receives factual terminal return only"
)
SELECTION_POLICIES = {
    "critical": CRITICAL_SELECTION_POLICY,
    "factual-critic": FACTUAL_CRITIC_SELECTION_POLICY,
}

FROZEN_QU_V2B_SHA256 = (
    "ec69a2db7660c38e6623e9711ed910718c650780a020369148836d40119a8447"
)
FROZEN_QU_V2A_PARENT_SHA256 = (
    "fe1e12fd912d1678ddc51fd07700588958b169d4bf435c568942e243b003187a"
)
DEFAULT_QU_V2B = ROOT / "agent" / "weights.npz"
DEFAULT_PARENT = (
    ROOT / "tools" / "checkpoints" / "qu-v2a-field-v1"
    / "candidate-qu-v2a-weights.npz"
)
DEFAULT_REPLAY_DIRS = (
    ROOT / "tools" / "checkpoints" / "qu-v2b-ladder" / "original",
    ROOT / "tools" / "checkpoints" / "qu-v2b-ladder" / "clone",
)
DEFAULT_OUT = (
    ROOT / "tools" / "checkpoints" / "qu-v2c-roots-v1" / "critical-ladder"
)
PROTECTED_OUTPUT_TREES = tuple(
    (ROOT / name).resolve() for name in ("agent", "data", "decks")
)
FORBIDDEN_PUBLIC_KEYS = frozenset({
    CFO.EXACT_HIDDEN_KEY,
    "search_begin_input",
    "visualize",
})


class MiningError(RuntimeError):
    """A replay, artifact, or output violated the root-mining contract."""


@dataclass(frozen=True)
class ActionRow:
    source_step: int
    answer_step: int
    observation: dict[str, Any]
    logged_action: tuple[int, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _atomic_bytes(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, mode)
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


def _jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(record) + b"\n" for record in records)


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MiningError(f"invalid replay JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MiningError(f"replay root is not an object: {path}")
    return value, _sha256_bytes(raw)


def _load_visual_trajectory(replay: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps = replay.get("steps")
    if (not isinstance(steps, list) or not steps
            or not isinstance(steps[0], list) or not steps[0]
            or not isinstance(steps[0][0], Mapping)):
        raise MiningError("replay has no step-zero visualization owner")
    visual: Any = steps[0][0].get("visualize")
    if isinstance(visual, str):
        try:
            visual = json.loads(visual)
        except json.JSONDecodeError as exc:
            raise MiningError("visualization string is invalid JSON") from exc
    if (not isinstance(visual, list)
            or not all(isinstance(row, dict) for row in visual)):
        raise MiningError("replay visualization is not an object trajectory")
    return visual


def _valid_reward(replay: Mapping[str, Any], seat: int) -> float:
    rewards = replay.get("rewards")
    if (not isinstance(rewards, list) or len(rewards) != 2
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) not in (-1.0, 0.0, 1.0)
                for value in rewards
            )
            or float(rewards[0]) != -float(rewards[1])):
        raise MiningError("replay has invalid terminal rewards")
    return float(rewards[seat])


def _optional_valid_reward(
    replay: Mapping[str, Any],
    seat: int,
) -> float | None:
    """Return a usable terminal reward, or ``None`` for incomplete episodes.

    Kaggle can retain a replay after one agent times out while leaving that
    seat's reward as ``null``.  Such a game is useful for ladder diagnostics
    but cannot supply a factual terminal-return target.  Keep the strict
    validator above and make the corpus-level exclusion explicit here.
    """
    try:
        return _valid_reward(replay, seat)
    except MiningError as exc:
        if str(exc) != "replay has invalid terminal rewards":
            raise
        return None


def action_rows(replay: Mapping[str, Any], seat: int) -> Iterable[ActionRow]:
    """Yield the exact Kaggle ``observation[t] -> action[t+1]`` pairing."""
    steps = replay.get("steps")
    if not isinstance(steps, list):
        return
    for answer_step in range(1, len(steps)):
        source_step = answer_step - 1
        if (not isinstance(steps[answer_step], list)
                or not isinstance(steps[source_step], list)
                or seat >= len(steps[answer_step])
                or seat >= len(steps[source_step])):
            continue
        answer = steps[answer_step][seat]
        source = steps[source_step][seat]
        if not isinstance(answer, Mapping) or not isinstance(source, Mapping):
            continue
        if source.get("status") == "INACTIVE":
            continue
        action = answer.get("action")
        if not isinstance(action, list) or len(action) == 60:
            continue
        obs = source.get("observation")
        if not isinstance(obs, dict):
            continue
        select = obs.get("select")
        current = obs.get("current")
        if (not isinstance(select, Mapping)
                or not isinstance(select.get("option"), list)
                or not isinstance(current, Mapping)
                or current.get("yourIndex") != seat):
            continue
        options = select["option"]
        minimum = select.get("minCount", 1)
        maximum = select.get("maxCount", 1)
        if (not isinstance(minimum, int) or isinstance(minimum, bool)
                or not isinstance(maximum, int) or isinstance(maximum, bool)
                or minimum < 0 or maximum < 0
                or (maximum > 0 and minimum > maximum)):
            continue
        if (any(
                not isinstance(index, int) or isinstance(index, bool)
                or not 0 <= index < len(options)
                for index in action
            )
                or len(set(action)) != len(action)
                or len(action) < min(minimum, len(options))
                or (maximum > 0 and len(action) > min(maximum, len(options)))):
            continue
        yield ActionRow(
            source_step=source_step,
            answer_step=answer_step,
            observation=obs,
            logged_action=tuple(action),
        )


def _visual_coordinate(value: Mapping[str, Any]) -> tuple[Any, Any, Any] | None:
    current = value.get("current")
    if not isinstance(current, Mapping):
        return None
    return (
        current.get("turn"),
        current.get("turnActionCount"),
        current.get("yourIndex"),
    )


def _observation_coordinate(obs: Mapping[str, Any]) -> tuple[Any, Any, Any] | None:
    current = obs.get("current")
    if not isinstance(current, Mapping):
        return None
    return (
        current.get("turn"),
        current.get("turnActionCount"),
        current.get("yourIndex"),
    )


def align_exact_payload(
    obs: Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
) -> tuple[int, dict[str, Any]]:
    """Return the unique visualization row passing strict public projection."""
    coordinate = _observation_coordinate(obs)
    matches: list[tuple[int, dict[str, Any]]] = []
    for index, visual in enumerate(trajectory):
        if _visual_coordinate(visual) != coordinate:
            continue
        try:
            payload = CFO.exact_hidden_payload(obs, visual)
        except ValueError:
            continue
        matches.append((index, payload))
    if len(matches) != 1:
        raise MiningError(
            f"expected one exact visualization alignment, found {len(matches)}"
        )
    return matches[0]


def _has_public_face_up_prize(obs: Mapping[str, Any]) -> bool:
    current = obs.get("current")
    players = current.get("players") if isinstance(current, Mapping) else None
    if not isinstance(players, list):
        return True
    for player in players:
        prize = player.get("prize") if isinstance(player, Mapping) else None
        if isinstance(prize, list) and any(card is not None for card in prize):
            return True
    return False


def _is_supported_root(obs: Mapping[str, Any]) -> bool:
    select = obs.get("select")
    if not isinstance(select, Mapping):
        return False
    options = select.get("option")
    return (
        select.get("type") == ST_MAIN
        and select.get("minCount", 1) == 1
        and select.get("maxCount", 1) == 1
        and isinstance(options, list)
        and 2 <= len(options) <= 12
        and not _has_public_face_up_prize(obs)
    )


def _policy_output(net: Any, obs: dict[str, Any],
                   learner_deck: Sequence[int]) -> dict[str, Any]:
    sample = QF.encode_public_observation(obs, learner_deck)
    QF.validate_public_features(sample)
    logits, value = net.forward(sample)
    logits = np.asarray(logits, dtype=np.float32)
    select = obs["select"]
    count = len(select["option"])
    if logits.shape != (count + 1,) or not np.isfinite(logits).all():
        raise MiningError("frozen Qu-v2 policy returned invalid logits")
    action = model.decode_qu_v2(
        logits, count, select.get("minCount", 1), select.get("maxCount", 1),
    )
    if len(action) != 1:
        raise MiningError("supported one-pick root decoded to a non-singleton")
    real = logits[:count].astype(np.float64)
    shifted = real - float(real.max())
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    ordered = np.sort(real)
    margin = float(ordered[-1] - ordered[-2])
    return {
        "action": action,
        "semantic_action": TS.semantic_action(obs, action),
        "logits": [float(value) for value in logits],
        "real_action_probabilities": [
            float(value) for value in probabilities
        ],
        "margin": margin,
        "value": float(value),
        "feature_fingerprint": _sha256_bytes(QF.feature_fingerprint(sample)),
    }


def _find_forbidden_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_PUBLIC_KEYS:
                return f"{path}.{key}"
            found = _find_forbidden_key(item, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_forbidden_key(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def sanitize_public_observation(
    obs: Mapping[str, Any], learner_deck: Sequence[int],
) -> dict[str, Any]:
    """Drop transport-only fields and prove deployable features are unchanged."""
    public = copy.deepcopy(dict(obs))
    public.pop("logs", None)
    public.pop("search_begin_input", None)
    public.pop(CFO.EXACT_HIDDEN_KEY, None)
    forbidden = _find_forbidden_key(public)
    if forbidden is not None:
        raise MiningError(f"public observation retains forbidden field {forbidden}")
    before = QF.encode_public_observation(dict(obs), learner_deck)
    after = QF.encode_public_observation(public, learner_deck)
    if QF.feature_fingerprint(before) != QF.feature_fingerprint(after):
        raise MiningError("public sanitization changed deployable Qu-v2 features")
    return public


def _semantic_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_semantic_json(item) for item in value]
    if isinstance(value, list):
        return [_semantic_json(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_json(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise MiningError(f"semantic token contains unsupported value {type(value)}")


def _resolved_learner_seat(
    replay: Mapping[str, Any],
    learner_multiset: tuple[int, ...],
    aliases: set[str],
) -> tuple[int | None, str, dict[int, list[int]]]:
    decks = il_dataset.decks_from_document(dict(replay))
    matches = [
        seat for seat, deck in decks.items()
        if tuple(sorted(deck)) == learner_multiset
    ]
    if len(matches) == 1:
        return matches[0], "deck", decks
    if len(matches) > 1:
        teams = (replay.get("info") or {}).get("TeamNames") or []
        named = [
            seat for seat in matches
            if seat < len(teams) and teams[seat] in aliases
        ]
        if len(named) == 1:
            return named[0], "deck+team", decks
        return None, "ambiguous", decks
    return None, "deck_missing", decks


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            text=True, capture_output=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _source_hashes() -> dict[str, str | None]:
    paths = {
        "miner": Path(__file__).resolve(),
        "production_model": Path(model.__file__).resolve(),
        "production_public_features": Path(QF.__file__).resolve(),
        "semantic_mapping": Path(TS.__file__).resolve(),
        "counterfactual_alignment": Path(CFO.__file__).resolve(),
        "ladder_analysis": Path(ALR.__file__).resolve(),
        "episode_loader": Path(il_dataset.__file__).resolve(),
        "privileged_features": Path(PF.__file__).resolve(),
        "engine_wrapper": Path(CABT.__file__).resolve(),
        "engine_library": Path(CABT._LIB_PATH).resolve(),
        "cards_data": ROOT / "data" / "cards.json",
        "attacks_data": ROOT / "data" / "attacks.json",
        "registered_deck": ROOT / "decks" / "deck.csv",
    }
    result: dict[str, str | None] = {}
    for label, path in paths.items():
        try:
            result[label] = _sha256_file(path)
        except OSError:
            result[label] = None
    return result


def _episode_id(path: Path, replay: Mapping[str, Any]) -> str:
    if path.stem.isdigit():
        return path.stem
    value = replay.get("id")
    return str(value) if value is not None else path.stem


def _opponent_name(replay: Mapping[str, Any], learner_seat: int) -> str | None:
    info = replay.get("info")
    teams = info.get("TeamNames") if isinstance(info, Mapping) else None
    other = 1 - learner_seat
    if isinstance(teams, list) and other < len(teams):
        value = teams[other]
        return str(value) if value is not None else None
    return None


def mine(
    replay_paths: Sequence[Path],
    learner_deck: Sequence[int],
    qu_v2b: Any,
    parent: Any,
    aliases: set[str],
    selection_mode: str = "critical",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if selection_mode not in SELECTION_POLICIES:
        raise MiningError(f"unsupported selection mode {selection_mode!r}")
    selection_policy = SELECTION_POLICIES[selection_mode]
    public_records: list[dict[str, Any]] = []
    privileged_records: list[dict[str, Any]] = []
    learner_multiset = tuple(sorted(learner_deck))
    counters = {
        "replay_files": len(replay_paths),
        "resolved_games": 0,
        "unresolved_games": 0,
        "resolved_losses": 0,
        "resolved_wins": 0,
        "resolved_draws": 0,
        "action_rows": 0,
        "loss_rows": 0,
        "win_rows": 0,
        "draw_rows": 0,
        "supported_roots": 0,
        "supported_loss_roots": 0,
        "supported_win_roots": 0,
        "supported_draw_roots": 0,
        "semantic_disagreements": 0,
        "loss_semantic_disagreements": 0,
        "critical_candidates": 0,
        "aligned_roots": 0,
        "alignment_failures": 0,
        "duplicate_roots": 0,
    }
    unresolved_reasons: dict[str, int] = {}
    replay_inputs: list[dict[str, Any]] = []
    seen_roots: set[str] = set()

    for path in replay_paths:
        replay, replay_sha = _load_json(path)
        seat, resolution, decks = _resolved_learner_seat(
            replay, learner_multiset, aliases)
        replay_inputs.append({
            "path": str(path.resolve()),
            "sha256": replay_sha,
            "source": path.parent.name,
            "episode_id": _episode_id(path, replay),
            "learner_seat": seat,
            "seat_resolution": resolution,
        })
        if seat is None:
            counters["unresolved_games"] += 1
            unresolved_reasons[resolution] = unresolved_reasons.get(resolution, 0) + 1
            continue
        reward = _optional_valid_reward(replay, seat)
        if reward is None:
            counters["unresolved_games"] += 1
            reason = "invalid_terminal_reward"
            unresolved_reasons[reason] = unresolved_reasons.get(reason, 0) + 1
            replay_inputs[-1]["terminal_reward_valid"] = False
            continue
        replay_inputs[-1]["terminal_reward_valid"] = True
        counters["resolved_games"] += 1
        if reward < 0:
            counters["resolved_losses"] += 1
            outcome = "loss"
        elif reward > 0:
            counters["resolved_wins"] += 1
            outcome = "win"
        else:
            counters["resolved_draws"] += 1
            outcome = "draw"
        if selection_mode == "critical" and reward >= 0:
            continue
        try:
            trajectory = _load_visual_trajectory(replay)
        except MiningError:
            # Count individual selected roots as failures below only when a
            # trajectory was present; a missing trajectory invalidates every
            # candidate in this game and therefore the complete mining run.
            raise
        opponent_deck = decks.get(1 - seat)
        if not isinstance(opponent_deck, list) or len(opponent_deck) != 60:
            raise MiningError(f"resolved replay has no opponent deck: {path}")
        opponent_deck_sha = _value_sha256(opponent_deck)
        archetype = ALR.archetype(opponent_deck)
        episode_id = _episode_id(path, replay)

        for row in action_rows(replay, seat):
            counters["action_rows"] += 1
            counters[f"{outcome}_rows"] += 1
            obs = row.observation
            if not _is_supported_root(obs):
                continue
            counters["supported_roots"] += 1
            counters[f"supported_{outcome}_roots"] += 1
            b_output = _policy_output(qu_v2b, obs, learner_deck)
            parent_output = _policy_output(parent, obs, learner_deck)
            disagrees = (
                b_output["semantic_action"] != parent_output["semantic_action"]
            )
            if disagrees:
                counters["semantic_disagreements"] += 1
                if reward < 0:
                    counters["loss_semantic_disagreements"] += 1
                    counters["critical_candidates"] += 1
            if selection_mode == "critical" and not disagrees:
                continue
            try:
                visual_index, hidden = align_exact_payload(obs, trajectory)
            except MiningError:
                counters["alignment_failures"] += 1
                continue
            CFO.validate_hidden_payload(
                {
                    **obs,
                    CFO.EXACT_HIDDEN_KEY: hidden,
                },
                hidden,
            )
            privileged_features = PF.encode_privileged_observation(
                obs, hidden, learner_deck)
            public_obs = sanitize_public_observation(obs, learner_deck)
            public_fingerprint = CFO.public_root_fingerprint(obs)
            root_material = {
                "schema": SCHEMA,
                "replay_sha256": replay_sha,
                "source_step": row.source_step,
                "learner_seat": seat,
                "public_root_fingerprint": public_fingerprint,
                "qu_v2b_sha256": FROZEN_QU_V2B_SHA256,
                "semantic_identity": SEMANTIC_IDENTITY,
            }
            root_id = _value_sha256(root_material)
            if root_id in seen_roots:
                counters["duplicate_roots"] += 1
                continue
            seen_roots.add(root_id)
            counters["aligned_roots"] += 1

            semantic_options = TS.semantic_options(obs)
            logged_semantic = TS.semantic_action(obs, row.logged_action)
            priority_tier = (
                "A" if reward < 0 and disagrees
                else "B" if reward < 0
                else "W-disagreement" if disagrees
                else "W-control" if reward > 0
                else "D-control"
            )
            public_record = {
                "schema": PUBLIC_SCHEMA,
                "root_id": root_id,
                "source": {
                    "episode_id": episode_id,
                    "source_submission": path.parent.name,
                    "source_step": row.source_step,
                    "answer_step": row.answer_step,
                    "learner_seat": seat,
                    "seat_resolution": resolution,
                    "learner_reward": reward,
                    "outcome": outcome,
                    "replay_sha256": replay_sha,
                    "opponent_name": _opponent_name(replay, seat),
                    "opponent_archetype": archetype,
                    "opponent_deck_sha256": opponent_deck_sha,
                },
                "identity": {
                    "public_root_fingerprint": public_fingerprint,
                    "qu_v2_feature_fingerprint": b_output[
                        "feature_fingerprint"
                    ],
                    "semantic_identity": SEMANTIC_IDENTITY,
                    "registered_learner_deck_sha256": _value_sha256(
                        list(learner_deck)),
                },
                "selection": {
                    "mode": selection_mode,
                    "policy": selection_policy,
                    "priority_tier": priority_tier,
                    "losing_game_query_only": reward < 0,
                    "hard_action_label": False,
                    "factual_terminal_return_label": (
                        selection_mode == "factual-critic"
                    ),
                    "supported_exact_root": True,
                    "frozen_b_parent_disagreement": disagrees,
                },
                "prompt": {
                    "turn": (obs.get("current") or {}).get("turn"),
                    "turn_action_count": (
                        obs.get("current") or {}).get("turnActionCount"),
                    "selecting_seat": seat,
                    "select_type": (obs.get("select") or {}).get("type"),
                    "min_count": (obs.get("select") or {}).get("minCount", 1),
                    "max_count": (obs.get("select") or {}).get("maxCount", 1),
                    "semantic_options": _semantic_json(semantic_options),
                },
                "logged": {
                    "action": list(row.logged_action),
                    "semantic_action": _semantic_json(logged_semantic),
                },
                "qu_v2b": {
                    key: (
                        _semantic_json(value)
                        if key == "semantic_action" else value
                    )
                    for key, value in b_output.items()
                    if key != "feature_fingerprint"
                },
                "parent": {
                    key: (
                        _semantic_json(value)
                        if key == "semantic_action" else value
                    )
                    for key, value in parent_output.items()
                    if key != "feature_fingerprint"
                },
                "public_observation": public_obs,
            }
            forbidden = _find_forbidden_key(public_record)
            if forbidden is not None:
                raise MiningError(
                    f"public root {root_id} contains forbidden field {forbidden}"
                )
            privileged_record = {
                "schema": PRIVILEGED_SCHEMA,
                "root_id": root_id,
                "source": {
                    "episode_id": episode_id,
                    "source_step": row.source_step,
                    "learner_seat": seat,
                    "visual_index": visual_index,
                    "replay_sha256": replay_sha,
                },
                "binding": {
                    "public_root_fingerprint": public_fingerprint,
                    "search_begin_sha256": hidden["search_begin_sha256"],
                    "exact_hidden_payload_sha256": _value_sha256(hidden),
                    "privileged_feature_sha256": (
                        privileged_features.canonical_hash()
                    ),
                    "privileged_feature_schema": PF.SCHEMA,
                    "public_projection_pass": True,
                    "native_begin_pass": False,
                },
                "search_begin_input": obs.get("search_begin_input"),
                "exact_hidden_payload": hidden,
            }
            if not isinstance(privileged_record["search_begin_input"], str):
                raise MiningError(
                    f"critical root has no SearchBegin input: {path} "
                    f"step {row.source_step}"
                )
            public_records.append(public_record)
            privileged_records.append(privileged_record)

    counters["unresolved_reasons"] = unresolved_reasons
    counters["unique_public_roots"] = len(public_records)
    if len(public_records) != len(privileged_records):
        raise MiningError("public and privileged root counts diverged")
    if counters["alignment_failures"]:
        raise MiningError(
            f"{counters['alignment_failures']} selected critical roots failed "
            "strict exact-state alignment"
        )
    if not public_records:
        raise MiningError("mining produced no critical roots")
    summary = {
        "selection_mode": selection_mode,
        "selection_policy": selection_policy,
        "counters": counters,
        "inputs": replay_inputs,
    }
    return public_records, privileged_records, summary


def _replay_paths(raw_dirs: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_dirs:
        directory = Path(raw).expanduser().resolve()
        if not directory.is_dir():
            raise MiningError(f"replay directory does not exist: {directory}")
        paths.extend(Path(path).resolve() for path in glob.glob(
            str(directory / "*.json")))
    unique = sorted(set(paths))
    if not unique:
        raise MiningError("no replay JSON files found")
    return unique


def _prepare_output(raw: str) -> Path:
    output = Path(raw).expanduser().resolve()
    if any(_inside(output, tree) for tree in PROTECTED_OUTPUT_TREES):
        raise MiningError("refusing to write Qu-v2C research data in production")
    if output.exists():
        if not output.is_dir():
            raise MiningError(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise MiningError(f"output directory is not empty: {output}")
    else:
        output.mkdir(parents=True, mode=0o755)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-dir", action="append", dest="replay_dirs",
        help="repeatable replay directory (defaults to Qu-v2B original+clone)",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--qu-v2b", default=str(DEFAULT_QU_V2B))
    parser.add_argument("--parent", default=str(DEFAULT_PARENT))
    parser.add_argument(
        "--selection", choices=tuple(SELECTION_POLICIES), default="critical",
        help="critical ladder queries or all supported factual critic roots",
    )
    parser.add_argument(
        "--team-alias", action="append", default=["増殖するG"],
        help="repeatable learner team name used to resolve exact-deck mirrors",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        replay_paths = _replay_paths(
            args.replay_dirs
            or [str(path) for path in DEFAULT_REPLAY_DIRS]
        )
        qu_v2b_path = Path(args.qu_v2b).expanduser().resolve()
        parent_path = Path(args.parent).expanduser().resolve()
        if _sha256_file(qu_v2b_path) != FROZEN_QU_V2B_SHA256:
            raise MiningError("--qu-v2b is not the frozen ec69a2db Qu-v2B")
        if _sha256_file(parent_path) != FROZEN_QU_V2A_PARENT_SHA256:
            raise MiningError("--parent is not the frozen fe1e12fd Qu-v2A")
        qu_v2b = model.load(str(qu_v2b_path))
        parent = model.load(str(parent_path))
        if (qu_v2b is None or parent is None
                or not getattr(qu_v2b, "is_qu_v2", False)
                or not getattr(parent, "is_qu_v2", False)):
            raise MiningError("could not load both frozen Qu-v2 policies")
        learner_deck = policy.load_deck()
        if len(learner_deck) != 60:
            raise MiningError("registered learner deck is not 60 cards")
        output = _prepare_output(args.out_dir)
        public, privileged, mining = mine(
            replay_paths, learner_deck, qu_v2b, parent,
            set(args.team_alias), args.selection,
        )
        public_path = output / "public-roots.jsonl"
        privileged_path = output / "privileged-roots.jsonl"
        public_bytes = _jsonl(public)
        privileged_bytes = _jsonl(privileged)
        _atomic_bytes(public_path, public_bytes, 0o644)
        _atomic_bytes(privileged_path, privileged_bytes, 0o600)
        manifest = {
            "schema": SCHEMA,
            "created_at": _utc_now(),
            "research_only": True,
            "contains_privileged_exact_hidden_state": True,
            "privileged_artifact_must_never_enter_actor_training": True,
            "selection_mode": args.selection,
            "selection_policy": SELECTION_POLICIES[args.selection],
            "semantic_identity": SEMANTIC_IDENTITY,
            "engine_rng_seedable": False,
            "native_branch_validation": (
                "pending; mining proves public/visual alignment only"
            ),
            "weights": {
                "qu_v2b": {
                    "path": str(qu_v2b_path),
                    "sha256": FROZEN_QU_V2B_SHA256,
                },
                "parent": {
                    "path": str(parent_path),
                    "sha256": FROZEN_QU_V2A_PARENT_SHA256,
                },
            },
            "registered_learner_deck": {
                "sha256": _value_sha256(learner_deck),
                "cards": learner_deck,
            },
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
            "mining": mining,
            "source_files_sha256": _source_hashes(),
            "git": _git_state(),
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
    except (OSError, ValueError, MiningError) as exc:
        parser.error(str(exc))

    counters = mining["counters"]
    print(
        f"Qu-v2C {args.selection} roots: "
        f"{len(public)} aligned from {counters['resolved_games']} games; "
        f"{counters['supported_roots']} supported roots; "
        f"{counters['semantic_disagreements']} B/parent disagreements",
        flush=True,
    )
    print(f"Artifacts: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
