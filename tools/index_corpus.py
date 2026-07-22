"""Build a deterministic, content-locked index of replay corpora.

This is research plumbing only.  It does not train a policy, alter generated
card/deck data, or write deployable weights.  Each input is a flat replay
directory specified as ``[LABEL=]DIRECTORY``.  Numeric Kaggle replay names and
``episode-<id>-replay.json`` names are recognized; downloader control files
such as ``.done_subs.json`` are ignored.

The index deliberately retains malformed records and identity conflicts with
explicit reasons.  Paths that share either an EpisodeId or byte-identical
content are one game group, so a regular file, symlink, hardlink, or copied
alias can never cross train/validation/test partitions.

Example::

    python tools/index_corpus.py \
        legacy=~/Desktop/ptcg_episodes \
        top=~/Desktop/ptcg_corpus_top \
        mid=~/Desktop/ptcg_corpus_mid \
        --json-out tools/checkpoints/corpus-index/corpus-v2.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

try:  # Running as ``python tools/index_corpus.py``.
    import il_dataset
except ImportError:  # Importing as ``tools.index_corpus`` from tests/tools.
    from tools import il_dataset  # type: ignore


SCHEMA = "ptcg-corpus-index-v2"
DEFAULT_SPLITS = (("train", 0.8), ("validation", 0.1), ("test", 0.1))
_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")
_NUMERIC_REPLAY = re.compile(r"^(?P<id>[0-9]+)\.json$")
_KAGGLE_REPLAY = re.compile(r"^episode-(?P<id>[0-9]+)-replay\.json$")
_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceSpec:
    label: str
    root: Path


class _DisjointSet:
    def __init__(self, count: int):
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deck_sha256(deck: Sequence[int]) -> str:
    canonical = ",".join(map(str, sorted(int(card) for card in deck)))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _safe_episode_id(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)


def episode_id_from_name(name: str) -> int | None:
    for pattern in (_NUMERIC_REPLAY, _KAGGLE_REPLAY):
        match = pattern.match(name)
        if match:
            return int(match.group("id"))
    return None


def parse_source_specs(values: Sequence[str]) -> tuple[SourceSpec, ...]:
    if not values:
        raise ValueError("at least one replay directory is required")
    sources: list[SourceSpec] = []
    labels: set[str] = set()
    roots: set[str] = set()
    for raw in values:
        label: str | None = None
        path_text = raw
        if "=" in raw:
            possible_label, possible_path = raw.split("=", 1)
            if possible_label and _LABEL.fullmatch(possible_label):
                label, path_text = possible_label, possible_path
        root = Path(path_text).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"replay source is not a directory: {root}")
        label = label or root.name
        if not label or not _LABEL.fullmatch(label):
            raise ValueError(f"invalid source label {label!r}")
        if label in labels:
            raise ValueError(f"duplicate source label {label!r}")
        root_key = os.path.normcase(str(root))
        if root_key in roots:
            raise ValueError(f"duplicate replay source root: {root}")
        labels.add(label)
        roots.add(root_key)
        sources.append(SourceSpec(label, root))
    return tuple(sorted(sources, key=lambda source: (source.label, str(source.root))))


def parse_split_spec(value: str | None) -> tuple[tuple[str, float], ...]:
    if value is None:
        return DEFAULT_SPLITS
    result: list[tuple[str, float]] = []
    labels: set[str] = set()
    for item in value.split(","):
        label, separator, raw_fraction = item.strip().partition("=")
        if not separator or not _LABEL.fullmatch(label):
            raise ValueError(f"invalid split entry {item!r}; expected LABEL=FRACTION")
        if label in labels:
            raise ValueError(f"duplicate split label {label!r}")
        try:
            fraction = float(raw_fraction)
        except ValueError as error:
            raise ValueError(f"invalid split fraction {raw_fraction!r}") from error
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
            raise ValueError("split fractions must be finite and strictly between 0 and 1")
        result.append((label, fraction))
        labels.add(label)
    if len(result) < 2:
        raise ValueError("at least two data splits are required")
    if not math.isclose(sum(fraction for _, fraction in result), 1.0,
                        rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("split fractions must sum to 1")
    return tuple(result)


def split_bucket_sha256(group_key: str, seed: int) -> str:
    material = f"{SCHEMA}\0split-bucket\0{int(seed)}\0{group_key}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def split_order_key(group_key: str, seed: int) -> str:
    """Compatibility alias for callers that only need the seeded hash."""
    return split_bucket_sha256(group_key, seed)


def assign_group_split(
        group_key: str, seed: int,
        splits: Sequence[tuple[str, float]] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    """Assign one game through an append-stable seeded hash bucket."""
    checked = parse_split_spec(",".join(f"{name}={fraction:.17g}"
                                        for name, fraction in splits))
    digest = split_bucket_sha256(group_key, seed)
    bucket = int(digest[:16], 16)
    bucket_space = 1 << 64
    cumulative = 0.0
    selected = checked[-1][0]
    for label, fraction in checked[:-1]:
        cumulative += fraction
        if bucket < math.floor(cumulative * bucket_space):
            selected = label
            break
    return {
        "split": selected,
        "split_bucket_sha256": digest,
        # Hex avoids accidental precision loss in non-Python JSON consumers.
        "split_bucket_u64_hex": f"{bucket:016x}",
        # Retain the consumer-facing order primitive, but unlike v1 this is a
        # per-game hash value—not a corpus-relative enumeration that changes
        # when unrelated games are appended.
        "split_rank": bucket,
    }


def assign_group_splits(
        group_keys: Iterable[str], seed: int,
        splits: Sequence[tuple[str, float]] = DEFAULT_SPLITS,
) -> dict[str, dict[str, Any]]:
    """Assign deterministic, append-stable splits to unique game groups.

    Each assignment depends only on ``(schema, seed, game_uid, fractions)``.
    Adding, removing, or reordering unrelated games therefore cannot move an
    existing game between train, validation, and test.
    """
    keys = list(group_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("group keys must be unique")
    return {
        key: assign_group_split(key, seed, splits)
        for key in sorted(keys)
    }


def _agent_name(info: Mapping[str, Any], seat: int) -> str | None:
    agents = info.get("Agents")
    if not isinstance(agents, list) or seat >= len(agents):
        return None
    agent = agents[seat]
    if isinstance(agent, Mapping):
        name = agent.get("Name")
        return name if isinstance(name, str) else None
    return agent if isinstance(agent, str) else None


def _team_name(info: Mapping[str, Any], seat: int) -> str | None:
    teams = info.get("TeamNames")
    if not isinstance(teams, list) or seat >= len(teams):
        return None
    return teams[seat] if isinstance(teams[seat], str) else None


def _normalized_deck(value: Any) -> list[int] | None:
    if (not isinstance(value, list) or len(value) != 60
            or any(isinstance(card, bool) or not isinstance(card, int) or card <= 0
                   for card in value)):
        return None
    return sorted(int(card) for card in value)


def _valid_rewards(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(reward, (int, float)) and not isinstance(reward, bool)
                and math.isfinite(float(reward)) and float(reward) in (-1.0, 0.0, 1.0)
                for reward in value)
        and float(value[0]) == -float(value[1])
    )


def _empty_action_audit() -> dict[str, Any]:
    return {
        "expected_prompt_rows": 0,
        "valid_action_rows": 0,
        "invalid_action_rows": 0,
        "ignored_inactive_prompt_rows": 0,
        "error_counts": {},
        "invalid_examples": [],
        "invalid_examples_truncated": 0,
    }


def audit_action_rows(document: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate every active prompt against its next-row action.

    Kaggle stores the answer to ``steps[t][seat].observation`` in
    ``steps[t + 1][seat].action``.  The imitation loader intentionally skips
    malformed pairs; an index used to define train/validation/test data must
    instead retain the game and fail it closed.  INACTIVE/DONE rows can carry a
    copied prompt for the acting seat, so only ACTIVE rows (or compact local
    rows with no status field) are actor prompts.
    """
    result = _empty_action_audit()
    steps = document.get("steps")
    if not isinstance(steps, list):
        return result
    error_counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    def record_error(reason: str) -> None:
        error_counts[reason] = error_counts.get(reason, 0) + 1

    for source_step, turn in enumerate(steps):
        if not isinstance(turn, list) or len(turn) < 2:
            continue
        for seat in (0, 1):
            source = turn[seat]
            if not isinstance(source, Mapping):
                continue
            observation = source.get("observation")
            select = observation.get("select") if isinstance(observation, Mapping) else None
            if select is None:
                continue
            status = source.get("status")
            if status in ("INACTIVE", "DONE"):
                result["ignored_inactive_prompt_rows"] += 1
                continue

            result["expected_prompt_rows"] += 1
            row_errors: list[str] = []
            if status not in (None, "ACTIVE"):
                row_errors.append("prompt_row_status_not_active")
            if not isinstance(select, Mapping):
                row_errors.append("select_not_object")
                options = None
                minimum = maximum = None
            else:
                options = select.get("option")
                if (not isinstance(options, list) or not options
                        or not all(isinstance(option, Mapping) for option in options)):
                    row_errors.append("select_options_invalid")
                    options = None
                minimum = select.get("minCount", 1)
                maximum = select.get("maxCount", 1)
                if (isinstance(minimum, bool) or not isinstance(minimum, int)
                        or minimum < 0):
                    row_errors.append("select_min_count_invalid")
                    minimum = None
                if (isinstance(maximum, bool) or not isinstance(maximum, int)
                        or maximum < 0):
                    row_errors.append("select_max_count_invalid")
                    maximum = None
                if (minimum is not None and maximum is not None
                        and maximum > 0 and maximum < minimum):
                    row_errors.append("select_count_range_invalid")

            current = observation.get("current") if isinstance(observation, Mapping) else None
            your_index = current.get("yourIndex") if isinstance(current, Mapping) else None
            if (isinstance(your_index, bool) or not isinstance(your_index, int)
                    or your_index != seat):
                row_errors.append("prompt_actor_mismatch")

            if source_step + 1 >= len(steps):
                row_errors.append("prompt_missing_followup_row")
                action = None
            else:
                target_turn = steps[source_step + 1]
                target = (target_turn[seat]
                          if isinstance(target_turn, list) and len(target_turn) > seat
                          else None)
                action = target.get("action") if isinstance(target, Mapping) else None
                if not isinstance(action, list):
                    row_errors.append("action_not_list")

            if (isinstance(action, list) and options is not None
                    and minimum is not None and maximum is not None):
                if len(action) == 60:
                    row_errors.append("registration_action_at_prompt")
                integer_indices = all(
                    not isinstance(index, bool) and isinstance(index, int)
                    for index in action)
                if not integer_indices:
                    row_errors.append("action_index_not_integer")
                else:
                    if any(index < 0 or index >= len(options) for index in action):
                        row_errors.append("action_index_out_of_range")
                    if len(action) != len(set(action)):
                        row_errors.append("action_indices_not_unique")
                if len(action) < min(minimum, len(options)):
                    row_errors.append("action_below_min_count")
                if maximum > 0 and len(action) > maximum:
                    row_errors.append("action_above_max_count")

            row_errors = sorted(set(row_errors))
            if row_errors:
                result["invalid_action_rows"] += 1
                for reason in row_errors:
                    record_error(reason)
                if len(examples) < 20:
                    examples.append({
                        "source_step": source_step,
                        "seat": seat,
                        "reasons": row_errors,
                    })
            else:
                result["valid_action_rows"] += 1

    result["error_counts"] = dict(sorted(error_counts.items()))
    result["invalid_examples"] = examples
    result["invalid_examples_truncated"] = max(
        0, result["invalid_action_rows"] - len(examples))
    return result


def inspect_document(raw: bytes) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        document = json.loads(raw, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {
            "document_episode_id": None,
            "valid": False,
            "valid_for_bc": False,
            "reasons": ["json_decode_error"],
            "warnings": [],
            "rewards": None,
            "statuses": None,
            "steps": 0,
            "decision_count": 0,
            "action_audit": _empty_action_audit(),
            "seats": [],
        }
    if not isinstance(document, dict):
        return {
            "document_episode_id": None,
            "valid": False,
            "valid_for_bc": False,
            "reasons": ["document_not_object"],
            "warnings": [],
            "rewards": None,
            "statuses": None,
            "steps": 0,
            "decision_count": 0,
            "action_audit": _empty_action_audit(),
            "seats": [],
        }

    info = document.get("info")
    info = info if isinstance(info, Mapping) else {}
    document_episode_id = _safe_episode_id(info.get("EpisodeId"))
    rewards = document.get("rewards")
    rewards_valid = _valid_rewards(rewards)
    if not rewards_valid:
        reasons.append("invalid_rewards")

    statuses = document.get("statuses")
    if statuses is None:
        warnings.append("terminal_status_missing")
        stored_statuses = None
    elif (not isinstance(statuses, list) or len(statuses) != 2
          or not all(isinstance(status, str) for status in statuses)):
        reasons.append("invalid_statuses")
        stored_statuses = statuses
    else:
        stored_statuses = list(statuses)
        if any(status != "DONE" for status in statuses):
            reasons.append("episode_not_terminal")

    steps = document.get("steps")
    steps_valid = (
        isinstance(steps, list) and bool(steps)
        and all(isinstance(turn, list) and len(turn) >= 2
                and isinstance(turn[0], dict) and isinstance(turn[1], dict)
                for turn in steps)
    )
    if not steps_valid:
        reasons.append("invalid_steps")

    registered: dict[int, list[int]] = {}
    if steps_valid:
        try:
            registered = il_dataset.decks_from_document(document)
        except Exception:
            reasons.append("deck_registration_parse_error")

    seats: list[dict[str, Any]] = []
    for seat in (0, 1):
        deck = _normalized_deck(registered.get(seat))
        if deck is None:
            reasons.append(f"invalid_registered_deck_seat_{seat}")
        reward = float(rewards[seat]) if rewards_valid else None
        seats.append({
            "seat": seat,
            "team_name": _team_name(info, seat),
            "agent_name": _agent_name(info, seat),
            "reward": reward,
            "registered_deck": deck,
            "registered_deck_sha256": deck_sha256(deck) if deck is not None else None,
        })

    action_audit = audit_action_rows(document) if steps_valid else _empty_action_audit()
    if action_audit["invalid_action_rows"]:
        reasons.append("invalid_action_rows")
    decision_count = int(action_audit["valid_action_rows"])
    if decision_count == 0:
        warnings.append("no_policy_decisions")

    reasons = sorted(set(reasons))
    warnings = sorted(set(warnings))
    valid = not reasons
    return {
        "document_episode_id": document_episode_id,
        "valid": valid,
        "valid_for_bc": valid and decision_count > 0,
        "reasons": reasons,
        "warnings": warnings,
        "rewards": [float(value) for value in rewards] if rewards_valid else rewards,
        "statuses": stored_statuses,
        "steps": len(steps) if isinstance(steps, list) else 0,
        "decision_count": decision_count,
        "action_audit": action_audit,
        "seats": seats,
    }


def _stable_read(path: Path) -> tuple[bytes, os.stat_result]:
    before = path.stat()
    with path.open("rb") as handle:
        raw = handle.read()
        opened = os.fstat(handle.fileno())
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise RuntimeError("file changed while it was being indexed")
    if any(getattr(before, field) != getattr(opened, field) for field in fields):
        raise RuntimeError("opened file identity did not match the discovered path")
    if len(raw) != before.st_size:
        raise RuntimeError("short read while indexing replay")
    return raw, before


def _discover(sources: Sequence[SourceSpec]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for source in sources:
        candidate_count = ignored = regular = symlinks = 0
        for path in sorted(source.root.iterdir(), key=lambda item: item.name):
            if not path.name.endswith(".json"):
                continue
            filename_episode_id = episode_id_from_name(path.name)
            if filename_episode_id is None:
                ignored += 1
                continue
            if not (path.is_file() or path.is_symlink()):
                ignored += 1
                continue
            candidate_count += 1
            is_symlink = path.is_symlink()
            symlinks += int(is_symlink)
            regular += int(not is_symlink)
            candidates.append({
                "source": source.label,
                "path": str(path.absolute()),
                "path_object": path,
                "filename_episode_id": filename_episode_id,
                "is_symlink": is_symlink,
            })
        source_rows.append({
            "label": source.label,
            "root": str(source.root),
            "candidate_paths": candidate_count,
            "regular_paths": regular,
            "symlink_paths": symlinks,
            "ignored_json_files": ignored,
        })
    candidates.sort(key=lambda item: (item["source"], item["path"]))
    return candidates, source_rows


def _game_uid(episode_ids: Sequence[int], content_hashes: Sequence[str]) -> str:
    return _json_sha256({
        "schema": "ptcg-corpus-game-identity-v1",
        "episode_ids": list(episode_ids),
        "content_sha256s": list(content_hashes),
    })


def _build_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dsu = _DisjointSet(len(records))
    by_content: dict[str, int] = {}
    by_episode: dict[int, int] = {}
    for index, record in enumerate(records):
        content_hash = record.get("content_sha256")
        if content_hash is not None:
            if content_hash in by_content:
                dsu.union(index, by_content[content_hash])
            else:
                by_content[content_hash] = index
        identity_ids = set()
        if record.get("filename_episode_id") is not None:
            identity_ids.add(record["filename_episode_id"])
        document = record.get("document")
        if document and document.get("document_episode_id") is not None:
            identity_ids.add(document["document_episode_id"])
        for episode_id in identity_ids:
            if episode_id in by_episode:
                dsu.union(index, by_episode[episode_id])
            else:
                by_episode[episode_id] = index

    members: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        members.setdefault(dsu.find(index), []).append(record)

    games: list[dict[str, Any]] = []
    for aliases in members.values():
        aliases.sort(key=lambda item: (item["source"], item["path"]))
        content_hashes = sorted({item["content_sha256"] for item in aliases
                                 if item.get("content_sha256") is not None})
        filename_ids = sorted({item["filename_episode_id"] for item in aliases
                               if item.get("filename_episode_id") is not None})
        document_ids = sorted({
            item["document"]["document_episode_id"]
            for item in aliases
            if item.get("document")
            and item["document"].get("document_episode_id") is not None
        })
        episode_ids = sorted(set(filename_ids) | set(document_ids))
        reasons: set[str] = set()
        warnings: set[str] = set()
        documents: dict[str, dict[str, Any]] = {}
        for alias in aliases:
            if alias.get("read_error"):
                reasons.add("replay_read_error")
            document = alias.get("document")
            if document is not None:
                reasons.update(document["reasons"])
                warnings.update(document["warnings"])
                documents.setdefault(alias["content_sha256"], document)
                document_id = document.get("document_episode_id")
                filename_id = alias.get("filename_episode_id")
                if document_id is not None and filename_id != document_id:
                    reasons.add("filename_document_episode_id_mismatch")
        if not episode_ids:
            reasons.add("episode_id_missing")
        if len(document_ids) > 1:
            reasons.add("content_episode_id_conflict")
        if len(content_hashes) > 1:
            reasons.add("episode_id_content_conflict")
        if len(filename_ids) > 1:
            reasons.add("alias_filename_episode_id_conflict")

        canonical_document = documents.get(content_hashes[0]) if len(content_hashes) == 1 else None
        canonical_episode_id = document_ids[0] if len(document_ids) == 1 else (
            filename_ids[0] if len(filename_ids) == 1 else None)
        uid = _game_uid(episode_ids, content_hashes)
        source_membership = sorted({alias["source"] for alias in aliases})
        group_valid = canonical_document is not None and canonical_document["valid"] and not reasons
        group_valid_for_bc = (
            group_valid and bool(canonical_document["valid_for_bc"])
        )
        alias_rows = []
        for alias in aliases:
            alias_rows.append({
                "source": alias["source"],
                "path": alias["path"],
                "resolved_path": alias.get("resolved_path"),
                "is_symlink": alias["is_symlink"],
                "filename_episode_id": alias.get("filename_episode_id"),
                "document_episode_id": (
                    alias["document"].get("document_episode_id")
                    if alias.get("document") else None),
                "content_sha256": alias.get("content_sha256"),
                "size": alias.get("size"),
                "mtime_ns": alias.get("mtime_ns"),
                "link_mtime_ns": alias.get("link_mtime_ns"),
                "physical_file_key": alias.get("physical_file_key"),
                "read_error": alias.get("read_error"),
            })
        games.append({
            "game_uid": uid,
            "source_membership": source_membership,
            "episode_id": canonical_episode_id,
            "episode_ids": episode_ids,
            "content_sha256": content_hashes[0] if len(content_hashes) == 1 else None,
            "content_sha256s": content_hashes,
            "valid": group_valid,
            "valid_for_bc": group_valid_for_bc,
            "reasons": sorted(reasons),
            "warnings": sorted(warnings),
            "rewards": canonical_document.get("rewards") if canonical_document else None,
            "statuses": canonical_document.get("statuses") if canonical_document else None,
            "steps": canonical_document.get("steps") if canonical_document else None,
            "decision_count": canonical_document.get("decision_count") if canonical_document else 0,
            "action_audit": (canonical_document.get("action_audit")
                             if canonical_document else _empty_action_audit()),
            "seats": canonical_document.get("seats") if canonical_document else [],
            # Normal games expose their canonical fields directly above.  Keep
            # full variants only for a fail-closed content conflict, avoiding a
            # second copy of both 60-card lists in every healthy manifest row.
            "content_variants": ([
                {"content_sha256": content_hash, **documents[content_hash]}
                for content_hash in sorted(documents)
            ] if len(content_hashes) > 1 else []),
            "aliases": alias_rows,
        })
    games.sort(key=lambda game: (
        game["episode_id"] is None,
        game["episode_id"] if game["episode_id"] is not None else 0,
        game["game_uid"],
    ))
    return games


def _corpus_content_hash(games: Sequence[Mapping[str, Any]]) -> str:
    return _json_sha256({
        "schema": "ptcg-corpus-content-v1",
        "games": [{
            "game_uid": game["game_uid"],
            "episode_ids": game["episode_ids"],
            "content_sha256s": game["content_sha256s"],
        } for game in games],
    })


def add_manifest_sha256(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = _json_sha256(result)
    return result


def verify_manifest(manifest: Mapping[str, Any]) -> bool:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str):
        return False
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return _json_sha256(payload) == expected


def build_index(
        sources: Sequence[SourceSpec], *, split_seed: int = 20260722,
        splits: Sequence[tuple[str, float]] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one source is required")
    sources = tuple(sorted(sources, key=lambda source: (source.label, str(source.root))))
    if len({source.label for source in sources}) != len(sources):
        raise ValueError("source labels must be unique")
    if len({str(source.root.resolve()) for source in sources}) != len(sources):
        raise ValueError("source roots must be unique")
    checked_splits = parse_split_spec(",".join(
        f"{label}={fraction:.17g}" for label, fraction in splits))
    candidates, source_rows = _discover(sources)
    records: list[dict[str, Any]] = []
    inode_cache: dict[tuple[int, int, int, int], tuple[str, dict[str, Any]]] = {}
    content_cache: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        path: Path = candidate.pop("path_object")
        record = dict(candidate)
        try:
            link_stat = path.lstat()
            discovered = path.stat()
            file_key_tuple = (
                discovered.st_dev, discovered.st_ino,
                discovered.st_size, discovered.st_mtime_ns,
            )
            cached = inode_cache.get(file_key_tuple)
            if cached is None:
                raw, stat_result = _stable_read(path)
                content_hash = hashlib.sha256(raw).hexdigest()
                document = content_cache.get(content_hash)
                if document is None:
                    document = inspect_document(raw)
                    content_cache[content_hash] = document
                inode_cache[file_key_tuple] = (content_hash, document)
            else:
                content_hash, document = cached
                stat_result = path.stat()
                current_key = (
                    stat_result.st_dev, stat_result.st_ino,
                    stat_result.st_size, stat_result.st_mtime_ns,
                )
                if current_key != file_key_tuple:
                    raise RuntimeError("file changed while resolving an alias")
            record.update({
                "resolved_path": str(path.resolve(strict=True)),
                "content_sha256": content_hash,
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "link_mtime_ns": link_stat.st_mtime_ns,
                "physical_file_key": f"{stat_result.st_dev}:{stat_result.st_ino}",
                "document": document,
                "read_error": None,
            })
        except (OSError, RuntimeError) as error:
            record.update({
                "resolved_path": None,
                "content_sha256": None,
                "size": None,
                "mtime_ns": None,
                "link_mtime_ns": None,
                "physical_file_key": None,
                "document": None,
                "read_error": f"{type(error).__name__}: {error}",
            })
        records.append(record)

    games = _build_groups(records)
    split_rows = assign_group_splits(
        [game["game_uid"] for game in games], split_seed, checked_splits)
    for game in games:
        game.update(split_rows[game["game_uid"]])

    split_counts = {
        label: sum(game["split"] == label for game in games)
        for label, _ in checked_splits
    }
    valid_split_counts = {
        label: sum(game["split"] == label and game["valid_for_bc"] for game in games)
        for label, _ in checked_splits
    }
    readable = [record for record in records if record.get("content_sha256")]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_only": True,
        "indexer_sha256": sha256_file(__file__),
        "loader_sha256": sha256_file(il_dataset.__file__),
        "sources": source_rows,
        "split": {
            "seed": int(split_seed),
            "fractions": [
                {"label": label, "fraction": fraction}
                for label, fraction in checked_splits
            ],
            "assignment": "seeded_sha256_independent_u64_bucket_v1",
            "append_stable": True,
            "bucket_bits": 64,
            "split_rank_semantics": "unsigned_u64_prefix_of_split_bucket_sha256",
        },
        "summary": {
            "candidate_paths": len(records),
            "readable_paths": len(readable),
            "unreadable_paths": len(records) - len(readable),
            "regular_paths": sum(not record["is_symlink"] for record in records),
            "symlink_paths": sum(record["is_symlink"] for record in records),
            "physical_files": len({record["physical_file_key"] for record in readable}),
            "unique_contents": len({record["content_sha256"] for record in readable}),
            "game_groups": len(games),
            "valid_games": sum(game["valid"] for game in games),
            "valid_bc_games": sum(game["valid_for_bc"] for game in games),
            "invalid_games": sum(not game["valid"] for game in games),
            "groups_with_aliases": sum(len(game["aliases"]) > 1 for game in games),
            "deduplicated_alias_paths": sum(max(0, len(game["aliases"]) - 1)
                                              for game in games),
            "content_conflict_groups": sum(len(game["content_sha256s"]) > 1
                                           for game in games),
            "split_games": split_counts,
            "split_valid_bc_games": valid_split_counts,
        },
        "clean": all(game["valid"] for game in games),
        "corpus_content_sha256": _corpus_content_hash(games),
        "games": games,
    }
    return add_manifest_sha256(manifest)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_candidate_output(path: os.PathLike[str] | str) -> Path:
    output = Path(path).expanduser().resolve(strict=False)
    if output.suffix.lower() != ".json":
        raise ValueError("corpus index output must be a .json file")
    for protected in (_ROOT / "agent", _ROOT / "decks", _ROOT / "data"):
        if _within(output, protected):
            raise ValueError(
                f"candidate-only corpus index refuses production/generated path {output}")
    return output


def write_index(manifest: Mapping[str, Any], path: os.PathLike[str] | str) -> Path:
    if not verify_manifest(manifest):
        raise ValueError("refusing to write a corpus index with an invalid manifest hash")
    output = validate_candidate_output(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "sources", nargs="+", metavar="[LABEL=]DIRECTORY",
        help="flat replay directories; labels must be unique",
    )
    parser.add_argument("--json-out", required=True,
                        help="candidate manifest path outside agent/, decks/, and data/")
    parser.add_argument("--split-seed", type=int, default=20260722)
    parser.add_argument(
        "--splits", default="train=0.8,validation=0.1,test=0.1",
        help="comma-separated deterministic game-grouped fractions",
    )
    args = parser.parse_args(argv)
    sources = parse_source_specs(args.sources)
    splits = parse_split_spec(args.splits)
    manifest = build_index(sources, split_seed=args.split_seed, splits=splits)
    output = write_index(manifest, args.json_out)
    summary = manifest["summary"]
    print(
        f"indexed {summary['candidate_paths']} paths -> {summary['game_groups']} games "
        f"({summary['valid_bc_games']} BC-valid, {summary['invalid_games']} invalid); "
        f"content {manifest['corpus_content_sha256'][:12]} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
