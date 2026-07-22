"""Diagnose a signed corpus index without opening any replay files.

The input must be a self-consistent ``ptcg-corpus-index-v2`` manifest.  Every
statistic is derived from metadata already embedded in that manifest; alias
paths and canonical replay paths are data, never inputs to a filesystem read.

Examples::

    python tools/research/analyze_corpus_index.py \
        tools/checkpoints/corpus-index/field-all-v2.json

    python tools/research/analyze_corpus_index.py INDEX.json \
        --format json --json-out tools/checkpoints/corpus-index/diagnostics.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


INDEX_SCHEMA = "ptcg-corpus-index-v2"
REPORT_SCHEMA = "ptcg-corpus-diagnostics-v1"
ROOT = Path(__file__).resolve().parents[2]
PROTECTED_OUTPUT_DIRS = (ROOT / "agent", ROOT / "data", ROOT / "decks")
QUANTILES = (("p10", 0.10), ("p25", 0.25), ("p50", 0.50),
             ("p75", 0.75), ("p90", 0.90))


class DiagnosticError(ValueError):
    """Raised when a corpus manifest cannot be diagnosed safely."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def verify_manifest(manifest: Mapping[str, Any]) -> bool:
    """Verify the indexer's canonical self-hash, without importing it."""
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    try:
        return _json_sha256(payload) == expected
    except (TypeError, ValueError):
        return False


def _with_report_hash(report: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(report)
    result.pop("diagnostics_sha256", None)
    result["diagnostics_sha256"] = _json_sha256(result)
    return result


def verify_report(report: Mapping[str, Any]) -> bool:
    expected = report.get("diagnostics_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    payload = dict(report)
    payload.pop("diagnostics_sha256", None)
    try:
        return _json_sha256(payload) == expected
    except (TypeError, ValueError):
        return False


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def load_manifest(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load exactly one manifest file. Replay paths are never dereferenced."""
    manifest_path = Path(path).expanduser().resolve(strict=True)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_nonfinite)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise DiagnosticError(f"cannot read corpus manifest {manifest_path}: {error}") from error
    if not isinstance(value, dict):
        raise DiagnosticError("corpus manifest must be a JSON object")
    _validate_manifest_header(value)
    return value


def _validate_manifest_header(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != INDEX_SCHEMA:
        raise DiagnosticError(
            f"expected schema {INDEX_SCHEMA!r}, got {manifest.get('schema')!r}")
    if not verify_manifest(manifest):
        raise DiagnosticError("corpus manifest self-hash is invalid")
    if not isinstance(manifest.get("games"), list):
        raise DiagnosticError("corpus manifest games must be a list")
    split = manifest.get("split")
    if not isinstance(split, Mapping) or not isinstance(split.get("fractions"), list):
        raise DiagnosticError("corpus manifest split metadata is missing")


def _split_labels(manifest: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    fractions = manifest["split"]["fractions"]
    for row in fractions:
        if not isinstance(row, Mapping) or not isinstance(row.get("label"), str):
            raise DiagnosticError("split fraction rows must have string labels")
        label = row["label"]
        if not label or label in labels:
            raise DiagnosticError(f"duplicate or empty split label {label!r}")
        labels.append(label)
    if len(labels) < 2:
        raise DiagnosticError("at least two indexed splits are required")
    return labels


def _checked_games(manifest: Mapping[str, Any], splits: Sequence[str]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, game in enumerate(manifest["games"]):
        if not isinstance(game, Mapping):
            raise DiagnosticError(f"game row {index} is not an object")
        uid = game.get("game_uid")
        if not isinstance(uid, str) or not uid or uid in seen:
            raise DiagnosticError(f"game row {index} has an invalid or duplicate game_uid")
        seen.add(uid)
        if game.get("split") not in splits:
            raise DiagnosticError(f"game {uid} has unknown split {game.get('split')!r}")
        for field in ("valid", "valid_for_bc"):
            if not isinstance(game.get(field), bool):
                raise DiagnosticError(f"game {uid} has non-boolean {field}")
        decisions = game.get("decision_count")
        if isinstance(decisions, bool) or not isinstance(decisions, int) or decisions < 0:
            raise DiagnosticError(f"game {uid} has invalid decision_count")
        if not isinstance(game.get("source_membership"), list) or not all(
                isinstance(value, str) and value for value in game["source_membership"]):
            raise DiagnosticError(f"game {uid} has invalid source_membership")
        if not isinstance(game.get("aliases"), list):
            raise DiagnosticError(f"game {uid} has invalid aliases")
        if not isinstance(game.get("reasons"), list) or not isinstance(
                game.get("warnings"), list):
            raise DiagnosticError(f"game {uid} has invalid reasons/warnings")
        if not isinstance(game.get("seats"), list):
            raise DiagnosticError(f"game {uid} has invalid seats")
        result.append(game)
    return result


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _reward_bucket(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "other"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "other"
    if numeric == -1.0:
        return "negative_one"
    if numeric == 0.0:
        return "zero"
    if numeric == 1.0:
        return "positive_one"
    return "other"


def _outcome_bucket(game: Mapping[str, Any]) -> str:
    rewards = game.get("rewards")
    if not isinstance(rewards, list) or len(rewards) != 2:
        return "missing"
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in rewards):
        return "nonstandard"
    left, right = map(float, rewards)
    if left > right:
        return "seat_0_win"
    if right > left:
        return "seat_1_win"
    if left == right:
        return "draw"
    return "nonstandard"


def _identity_summary(games: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("team_name", "agent_name", "registered_deck_sha256")
    values = {field: set() for field in fields}
    missing = Counter()
    seats = 0
    for game in games:
        for seat in game.get("seats", []):
            if not isinstance(seat, Mapping):
                continue
            seats += 1
            for field in fields:
                value = seat.get(field)
                if isinstance(value, str) and value:
                    values[field].add(value)
                else:
                    missing[field] += 1
    return {
        "seat_rows": seats,
        "unique_team_names": len(values["team_name"]),
        "unique_agent_names": len(values["agent_name"]),
        "unique_registered_deck_sha256s": len(values["registered_deck_sha256"]),
        "missing_team_names": missing["team_name"],
        "missing_agent_names": missing["agent_name"],
        "missing_registered_deck_sha256s": missing["registered_deck_sha256"],
    }


def _game_stats(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [game for game in games if game["valid"]]
    invalid = [game for game in games if not game["valid"]]
    bc = [game for game in games if game["valid_for_bc"]]
    valid_not_bc = [game for game in games if game["valid"] and not game["valid_for_bc"]]
    rewards = Counter()
    seat_indices = Counter()
    per_seat_rewards: dict[str, Counter[str]] = defaultdict(Counter)
    seat_rows = 0
    for game in games:
        for seat in game.get("seats", []):
            if not isinstance(seat, Mapping):
                continue
            seat_rows += 1
            seat_index = seat.get("seat")
            seat_key = str(seat_index) if isinstance(seat_index, int) else "missing"
            seat_indices[seat_key] += 1
            bucket = _reward_bucket(seat.get("reward"))
            rewards[bucket] += 1
            per_seat_rewards[seat_key][bucket] += 1
    outcomes = Counter(_outcome_bucket(game) for game in games)
    alias_paths = sum(len(game["aliases"]) for game in games)
    return {
        "games": {
            "total": len(games),
            "valid": len(valid),
            "invalid": len(invalid),
            "valid_for_bc": len(bc),
            "valid_not_for_bc": len(valid_not_bc),
        },
        "decisions": {
            "all": sum(game["decision_count"] for game in games),
            "valid": sum(game["decision_count"] for game in valid),
            "invalid": sum(game["decision_count"] for game in invalid),
            "valid_for_bc": sum(game["decision_count"] for game in bc),
            "valid_not_for_bc": sum(game["decision_count"] for game in valid_not_bc),
        },
        "aliases": {
            "paths": alias_paths,
            "deduplicated_paths": alias_paths - len(games),
            "groups_with_aliases": sum(len(game["aliases"]) > 1 for game in games),
            "cross_source_groups": sum(len(set(game["source_membership"])) > 1
                                       for game in games),
        },
        "seats": {
            "rows": seat_rows,
            "seat_index_counts": _counter_dict(seat_indices),
            "reward_counts": _counter_dict(rewards),
            "reward_counts_by_seat": {
                key: _counter_dict(per_seat_rewards[key])
                for key in sorted(per_seat_rewards)
            },
        },
        "outcomes": _counter_dict(outcomes),
        "unique_identities": {
            "all_games": _identity_summary(games),
            "valid_for_bc_games": _identity_summary(bc),
        },
    }


def _rate_row(left_name: str, right_name: str,
              left: set[str], right: set[str]) -> dict[str, Any]:
    intersection = left & right
    union = left | right
    return {
        "left": left_name,
        "right": right_name,
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard_rate": len(intersection) / len(union) if union else 0.0,
        "left_overlap_rate": len(intersection) / len(left) if left else 0.0,
        "right_overlap_rate": len(intersection) / len(right) if right else 0.0,
    }


def _source_diagnostics(manifest: Mapping[str, Any],
                        games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = {
        row.get("label") for row in manifest.get("sources", [])
        if isinstance(row, Mapping) and isinstance(row.get("label"), str)
    }
    labels.update(source for game in games for source in game["source_membership"])
    labels = set(filter(None, labels))
    per_source = []
    source_games: dict[str, set[str]] = {}
    for label in sorted(labels):
        selected = [game for game in games if label in game["source_membership"]]
        source_games[label] = {game["game_uid"] for game in selected}
        alias_paths = 0
        for game in selected:
            alias_paths += sum(
                isinstance(alias, Mapping) and alias.get("source") == label
                for alias in game["aliases"])
        per_source.append({
            "source": label,
            "games": len(selected),
            "valid_games": sum(game["valid"] for game in selected),
            "invalid_games": sum(not game["valid"] for game in selected),
            "valid_for_bc_games": sum(game["valid_for_bc"] for game in selected),
            "alias_paths": alias_paths,
        })
    pairs = [
        _rate_row(left, right, source_games[left], source_games[right])
        for left_index, left in enumerate(sorted(labels))
        for right in sorted(labels)[left_index + 1:]
    ]
    combinations: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for game in games:
        combinations[tuple(sorted(set(game["source_membership"])))].append(game)
    combination_rows = []
    for membership in sorted(combinations):
        selected = combinations[membership]
        aliases = sum(len(game["aliases"]) for game in selected)
        combination_rows.append({
            "sources": list(membership),
            "games": len(selected),
            "valid_games": sum(game["valid"] for game in selected),
            "invalid_games": sum(not game["valid"] for game in selected),
            "valid_for_bc_games": sum(game["valid_for_bc"] for game in selected),
            "alias_paths": aliases,
            "deduplicated_alias_paths": aliases - len(selected),
        })
    return {
        "sources": per_source,
        "membership_combinations": combination_rows,
        "pairwise_game_overlap": pairs,
    }


def _identity_sets(games: Iterable[Mapping[str, Any]], field: str) -> set[str]:
    result: set[str] = set()
    for game in games:
        if not game["valid_for_bc"]:
            continue
        for seat in game.get("seats", []):
            value = seat.get(field) if isinstance(seat, Mapping) else None
            if isinstance(value, str) and value:
                result.add(value)
    return result


def _identity_overlap(games: Sequence[Mapping[str, Any]],
                      splits: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("team_name", "agent_name", "registered_deck_sha256"):
        per_split = {
            split: _identity_sets(
                (game for game in games if game["split"] == split), field)
            for split in splits
        }
        result[field] = {
            "scope": "valid_for_bc_games_registered_seats",
            "unique_by_split": {split: len(per_split[split]) for split in splits},
            "pairs": [
                _rate_row(left, right, per_split[left], per_split[right])
                for left_index, left in enumerate(splits)
                for right in splits[left_index + 1:]
            ],
        }
    return result


def _type7_quantile(sorted_values: Sequence[int], fraction: float) -> int | float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    value = (sorted_values[lower] * (upper - position)
             + sorted_values[upper] * (position - lower))
    return int(value) if value.is_integer() else value


def _episode_quantiles(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(
        game["episode_id"] for game in games
        if isinstance(game.get("episode_id"), int)
        and not isinstance(game.get("episode_id"), bool)
    )
    return {
        "game_count": len(games),
        "episode_id_count": len(values),
        "missing_episode_id_count": len(games) - len(values),
        "minimum": values[0] if values else None,
        "median": _type7_quantile(values, 0.5),
        "maximum": values[-1] if values else None,
        "quantiles": {
            label: _type7_quantile(values, fraction)
            for label, fraction in QUANTILES
        },
    }


def _temporal_diagnostics(games: Sequence[Mapping[str, Any]],
                          splits: Sequence[str]) -> dict[str, Any]:
    def row(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "all_games": _episode_quantiles(selected),
            "valid_for_bc_games": _episode_quantiles(
                [game for game in selected if game["valid_for_bc"]]),
        }
    return {
        "field": "episode_id",
        "quantile_method": "Hyndman-Fan type 7 linear interpolation",
        "interpretation": "collection-time proxy only",
        "overall": row(games),
        "by_split": {
            split: row([game for game in games if game["split"] == split])
            for split in splits
        },
    }


def _top_decks(games: Sequence[Mapping[str, Any]], limit: int,
               splits: Sequence[str]) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for game in games:
        if not game["valid_for_bc"]:
            continue
        seen_in_game: set[str] = set()
        for seat in game.get("seats", []):
            if not isinstance(seat, Mapping):
                continue
            deck_hash = seat.get("registered_deck_sha256")
            if not isinstance(deck_hash, str) or not deck_hash:
                continue
            row = aggregate.setdefault(deck_hash, {
                "registered_deck_sha256": deck_hash,
                "registered_deck": seat.get("registered_deck"),
                "seat_count": 0,
                "game_count": 0,
                "split_seat_counts": Counter(),
                "team_names": set(),
                "agent_names": set(),
            })
            if row["registered_deck"] != seat.get("registered_deck"):
                raise DiagnosticError(
                    f"deck hash {deck_hash} maps to multiple registered deck lists")
            row["seat_count"] += 1
            row["split_seat_counts"][game["split"]] += 1
            if deck_hash not in seen_in_game:
                row["game_count"] += 1
                seen_in_game.add(deck_hash)
            for source, target in (("team_name", "team_names"),
                                   ("agent_name", "agent_names")):
                value = seat.get(source)
                if isinstance(value, str) and value:
                    row[target].add(value)
    total_seats = sum(row["seat_count"] for row in aggregate.values())
    rows = []
    for deck_hash, row in aggregate.items():
        rows.append({
            "registered_deck_sha256": deck_hash,
            "registered_deck": row["registered_deck"],
            "seat_count": row["seat_count"],
            "game_count": row["game_count"],
            "seat_rate": row["seat_count"] / total_seats if total_seats else 0.0,
            "split_seat_counts": {
                split: row["split_seat_counts"][split] for split in splits
            },
            "team_names": sorted(row["team_names"]),
            "agent_names": sorted(row["agent_names"]),
        })
    rows.sort(key=lambda row: (-row["seat_count"], row["registered_deck_sha256"]))
    return rows[:limit]


def _action_audit(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numeric_fields = (
        "expected_prompt_rows", "valid_action_rows", "invalid_action_rows",
        "ignored_inactive_prompt_rows", "invalid_examples_truncated",
    )
    totals = Counter()
    errors = Counter()
    examples_retained = 0
    games_with_invalid_rows = 0
    for game in games:
        audit = game.get("action_audit")
        if not isinstance(audit, Mapping):
            continue
        for field in numeric_fields:
            value = audit.get(field, 0)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[field] += value
        if isinstance(audit.get("invalid_examples"), list):
            examples_retained += len(audit["invalid_examples"])
        error_counts = audit.get("error_counts")
        if isinstance(error_counts, Mapping):
            for reason, count in error_counts.items():
                if (isinstance(reason, str) and isinstance(count, int)
                        and not isinstance(count, bool) and count >= 0):
                    errors[reason] += count
        if isinstance(audit.get("invalid_action_rows"), int) and audit.get(
                "invalid_action_rows", 0) > 0:
            games_with_invalid_rows += 1
    return {
        **{field: totals[field] for field in numeric_fields},
        "games_with_invalid_action_rows": games_with_invalid_rows,
        "invalid_examples_retained": examples_retained,
        "error_counts": _counter_dict(errors),
    }


def _invalidity(games: Sequence[Mapping[str, Any]],
                 splits: Sequence[str]) -> dict[str, Any]:
    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        invalid = [game for game in selected if not game["valid"]]
        valid_not_bc = [game for game in selected
                        if game["valid"] and not game["valid_for_bc"]]
        reasons = Counter(
            reason for game in invalid for reason in game.get("reasons", [])
            if isinstance(reason, str))
        warnings = Counter(
            warning for game in selected for warning in game.get("warnings", [])
            if isinstance(warning, str))
        return {
            "invalid_games": len(invalid),
            "valid_not_for_bc_games": len(valid_not_bc),
            "reason_counts": _counter_dict(reasons),
            "warning_counts": _counter_dict(warnings),
            "action_audit_all_games": _action_audit(selected),
            "action_audit_invalid_games": _action_audit(invalid),
        }
    return {
        "overall": summarize(games),
        "by_split": {
            split: summarize([game for game in games if game["split"] == split])
            for split in splits
        },
    }


def analyze_manifest(manifest: Mapping[str, Any], *, top_decks: int = 20) -> dict[str, Any]:
    """Return deterministic, manifest-only diagnostics for a signed v2 index."""
    _validate_manifest_header(manifest)
    if isinstance(top_decks, bool) or not isinstance(top_decks, int) or top_decks <= 0:
        raise DiagnosticError("top_decks must be a positive integer")
    splits = _split_labels(manifest)
    games = _checked_games(manifest, splits)
    by_split = {
        split: _game_stats([game for game in games if game["split"] == split])
        for split in splits
    }
    overlap = _identity_overlap(games, splits)
    overlap_count = sum(
        row["intersection_count"]
        for field in overlap.values()
        for row in field["pairs"]
    )
    report = {
        "schema": REPORT_SCHEMA,
        "candidate_only": True,
        "input": {
            "schema": manifest["schema"],
            "manifest_sha256": manifest["manifest_sha256"],
            "corpus_content_sha256": manifest.get("corpus_content_sha256"),
            "split_seed": manifest["split"].get("seed"),
            "split_assignment": manifest["split"].get("assignment"),
            "split_order": splits,
        },
        "scope": {
            "filesystem_reads": "input manifest only; replay and alias paths are never opened",
            "identity_overlap": "valid_for_bc games and their registered seats",
            "top_exact_decks": "valid_for_bc games and their registered seats",
            "validity_and_outcomes": "all indexed game groups",
        },
        "warnings": [
            (
                "Random held-out NLL is an interpolation diagnostic when team, agent, "
                "or exact-deck identities overlap across splits; it is not an independent "
                "ladder-transfer estimate."
            ),
            (
                "EpisodeId is only a collection-time proxy; its quantiles do not establish "
                "chronology, meta freshness, or ladder-transfer performance."
            ),
        ],
        "identity_overlap_detected": overlap_count > 0,
        "overall": _game_stats(games),
        "by_split": by_split,
        "source_membership_and_aliases": _source_diagnostics(manifest, games),
        "split_identity_overlap": overlap,
        "episode_id_temporal_proxy": _temporal_diagnostics(games, splits),
        "top_exact_decks": {
            "limit": top_decks,
            "overall": _top_decks(games, top_decks, splits),
            "by_split": {
                split: _top_decks(
                    [game for game in games if game["split"] == split],
                    top_decks,
                    splits,
                ) for split in splits
            },
        },
        "invalidity_and_action_audit": _invalidity(games, splits),
    }
    return _with_report_hash(report)


def render_table(report: Mapping[str, Any]) -> str:
    """Render a stable, compact human-readable view of a diagnostics report."""
    if not verify_report(report):
        raise DiagnosticError("cannot render diagnostics with an invalid self-hash")
    splits = report["input"]["split_order"]
    lines = [
        f"Corpus diagnostics {report['input']['manifest_sha256']}",
        "",
        "Split         games valid invalid bc_games decisions valid_dec invalid_dec bc_dec",
    ]
    for split in splits:
        row = report["by_split"][split]
        games = row["games"]
        decisions = row["decisions"]
        lines.append(
            f"{split:<13} {games['total']:>5} {games['valid']:>5} "
            f"{games['invalid']:>7} {games['valid_for_bc']:>8} "
            f"{decisions['all']:>9} {decisions['valid']:>9} "
            f"{decisions['invalid']:>11} {decisions['valid_for_bc']:>6}"
        )
    lines.extend([
        "",
        "Source membership and aliases",
        "source        games valid invalid bc_games alias_paths",
    ])
    sources = report["source_membership_and_aliases"]
    for row in sources["sources"]:
        lines.append(
            f"{row['source']:<13} {row['games']:>5} {row['valid_games']:>5} "
            f"{row['invalid_games']:>7} {row['valid_for_bc_games']:>8} "
            f"{row['alias_paths']:>11}"
        )
    for row in sources["pairwise_game_overlap"]:
        lines.append(
            f"source overlap {row['left']} / {row['right']}: "
            f"{row['intersection_count']} shared, Jaccard={row['jaccard_rate']:.4f}"
        )
    overall = report["overall"]
    identity = overall["unique_identities"]["valid_for_bc_games"]
    lines.extend([
        "",
        "Outcomes and registered seats (all indexed games)",
        "outcomes: " + ", ".join(
            f"{key}={value}" for key, value in overall["outcomes"].items()),
        "seat rewards: " + ", ".join(
            f"{key}={value}"
            for key, value in overall["seats"]["reward_counts"].items()),
        f"seat rows={overall['seats']['rows']}; BC-eligible unique identities: "
        f"teams={identity['unique_team_names']}, agents={identity['unique_agent_names']}, "
        f"exact_decks={identity['unique_registered_deck_sha256s']}",
        "",
        "BC-eligible identity overlap",
    ])
    for field, section in report["split_identity_overlap"].items():
        for row in section["pairs"]:
            lines.append(
                f"{field:<26} {row['left']} / {row['right']}: "
                f"{row['intersection_count']} shared, Jaccard={row['jaccard_rate']:.4f}"
            )
    lines.extend(["", "EpisodeId proxy (BC-eligible games)"])
    for split in splits:
        row = report["episode_id_temporal_proxy"]["by_split"][split][
            "valid_for_bc_games"]
        lines.append(
            f"{split:<13} n={row['episode_id_count']} min={row['minimum']} "
            f"median={row['median']} max={row['maximum']}"
        )
    lines.extend(["", "Top exact decks (BC-eligible seats)"])
    for row in report["top_exact_decks"]["overall"]:
        lines.append(
            f"{row['registered_deck_sha256']} seats={row['seat_count']} "
            f"rate={row['seat_rate']:.4f}"
        )
    invalid = report["invalidity_and_action_audit"]["overall"]
    action_audit = invalid["action_audit_all_games"]
    lines.extend([
        "",
        f"Invalid games: {invalid['invalid_games']}; "
        f"valid but not BC-eligible: {invalid['valid_not_for_bc_games']}",
        f"Action audit: expected={action_audit['expected_prompt_rows']} "
        f"valid={action_audit['valid_action_rows']} "
        f"invalid={action_audit['invalid_action_rows']}",
    ])
    for reason, count in invalid["reason_counts"].items():
        lines.append(f"invalid reason {reason}: {count}")
    for reason, count in action_audit["error_counts"].items():
        lines.append(f"action-audit error {reason}: {count}")
    lines.extend(["", *[f"WARNING: {warning}" for warning in report["warnings"]]])
    return "\n".join(lines) + "\n"


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_candidate_output(path: os.PathLike[str] | str) -> Path:
    output = Path(path).expanduser().resolve(strict=False)
    if output.suffix.lower() != ".json":
        raise DiagnosticError("diagnostics output must be a .json file")
    for protected in PROTECTED_OUTPUT_DIRS:
        if _within(output, protected.resolve(strict=False)):
            raise DiagnosticError(
                f"candidate-only diagnostics refuse protected path {output}")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing diagnostics {output}")
    return output


def write_report(report: Mapping[str, Any], path: os.PathLike[str] | str) -> Path:
    """Atomically publish a report, failing if the destination already exists."""
    if not verify_report(report):
        raise DiagnosticError("refusing to write diagnostics with an invalid self-hash")
    output = validate_candidate_output(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Re-resolve after creating parents so a newly materialized symlink cannot
    # redirect the output into a protected production/generated directory.
    output = validate_candidate_output(output)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent,
                prefix=f".{output.name}.tmp-", delete=False) as handle:
            temporary_name = handle.name
            json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # link(2) is an atomic no-replace publication on the same filesystem.
        os.link(temporary_name, output)
        try:
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help=f"signed {INDEX_SCHEMA} JSON manifest")
    parser.add_argument("--top-decks", type=int, default=20)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument(
        "--json-out",
        help="optional candidate-only JSON path outside agent/, data/, and decks/; "
             "existing files are never overwritten",
    )
    args = parser.parse_args(argv)
    report = analyze_manifest(load_manifest(args.manifest), top_decks=args.top_decks)
    if args.json_out:
        write_report(report, args.json_out)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False))
    else:
        print(render_table(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
