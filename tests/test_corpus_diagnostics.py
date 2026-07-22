"""Manifest-only corpus diagnostics contracts."""

import json
from pathlib import Path
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import index_corpus as INDEX  # noqa: E402
from tools.research import analyze_corpus_index as DIAG  # noqa: E402


def deck(card):
    cards = [card] * 60
    return cards, INDEX.deck_sha256(cards)


DECK_1, HASH_1 = deck(1)
DECK_2, HASH_2 = deck(2)
DECK_3, HASH_3 = deck(3)
DECK_4, HASH_4 = deck(4)


def seat(index, team, agent, registered_deck, deck_hash, reward):
    return {
        "seat": index,
        "team_name": team,
        "agent_name": agent,
        "reward": reward,
        "registered_deck": registered_deck,
        "registered_deck_sha256": deck_hash,
    }


def audit(*, expected=1, valid=1, invalid=0, errors=None):
    return {
        "expected_prompt_rows": expected,
        "valid_action_rows": valid,
        "invalid_action_rows": invalid,
        "ignored_inactive_prompt_rows": 0,
        "error_counts": errors or {},
        "invalid_examples": ([{"source_step": 1, "seat": 0}]
                             if invalid else []),
        "invalid_examples_truncated": max(0, invalid - 1),
    }


def game(uid, episode_id, split, decisions, seats, *, sources,
         valid=True, valid_for_bc=True, reasons=(), warnings=(), aliases=None,
         action_audit=None):
    aliases = aliases or [
        {"source": sources[0], "path": f"/replays/MUST_NOT_OPEN_{uid}.json"},
    ]
    return {
        "game_uid": uid,
        "source_membership": list(sources),
        "episode_id": episode_id,
        "valid": valid,
        "valid_for_bc": valid_for_bc,
        "reasons": list(reasons),
        "warnings": list(warnings),
        "rewards": [row["reward"] for row in seats] if len(seats) == 2 else None,
        "decision_count": decisions,
        "action_audit": action_audit or audit(expected=decisions, valid=decisions),
        "seats": seats,
        "aliases": aliases,
        "split": split,
    }


def synthetic_manifest():
    games = [
        game(
            "g-train", 100, "train", 10,
            [seat(0, "alpha", "agent-a", DECK_1, HASH_1, 1.0),
             seat(1, "beta", "agent-b", DECK_2, HASH_2, -1.0)],
            sources=("top",),
            aliases=[
                {"source": "top", "path": "/replays/MUST_NOT_OPEN_train_a.json"},
                {"source": "top", "path": "/replays/MUST_NOT_OPEN_train_b.json"},
            ],
        ),
        game(
            "g-validation", 200, "validation", 5,
            [seat(0, "alpha", "agent-a", DECK_1, HASH_1, -1.0),
             seat(1, "gamma", "agent-c", DECK_3, HASH_3, 1.0)],
            sources=("mid", "top"),
            aliases=[
                {"source": "mid", "path": "/replays/MUST_NOT_OPEN_val_mid.json"},
                {"source": "top", "path": "/replays/MUST_NOT_OPEN_val_top.json"},
            ],
        ),
        game(
            "g-invalid", 300, "test", 2,
            [seat(0, "ignored", "ignored-agent", DECK_3, HASH_3, 1.0),
             seat(1, "ignored-2", "ignored-agent-2", DECK_4, HASH_4, -1.0)],
            sources=("mid",), valid=False, valid_for_bc=False,
            reasons=("invalid_action_rows",),
            action_audit=audit(
                expected=5, valid=3, invalid=2,
                errors={"action_index_out_of_range": 2}),
        ),
        game(
            "g-empty", 400, "test", 0,
            [seat(0, None, None, DECK_3, HASH_3, 0.0),
             seat(1, None, None, DECK_4, HASH_4, 0.0)],
            sources=("legacy",), valid=True, valid_for_bc=False,
            warnings=("no_policy_decisions",),
            action_audit=audit(expected=0, valid=0),
        ),
        game(
            "g-test", 500, "test", 8,
            [seat(0, "beta", "agent-b", DECK_2, HASH_2, -1.0),
             seat(1, "delta", "agent-d", DECK_4, HASH_4, 1.0)],
            sources=("mid", "top"),
            aliases=[
                {"source": "mid", "path": "/replays/MUST_NOT_OPEN_test_mid_a.json"},
                {"source": "mid", "path": "/replays/MUST_NOT_OPEN_test_mid_b.json"},
                {"source": "top", "path": "/replays/MUST_NOT_OPEN_test_top.json"},
            ],
        ),
    ]
    return INDEX.add_manifest_sha256({
        "schema": "ptcg-corpus-index-v2",
        "candidate_only": True,
        "sources": [
            {"label": "legacy"}, {"label": "mid"}, {"label": "top"},
        ],
        "split": {
            "seed": 77,
            "assignment": "synthetic",
            "fractions": [
                {"label": "train", "fraction": 0.8},
                {"label": "validation", "fraction": 0.1},
                {"label": "test", "fraction": 0.1},
            ],
        },
        "corpus_content_sha256": "c" * 64,
        "games": games,
    })


def pair(report, field, left, right):
    rows = report["split_identity_overlap"][field]["pairs"]
    return next(row for row in rows if row["left"] == left and row["right"] == right)


def test_exact_counts_overlap_temporal_and_deck_distribution():
    report = DIAG.analyze_manifest(synthetic_manifest(), top_decks=10)
    assert DIAG.verify_report(report)
    assert report == DIAG.analyze_manifest(synthetic_manifest(), top_decks=10)

    assert report["overall"]["games"] == {
        "total": 5,
        "valid": 4,
        "invalid": 1,
        "valid_for_bc": 3,
        "valid_not_for_bc": 1,
    }
    assert report["overall"]["decisions"] == {
        "all": 25,
        "valid": 23,
        "invalid": 2,
        "valid_for_bc": 23,
        "valid_not_for_bc": 0,
    }
    assert report["by_split"]["test"]["games"] == {
        "total": 3,
        "valid": 2,
        "invalid": 1,
        "valid_for_bc": 1,
        "valid_not_for_bc": 1,
    }
    assert report["by_split"]["test"]["decisions"] == {
        "all": 10,
        "valid": 8,
        "invalid": 2,
        "valid_for_bc": 8,
        "valid_not_for_bc": 0,
    }
    assert report["overall"]["outcomes"] == {
        "draw": 1, "seat_0_win": 2, "seat_1_win": 2,
    }
    assert report["overall"]["seats"]["reward_counts"] == {
        "negative_one": 4, "positive_one": 4, "zero": 2,
    }
    identities = report["overall"]["unique_identities"]["valid_for_bc_games"]
    assert identities["unique_team_names"] == 4
    assert identities["unique_agent_names"] == 4
    assert identities["unique_registered_deck_sha256s"] == 4

    for field in ("team_name", "agent_name", "registered_deck_sha256"):
        train_validation = pair(report, field, "train", "validation")
        assert train_validation["intersection_count"] == 1
        assert train_validation["union_count"] == 3
        assert train_validation["jaccard_rate"] == 1 / 3
        train_test = pair(report, field, "train", "test")
        assert train_test["intersection_count"] == 1
        assert train_test["jaccard_rate"] == 1 / 3
        validation_test = pair(report, field, "validation", "test")
        assert validation_test["intersection_count"] == 0

    temporal = report["episode_id_temporal_proxy"]
    assert temporal["overall"]["all_games"]["median"] == 300
    assert temporal["overall"]["all_games"]["quantiles"]["p10"] == 140
    assert temporal["by_split"]["test"]["all_games"]["quantiles"]["p10"] == 320
    assert temporal["by_split"]["test"]["valid_for_bc_games"]["minimum"] == 500

    decks = {row["registered_deck_sha256"]: row
             for row in report["top_exact_decks"]["overall"]}
    assert decks[HASH_1]["seat_count"] == 2
    assert decks[HASH_1]["game_count"] == 2
    assert decks[HASH_1]["seat_rate"] == 2 / 6
    assert decks[HASH_2]["seat_count"] == 2
    assert decks[HASH_3]["seat_count"] == 1
    assert decks[HASH_4]["seat_count"] == 1

    invalid = report["invalidity_and_action_audit"]["overall"]
    assert invalid["reason_counts"] == {"invalid_action_rows": 1}
    assert invalid["warning_counts"] == {"no_policy_decisions": 1}
    assert invalid["action_audit_invalid_games"]["invalid_action_rows"] == 2
    assert invalid["action_audit_invalid_games"]["error_counts"] == {
        "action_index_out_of_range": 2,
    }


def test_source_membership_alias_overlap_and_table_warnings():
    report = DIAG.analyze_manifest(synthetic_manifest())
    sources = {row["source"]: row
               for row in report["source_membership_and_aliases"]["sources"]}
    assert sources["top"]["games"] == 3
    assert sources["top"]["alias_paths"] == 4
    assert sources["mid"]["games"] == 3
    assert sources["mid"]["alias_paths"] == 4
    pairs = report["source_membership_and_aliases"]["pairwise_game_overlap"]
    top_mid = next(row for row in pairs
                   if {row["left"], row["right"]} == {"top", "mid"})
    assert top_mid["intersection_count"] == 2
    assert top_mid["union_count"] == 4
    assert top_mid["jaccard_rate"] == 0.5
    combinations = {
        tuple(row["sources"]): row
        for row in report["source_membership_and_aliases"]["membership_combinations"]
    }
    assert combinations[("mid", "top")]["games"] == 2
    assert combinations[("mid", "top")]["alias_paths"] == 5
    assert combinations[("top",)]["deduplicated_alias_paths"] == 1

    table = DIAG.render_table(report)
    assert "Random held-out NLL is an interpolation diagnostic" in table
    assert "EpisodeId is only a collection-time proxy" in table
    assert "registered_deck_sha256" in table


def test_manifest_hash_schema_rejection_and_no_replay_opens():
    manifest = synthetic_manifest()
    tampered = json.loads(json.dumps(manifest))
    tampered["games"][0]["decision_count"] = 999
    try:
        DIAG.analyze_manifest(tampered)
    except DIAG.DiagnosticError as error:
        assert "self-hash" in str(error)
    else:
        raise AssertionError("accepted a tampered manifest")

    wrong_schema = dict(manifest)
    wrong_schema["schema"] = "ptcg-corpus-index-v1"
    wrong_schema = INDEX.add_manifest_sha256(wrong_schema)
    try:
        DIAG.analyze_manifest(wrong_schema)
    except DIAG.DiagnosticError as error:
        assert "schema" in str(error)
    else:
        raise AssertionError("accepted the wrong corpus schema")

    with tempfile.TemporaryDirectory() as temporary:
        manifest_path = Path(temporary) / "index.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        original_open = Path.open
        opened = []

        def guarded_open(path, *args, **kwargs):
            opened.append(path.resolve(strict=False))
            if path.resolve(strict=False) != manifest_path.resolve():
                raise AssertionError(f"opened non-manifest path {path}")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            loaded = DIAG.load_manifest(manifest_path)
            report = DIAG.analyze_manifest(loaded)
        assert DIAG.verify_report(report)
        assert opened == [manifest_path.resolve()]


def test_atomic_output_guards_and_no_overwrite():
    report = DIAG.analyze_manifest(synthetic_manifest())
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "candidate" / "diagnostics.json"
        assert DIAG.write_report(report, output) == output.resolve()
        assert json.loads(output.read_text(encoding="utf-8")) == report
        try:
            DIAG.write_report(report, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("overwrote an existing diagnostics report")
        assert json.loads(output.read_text(encoding="utf-8")) == report

        try:
            DIAG.validate_candidate_output(root / "not-json.txt")
        except DIAG.DiagnosticError:
            pass
        else:
            raise AssertionError("accepted a non-JSON output")

        tampered = dict(report)
        tampered["identity_overlap_detected"] = False
        try:
            DIAG.write_report(tampered, root / "tampered.json")
        except DIAG.DiagnosticError:
            pass
        else:
            raise AssertionError("wrote diagnostics with a bad self-hash")

    for protected in (ROOT / "agent", ROOT / "data", ROOT / "decks"):
        try:
            DIAG.validate_candidate_output(protected / "corpus-diagnostics.json")
        except DIAG.DiagnosticError:
            pass
        else:
            raise AssertionError(f"accepted protected output path {protected}")


if __name__ == "__main__":
    test_exact_counts_overlap_temporal_and_deck_distribution()
    test_source_membership_alias_overlap_and_table_warnings()
    test_manifest_hash_schema_rejection_and_no_replay_opens()
    test_atomic_output_guards_and_no_overwrite()
    print("all corpus diagnostics tests passed")
