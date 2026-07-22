"""Corpus-index identity, validity, provenance, and split contracts."""

import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import index_corpus as INDEX  # noqa: E402


def replay(episode_id, *, reward=(1, -1), team=("alpha", "beta"),
           deck_offset=0, statuses=("DONE", "DONE"), with_decision=True):
    decks = [
        [1 + deck_offset + (index % 4) for index in range(60)],
        [101 + deck_offset + (index % 4) for index in range(60)],
    ]
    select = {
        "type": 0,
        "minCount": 1,
        "maxCount": 1,
        "option": [{"type": 14}],
    }
    first = [
        {
            "action": decks[0],
            "status": "ACTIVE",
            "observation": {
                "current": {"yourIndex": 0},
                "select": select,
            },
        },
        {
            "action": decks[1],
            "status": "INACTIVE",
            "observation": {
                "current": {"yourIndex": 1},
                "select": None,
            },
        },
    ]
    steps = [first]
    if with_decision:
        steps.append([
            {"action": [0], "status": "ACTIVE", "observation": {}},
            {"action": [], "status": "INACTIVE", "observation": {}},
        ])
    return {
        "info": {
            "EpisodeId": episode_id,
            "TeamNames": list(team),
            "Agents": [{"Name": f"agent-{team[0]}"}, {"Name": f"agent-{team[1]}"}],
        },
        "rewards": list(reward),
        "statuses": list(statuses) if statuses is not None else None,
        "steps": steps,
    }


def write_replay(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True)


def source(label, path):
    return INDEX.SourceSpec(label, path.resolve())


def test_aliases_dedupe_to_one_content_locked_game():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        legacy = root / "legacy"
        top = root / "top"
        copy = root / "copy"
        original = legacy / "101.json"
        write_replay(original, replay(101))
        top.mkdir()
        os.symlink(original, top / "101.json")
        copy.mkdir()
        (copy / "episode-101-replay.json").write_bytes(original.read_bytes())
        (top / ".done_subs.json").write_text("[]", encoding="utf-8")

        manifest = INDEX.build_index((
            source("top", top), source("legacy", legacy), source("copy", copy),
        ), split_seed=17)
        reordered = INDEX.build_index((
            source("copy", copy), source("legacy", legacy), source("top", top),
        ), split_seed=17)
        assert reordered == manifest
        assert INDEX.verify_manifest(manifest)
        assert manifest["schema"] == "ptcg-corpus-index-v2"
        assert manifest["clean"]
        game = manifest["games"][0]
        split_counts = {"train": 0, "validation": 0, "test": 0}
        split_counts[game["split"]] = 1
        assert manifest["summary"] == {
            "candidate_paths": 3,
            "readable_paths": 3,
            "unreadable_paths": 0,
            "regular_paths": 2,
            "symlink_paths": 1,
            "physical_files": 2,
            "unique_contents": 1,
            "game_groups": 1,
            "valid_games": 1,
            "valid_bc_games": 1,
            "invalid_games": 0,
            "groups_with_aliases": 1,
            "deduplicated_alias_paths": 2,
            "content_conflict_groups": 0,
            "split_games": split_counts,
            "split_valid_bc_games": split_counts,
        }
        assert game["episode_id"] == 101
        assert game["source_membership"] == ["copy", "legacy", "top"]
        assert len(game["aliases"]) == 3
        assert len(game["content_sha256s"]) == 1
        assert game["content_sha256"] == game["content_sha256s"][0]
        assert game["decision_count"] == 1
        assert game["seats"][0]["team_name"] == "alpha"
        assert game["seats"][1]["agent_name"] == "agent-beta"
        assert len(game["seats"][0]["registered_deck"]) == 60
        assert len(game["seats"][0]["registered_deck_sha256"]) == 64
        assert game["split"] in {"train", "validation", "test"}
        assert len(game["split_bucket_sha256"]) == 64
        assert len(game["split_bucket_u64_hex"]) == 16
        assert game["split_rank"] == int(game["split_bucket_u64_hex"], 16)
        by_source = {row["label"]: row for row in manifest["sources"]}
        assert by_source["top"]["ignored_json_files"] == 1


def test_same_episode_id_with_different_content_fails_closed():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        left, right = root / "left", root / "right"
        write_replay(left / "202.json", replay(202, team=("one", "two")))
        write_replay(right / "202.json", replay(
            202, team=("three", "four"), deck_offset=10))
        manifest = INDEX.build_index((source("left", left), source("right", right)))
        assert INDEX.verify_manifest(manifest)
        assert not manifest["clean"]
        assert manifest["summary"]["game_groups"] == 1
        assert manifest["summary"]["content_conflict_groups"] == 1
        game = manifest["games"][0]
        assert not game["valid"]
        assert not game["valid_for_bc"]
        assert game["content_sha256"] is None
        assert len(game["content_sha256s"]) == 2
        assert "episode_id_content_conflict" in game["reasons"]
        assert len(game["content_variants"]) == 2


def test_invalid_replay_is_retained_with_explicit_reasons():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "bad"
        document = replay(303, reward=(None, None), statuses=("ACTIVE", "DONE"))
        document["steps"][0][1]["action"] = []
        write_replay(root / "303.json", document)
        manifest = INDEX.build_index((source("bad", root),))
        game = manifest["games"][0]
        assert not game["valid"]
        assert not game["valid_for_bc"]
        assert set(game["reasons"]) >= {
            "invalid_rewards", "episode_not_terminal",
            "invalid_registered_deck_seat_1",
        }
        assert manifest["summary"]["invalid_games"] == 1
        assert manifest["summary"]["valid_bc_games"] == 0
        assert INDEX.verify_manifest(manifest)


def test_valid_game_without_decisions_is_not_a_bc_game():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "empty"
        document = replay(304, statuses=None, with_decision=False)
        document["steps"][0][0]["observation"]["select"] = None
        write_replay(root / "304.json", document)
        manifest = INDEX.build_index((source("empty", root),))
        game = manifest["games"][0]
        assert game["valid"]
        assert not game["valid_for_bc"]
        assert game["reasons"] == []
        assert set(game["warnings"]) == {
            "no_policy_decisions", "terminal_status_missing",
        }
        assert manifest["clean"]
        assert manifest["summary"]["valid_games"] == 1
        assert manifest["summary"]["valid_bc_games"] == 0


def test_split_is_append_stable_game_grouped_and_order_independent():
    keys = [f"game-{index}" for index in range(1000)]
    forward = INDEX.assign_group_splits(keys, 55)
    reverse = INDEX.assign_group_splits(reversed(keys), 55)
    assert forward == reverse
    counts = {label: sum(row["split"] == label for row in forward.values())
              for label in ("train", "validation", "test")}
    assert 730 <= counts["train"] <= 870
    assert 50 <= counts["validation"] <= 150
    assert 50 <= counts["test"] <= 150

    # Exact-cardinality/rank splitting violated this property: unrelated
    # future games cannot alter an existing game's assignment or metadata.
    extended = INDEX.assign_group_splits(
        keys + [f"unrelated-{index}" for index in range(500)], 55)
    assert all(extended[key] == forward[key] for key in keys)
    reduced = INDEX.assign_group_splits(keys[::2], 55)
    assert all(reduced[key] == forward[key] for key in keys[::2])
    assert INDEX.assign_group_splits(keys, 56) != forward

    # Tiny synthetic consumers should choose fixtures by their stable bucket,
    # rather than assuming every small sample has an exact 80/10/10 split.
    selected = {"train": [], "validation": [], "test": []}
    candidate = 0
    while min(map(len, selected.values())) < 5:
        key = f"fixture-{candidate}"
        row = INDEX.assign_group_split(key, 99)
        if len(selected[row["split"]]) < 5:
            selected[row["split"]].append(key)
        candidate += 1
        assert candidate < 10000
    assert all(len(values) == 5 for values in selected.values())


def test_strict_action_audit_rejects_one_illegal_active_pair():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "mixed"
        document = replay(305)
        prompt = document["steps"][0][0]["observation"]["select"]
        # Step 1's legal [0] answers the first prompt, while its observation
        # requests a second action. Step 2 answers that prompt illegally.
        document["steps"][1][0].update({
            "status": "ACTIVE",
            "observation": {
                "current": {"yourIndex": 0},
                "select": prompt,
            },
        })
        document["steps"].append([
            {"action": [7], "status": "ACTIVE", "observation": {}},
            {"action": [], "status": "INACTIVE", "observation": {}},
        ])
        write_replay(root / "305.json", document)
        manifest = INDEX.build_index((source("mixed", root),))
        game = manifest["games"][0]
        assert not game["valid"]
        assert not game["valid_for_bc"]
        assert "invalid_action_rows" in game["reasons"]
        assert game["decision_count"] == 1
        assert game["action_audit"]["expected_prompt_rows"] == 2
        assert game["action_audit"]["valid_action_rows"] == 1
        assert game["action_audit"]["invalid_action_rows"] == 1
        assert game["action_audit"]["error_counts"] == {
            "action_index_out_of_range": 1,
        }


def test_strict_action_audit_preserves_optional_stop_and_registrations():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "stop"
        document = replay(306)
        select = document["steps"][0][0]["observation"]["select"]
        select["minCount"] = 0
        document["steps"][1][0]["action"] = []
        write_replay(root / "306.json", document)
        manifest = INDEX.build_index((source("stop", root),))
        game = manifest["games"][0]
        assert game["valid"]
        assert game["valid_for_bc"]
        assert game["action_audit"]["expected_prompt_rows"] == 1
        assert game["action_audit"]["valid_action_rows"] == 1
        assert game["action_audit"]["invalid_action_rows"] == 0
        assert len(game["seats"][0]["registered_deck"]) == 60
        assert len(game["seats"][1]["registered_deck"]) == 60


def test_manifest_tampering_and_production_output_are_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "corpus"
        write_replay(corpus / "404.json", replay(404))
        manifest = INDEX.build_index((source("corpus", corpus),))
        output = root / "candidate" / "index.json"
        assert INDEX.write_index(manifest, output) == output
        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded == manifest
        assert INDEX.verify_manifest(loaded)
        loaded["summary"]["game_groups"] = 99
        assert not INDEX.verify_manifest(loaded)
        try:
            INDEX.write_index(loaded, root / "tampered.json")
        except ValueError:
            pass
        else:
            raise AssertionError("wrote a tampered manifest")

    try:
        INDEX.validate_candidate_output(ROOT / "agent" / "corpus-index.json")
    except ValueError:
        pass
    else:
        raise AssertionError("candidate index accepted an agent/ output path")


if __name__ == "__main__":
    test_aliases_dedupe_to_one_content_locked_game()
    test_same_episode_id_with_different_content_fails_closed()
    test_invalid_replay_is_retained_with_explicit_reasons()
    test_valid_game_without_decisions_is_not_a_bc_game()
    test_split_is_append_stable_game_grouped_and_order_independent()
    test_strict_action_audit_rejects_one_illegal_active_pair()
    test_strict_action_audit_preserves_optional_stop_and_registrations()
    test_manifest_tampering_and_production_output_are_rejected()
    print("all corpus index tests passed")
