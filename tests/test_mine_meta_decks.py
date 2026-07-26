import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import mine_meta_decks as MINE  # noqa: E402


def _episode(path: Path, episode_id: int, decks: tuple[list[int], list[int]]) -> None:
    document = {
        "info": {
            "EpisodeId": episode_id,
            "TeamNames": [f"team-{episode_id}-0", f"team-{episode_id}-1"],
        },
        "steps": [[
            {"action": decks[0]},
            {"action": decks[1]},
        ]],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def test_recent_window_exclusively_controls_membership_order_and_weight(tmp_path):
    history = tmp_path / "history"
    recent = tmp_path / "recent"
    history.mkdir()
    recent.mkdir()
    extinct = [101] * 60
    live_leader = [202] * 60
    live_second = [303] * 60

    for index in range(5):
        _episode(history / f"{index}.json", index, (extinct, extinct))
    _episode(recent / "10.json", 10, (live_second, live_leader))
    _episode(recent / "11.json", 11, (live_leader, live_leader))

    output = MINE.build(history, recent)

    assert [entry["deck"] for entry in output] == [
        live_leader,
        live_second,
    ]
    assert [entry["count"] for entry in output] == [3, 1]
    assert [entry["recent_count"] for entry in output] == [3, 1]
    assert [entry["historical_count"] for entry in output] == [0, 0]
    assert all(entry["deck"] != extinct for entry in output)


def test_history_is_diagnostic_and_never_breaks_recent_frequency_ties(tmp_path):
    history = tmp_path / "history"
    recent = tmp_path / "recent"
    history.mkdir()
    recent.mkdir()
    lower_tuple = [111] * 60
    higher_tuple = [222] * 60

    _episode(history / "1.json", 1, (higher_tuple, higher_tuple))
    _episode(history / "2.json", 2, (higher_tuple, lower_tuple))
    _episode(recent / "3.json", 3, (higher_tuple, lower_tuple))

    output = MINE.build(history, recent)

    assert [entry["deck"] for entry in output] == [lower_tuple, higher_tuple]
    assert [entry["count"] for entry in output] == [1, 1]
    assert [entry["historical_count"] for entry in output] == [1, 3]


def test_single_directory_mode_uses_that_directory_as_the_window(tmp_path):
    _episode(tmp_path / "1.json", 1, ([444] * 60, [555] * 60))

    output = MINE.build(tmp_path)

    assert len(output) == 2
    assert all(entry["count"] == entry["recent_count"] for entry in output)
    assert all(
        entry["historical_count"] == entry["recent_count"] for entry in output
    )
