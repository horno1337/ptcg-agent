import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.research import analyze_md_v1_mirror_routes as ANALYZE  # noqa: E402


def _observation(seat: int, select_type: int, context: int) -> dict:
    return {
        "current": {"yourIndex": seat},
        "select": {
            "type": select_type,
            "context": context,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 1}],
        },
    }


def test_analyzer_resolves_learner_seat_and_keeps_loss_families_descriptive(
    tmp_path,
):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    (replay_dir / "candidate-vs-base-episode-000000.json").write_text(
        json.dumps([
            _observation(0, 0, 0),
            _observation(0, 1, 7),
            _observation(1, 1, 7),
        ]),
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps({
            "schema": "ptcg-eval-ab-v2",
            "args": {"candidate_select_type": 0, "games": 1},
            "results": [{
                "summary": {"gate_valid": True, "invalid": 0},
                "records": [{
                    "episode_id": 0,
                    "learner_seat": 0,
                    "result": "loss",
                }],
            }],
        }),
        encoding="utf-8",
    )

    report = ANALYZE.analyze(eval_path, replay_dir)

    assert report["prompts"]["total"] == 2
    assert report["prompts"]["route_counts"] == {"md_v1": 1, "qu_v2b": 1}
    assert report["games"]["by_result"] == {"loss": 1}
    losses = report["non_st_main_families"]["losses"]
    assert len(losses) == 1 and losses[0]["select_type"] == "card"
    assert "not action-quality labels" in report["methodology"]


def test_analyzer_normalizes_serialized_enum_names(tmp_path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    observation = _observation(1, 0, 0)
    observation["select"]["type"] = "Main"
    observation["select"]["option"][0]["type"] = "End"
    (replay_dir / "candidate-vs-base-episode-000000.json").write_text(
        json.dumps([observation]),
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps({
            "schema": "ptcg-eval-ab-v2",
            "args": {"candidate_select_type": 0, "games": 1},
            "results": [{
                "summary": {"gate_valid": True, "invalid": 0},
                "records": [{
                    "episode_id": 0,
                    "learner_seat": 1,
                    "result": "win",
                }],
            }],
        }),
        encoding="utf-8",
    )

    report = ANALYZE.analyze(eval_path, replay_dir)

    assert report["prompts"]["route_counts"] == {"md_v1": 1}
    assert report["prompts"]["select_type_counts"] == {"main": 1}
