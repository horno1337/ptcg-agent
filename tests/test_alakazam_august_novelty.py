from __future__ import annotations

from tools.research import audit_alakazam_august_novelty as A


def rows(
    main_train=(2_000, 300, 100), main_valid=(500, 50, 30),
    card_train=(2_000, 100, 80), card_valid=(500, 40, 30),
):
    def one(values):
        decisions, disagreements, games = values
        return {
            "valid_decisions": decisions,
            "semantic_disagreements": disagreements,
            "games_with_disagreement": games,
        }
    return {
        "MAIN": {"train": one(main_train), "validation": one(main_valid)},
        "CARD": {"train": one(card_train), "validation": one(card_valid)},
    }


def test_aggregate_corpus_gate_is_binding():
    result = A.adjudicate(rows(), games=A.MIN_NEW_GAMES - 1)
    assert result["aggregate_corpus_gate"]["pass"] is False
    assert not result["heads"]["MAIN"]["qualifies_for_training"]
    assert result["stop_without_training"]


def test_only_independently_qualifying_head_is_authorized():
    result = A.adjudicate(rows(), games=500)
    assert result["aggregate_corpus_gate"]["pass"] is True
    assert result["heads"]["MAIN"]["qualifies_for_training"]
    assert not result["heads"]["CARD"]["qualifies_for_training"]
    assert result["any_head_qualifies"]
    assert not result["stop_without_training"]


def test_validation_threshold_cannot_be_rescued_by_training_novelty():
    data = rows(main_valid=(500, 49, 40))
    result = A.adjudicate(data, games=500)
    assert result["heads"]["MAIN"]["train_pass"]
    assert not result["heads"]["MAIN"]["validation_pass"]
    assert not result["heads"]["MAIN"]["qualifies_for_training"]


def test_exact_thresholds_pass():
    data = rows(
        main_train=(2_500, 300, 75),
        main_valid=(500, 50, 20),
        card_train=(1_000, 0, 0),
        card_valid=(100, 0, 0),
    )
    result = A.adjudicate(data, games=400)
    assert result["heads"]["MAIN"]["qualifies_for_training"]
