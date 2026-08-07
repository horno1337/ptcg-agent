from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agent import dobi_v1_card, md_v2_card, model, policy, qu_v2_features as QF
from agent.obsview import ST_CARD, ST_MAIN, ObsView
from tools import analyze_ladder_replays as LADDER
from tools.research import analyze_dobi_v1_elite_card_disagreement as RESEARCH


ROOT = Path(__file__).resolve().parents[1]
REAL_REPLAY = ROOT / (
    "tools/checkpoints/dobi-v1-top-grim-comparison-v1/replays/"
    "sixth-sense/89766704.json"
)


def _observation(*, effect: int | None, context: int = 7) -> dict:
    return {
        "select": {
            "type": ST_CARD,
            "context": context,
            "effect": {"id": effect} if effect is not None else None,
            "option": [{"cardId": 7}, {"cardId": 112}],
            "minCount": 1,
            "maxCount": 1,
        },
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": []},
                {"active": [{"id": 648}], "bench": []},
            ],
        },
    }


class _Candidate:
    def __init__(self, logits=None, error: Exception | None = None):
        self.logits = (
            # Two option logits followed by the virtual STOP logit.
            np.asarray([0.0, 9.0, -9.0], dtype=np.float32)
            if logits is None
            else np.asarray(logits, dtype=np.float32)
        )
        self.error = error
        self.calls = 0

    def forward(self, _sample):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.logits, 0.0


def test_runtime_family_inventory_exactly_matches_locked_research_semantics():
    assert dobi_v1_card.MUNKIDORI == RESEARCH.MUNKIDORI
    assert dobi_v1_card.FAMILY_EFFECT_IDS == RESEARCH.FAMILY_EFFECT_IDS
    assert dobi_v1_card.FIXED_FAMILIES == RESEARCH.FIXED_FAMILIES


def test_real_replay_rows_have_exact_classifier_and_route_parity():
    if not REAL_REPLAY.is_file():
        pytest.skip("bound comparison replay is not present in this checkout")
    document = json.loads(REAL_REPLAY.read_text(encoding="utf-8"))
    seat = document["info"]["TeamNames"].index("Sixth Sense")
    rows = [
        view
        for view, _action in LADDER.action_rows(document, seat)
        if view.select_type == ST_CARD
    ]
    assert len(rows) >= 40
    assert {
        RESEARCH.classify_family(view) for view in rows
    } >= set(RESEARCH.FIXED_FAMILIES)

    deck = list(md_v2_card.TARGET_DECK)
    for view in rows:
        family = RESEARCH.classify_family(view)
        assert dobi_v1_card.classify_family(view) == family
        assert dobi_v1_card.supports_view(view, deck) is (
            md_v2_card.supports_view(view, deck)
            and family in RESEARCH.FIXED_FAMILIES
        )


def test_target_family_uses_candidate_and_other_card_family_is_parent_identity():
    candidate = _Candidate()
    target = ObsView(_observation(effect=dobi_v1_card.FAMILY_EFFECT_IDS["poke_pad"]))
    parent_calls = []

    def parent(sample, view, deck):
        parent_calls.append((sample, view, deck))
        return [0]

    result = dobi_v1_card.decide_layered(
        object(), target, list(md_v2_card.TARGET_DECK), candidate,
        parent_decider=parent,
    )
    assert result == [1]
    assert candidate.calls == 1
    assert parent_calls == []

    other = ObsView(_observation(effect=1231))
    result = dobi_v1_card.decide_layered(
        object(), other, list(md_v2_card.TARGET_DECK), candidate,
        parent_decider=parent,
    )
    assert result == [0]
    assert candidate.calls == 1
    assert len(parent_calls) == 1


@pytest.mark.parametrize("scope_miss", ("deck", "signature", "select_type"))
def test_scope_miss_is_exact_parent_fallback(scope_miss: str):
    obs = _observation(effect=dobi_v1_card.FAMILY_EFFECT_IDS["boss"])
    deck = list(md_v2_card.TARGET_DECK)
    if scope_miss == "deck":
        deck[0] = 999
    elif scope_miss == "signature":
        obs["current"]["players"][1]["active"] = [{"id": 999}]
    else:
        obs["select"]["type"] = ST_MAIN
    view = ObsView(obs)
    candidate = _Candidate(error=AssertionError("candidate must not run"))
    sentinel = [0]
    assert dobi_v1_card.decide_layered(
        object(), view, deck, candidate,
        parent_decider=lambda *_args: sentinel,
    ) is sentinel
    assert candidate.calls == 0


def test_candidate_exception_falls_through_to_real_frozen_parent_action():
    view = ObsView(_observation(effect=dobi_v1_card.FAMILY_EFFECT_IDS["petrel"]))
    deck = list(md_v2_card.TARGET_DECK)
    sample = QF.encode_public_observation(view.obs, deck)
    expected = md_v2_card.decide(sample, view, deck)
    assert expected is not None

    candidate = _Candidate(error=RuntimeError("candidate inference fault"))
    assert dobi_v1_card.decide_layered(
        sample, view, deck, candidate,
    ) == expected
    assert candidate.calls == 1


def test_parent_exception_also_fails_soft_to_outer_runtime():
    view = ObsView(_observation(effect=1231))
    candidate = _Candidate(error=AssertionError("candidate must not run"))

    def broken_parent(*_args):
        raise RuntimeError("parent fault")

    assert dobi_v1_card.decide_layered(
        object(), view, list(md_v2_card.TARGET_DECK), candidate,
        parent_decider=broken_parent,
    ) is None
    assert candidate.calls == 0


def test_unbound_worktree_candidate_cannot_load_or_activate(monkeypatch):
    monkeypatch.setattr(dobi_v1_card, "_candidate", None)
    monkeypatch.setattr(dobi_v1_card, "_load_attempted", False)
    assert len(dobi_v1_card.WEIGHTS_SHA256) != 64
    assert dobi_v1_card._load() is None


def test_production_dispatcher_invokes_candidate_before_frozen_card(
    monkeypatch,
):
    obs = _observation(effect=dobi_v1_card.FAMILY_EFFECT_IDS["poke_pad"])
    view = ObsView(obs)
    candidate = _Candidate()
    base = model.load()
    assert base is not None
    monkeypatch.setattr(policy, "load_deck", lambda: list(md_v2_card.TARGET_DECK))
    monkeypatch.setattr(model, "load", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(dobi_v1_card, "_load", lambda: candidate)
    monkeypatch.setenv("PTCG_DOBI_V1_CARD", "1")
    monkeypatch.setenv("PTCG_MD_V2_CARD", "1")
    assert policy._model_decide(view) == [1]
    assert candidate.calls == 1


def test_production_dispatcher_candidate_fault_falls_to_frozen_card(
    monkeypatch,
):
    obs = _observation(effect=dobi_v1_card.FAMILY_EFFECT_IDS["petrel"])
    view = ObsView(obs)
    deck = list(md_v2_card.TARGET_DECK)
    base = model.load()
    assert base is not None
    sample = QF.encode_public_observation(obs, deck)
    expected = md_v2_card.decide(sample, view, deck)
    assert expected is not None
    candidate = _Candidate(error=RuntimeError("candidate inference fault"))
    monkeypatch.setattr(policy, "load_deck", lambda: deck)
    monkeypatch.setattr(model, "load", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(dobi_v1_card, "_load", lambda: candidate)
    monkeypatch.setenv("PTCG_DOBI_V1_CARD", "1")
    monkeypatch.setenv("PTCG_MD_V2_CARD", "1")
    assert policy._model_decide(view) == expected
    assert candidate.calls == 1
