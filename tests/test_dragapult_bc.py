from __future__ import annotations

import numpy as np

from agent import dragapult_bc as BC
from agent.obsview import (
    AREA_ACTIVE,
    AREA_BENCH,
    CTX_DAMAGE_COUNTER_ANY,
    CTX_SWITCH,
    OT_ABILITY,
    OT_ATTACH,
    OT_ATTACK,
    OT_END,
    OT_EVOLVE,
    OT_PLAY,
    ST_CARD,
    ST_MAIN,
    ObsView,
)


def _pokemon(card_id: int, hp: int, *, energy=()) -> dict:
    return {
        "id": card_id,
        "hp": hp,
        "maxHp": max(hp, 100),
        "energies": list(energy),
        "energyCards": [{"id": card_id} for card_id in energy],
        "tools": [],
    }


def _view(hps: list[int], *, context: int = CTX_DAMAGE_COUNTER_ANY,
          effect: int = BC.DRAGAPULT_EX) -> ObsView:
    options = [
        {
            "type": 3,
            "area": AREA_BENCH,
            "index": index,
            "playerIndex": 1,
        }
        for index in range(len(hps))
    ]
    return ObsView({
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": []},
                {
                    "active": [],
                    "bench": [
                        _pokemon(200 + index, hp)
                        for index, hp in enumerate(hps)
                    ],
                },
            ],
        },
        "select": {
            "type": ST_CARD,
            "context": context,
            "effect": {"id": effect},
            "remainDamageCounter": 5,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        },
    })


def _main_view(*, active, bench=(), hand=(), opponent_active=None,
               opponent_bench=(), options=(), deck_count=30,
               select_type=ST_MAIN, context=0, effect=None, turn=5,
               prizes=6, energy_attached=False) -> ObsView:
    return ObsView({
        "current": {
            "yourIndex": 0,
            "turn": turn,
            "energyAttached": energy_attached,
            "players": [
                {
                    "active": [active],
                    "bench": list(bench),
                    "hand": [{"id": card_id} for card_id in hand],
                    "handCount": len(hand),
                    "deckCount": deck_count,
                    "discard": [],
                    "prize": [None] * prizes,
                },
                {
                    "active": [opponent_active or _pokemon(200, 250)],
                    "bench": list(opponent_bench),
                    "hand": None,
                    "handCount": 0,
                    "deckCount": 30,
                    "discard": [],
                    "prize": [None] * 6,
                },
            ],
        },
        "select": {
            "type": select_type,
            "context": context,
            "effect": {"id": effect} if effect is not None else None,
            "minCount": 1,
            "maxCount": 1,
            "option": list(options),
        },
    })


def test_phantom_dive_preserves_a_live_bc_target():
    view = _view([20, 10, 80])
    assert BC._guard_phantom_dive_target(view, [0]) == [0]


def test_phantom_dive_retargets_dead_choice_to_lowest_live_hp():
    view = _view([0, 40, 10, -20])
    assert BC._guard_phantom_dive_target(view, [0]) == [2]


def test_phantom_dive_keeps_choice_when_no_live_target_exists():
    view = _view([0, -10])
    assert BC._guard_phantom_dive_target(view, [0]) == [0]


def test_phantom_secure_prize_finishes_higher_value_lucario_target():
    view = _main_view(
        active=_pokemon(BC.DRAGAPULT_EX, 320),
        opponent_active=_pokemon(678, 340),
        opponent_bench=[_pokemon(675, 50), _pokemon(678, 30)],
        select_type=ST_CARD,
        context=CTX_DAMAGE_COUNTER_ANY,
        effect=BC.DRAGAPULT_EX,
        options=[
            {"type": 3, "area": AREA_BENCH, "index": 0,
             "playerIndex": 1},
            {"type": 3, "area": AREA_BENCH, "index": 1,
             "playerIndex": 1},
        ],
    )
    view.select["remainDamageCounter"] = 6
    assert BC._guard_phantom_secure_prize(view, [0]) == [1]


def test_phantom_secure_prize_preserves_best_ko_and_non_lucario_states():
    view = _main_view(
        active=_pokemon(BC.DRAGAPULT_EX, 320),
        opponent_active=_pokemon(678, 340),
        opponent_bench=[_pokemon(675, 50), _pokemon(677, 40)],
        select_type=ST_CARD, context=CTX_DAMAGE_COUNTER_ANY,
        effect=BC.DRAGAPULT_EX,
        options=[
            {"type": 3, "area": AREA_BENCH, "index": 0,
             "playerIndex": 1},
            {"type": 3, "area": AREA_BENCH, "index": 1,
             "playerIndex": 1},
        ],
    )
    view.select["remainDamageCounter"] = 5
    assert BC._guard_phantom_secure_prize(view, [0]) == [0]
    view.opp["active"][0]["id"] = 200
    assert BC._guard_phantom_secure_prize(view, [1]) == [1]


def _protection_view(entries: list[dict], *, remain: int = 6) -> ObsView:
    view = _main_view(
        active=_pokemon(BC.DRAGAPULT_EX, 320),
        opponent_bench=entries,
        select_type=ST_CARD,
        context=CTX_DAMAGE_COUNTER_ANY,
        effect=BC.DRAGAPULT_EX,
        options=[
            {"type": 3, "area": AREA_BENCH, "index": index,
             "playerIndex": 1}
            for index in range(len(entries))
        ],
    )
    view.select["remainDamageCounter"] = remain
    return view


def test_phantom_protection_guard_uses_head_ranked_safe_target():
    view = _protection_view([
        _pokemon(BC.TEAM_ROCKET_ARTICUNO, 120),
        _pokemon(BC.DREEPY, 70),
        _pokemon(BC.MUNKIDORI, 110),
    ])
    logits = np.asarray([9.0, 2.0, 4.0, 0.0])
    assert BC._guard_phantom_protected_target(view, logits, [0]) == [2]


def test_phantom_protection_guard_honors_mist_and_typed_rock_energy():
    mist = _pokemon(200, 100, energy=(BC.MIST_ENERGY,))
    fighting = _pokemon(58, 140, energy=(BC.ROCK_FIGHTING_ENERGY,))
    not_fighting = _pokemon(BC.DREEPY, 70, energy=(BC.ROCK_FIGHTING_ENERGY,))
    safe = _pokemon(BC.MUNKIDORI, 110)
    assert BC._phantom_effect_protected(mist)
    assert BC._phantom_effect_protected(fighting)
    assert not BC._phantom_effect_protected(not_fighting)
    for protected in (mist, fighting):
        view = _protection_view([protected, safe])
        assert BC._guard_phantom_protected_target(
            view, np.asarray([8.0, 1.0, 0.0]), [0],
        ) == [1]


def test_phantom_protection_guard_preserves_dump_when_alternative_would_ko():
    view = _protection_view([
        _pokemon(BC.TEAM_ROCKET_ARTICUNO, 120),
        _pokemon(BC.DREEPY, 60),
    ])
    assert BC._guard_phantom_protected_target(
        view, np.asarray([8.0, 1.0, 0.0]), [0],
    ) == [0]


def test_phantom_protection_guard_does_not_implement_battle_cage():
    view = _protection_view([
        _pokemon(BC.DREEPY, 70), _pokemon(BC.MUNKIDORI, 110),
    ])
    view.current["stadium"] = [{"id": 1264, "playerIndex": 1}]
    assert BC._guard_phantom_protected_target(
        view, np.asarray([8.0, 1.0, 0.0]), [0],
    ) == [0]


def _battle_cage_view(*, opponent_bench=()):
    view = _main_view(
        active=_pokemon(BC.DRAGAPULT_EX, 320,
                        energy=(BC.FIRE_ENERGY, BC.PSYCHIC_ENERGY)),
        opponent_bench=opponent_bench,
        options=[
            {"type": OT_ATTACK, "attackId": BC.PHANTOM_DIVE},
            {"type": OT_PLAY, "index": 0},
            {"type": OT_END},
        ],
    )
    view.me["hand"] = [{"id": BC.JAMMING_TOWER}]
    view.current["stadium"] = [{"id": BC.BATTLE_CAGE, "playerIndex": 1}]
    return view


def test_battle_cage_guard_replaces_before_attack_or_end():
    view = _battle_cage_view()
    assert BC._guard_battle_cage_replacement(view, [0]) == [1]
    assert BC._guard_battle_cage_replacement(view, [2]) == [1]


def test_battle_cage_guard_preserves_free_setup_and_visible_spread():
    view = _battle_cage_view()
    view.options.append({"type": OT_ABILITY, "area": AREA_ACTIVE, "index": 0})
    assert BC._guard_battle_cage_replacement(view, [3]) == [3]
    for source in (
        _pokemon(BC.DRAGAPULT_EX, 320),
        _pokemon(BC.FROSLASS, 90),
        _pokemon(BC.MUNKIDORI, 110, energy=(BC.DARK_ENERGY,)),
    ):
        guarded = _battle_cage_view(opponent_bench=[source])
        assert BC._guard_battle_cage_replacement(guarded, [0]) == [0]


def test_battle_cage_guard_requires_cage_tower_and_phantom():
    view = _battle_cage_view()
    view.current["stadium"] = [{"id": BC.JAMMING_TOWER}]
    assert BC._guard_battle_cage_replacement(view, [0]) == [0]
    view = _battle_cage_view()
    view.options[0] = {"type": OT_ATTACK, "attackId": BC.JET_HEADBUTT}
    assert BC._guard_battle_cage_replacement(view, [0]) == [0]


def test_guard_is_scoped_to_dragapult_phantom_dive():
    assert BC._guard_phantom_dive_target(
        _view([0, 10], context=13), [0],
    ) == [0]
    assert BC._guard_phantom_dive_target(
        _view([0, 10], effect=120), [0],
    ) == [0]


def test_decide_applies_guard_after_card_head(monkeypatch):
    class _Net:
        def forward(self, _sample):
            return object(), None

    monkeypatch.setattr(BC, "_load_head", lambda _name: _Net())
    monkeypatch.setattr(
        BC.qu_v2_features, "encode_public_observation",
        lambda _obs, _deck: object(),
    )
    monkeypatch.setattr(BC.model, "decode_qu_v2", lambda *_args: [0])
    monkeypatch.setattr(BC, "ENABLE_PHANTOM_TARGET_GUARD", True)

    assert BC.decide(_view([0, 30, 10]), BC.TARGET_DECK) == [2]


def test_rejected_phantom_target_guard_is_disabled_by_default(monkeypatch):
    class _Net:
        def forward(self, _sample):
            return object(), None

    monkeypatch.setattr(BC, "_load_head", lambda _name: _Net())
    monkeypatch.setattr(
        BC.qu_v2_features, "encode_public_observation",
        lambda _obs, _deck: object(),
    )
    monkeypatch.setattr(BC.model, "decode_qu_v2", lambda *_args: [0])
    monkeypatch.setattr(BC, "ENABLE_PHANTOM_TARGET_GUARD", False)

    assert BC.decide(_view([0, 30, 10]), BC.TARGET_DECK) == [0]


def _energy_route_view(*, ready=True, backup_energy=(), deck_count=30,
                       include_recon=False, turn=3):
    active_energy = (
        (BC.FIRE_ENERGY, BC.PSYCHIC_ENERGY)
        if ready else (BC.FIRE_ENERGY,)
    )
    options = [
        {
            "type": OT_ATTACH,
            "index": 0,
            "inPlayArea": AREA_BENCH,
            "inPlayIndex": 0,
        },
        {
            "type": OT_ATTACH,
            "index": 1,
            "inPlayArea": AREA_BENCH,
            "inPlayIndex": 1,
        },
        {"type": OT_ATTACK, "attackId": BC.PHANTOM_DIVE},
        {"type": OT_END},
    ]
    if include_recon:
        options.append({"type": OT_ABILITY, "area": AREA_BENCH, "index": 1})
    return _main_view(
        active=_pokemon(BC.DRAGAPULT_EX, 320, energy=active_energy),
        bench=[
            _pokemon(BC.MUNKIDORI, 110),
            _pokemon(BC.DRAKLOAK, 90, energy=backup_energy),
        ],
        hand=[BC.DARK_ENERGY, BC.FIRE_ENERGY],
        options=options,
        deck_count=deck_count,
        turn=turn,
    )


def test_early_dark_to_munk_is_masked_until_backup_has_energy():
    view = _energy_route_view()
    logits = np.asarray([10.0, 9.0, 2.0, 1.0, 0.0], dtype=np.float32)
    assert BC._apply_main_route_guards(view, logits, [0]) == [1]


def test_dark_to_munk_is_allowed_after_attacker_and_backup_are_started():
    view = _energy_route_view(backup_energy=(BC.PSYCHIC_ENERGY,))
    logits = np.asarray([10.0, 9.0, 2.0, 1.0, 0.0], dtype=np.float32)
    assert BC._apply_main_route_guards(view, logits, [0]) == [0]


def test_backup_energy_does_not_unlock_munk_before_phantom_is_ready():
    view = _energy_route_view(
        ready=False, backup_energy=(BC.PSYCHIC_ENERGY,),
    )
    logits = np.asarray([10.0, 9.0, 2.0, 1.0, 0.0], dtype=np.float32)
    assert BC._apply_main_route_guards(view, logits, [0]) == [1]


def test_recon_is_preserved_and_used_as_safe_blocked_attachment_replacement():
    view = _energy_route_view(include_recon=True)
    logits = np.asarray(
        [10.0, 9.0, 8.0, 7.0, 6.0, 0.0], dtype=np.float32,
    )
    assert BC._apply_main_route_guards(view, logits, [2]) == [2]
    assert BC._apply_main_route_guards(view, logits, [4]) == [4]
    assert BC._apply_main_route_guards(view, logits, [0]) == [4]


def test_recon_guard_preserves_low_deck_attack_choice():
    view = _energy_route_view(deck_count=BC.RECON_SAFE_DECK_COUNT,
                              include_recon=True)
    logits = np.asarray(
        [10.0, 9.0, 8.0, 7.0, 6.0, 0.0], dtype=np.float32,
    )
    assert BC._apply_main_route_guards(view, logits, [2]) == [2]


def test_dark_to_munk_is_allowed_after_early_setup_window():
    view = _energy_route_view(turn=BC.EARLY_SETUP_LAST_TURN + 1)
    logits = np.asarray([10.0, 9.0, 2.0, 1.0, 0.0], dtype=np.float32)
    assert BC._apply_main_route_guards(view, logits, [0]) == [0]


def _completion_view(*, energy=(BC.FIRE_ENERGY,)):
    return _main_view(
        active=_pokemon(BC.DRAGAPULT_EX, 320, energy=energy),
        bench=[_pokemon(BC.DREEPY, 70), _pokemon(BC.MUNKIDORI, 110)],
        hand=[BC.PSYCHIC_ENERGY, BC.DARK_ENERGY],
        options=[
            {
                "type": OT_ATTACH,
                "index": 0,
                "inPlayArea": AREA_ACTIVE,
                "inPlayIndex": 0,
            },
            {
                "type": OT_ATTACH,
                "index": 0,
                "inPlayArea": AREA_BENCH,
                "inPlayIndex": 0,
            },
            {
                "type": OT_ATTACH,
                "index": 1,
                "inPlayArea": AREA_BENCH,
                "inPlayIndex": 1,
            },
            {"type": OT_ABILITY, "area": AREA_BENCH, "index": 0},
            {"type": OT_ATTACK, "attackId": BC.JET_HEADBUTT},
            {"type": OT_END},
        ],
    )


def test_phantom_completion_redirects_conflicting_commitments():
    view = _completion_view()
    assert BC._phantom_completion_options(view) == [0]
    assert BC._guard_phantom_completion(view, [1]) == [0]
    assert BC._guard_phantom_completion(view, [2]) == [0]
    assert BC._guard_phantom_completion(view, [4]) == [0]
    assert BC._guard_phantom_completion(view, [5]) == [0]


def test_phantom_completion_preserves_free_sequencing_and_completed_choice():
    view = _completion_view()
    assert BC._guard_phantom_completion(view, [0]) == [0]
    assert BC._guard_phantom_completion(view, [3]) == [3]


def test_phantom_completion_requires_exactly_one_missing_energy_type():
    assert BC._guard_phantom_completion(_completion_view(energy=()), [4]) == [4]
    assert BC._guard_phantom_completion(
        _completion_view(energy=(BC.FIRE_ENERGY, BC.PSYCHIC_ENERGY)), [4],
    ) == [4]


def _hammer_sequence_view():
    return _main_view(
        active=_pokemon(235, 30),
        bench=[_pokemon(BC.DREEPY, 70)],
        hand=[
            BC.CRUSHING_HAMMER, 1086, BC.ULTRA_BALL,
            BC.UNFAIR_STAMP, BC.JAMMING_TOWER, BC.DREEPY,
        ],
        options=[
            {"type": OT_PLAY, "index": 0},
            {"type": OT_PLAY, "index": 1},
            {"type": OT_PLAY, "index": 2},
            {"type": OT_PLAY, "index": 3},
            {"type": OT_PLAY, "index": 4},
            {"type": OT_PLAY, "index": 5},
            {"type": OT_EVOLVE, "index": 0, "area": AREA_BENCH, "inPlayIndex": 0},
            {"type": OT_ABILITY, "area": AREA_BENCH, "index": 0},
            {"type": OT_ATTACH, "index": 0, "area": AREA_BENCH, "inPlayIndex": 0},
            {"type": OT_ATTACK, "attackId": 323},
            {"type": OT_END},
        ],
    )


def test_hammer_sequence_delays_only_to_safe_setup():
    view = _hammer_sequence_view()
    logits = np.asarray([
        10.0, 4.0, 9.9, 9.8, 5.0, 6.0, 8.0, 7.0, 9.7, 9.6, 9.5, 0.0,
    ])
    # Ultra Ball, Stamp, attachment, attack and END all outrank the evolve,
    # but none may replace Hammer. The highest safe setup is the evolve.
    assert BC._guard_hammer_sequencing(view, logits, [0]) == [6]


def test_hammer_sequence_preserves_hammer_when_no_safe_setup_exists():
    view = _main_view(
        active=_pokemon(235, 30),
        hand=[BC.CRUSHING_HAMMER, BC.ULTRA_BALL, BC.UNFAIR_STAMP],
        options=[
            {"type": OT_PLAY, "index": 0},
            {"type": OT_PLAY, "index": 1},
            {"type": OT_PLAY, "index": 2},
            {"type": OT_ATTACK, "attackId": 323},
            {"type": OT_END},
        ],
    )
    logits = np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 0.0])
    assert BC._guard_hammer_sequencing(view, logits, [0]) == [0]


def _boss_setup_mate_main_view(*, prizes=1, target_hp=110,
                               active_hp=340):
    return _main_view(
        active=_pokemon(BC.DRAGAPULT_EX, 320, energy=(BC.FIRE_ENERGY,)),
        hand=[BC.BOSS, BC.PSYCHIC_ENERGY],
        opponent_active=_pokemon(678, active_hp),
        opponent_bench=[_pokemon(676, target_hp)],
        prizes=prizes,
        options=[
            {"type": OT_PLAY, "index": 0},
            {"type": OT_ATTACH, "index": 1, "area": AREA_ACTIVE,
             "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0,
             "playerIndex": 0},
            {"type": OT_ATTACK, "attackId": BC.JET_HEADBUTT},
            {"type": OT_END},
        ],
    )


def test_boss_setup_mate_forces_only_visible_final_prize_line():
    view = _boss_setup_mate_main_view()
    assert BC._boss_setup_mate_main(view) == [0]
    assert BC._boss_setup_mate_main(_boss_setup_mate_main_view(prizes=2)) is None
    assert BC._boss_setup_mate_main(
        _boss_setup_mate_main_view(target_hp=210),
    ) is None
    assert BC._boss_setup_mate_main(
        _boss_setup_mate_main_view(active_hp=190),
    ) is None


def test_boss_setup_mate_selects_target_before_completing_attachment():
    view = _main_view(
        active=_pokemon(BC.DRAGAPULT_EX, 320, energy=(BC.FIRE_ENERGY,)),
        hand=[BC.PSYCHIC_ENERGY],
        opponent_active=_pokemon(678, 340),
        opponent_bench=[_pokemon(676, 110), _pokemon(678, 340)],
        prizes=1,
        select_type=ST_CARD,
        context=CTX_SWITCH,
        effect=BC.BOSS,
        options=[
            {"type": 3, "area": AREA_BENCH, "index": 0,
             "playerIndex": 1},
            {"type": 3, "area": AREA_BENCH, "index": 1,
             "playerIndex": 1},
        ],
    )
    assert BC._boss_setup_mate_target(view) == [0]
    spent = _main_view(
        active=_pokemon(BC.DRAGAPULT_EX, 320, energy=(BC.FIRE_ENERGY,)),
        hand=[BC.PSYCHIC_ENERGY], opponent_active=_pokemon(678, 340),
        opponent_bench=[_pokemon(676, 110)], prizes=1,
        energy_attached=True, select_type=ST_CARD, context=CTX_SWITCH,
        effect=BC.BOSS,
        options=[{"type": 3, "area": AREA_BENCH, "index": 0,
                  "playerIndex": 1}],
    )
    assert BC._boss_setup_mate_target(spent) is None


def _boss_main_view(*, active_hp=250, bench=()):
    return _main_view(
        active=_pokemon(
            BC.DRAGAPULT_EX, 320,
            energy=(BC.FIRE_ENERGY, BC.PSYCHIC_ENERGY),
        ),
        hand=[BC.BOSS],
        opponent_active=_pokemon(200, active_hp),
        opponent_bench=bench or [_pokemon(24, 190)],
        options=[
            {"type": OT_PLAY, "index": 0},
            {"type": OT_ATTACK, "attackId": BC.PHANTOM_DIVE},
            {"type": OT_END},
        ],
    )


def test_boss_guard_fires_only_when_bench_ko_is_unavailable_on_active():
    assert BC._boss_immediate_prize_main(_boss_main_view()) == [0]
    assert BC._boss_immediate_prize_main(
        _boss_main_view(active_hp=180),
    ) is None
    assert BC._boss_immediate_prize_main(
        _boss_main_view(bench=[_pokemon(201, 220)]),
    ) is None


def test_boss_guard_ignores_ordinary_single_prizer_but_targets_engine():
    assert BC._boss_immediate_prize_main(
        _boss_main_view(bench=[_pokemon(201, 100)]),
    ) is None
    assert BC._boss_immediate_prize_main(
        _boss_main_view(bench=[_pokemon(BC.DRAKLOAK, 90)]),
    ) == [0]


def test_boss_target_prefers_higher_prize_reachable_pokemon():
    view = _main_view(
        active=_pokemon(
            BC.DRAGAPULT_EX, 320,
            energy=(BC.FIRE_ENERGY, BC.PSYCHIC_ENERGY),
        ),
        opponent_active=_pokemon(200, 250),
        opponent_bench=[_pokemon(201, 50), _pokemon(24, 190)],
        select_type=ST_CARD,
        context=CTX_SWITCH,
        effect=BC.BOSS,
        options=[
            {"type": 3, "area": AREA_BENCH, "index": 0, "playerIndex": 1},
            {"type": 3, "area": AREA_BENCH, "index": 1, "playerIndex": 1},
        ],
    )
    assert BC._boss_immediate_prize_target(view) == [1]


def test_decide_applies_boss_guard_before_loading_head(monkeypatch):
    monkeypatch.setattr(BC, "ENABLE_EXPERIMENTAL_ROUTE_GUARDS", True)
    monkeypatch.setattr(
        BC, "_load_head",
        lambda _name: (_ for _ in ()).throw(AssertionError("head must not load")),
    )
    assert BC.decide(_boss_main_view(), BC.TARGET_DECK) == [0]


def test_rejected_route_guards_are_disabled_in_shipped_decision_path(monkeypatch):
    class _Net:
        def forward(self, _sample):
            return object(), None

    monkeypatch.setattr(BC, "ENABLE_EXPERIMENTAL_ROUTE_GUARDS", False)
    monkeypatch.setattr(BC, "_load_head", lambda _name: _Net())
    monkeypatch.setattr(
        BC.qu_v2_features, "encode_public_observation",
        lambda _obs, _deck: object(),
    )
    monkeypatch.setattr(BC.model, "decode_qu_v2", lambda *_args: [1])

    assert BC.decide(_boss_main_view(), BC.TARGET_DECK) == [1]
