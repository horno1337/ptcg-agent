"""Unit tests for the deterministic count-to-lethal helper (agent/lethal.py).

No engine, no net — pure obs-dict fixtures. Run: python tests/test_lethal.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent import lethal
from agent.obsview import ObsView, ST_ATTACK, ST_MAIN, OT_ATTACK, OT_END


def mk(hand, opp_active, select):
    """Minimal obs: us index 0, opponent index 1 with a given active entry."""
    return {
        "current": {
            "yourIndex": 0,
            "players": [
                {"handCount": hand, "active": [], "bench": []},
                {"active": [opp_active] if opp_active else [], "bench": []},
            ],
        },
        "select": select,
    }


def active(hp, cid=200, energies=None, energy_cards=None):
    e = {"id": cid, "hp": hp, "maxHp": hp}
    if energies is not None:
        e["energies"] = energies
    if energy_cards is not None:
        e["energyCards"] = energy_cards
    return e


ATTACK_SEL = {"type": ST_ATTACK, "option": [
    {"attackId": 999}, {"attackId": lethal.POWERFUL_HAND}]}
MAIN_SEL = {"type": ST_MAIN, "option": [
    {"type": OT_END}, {"type": OT_ATTACK}]}

n = 0


def check(cond, msg):
    global n
    assert cond, "FAIL: " + msg
    n += 1


def run():
    # --- lethal math: 20 * hand vs hp ---
    a = lethal.count_to_lethal(ObsView(mk(9, active(170), ATTACK_SEL)))
    check(a["need_for_ko"] == 9 and a["lethal_now"], "170hp needs 9, hand 9 -> lethal")

    a = lethal.count_to_lethal(ObsView(mk(8, active(170), ATTACK_SEL)))
    check(a["need_for_ko"] == 9 and not a["lethal_now"], "hand 8 < 9 -> not lethal")

    a = lethal.count_to_lethal(ObsView(mk(6, active(101), ATTACK_SEL)))
    check(a["need_for_ko"] == 6 and a["lethal_now"], "101hp -> ceil = 6 cards")

    # --- protection blocks the claim even with enough damage ---
    a = lethal.count_to_lethal(ObsView(mk(20, active(170, energies=[11]), ATTACK_SEL)))
    check(a["protected"] and not a["lethal_now"], "Mist Energy (11) blocks")

    a = lethal.count_to_lethal(ObsView(mk(20, active(170, energy_cards=[{"id": 20}]), ATTACK_SEL)))
    check(a["protected"] and not a["lethal_now"], "Rock Fighting Energy (20) blocks")

    a = lethal.count_to_lethal(ObsView(mk(20, active(120, cid=414), ATTACK_SEL)))
    check(a["protected"] and not a["lethal_now"], "Team Rocket's Articuno (414) blocks")

    # Cornerstone Ogerpon does NOT block
    a = lethal.count_to_lethal(ObsView(mk(20, active(210, cid=117), ATTACK_SEL)))
    check(not a["protected"] and a["lethal_now"], "Cornerstone Ogerpon (117) does not block")

    # --- unknowns -> conservative None ---
    check(lethal.count_to_lethal(ObsView(mk(9, None, ATTACK_SEL))) is None, "no active -> None")
    check(lethal.count_to_lethal(ObsView(mk(9, active(0), ATTACK_SEL))) is None, "hp 0 -> None")

    # --- override wiring ---
    check(lethal.attack_override(ObsView(mk(9, active(170), ATTACK_SEL))) == [1],
          "ST_ATTACK lethal -> picks Powerful Hand option (index 1)")
    check(lethal.attack_override(ObsView(mk(9, active(170), MAIN_SEL))) == [1],
          "ST_MAIN lethal -> routes into attack (index 1)")
    check(lethal.attack_override(ObsView(mk(8, active(170), ATTACK_SEL))) is None,
          "not lethal -> no override")
    check(lethal.attack_override(ObsView(mk(20, active(170, energies=[11]), ATTACK_SEL))) is None,
          "protected -> no override")

    print(f"all {n} lethal tests passed")


if __name__ == "__main__":
    run()
