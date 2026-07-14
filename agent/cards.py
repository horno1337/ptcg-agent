"""Card database: static lookups over the engine's card pool.

Data files (cards.json / attacks.json) are dumped from the engine itself via
tools/dump_cards.py, so they are always in sync with the installed
kaggle-environments version. They are shipped inside the submission bundle.
"""

import json
import os
from functools import lru_cache

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# CardType enum values (see cabt docs)
POKEMON, ITEM, TOOL, SUPPORTER, STADIUM, BASIC_ENERGY, SPECIAL_ENERGY = range(7)


@lru_cache(maxsize=1)
def card_db() -> dict[int, dict]:
    with open(os.path.join(_DATA_DIR, "cards.json")) as f:
        return {c["cardId"]: c for c in json.load(f)}


@lru_cache(maxsize=1)
def attack_db() -> dict[int, dict]:
    with open(os.path.join(_DATA_DIR, "attacks.json")) as f:
        return {a["attackId"]: a for a in json.load(f)}


def card(card_id: int | None) -> dict | None:
    if card_id is None:
        return None
    return card_db().get(card_id)


def attack(attack_id: int | None) -> dict | None:
    if attack_id is None:
        return None
    return attack_db().get(attack_id)


def is_pokemon(card_id: int) -> bool:
    c = card(card_id)
    return bool(c) and c["cardType"] == POKEMON


def is_basic_pokemon(card_id: int) -> bool:
    c = card(card_id)
    return bool(c) and c["cardType"] == POKEMON and c.get("basic", False)


def is_energy(card_id: int) -> bool:
    c = card(card_id)
    return bool(c) and c["cardType"] in (BASIC_ENERGY, SPECIAL_ENERGY)


def hp(card_id: int) -> int:
    c = card(card_id)
    return c["hp"] if c else 0


def max_attack_damage(card_id: int) -> int:
    """Best printed damage across a Pokemon's attacks (rough card strength proxy)."""
    c = card(card_id)
    if not c:
        return 0
    best = 0
    for aid in c.get("attacks", []):
        a = attack(aid)
        if a:
            best = max(best, a.get("damage", 0))
    return best
