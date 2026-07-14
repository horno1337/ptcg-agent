"""Lightweight read-only view over the raw observation dict.

The engine gives plain dicts; this wraps the common access patterns so the
policy code stays readable. No copying, no mutation.

Enum constants mirror the cabt docs (SelectType / SelectContext / OptionType /
AreaType). Keep them here as ints so there is zero import cost at runtime.
"""

# --- SelectType ---
ST_MAIN, ST_CARD, ST_ATTACHED_CARD, ST_CARD_OR_ATTACHED, ST_ENERGY, ST_SKILL, \
    ST_ATTACK, ST_EVOLVE, ST_COUNT, ST_YES_NO, ST_SPECIAL_CONDITION = range(11)

# --- OptionType ---
OT_NUMBER, OT_YES, OT_NO, OT_CARD, OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY, \
    OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ABILITY, OT_DISCARD, OT_RETREAT, \
    OT_ATTACK, OT_END, OT_SKILL, OT_SPECIAL_CONDITION = range(17)

# --- AreaType ---
AREA_DECK, AREA_HAND, AREA_DISCARD, AREA_ACTIVE, AREA_BENCH, AREA_PRIZE, \
    AREA_STADIUM, AREA_ENERGY, AREA_TOOL, AREA_PRE_EVOLUTION, AREA_PLAYER, \
    AREA_LOOKING = range(1, 13)

# --- SelectContext (subset we branch on) ---
CTX_MAIN = 0
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH = 2
CTX_SWITCH = 3
CTX_TO_ACTIVE = 4
CTX_TO_BENCH = 5
CTX_TO_FIELD = 6
CTX_TO_HAND = 7
CTX_DISCARD = 8
CTX_DAMAGE_COUNTER = 13
CTX_DAMAGE_COUNTER_ANY = 14
CTX_DAMAGE = 15
CTX_REMOVE_DAMAGE_COUNTER = 16
CTX_HEAL = 17
CTX_EVOLVES_FROM = 18
CTX_EVOLVES_TO = 19
CTX_ATTACH_FROM = 21
CTX_ATTACH_TO = 22
CTX_EFFECT_TARGET = 25
CTX_ATTACK = 35
CTX_DRAW_COUNT = 38
CTX_IS_FIRST = 41
CTX_MULLIGAN = 42
CTX_ACTIVATE = 43
CTX_COIN_HEAD = 46


class ObsView:
    def __init__(self, obs: dict):
        self.obs = obs
        self.current = obs.get("current")
        self.select = obs.get("select")

    # --- phases ---
    @property
    def is_deck_selection(self) -> bool:
        return self.select is None

    # --- select accessors ---
    @property
    def options(self) -> list[dict]:
        return self.select.get("option", []) if self.select else []

    @property
    def select_type(self) -> int:
        return self.select.get("type", -1) if self.select else -1

    @property
    def context(self) -> int:
        return self.select.get("context", -1) if self.select else -1

    @property
    def min_count(self) -> int:
        return self.select.get("minCount", 1) if self.select else 0

    @property
    def max_count(self) -> int:
        return self.select.get("maxCount", 1) if self.select else 0

    # --- state accessors ---
    @property
    def my_index(self) -> int:
        return self.current["yourIndex"] if self.current else 0

    @property
    def me(self) -> dict | None:
        if not self.current:
            return None
        return self.current["players"][self.my_index]

    @property
    def opp(self) -> dict | None:
        if not self.current:
            return None
        return self.current["players"][1 - self.my_index]

    @property
    def turn(self) -> int:
        return self.current.get("turn", 0) if self.current else 0

    def option_card_id(self, opt: dict) -> int | None:
        """Best-effort card id referenced by an option (may be None)."""
        return opt.get("cardId")

    def hand_card_id(self, hand_index: int) -> int | None:
        me = self.me
        if not me:
            return None
        hand = me.get("hand") or []
        if 0 <= hand_index < len(hand):
            c = hand[hand_index]
            return c.get("id") if isinstance(c, dict) else None
        return None
