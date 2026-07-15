"""ctypes bindings + local battle runner over the official cabt engine.

Build engine/libcg.so first (tools/build_engine.sh); the engine source is
competition-use-only and lives outside the repo.

The runner mimics the kaggle-environments contract so `agent(obs) -> list[int]`
code runs unchanged:
  * first call per agent has obs["select"] is None -> return 60 card ids
  * then option-index selections against obs["select"]["option"]
  * an invalid selection loses the game on the spot, same as the ladder
"""
import ctypes
import json
import os
import random

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "libcg.so")
_SAMPLE_DECK_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "decks", "sample.csv")

DECK_SIZE = 60
# BattleStart deck-validation errors (Api.h)
DECK_ERRORS = {1: "unknown card id", 2: "more than 4 copies of a name",
               3: "no basic pokemon", 4: "duplicate ACE SPEC"}


class _StartData(ctypes.Structure):
    _fields_ = [("battlePtr", ctypes.c_void_p),
                ("errorPlayer", ctypes.c_int),
                ("errorType", ctypes.c_int)]


class _SerialData(ctypes.Structure):
    _fields_ = [("json", ctypes.c_char_p),
                ("data", ctypes.c_void_p),   # base64 state, NOT null-terminated: use count
                ("count", ctypes.c_int),
                ("selectPlayer", ctypes.c_int)]


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(f"{_LIB_PATH} missing - run tools/build_engine.sh")
        L = ctypes.CDLL(_LIB_PATH)
        L.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
        L.BattleStart.restype = _StartData
        L.GetBattleData.argtypes = [ctypes.c_void_p]
        L.GetBattleData.restype = _SerialData
        L.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.Select.restype = ctypes.c_int
        L.VisualizeData.argtypes = [ctypes.c_void_p]
        L.VisualizeData.restype = ctypes.c_char_p
        L.BattleFinish.argtypes = [ctypes.c_void_p]
        L.AllCard.restype = ctypes.c_char_p
        L.AllAttack.restype = ctypes.c_char_p
        # agent-side search API (SearchBegin/SearchStep over a serialized obs)
        L.AgentStart.restype = ctypes.c_void_p
        L.SearchBegin.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        L.SearchBegin.restype = ctypes.c_char_p
        L.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                 ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.SearchStep.restype = ctypes.c_char_p
        L.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
        L.SearchEnd.argtypes = [ctypes.c_void_p]
        import hashlib
        flag = "_PTCG_INIT_" + hashlib.md5(
            os.path.realpath(_LIB_PATH).encode()).hexdigest()[:12]
        if os.environ.get(flag) != str(os.getpid()):
            L.GameInitialize()
            os.environ[flag] = str(os.getpid())
        _lib = L
    return _lib


class AgentSearch:
    """Agent-side determinized search over a serialized observation.

    Mirrors the official cg.api search wrappers: begin() builds a concrete
    world from our predictions of every hidden zone; step() advances it and
    returns the next (obs, search_id). Errors return None instead of raising
    so callers can fall back to reflex play.
    """

    def __init__(self):
        self._ptr = lib().AgentStart()

    @staticmethod
    def _arr(xs):
        return (ctypes.c_int * max(len(xs), 1))(*xs)

    def begin(self, obs: dict, my_deck, my_prize, opp_deck, opp_prize,
              opp_hand, opp_active=(), manual_coin=False):
        sbi = obs.get("search_begin_input")
        if not sbi:
            return None
        raw = lib().SearchBegin(
            self._ptr, sbi.encode("ascii"), len(sbi),
            self._arr(list(my_deck)), self._arr(list(my_prize)),
            self._arr(list(opp_deck)), self._arr(list(opp_prize)),
            self._arr(list(opp_hand)), self._arr(list(opp_active)),
            int(manual_coin))
        return self._parse(raw)

    def step(self, search_id: int, select: list[int]):
        raw = lib().SearchStep(self._ptr, search_id,
                               self._arr(list(select)), len(select))
        return self._parse(raw)

    def release(self, search_id: int):
        lib().SearchRelease(self._ptr, search_id)

    def end(self):
        lib().SearchEnd(self._ptr)

    @staticmethod
    def _parse(raw):
        try:
            r = json.loads(raw.decode()) if isinstance(raw, bytes) else json.loads(raw)
        except Exception:
            return None
        if r.get("error"):
            return None
        return r.get("state")  # {"observation": {...}, "searchId": int}


class DeckError(ValueError):
    def __init__(self, player: int, error_type: int):
        self.player, self.error_type = player, error_type
        super().__init__(f"deck of player {player} rejected: "
                         f"{DECK_ERRORS.get(error_type, error_type)}")


class Battle:
    """One engine battle. Use as a context manager or call close()."""

    def __init__(self, deck0: list[int], deck1: list[int]):
        if len(deck0) != DECK_SIZE or len(deck1) != DECK_SIZE:
            raise DeckError(0 if len(deck0) != DECK_SIZE else 1, 1)
        arr = (ctypes.c_int * (2 * DECK_SIZE))(*deck0, *deck1)
        sd = lib().BattleStart(arr)
        if not sd.battlePtr:
            raise DeckError(sd.errorPlayer, sd.errorType)
        self._ptr = sd.battlePtr

    def obs(self) -> tuple[dict, int]:
        """Current observation (for the selecting player) and that player's index."""
        sd = lib().GetBattleData(self._ptr)
        o = json.loads(ctypes.string_at(sd.json).decode())
        if sd.data and sd.count > 0:
            # same field the kaggle wrapper provides: input for SearchBegin
            o["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
        return o, sd.selectPlayer

    def select(self, indices: list[int]) -> int:
        """Apply a selection; returns 0 on success, engine error code otherwise."""
        arr = (ctypes.c_int * max(len(indices), 1))(*indices)
        return lib().Select(self._ptr, arr, len(indices))

    def visualize(self) -> str:
        """Replay JSON (array of per-select vis states) for the whole battle so far."""
        return ctypes.string_at(lib().VisualizeData(self._ptr)).decode()

    def close(self):
        if self._ptr:
            lib().BattleFinish(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Battle runner (kaggle-compatible agent contract)
# ---------------------------------------------------------------------------

_DECK_REGISTRATION_OBS = {"select": None, "current": None, "logs": []}
_MAX_SELECTS = 5000  # hard cap; the engine's own turn limit should end games first


def run_battle(agent0, agent1, collect_replay: bool = False) -> dict:
    """Run one battle. Returns:
    {"result": 0|1|2, "selects": int, "error": None | (player, description),
     "replay": str|None}
    result is the winning player index, 2 = draw. An agent crash or illegal
    selection loses immediately, mirroring the ladder rules.
    """
    agents = (agent0, agent1)
    decks = []
    for p, a in enumerate(agents):
        try:
            decks.append(list(a(dict(_DECK_REGISTRATION_OBS))))
        except Exception as e:
            return {"result": 1 - p, "selects": 0, "replay": None,
                    "error": (p, f"deck registration crashed: {e!r}")}

    try:
        battle = Battle(decks[0], decks[1])
    except DeckError as e:
        return {"result": 1 - e.player, "selects": 0, "replay": None,
                "error": (e.player, str(e))}

    with battle:
        for n in range(_MAX_SELECTS):
            obs, sp = battle.obs()
            if obs["current"]["result"] != -1:
                return {"result": obs["current"]["result"], "selects": n,
                        "error": None,
                        "replay": battle.visualize() if collect_replay else None}
            obs["remainingOverageTime"] = 600
            try:
                action = agents[sp](obs)
            except Exception as e:
                return {"result": 1 - sp, "selects": n, "replay": None,
                        "error": (sp, f"agent crashed: {e!r}")}
            err = battle.select(list(action))
            if err:
                return {"result": 1 - sp, "selects": n, "replay": None,
                        "error": (sp, f"illegal selection {action} (engine code {err})")}
        return {"result": 2, "selects": _MAX_SELECTS, "replay": None,
                "error": (None, "select cap reached")}


# ---------------------------------------------------------------------------
# Baseline agents (mirror kaggle_environments.envs.cabt baselines)
# ---------------------------------------------------------------------------

def _sample_deck() -> list[int]:
    with open(_SAMPLE_DECK_CSV) as f:
        return [int(line) for line in f if line.strip()]


def first_agent(obs: dict) -> list[int]:
    if obs["select"] is None:
        return _sample_deck()
    n = len(obs["select"]["option"])
    return list(range(min(obs["select"].get("minCount", 1) or 1, n)))


def random_agent(obs: dict) -> list[int]:
    if obs["select"] is None:
        return _sample_deck()
    sel = obs["select"]
    n = len(sel["option"])
    lo = max(min(sel.get("minCount", 1), n), 0)
    hi = max(min(sel.get("maxCount", 1), n), lo)
    k = random.randint(lo, hi) if hi > lo else lo
    return random.sample(range(n), k)
