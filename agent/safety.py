"""Safety wrapper: the ladder contract is "never crash, never time out".

Everything the policy returns is validated here:
  * exception in policy       -> legal fallback
  * wrong length / bad index  -> repaired to a legal selection
  * time budget nearly spent  -> skip policy, answer instantly

An invalid action or a crash is an instant game loss on the ladder, so this
module is the one place where paranoia is the point.
"""

import time
from . import policy

_TOTAL_BUDGET_S = 600.0        # 10 min per player per game
_PANIC_RESERVE_S = 30.0        # once this little remains, stop thinking
_spent = 0.0


def _fallback(obs: dict) -> list[int]:
    select = obs.get("select")
    if select is None:
        return policy.load_deck()
    options = select.get("option", [])
    max_count = select.get("maxCount", 1)
    # mirror the organizer's `first_agent`, which is known-legal
    return list(range(min(max_count, max(len(options), 1)) if options else max_count))


def _repair(action, obs: dict) -> list[int]:
    select = obs.get("select")
    if select is None:
        if isinstance(action, list) and len(action) == 60:
            return action
        return policy.load_deck()

    options = select.get("option", [])
    n = len(options)
    min_c = select.get("minCount", 1)
    max_c = select.get("maxCount", 1)

    if not isinstance(action, list):
        return _fallback(obs)
    # dedupe, keep order, drop out-of-range
    seen, cleaned = set(), []
    for a in action:
        if isinstance(a, int) and 0 <= a < n and a not in seen:
            seen.add(a)
            cleaned.append(a)
    # pad up to min_count with unused indices
    if len(cleaned) < min_c:
        for i in range(n):
            if i not in seen:
                cleaned.append(i)
                seen.add(i)
            if len(cleaned) >= min_c:
                break
    return cleaned[:max_c] if max_c > 0 else cleaned


def agent(obs: dict) -> list[int]:
    global _spent
    t0 = time.monotonic()
    try:
        if _spent > _TOTAL_BUDGET_S - _PANIC_RESERVE_S:
            action = _fallback(obs)
        else:
            action = policy.decide(obs)
        action = _repair(action, obs)
    except Exception:
        try:
            action = _fallback(obs)
        except Exception:
            action = [0]
    finally:
        _spent += time.monotonic() - t0
    return action
