# Turn-search gate — FUTILITY STOP

## Result

```
candidate  3357W-733L-6D   82.03%      (frozen Dobi-v2 + belief turn search, 7.0s)
control    3419W-672L-5D   83.53%      (packaged frozen Dobi-v2 router, unchanged)

delta  -1.50 pp   CI95 [-3.05,+0.05]   SE 0.790 pp   n=4096/arm
```

**Preregistered futility boundary: stop if delta < -1.00 pp. Observed -1.50 pp.
The gate STOPS. The run does not extend to 8,192.**

## The gate was valid

All 16 shards of both arms passed `arm_valid`. Zero agent, engine,
infrastructure, truncation, overlay and field faults. Field: 0 exceptions,
0 fallbacks. `unknown_opponent` 0.00%. Budget assertion held at 7.0s.

The planner was genuinely working, which is what makes this informative:

| | |
|---|---|
| searched roots | 155,357 |
| robust overrides | 16,411 (10.56%) |
| evidence coverage | 88.3% |
| control searched | 0 (correct: unchanged router) |

This is not the earlier failure mode of a planner that never fired. Search
made 16,411 real decision changes across 4,096 games and lost 1.50 pp doing it.

## Reading

The CI upper bound touches +0.05, so the effect is not decisively negative in
interval terms. That does not reopen the question: the boundary was
preregistered on the point estimate, before any gameplay was observed,
precisely so a near-miss interval could not be used to argue for continuing.
Re-reading it now as "inconclusive, therefore extend" is the post-hoc rescue
the preregistration exists to prevent.

## What this does and does not establish

Establishes: at the frozen operating point, on the frozen current field,
belief turn search wrapping frozen Dobi-v2 does not beat frozen Dobi-v2, and
is more likely mildly harmful than helpful.

Does NOT establish which component is responsible. Leaf evaluation, belief
reconstruction, the synchronized beam and the override gate (mean margin plus
unanimity) compose into one number. The evaluator is in the causal path of
every override and is a leading suspect, but this gate does not isolate it.

Does NOT establish anything about ladder strength, other decks, other budgets,
or a different evaluator.

## Disposition

- Do not integrate, package or upload the planner.
- Do not retune the budget, beam width, evidence floor or override margin on
  this opened result.
- Frozen Dobi-v2 remains the authoritative Grimmsnarl agent, unchanged.
- The infrastructure is retained: seat-aware policy, per-particle belief
  binding, packaged-runtime field, transactional leaf collection and the hard
  preflight are all reusable and independently verified.
- The evaluator comparison (`turn-search-evaluator-20260813`, Appendix C) is
  now the highest-value diagnostic route rather than a parked curiosity.
