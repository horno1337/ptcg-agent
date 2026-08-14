# Turn-search gameplay gate — preregistration

Written before any gate game was played. Adapter identity binds
`freeze.json` by hash, so a shard cannot be resumed under another budget.

## Question

Does Dobi-aware belief turn search beat the frozen Dobi-v2 router it wraps,
on the frozen current-field schedule?

- **Candidate:** frozen Dobi-v2 + planner at the frozen operating point.
- **Control:** the packaged frozen Dobi-v2 router, unchanged — the agent that
  ships, not candidate code with search switched off.

## Frozen operating point

| | |
|---|---|
| budget | **7.0 s** (`FROZEN_BUDGET_S`) |
| particles | 8 |
| evidence floor | 5 |
| topology | 8 workers x 2 threads |
| freeze | `1d5bc5cfcb86f1cf888881d260d8ff77ff5d1b7ea10a082415e0d0731db62def` |
| freeze file | `freeze.json`, sha256 `aa105a2145155e5c…cea8a` |

Selected outcome-blind from a 4/5/6/7/8 s sweep; plateau `[7.0, 8.0]`,
smallest within 2 pp of best coverage. **The freeze artifact stores
`plateau_budgets`, not a scalar budget**, so the authoritative scalar is
recorded here: **7.0**. `arm_valid` asserts `budget_s == 7.0`.

## Design

- **Seed 20260821**, fixed here before any gameplay.
- **4,096 games/arm.** This is the futility interim, not the full gate.
- **Futility boundary: stop if candidate − control < −1.00 pp.**
- **No automatic extension to 8,192.** Surviving the interim authorises
  nothing by itself; continuing requires a fresh explicit decision.
- Promotion requires the full-sample 8,192 interval. The 4,096 point estimate
  is never promotion evidence, whatever its sign.

## Validity

Every shard of both arms must satisfy `arm_valid`: frozen budget asserted,
packaged-runtime field with zero fallbacks and zero exceptions, zero learner
exceptions and overlay faults, zero `disabled`, searched/evidence-floor roots
and overrides all > 0, `unknown_opponent` below the locked 0.35 ceiling.
Any failure invalidates the gate rather than producing a result.

## Known limits

- Local current-field win rate is not ladder strength.
- The planner's leaf evaluator is a nine-term hand-weighted heuristic whose
  quality is untested; the evaluator comparison is parked at Appendix C of
  `turn-search-evaluator-20260813`.
- 8-game preflight win rates carry no evidential value and are not reported
  as results.
