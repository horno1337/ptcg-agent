# Leaf-evaluator comparison — preregistration

Written before any leaf was scored. Commits `aa278e3`, `29e6f75`, `ff7be15`.

## Question

**Which evaluator ranks root actions better _within the heuristic-selected
beam_?**

This is deliberately NOT a general evaluator comparison. Collected leaves are
the winners of beams that `evaluate_turn` itself ranked and pruned
(`_plan_rank` sorts `expanded` before truncating to `BEAM_WIDTH`). The leaf
population is therefore selected by the incumbent, which structurally favours
it. Escaping that bias requires re-running the beam per arm, which is not in
scope. A result here licenses only: "given the heuristic's beam, evaluator X
ranks the root actions better." It does not license "X is a better evaluator."

## Arms

1. `heuristic` — `evaluate_turn`, unchanged incumbent.
2. `qu_v2b` — value head of packaged `weights.npz`
   (`value1_weight|bias`, `value2_weight|bias`, sha `007c09ef914404f3`).
   Shared byte-identically by both CARD nets.
3. `md_v1` — value head of `md_v1_weights.npz` (sha `85ad6980e0fb8b53`).
4. `blend` — root-relative, discovery-calibrated, then frozen.

Value is state-level: every arm scores every leaf. Arms are NOT selected by
the leaf's prompt routing, and a turn-boundary leaf need not carry a MAIN or
CARD prompt at all.

## Normalization

Neural values are bounded near +/-1; heuristic leaves span roughly +/-25
because the prize term alone is `2.50 * (opp_prizes - my_prizes)`. A raw
weighted average would be ~96% heuristic and would measure almost nothing.
All arms are therefore compared on **root-relative action deltas**: per root,
subtract the mean across actions, then rank. Blend weights are fitted on
discovery roots only and frozen before confirmation is opened.

## Primary and secondary outcomes

- **Primary: non-mirror roots.** Grim is ~6.5% of current-field seats, so
  non-mirror is the deployment-relevant population.
- **Secondary: Grim mirror roots**, reported separately, never pooled.

**Preregistered directional prediction.** `md_v1`'s value was shaped by
300k steps of Grim mirror self-play (`dobi-v1-mirror-league-300k`,
`value_coefficient: 0.5`) and then frozen by BC specialization at 0.0.
We predict *in advance* that if `md_v1` wins anywhere it wins in the mirror
slice, and that it underperforms `qu_v2b` on non-mirror roots. A mirror-only
win is the outcome its training history predicts and must not be reported as
a general result.

## Labels

Paired terminal rollouts at learner-reached roots. Retain only action pairs
with `abs(delta) > 1.96 * paired_SE`; unresolved pairs are discarded, not
counted as ties. Rationale: this project's two previous critics scored 0.414
and 0.497 pairwise concordance — at or below chance — and the repeated-panel
diagnostic failed its all-pair sign gate at 0.668 versus 0.70. Dense action
values here are known to be noisy; the confidence filter is the one labelling
protocol with a demonstrated repeatability number (115/127 signs preserved).

## Discipline

- Discovery and confirmation roots are **episode-disjoint**, split before
  outcomes are opened.
- Blend parameters are fitted on discovery only and frozen; confirmation is
  opened once.
- Only roots passing `complete_roots()` are scoreable.
- Pilot agreement is **not** an outcome. It may be rerun as a sanity check
  after the fact; it can neither promote nor veto an arm.

## Gate on integration

A confirmed arm authorizes only: re-measure coverage, re-run the post-fix
3-6s budget sweep, refreeze the budget, then run the 4,096/8,192 gameplay
gate with the registered futility stop. It does not authorize packaging or
upload.

## Known confounds

- Leaf population selected by the incumbent (see Question).
- The 5.0s freeze (`5f3c6131e0f5a637e5f2177e6c55b765`) is **stale**: it
  predates per-particle deck binding, which changes opponent scoring in every
  particle. Coverage must be re-measured before any budget is refrozen.
- Collection-on vs collection-off coverage is unverified; if collection costs
  measurable coverage, the comparison is contaminated and must be rerun.

---

# Appendix — specifications completed before any leaf was scored

## A1. Normalization

Root-centering alone does not fix the ~25x scale gap. Per root `r` and
evaluator `e`, over that root's `n_root` actions:

```
c_e(r,a) = s_e(r,a) - mean_a s_e(r,a)
rms_e(r) = sqrt( mean_a c_e(r,a)^2 )          # population, not sample
z_e(r,a) = c_e(r,a) / rms_e(r)
```

`z` is therefore dimensionless and unit-variance per root, so heuristic and
neural arms are directly comparable and blendable.

**Degenerate roots.** If `rms_e(r) < 1e-9` for ANY arm, the root is dropped
from EVERY arm. Dropping per-arm would give each arm a different population
and silently favour whichever arm is least often flat. The dropped count is
reported per arm and in total.

## A2. Orientation

Each leaf record carries `leaf_seat` and `root_player`. Neural value is
oriented to the root player:

```
v_root = v if leaf_seat == root_player else -v
```

**This convention is verified, not assumed.** Leaves with
`current.result != -1` have a known winner. Before any arm is scored, the
oriented value is checked against that known outcome on terminal leaves; the
run aborts loudly if mean oriented value on root-player wins is not positive
and on losses not negative. An inverted sign convention would otherwise
produce a confident, exactly-wrong evaluator.

## A3. Blend family and frozen grid

```
z_blend(r,a) = (1 - L) * z_heuristic(r,a) + L * z_neural(r,a)
L in {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0}      # frozen, 9 points
neural in {qu_v2b, md_v1}                                        # frozen, 2 heads
```

The grid subsumes the pure arms: `L=0` is the heuristic, `L=1` the pure
neural head. Total discovery configurations: 17 distinct (`L=0` is shared).

## A4. Discovery selection rule

Select ONE challenger by maximizing concordance on **discovery non-mirror
pairs**. Deterministic tie-break, applied in order:

1. higher discovery non-mirror concordance;
2. smaller `L` (ties resolve toward the incumbent);
3. head order `qu_v2b` before `md_v1`.

The selected `(head, L)` is written to `frozen-challenger.json` with a hash
before confirmation is opened. Exactly one challenger is carried forward;
confirmation never re-selects.

## A5. Ranking statistic and inference

- **Pairs.** Learner-reached roots, paired terminal rollouts per action.
  Retain a pair only if `abs(delta) > 1.96 * paired_SE`. Unresolved pairs are
  discarded, never counted as ties.
- **Statistic.** Concordance `C_e` = fraction of retained pairs where
  `sign(z_e(r,a) - z_e(r,b))` matches the sign of the terminal delta. Null is
  0.50.
- **Primary quantity.** Paired difference on identical pairs:
  `D = C_challenger - C_heuristic`.
- **Clustering unit: the episode.** Pairs nest in roots and roots nest in
  episodes, so neither pairs nor roots are independent. 10,000-resample
  cluster bootstrap over episodes, seed 20260813, percentile 95% interval.
- **Minimum usable evidence** (each of discovery and confirmation, non-mirror):
  `>= 300` retained pairs across `>= 60` distinct episodes, and `>= 150`
  usable roots. Below any of these the stage is **invalid** — it is re-collected,
  not reported as a failure. At the ~4 confirmed pairs/game yield this implies
  roughly 75 games minimum, 150 target, per stage.
- **Confirmation pass threshold.** Pass iff the episode-clustered 95% lower
  bound on `D` is **strictly greater than 0** on non-mirror confirmation
  pairs. A positive point estimate with a lower bound at or below 0 is a
  fail, not a partial pass.
- **Mirror slice.** Reported separately with its own interval. It can neither
  promote nor veto. Its only preregistered role is testing the A4 prediction
  about MD-v1's mirror-PPO lineage.

## A6. Collection-on vs collection-off coverage

Runs **before** the scorer, since contaminated leaves would invalidate all
downstream work.

- 128 games per arm, identical seed and schedule, one arm with
  `collect_leaves` active and one without.
- **Pass:** coverage delta within `+/-1.0 pp` and search-seconds/game within
  2%, with the 95% interval contained in `+/-2.0 pp`.
- **Ambiguous** (point estimate inside tolerance, interval not contained):
  extend ONCE by a blind `64 + 64` games per arm and re-evaluate on the
  pooled 192. No further extension; a still-ambiguous result fails.
- **Fail:** collection is made cheaper or moved fully out-of-process before
  any leaf is scored.
