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
