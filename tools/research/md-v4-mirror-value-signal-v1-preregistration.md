# MD-v4 mirror value-signal v1 preregistration

Date locked: 2026-07-30, before computing any probe outcome.

## Question

Can the public, actor-relative MD-v4 resource state predict long-horizon
Grimmsnarl-mirror outcomes well enough to justify a learned leaf evaluator for
paired-belief turn search?

This is a feasibility screen, not a model candidate and not promotion
evidence. It cannot authorize an upload.

## Fixed cohort

- Use only the already locked through-July-28 MD-v4 development corpus.
- Use exactly the existing game-identity-disjoint train/validation split.
- Include only exact-list Grimmsnarl mirrors:
  3,110 training games and 350 validation games.
- July 29 and all later episodes remain unopened.
- Each resolved seat-game has equal weight; its eligible `ST_MAIN` decisions
  divide that weight equally. Draws may train the regressor but are excluded
  from binary AUC.

## Fixed representation and probe

Freeze the exact epoch-4 explicit-FP32 MD-v4 actor. For every eligible
`ST_MAIN` callback, store the detached 192-value representation already
defined for resource PPO: frozen MD-v3 state `[160]` concatenated with MD-v4
resource/history fusion `[32]`. No action, future state, hidden information,
team identity, rating, or date is an input.

Standardize with training-only mean and variance. Train three fixed
`Linear(192,64)-ReLU-Linear(64,1)-tanh` probes with seeds
`202607301`, `202607302`, and `202607303`, Adam learning rate `1e-3`,
weight decay `1e-4`, batch size `1024`, and 12 terminal epochs. There is no
epoch, seed, or hyperparameter selection. The three predictions are averaged.
The fixed terminal probe mappings and training-only normalization are written
once before the result is reported, regardless of whether the screen passes.

The frozen MD-v3 value output on the identical callbacks is the baseline.

## Fixed metrics and decision rule

Measure on all 350 validation mirror games:

1. seat-game-normalized weighted AUC over resolved callbacks;
2. the same AUC over the first half of each resolved seat-game's eligible
   callbacks, preventing late-game inevitability from carrying the screen;
3. seat-game-normalized Brier score after mapping signed values and rewards
   from `[-1,1]` to probabilities and binary labels in `[0,1]`; and
4. the same three metrics for frozen MD-v3.

The value-guided search route advances only if all are true:

- ensemble validation AUC is at least `0.60`;
- ensemble early-half validation AUC is at least `0.56`;
- ensemble Brier score is at least 5% lower than frozen MD-v3's Brier score;
- every individual seed has validation AUC at least `0.56`; and
- all rows, games, labels, representations, and predictions are finite and the
  cohort counts match 3,110/350 exactly.

Failure retires the learned-value search route. Thresholds will not be lowered,
seeds dropped, or strata changed after results are read.

## If the screen passes

Create a separate prospective gameplay experiment. Use the frozen ensemble
only as a leaf evaluator in semantic-action, same-particle paired-belief
end-of-turn search, initially restricted to exact-list Grimmsnarl mirrors.
The runtime must remain fail-soft and must demonstrate a material action-change
rate before a large gameplay A/B. The old deterministic-heuristic PIMC result,
the failed turn-search distillation, and this feasibility screen cannot be
pooled as evidence.
