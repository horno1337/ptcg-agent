# MD-v4 one-shot training and evaluation preregistration

Status: prospective and rejection-only. This document and
`lock_md_v4_training.py` must be committed before the official training lock is
created. Neither file authorizes production changes, packaging, or upload.

## Candidate

MD-v4 v1 is one exact-deck `ST_MAIN` overlay on complete frozen MD-v3. It adds
the already shadow-audited stateless public resource/current-log encoder and a
small zero-initialized residual. The MD-v3 state/option backbone, value head,
`ST_CARD` specialist, Qu-v2B residual route, deck, and fallback remain frozen.
At initialization, policy logits, decoded actions, and value must equal the
parent exactly.

There is one seed (`202607304`), one architecture, one training run, and one
candidate: the terminal checkpoint after epoch 4. Epochs 1–3 are recovery
checkpoints only. No validation-selected epoch, early stop, sweep, or rerun is
permitted. The official run is fixed to CUDA and to the canonical
`tools/checkpoints/md-v4-public-window-v1/{cache,model}` namespace; an alternate
device or output directory is not a second candidate path.

## Development population

The complete through-July-28 corpus is fixed by manifest, corpus-content, and
per-partition digests. Its existing split is preserved exactly:

- train: 18,793 games;
- validation: 2,090 games;
- test: empty.

Game UID and content overlap between train and validation must both be zero.
All valid exact-target-deck acting-seat `ST_MAIN` callbacks are eligible. The
natural matchup distribution is retained. Every example has weight 1.0 and
then receives only the deterministic inverse eligible-decision factor needed
to give each game total optimization weight 1; winners, mirrors, opponents,
ranks, sources, and days are not reweighted.

The first complete official daily dataset dated July 29 or later with zero UID
and content overlap is the deferred temporal cohort. Its inventory and hashes
must be locked before actions, rewards, or outcomes are opened. The fixed
candidate cannot be retrained or reselected after that point.

## Optimization

Only new MD-v4 parameters train. The optimizer is AdamW with learning rate
`3e-4`, weight decay `1e-4`, batch size 128, shuffle buffer 2,048, gradient
clip 1.0, and four fixed epochs. The loss is logged sequential policy NLL plus
`2.0 * KL(frozen parent || candidate)`. Value, entropy, PPO/policy-gradient,
auxiliary, winner, and reward terms are all zero.

Decision streaming uses a fixed, predeclared estimator for the equal-per-game
objective. If a split has `N` eligible callbacks and `G` games, callback `i`
from a game with `n_i` eligible callbacks receives factor `1/n_i`, and each
minibatch optimizes `sum_i[(1/n_i) * loss_i] * (N/G) / 128`. The denominator
stays 128 for the short final minibatch; the random minibatch weight sum is
never used as a denominator. Every split pass must end with exactly `N`
callbacks, `G` unique games, and total normalization mass `G`.

The epoch-4 checkpoint, NumPy weights, and self-hashed manifest are first
written and verified together in one private staging directory. Only that
complete directory may be atomically published, without replacement, as
`model/candidate-md-v4-final`. A partial staging directory and every recovery
checkpoint are ineligible as candidates.

The existing 20,883-shard Qu-v2A cache is reused only after its complete
21,916,387,995-byte inventory is bound by whole-file SHA-256 and every loaded
shard independently passes its internal header/payload hashes and
game/content/deck/source/parent checks. A new namespaced, pickle-free thin NPZ
cache stores only MD-v4 extension tensors and source-row indices; it does not
duplicate base tensors, labels, or parent logits. Its namespace binds the
corpus and partition, feature/model/trainer contracts, exact deck, `ST_MAIN`,
parent hashes, and frozen base-cache contract. Each thin shard is
split-specific, atomically published, mode `0600`, and cryptographically linked
to its base payload and the training lock. A stale or mismatched cache is
refused, not repaired or silently reused. The all-callback uncompressed upper
bound is 22,538,038,972 bytes; actual exact-deck `ST_MAIN` compressed use is
strictly smaller. Require 31,127,973,564 free bytes before materialization and
retain at least 8 GiB afterward. Capacity never justifies dropping examples.

## Offline rejection gates

Before training, initialization must be exactly parent-equivalent over every
validation example. After epoch 4:

- all four epochs completed with finite parameters and no parent mutation,
  private/future features, split overlap, pairing faults, non-unit raw
  outcome/matchup weights, or inverse eligible-decision normalization failures;
- game-normalized validation `KL(parent || candidate) <= 0.02`;
- deterministic full-action disagreement on validation `ST_MAIN` is at least
  3%; and
- at least 50% of validation games are touched by one disagreement.

These gates can only reject. They are not evidence of stronger play. Failure
retires this run without testing another epoch, seed, or threshold.

## Gameplay and promotion order

If all offline gates pass, run:

1. frozen-vs-frozen harness sanity with zero controller faults;
2. 2,560 exact-mirror games against complete frozen MD-v3: 1,280 fixed seeds,
   each run with physical seats swapped. Score is
   `(wins + 0.5 * draws) / 2560`; the 95% Wilson lower bound must be strictly
   above 50%, with zero fallbacks, repairs, exceptions, timeouts, invalid games,
   or artifact mismatches;
3. a separately locked 1,280-game-per-arm refreshed recent-frequency field A/B
   on an identical schedule. Candidate-control point delta must be non-negative
   and the conservative independent Wilson-difference lower bound must exceed
   −5 percentage points;
4. rejection-only corroboration on the untouched temporal cohort; and
5. safety tests, 200 fault-free random smoke games, and an exact-extracted
   tarball audit under a non-owner UID.

A failed stage stops all later stages. Offline or temporal metrics cannot rescue
a gameplay failure. No production file is edited and no package is uploaded
without every gate passing and the user explicitly naming the submission and
approving the FIFO retirement risk.
