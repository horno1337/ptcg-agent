# MD-v4 epoch-4 runtime-parity salvage preregistration

Status: prospective and rejection-only. This document does not reverse the
failure of MD-v4 public-window v1 and does not authorize a production edit,
package, tag, or upload.

## Incident and immutable candidate

The one allowed MD-v4 training run completed its fixed fourth epoch but failed
the locked Torch/NumPy parity gate before a final candidate bundle was
published. MD-v4 public-window v1 therefore remains failed under
`md-v4-training-preregistration.md`.

This salvage may use only the recovery checkpoint already emitted by that
failed run:

- path:
  `tools/checkpoints/md-v4-public-window-v1/model/candidate-md-v4-recovery.pt`;
- file SHA-256:
  `ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad`;
- schema: `ptcg.md-v4.fixed-checkpoint.v1`;
- epoch and selected epoch: 4;
- embedded state-dict SHA-256:
  `6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908`;
- frozen-parent state SHA-256:
  `5976a58846bacb4e06e86365f2ad9603c7dd333fff369a6ebbb1539a1175bb36`;
- training-lock SHA-256:
  `7adf2753a303bdfd39fad8872fa277248307fa5ae6637bae3abe33d4c75c3f96`;
- seed: `202607304`.

Every tensor in the epoch-4 state dict is immutable. The optimizer state is
ineligible. There is no retraining, resume, alternate epoch, alternate seed,
weight edit, quantization, calibration, distillation, output rounding, or
action override. The original failed bundle is never relabeled as passing.

## Permitted repair

There is one new candidate:
`md-v4-runtime-parity-salvage-v1`. It may change only the NumPy realization of
the already-declared MD-v4 forward equations. The Torch reference model,
feature encoder, architecture, trained tensors, frozen parent, router, deck,
and all gameplay behavior implied by the Torch reference remain fixed.

The implementation must be selected using algebra review and deterministic
synthetic conformance tests only. Development may not read a development
replay callback, its features, logits, action, reward, or outcome to select
between numerical implementations. In particular, the previously observed
64-callback failure is incident evidence, not a tuning set. No tolerance,
cohort, or decision rule may be changed in response to it.

One repair implementation is permitted. Its source, tests, checkpoint
identity, dependency hashes, and the rules in this document must be committed
and prospectively sealed before the official salvage evaluation is opened.
If it fails, this salvage route retires; another NumPy variant is not tested.

## Locked parity gate

The reference is the fixed epoch-4 Torch model on CUDA, as in the original
run. The candidate is the repaired single-observation NumPy runtime using an
unmodified export of the same tensors.

The official cohort is every one of the 99,946 eligible callbacks from all
2,090 games in the already locked validation split, in its canonical order.
For every callback:

- policy logits must satisfy NumPy `allclose` against the valid Torch logits
  with `atol=3e-5` and `rtol=1e-5`;
- absolute value-head difference must be strictly below `2e-5`; and
- decoded full actions must match exactly.

The population size must be exactly 99,946 callbacks and 2,090 games.
Non-finite output, an exception, a missing callback, an identity mismatch, one
numeric failure, or one decoded-action mismatch fails the gate. Passing is
required but is only numerical conformance, not evidence of stronger play.
The result records maxima and failure counts over the complete cohort.

## Unchanged offline rejection gates

Only after full parity passes, evaluate the fixed epoch-4 Torch model once on
the same complete validation population using the original equal-per-game
estimator. The original thresholds remain:

- game-normalized `KL(parent || candidate) <= 0.02`;
- deterministic full-action disagreement at least 3%; and
- at least 50% of validation games touched by a disagreement.

All original integrity checks remain mandatory, including exact parent bytes,
four completed epochs, finite parameters, zero split overlap, unit raw
outcome/matchup weights, and correct inverse eligible-decision normalization.
Offline metrics can only reject.

## Temporal seal and gameplay order

The July 29 archive remains unopened throughout implementation, parity, and
offline evaluation. Its already recorded archive and central-directory
identities may be bound, but JSON replay content, actions, rewards, and outcomes
must not be read until a fixed candidate has passed the preceding gates.

If parity and offline gates pass, the original gameplay order is unchanged:

1. 20-game frozen-vs-frozen harness sanity with zero controller faults;
2. 2,560 exact-mirror games against complete frozen MD-v3 using 1,280 fixed
   seeds with physical seats swapped; the 95% Wilson lower bound must be
   strictly above 50%, with zero fallbacks, repairs, exceptions, timeouts,
   invalid games, or artifact mismatches;
3. separately locked 1,280-game-per-arm refreshed recent-frequency field A/B,
   requiring non-negative point delta and conservative Wilson-difference lower
   bound above -5 percentage points;
4. rejection-only corroboration on the untouched temporal cohort; and
5. safety tests, 200 fault-free random smoke games, and exact-extracted-tarball
   audit under a non-owner UID.

A failed stage stops all later stages. No production file is edited and no
package is uploaded unless all gates pass and the user explicitly names and
approves the submission.
