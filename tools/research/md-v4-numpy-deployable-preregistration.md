# MD-v4 NumPy-deployable gameplay-candidate preregistration

Status: prospective and rejection-only. This is a new qualification route for
one already frozen policy. It does not reverse or relabel the failed original
MD-v4 publication, the failed TF32-emulation salvage, or the failed
NumPy-authoritative cross-engine identity experiment. It authorizes no
production edit, package, tag, or upload.

## Why this is a distinct experiment

The submission environment cannot contain Torch. The deployable MD-v4 policy
is therefore the existing deterministic NumPy FP32 runtime, not a fused
CUDA/cuDNN execution of the same equations. The completed cross-engine
experiment measured the difference over every locked validation callback:

- 99,946 callbacks from 2,090 games;
- zero numeric-tolerance failures;
- maximum absolute logit difference
  `9.059906005859375e-06`;
- maximum absolute value difference
  `1.2665987014770508e-06`; and
- 808 decoded-action mismatches.

That experiment failed its prospectively locked zero-action-mismatch rule and
remains failed. Its result is retained as a diagnostic of two different
floating-point execution engines. It is neither a rejection gate nor positive
evidence for this new gameplay candidate. This route makes no claim that Torch
and NumPy define byte-identical policies.

## One immutable candidate

The candidate is `md-v4-numpy-deployable-v1`. It is the original, unmodified
NumPy export of:

- recovery checkpoint
  `tools/checkpoints/md-v4-public-window-v1/model/candidate-md-v4-recovery.pt`,
  file SHA-256
  `ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad`;
- epoch-4 state-dict SHA-256
  `6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908`;
- frozen-parent state SHA-256
  `5976a58846bacb4e06e86365f2ad9603c7dd333fff369a6ebbb1539a1175bb36`;
  and
- NumPy array-mapping SHA-256
  `e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01`.

No tensor, equation, decoder, tie rule, epsilon, quantization, calibration, or
runtime override may change. There is no retraining, resume, alternate epoch,
seed, action canonicalization, or validation-selected implementation. The
optimizer state is ineligible.

## Exact research-bundle artifact identity gate

Before validation is opened, the fixed array mapping is serialized to the
exact compressed NPZ that will be published in the research bundle if the
gates pass. That exact file is reloaded through the strict research
`NumpyMDV4` reader. The in-memory research mapping and the reloaded
staged-research mapping must have identical:

- field names;
- shapes and dtypes;
- C-order array bytes; and
- mapping SHA-256, equal to
  `e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01`.

Over every one of the 99,946 eligible callbacks from all 2,090 games in the
already locked validation split, in canonical order, two independently
constructed `NumpyMDV4` instances—one from the research mapping and one from
the reloaded staged-research mapping—must produce bit-identical float32 logit
arrays and bit-identical float32 value outputs. Both must be finite. The full
decoded actions must consequently be identical. Any mapping, output,
population, or decode mismatch, inference exception, decode failure, or metric
failure retires this route and is recorded in the self-hashed result. No
alternate serialization or implementation is tried.

The staged NPZ is the exact file moved into the published research bundle; it
is not regenerated after evaluation. Publication is exclusive and atomic.
This establishes serialization/reload identity for the same research runtime
in the prospectively locked local NumPy/BLAS environment. It does not establish
identity with a later vendored submission runtime or across a different BLAS
implementation. Before promotion, the exact extracted submission must
separately bind this NPZ and prove conformance of its vendored runtime; that
package gate may reject but cannot alter the candidate.

## Unchanged offline rejection gates

During the same complete pass, compute metrics from the reloaded staged
research NumPy runtime and the already cached frozen-parent logits using the
original float32 sequential equations and equal-per-game estimator. Only after
the complete research-bundle artifact identity gate passes, require:

- game-normalized `KL(parent || candidate) <= 0.02`;
- deterministic full-action disagreement from frozen MD-v3 at least 3%; and
- at least 50% of validation games touched by one disagreement.

All original integrity checks remain mandatory: exact frozen-parent bytes,
four completed epochs, finite parameters, zero split overlap, unit raw
outcome/matchup weights, and correct inverse eligible-decision normalization.
Training-history metrics are provenance only. Offline metrics are
rejection/behavior-sizing screens and never promotion evidence.

There is exactly one official attempt and one exclusive output namespace.
Failure of identity or any offline screen publishes a self-hashed failure
result but no candidate bundle and opens no later cohort.

## Gameplay and promotion sequence

If and only if the exact research-bundle artifact identity gate and every
unchanged offline screen pass, atomically publish
`candidate-md-v4-numpy-deployable-v1`. The bundle records the fixed recovery
and tensor identities, exact evaluated NPZ, complete identity result, offline
result, and all prior failed routes.

Then run, in order, under separately prospective locks:

1. 20-game frozen-vs-frozen harness sanity with zero controller faults;
2. 2,560 exact-mirror games against complete frozen MD-v3 using 1,280 fixed
   seeds with physical seats swapped; the candidate score's 95% Wilson lower
   bound must be strictly above 50%, with zero fallbacks, repairs, exceptions,
   timeouts, invalid games, or artifact mismatches;
3. separately locked 1,280-game-per-arm refreshed recent-frequency field A/B,
   requiring non-negative point delta and a conservative independent
   Wilson-difference lower bound above -5 percentage points;
4. rejection-only corroboration on the untouched July 29 temporal cohort; and
5. bit-exact conformance between the research runtime and the exact-extracted
   vendored runtime over all 99,946 callbacks from the same 2,090 locked
   validation games, in canonical order, with zero logit-bit, value-bit,
   decoded-action, non-finite, or exception mismatches; then safety tests,
   200 fault-free random smoke games, and the exact-extracted-tarball audit
   under a non-owner UID. This proves the port in the locked local environment,
   not cross-BLAS or Kaggle-environment bit identity.

A failed stage stops all later stages. Local gameplay, not Torch agreement or
offline loss, decides whether this NumPy policy improves on MD-v3. Production
files remain unchanged and nothing is uploaded without every gate passing and
the user explicitly naming and approving the submission.

## Temporal seal

The July 29 archive remains unopened until this candidate is fixed, passes
research-bundle artifact identity and offline screens, and reaches its declared
temporal stage. Its existing archive and central-directory hashes may be
rebound prospectively, but no replay JSON, action, reward, or outcome may be
read before then.
