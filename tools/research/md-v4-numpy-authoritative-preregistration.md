# MD-v4 NumPy-authoritative frozen-weight candidate preregistration

Status: prospective and rejection-only. This document does not reverse or
relabel either the failed MD-v4 public-window v1 candidate or the failed TF32
NumPy-emulation salvage. It authorizes no production edit, package, tag, or
upload.

## Engineering diagnosis

The fixed MD-v4 training run completed epoch 4. Its candidate publication
failed because the Torch oracle evaluated `nn.GRU` through a fused cuDNN TF32
kernel while the submission runtime is necessarily Torch-free NumPy FP32.
The declared GRU equations were the same, but their numerical kernels were
not. A later attempt to approximate cuDNN TF32 inside NumPy also failed its
separately locked complete-population parity gate.

Torch is prohibited in the submission archive. Consequently, hardware-specific
cuDNN output is not a deployable policy definition. This experiment fixes the
reference boundary: the existing deterministic NumPy FP32 implementation is
the candidate policy, and an explicit Torch FP32 evaluation twin executes the
same declared GRU equations without dispatching to cuDNN.

## One immutable candidate

The candidate is `md-v4-numpy-authoritative-v1`. It uses only:

- recovery checkpoint
  `tools/checkpoints/md-v4-public-window-v1/model/candidate-md-v4-recovery.pt`,
  file SHA-256
  `ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad`;
- epoch 4 state-dict SHA-256
  `6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908`;
- frozen-parent state SHA-256
  `5976a58846bacb4e06e86365f2ad9603c7dd333fff369a6ebbb1539a1175bb36`;
- the original strict `NumpyMDV4` equations and unmodified export of those
  tensors.

No tensor may change. There is no retraining, resume, alternate epoch, seed,
quantization, calibration, TF32 emulation, output rounding, action override,
or validation-selected implementation. The optimizer state is ineligible.

There is one explicit Torch reference implementation and one evaluation
attempt. Its GRU must directly execute the original reset/update/new equations
with float32 Torch operations and `torch.backends.cuda.matmul.allow_tf32=False`.
It may not call `nn.GRU.forward`, a cuDNN RNN operation, the failed TF32 NumPy
repair, or any sample-specific numerical path. All non-GRU operations remain
the original MD-v4 Torch implementation. Source, tests, artifact identities,
and this contract must be committed and prospectively sealed before the
official validation population is opened.

## Complete deployment-parity gate

The official cohort is every one of the 99,946 eligible callbacks from all
2,090 games in the already locked validation split, in canonical order.
For every callback, the explicit Torch FP32 reference and the unmodified
deployable NumPy candidate must satisfy:

- valid policy logits: NumPy `allclose(atol=3e-5, rtol=1e-5)`;
- value-head absolute difference strictly below `2e-5`; and
- identical decoded full action.

There must be zero numeric failures, zero action mismatches, zero non-finite
outputs, and exactly the locked population. The report records complete-cohort
maxima and counts. Failure retires this route; no alternate reference,
tolerance, or NumPy implementation is tried.

The prior cuDNN TF32 outputs are incident evidence, not the candidate
definition. Their mismatch cannot reject or rescue this policy.

## Unchanged offline rejection gates

Only after complete deployment parity passes, run the fixed epoch-4 weights
through the explicit FP32 reference on the same complete validation population
with the original equal-per-game estimator. Require:

- game-normalized `KL(parent || candidate) <= 0.02`;
- deterministic full-action disagreement from frozen MD-v3 at least 3%; and
- at least 50% of validation games touched by one disagreement.

All original integrity checks remain mandatory: exact frozen-parent bytes,
four completed epochs, finite parameters, zero split overlap, unit raw
outcome/matchup weights, and correct inverse eligible-decision normalization.
Training-history metrics are provenance only; this new candidate is selected
neither by them nor by validation.

## Candidate publication and gameplay

If and only if deployment parity and all offline screens pass, atomically
publish a new self-hashed research bundle under a new
`candidate-md-v4-numpy-authoritative-v1` directory. The bundle must retain the
recovery provenance, immutable tensor hash, exact NumPy arrays, explicit
reference identity, complete parity result, and offline result. It must state
that both earlier candidates failed and grant no promotion or upload authority.

Then run, in order:

1. 20-game frozen-vs-frozen harness sanity with zero controller faults;
2. 2,560 exact-mirror games against complete frozen MD-v3 using 1,280 fixed
   seeds with physical seats swapped; the candidate score's 95% Wilson lower
   bound must be strictly above 50%, with zero fallbacks, repairs, exceptions,
   timeouts, invalid games, or artifact mismatches;
3. separately locked 1,280-game-per-arm refreshed recent-frequency field A/B,
   requiring non-negative point delta and conservative independent
   Wilson-difference lower bound above -5 percentage points;
4. rejection-only corroboration on the untouched July 29 temporal cohort; and
5. safety tests, 200 fault-free random smoke games, and exact-extracted-tarball
   audit under a non-owner UID.

A failed stage stops all later stages. Local gameplay, not offline loss,
determines whether the deployable NumPy policy improves on MD-v3. No production
file is edited and no package is uploaded without every gate passing and the
user explicitly naming and approving the submission.

## Temporal seal

The July 29 archive remains unopened until this candidate is fixed and passes
deployment parity and offline screens. Its existing archive and
central-directory hashes may be rebound prospectively, but no replay JSON,
action, reward, or outcome may be read before then.
