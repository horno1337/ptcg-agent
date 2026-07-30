# MD-v4 deployed-parent correction preregistration

Status: prospective and unopened. This document grants no gameplay,
production, packaging, tag, or upload authority.

## Why this is a separate experiment

The completed `md-v4-numpy-deployable-v1` attempt remains an immutable failed
attempt. Its official result compared the staged NumPy MD-v4 candidate with
the `parent_logits` stored in the pre-existing encoded cache. Those logits
were produced by the research Torch Qu-v2A runtime. The result therefore
failed its literal implementation at 2,951 disagreements out of 99,946
callbacks and will not be rerun, relabelled, rounded, or reversed.

The operational wording of that screen was nevertheless "disagreement from
frozen MD-v3." Frozen MD-v3 is deployed through the Torch-free production
runtime `agent.model.QuV2Net`, not through the Torch cache producer. The two
runtimes hold the same tensors but are not guaranteed to decode identically
near ties. This new, independently locked correction answers only the
operational question that the prior implementation did not answer.

## Frozen inputs

- Candidate: the unchanged epoch-4 MD-v4 recovery checkpoint and unchanged
  NumPy mapping used by the failed attempt.
- Candidate state-dict SHA-256:
  `6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908`.
- Candidate NumPy mapping SHA-256:
  `e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01`.
- Cohort: all 99,946 eligible ST_MAIN callbacks from all 2,090 games in the
  already locked validation split, in canonical order.
- Deployed parent weights:
  `tools/checkpoints/md-v2-allthrough26/model/candidate-qu-v2a-weights.npz`,
  file SHA-256
  `76420fc2e031127143f5816e64973ec49e150da22b84604308b649c9a4dc8df8`.
- Deployed parent equations: `agent.model.QuV2Net`.
- Deployed decoder: `agent.model.decode_qu_v2`.
- Frozen MD-v3 package:
  `submission-md-v2-card-v1-experimental-unsigned.tar.gz`, SHA-256
  `adb0e8c1da13a56468f6855bee197eab3d4b58f378e0a0a9b1ea6cc8b604ac42`.
  Its regular members must bind `agent/md_v1_weights.npz` to
  `76420fc2...8df8`, `agent/model.py` to `238a21d2...6ef`, and
  `agent/qu_v2_features.py` to `b2225e00...94d7`; the current source bytes
  used for evaluation must equal those package members.

The working-tree file `agent/md_v1_weights.npz` is the older MD-v1 artifact
(`5784b7ea...be78c`) and is not the frozen MD-v3 main artifact. MD-v3 packages
the locked `76420fc2...8df8` artifact under that submission-local filename.
This experiment binds the locked source artifact, not the unrelated
working-tree filename.

No weights, epoch, seed, examples, split membership, threshold, optimizer
state, equations, or features may change in this correction.

## Exact deployed-parent conformance gate

Before the corrected offline metrics can have authority, the complete locked
population must satisfy all of the following:

1. The direct frozen export and exact staged/reloaded candidate NPZ are
   bit-identical in arrays and outputs.
2. The embedded research NumPy parent inside the staged MD-v4 runtime and
   `agent.model.QuV2Net` loaded from the exact deployed-parent NPZ produce
   bit-identical logits and values on every callback.
3. The research sequential decoder and `agent.model.decode_qu_v2` produce
   identical candidate actions and identical parent actions on every
   callback.
4. Population counts are exactly 99,946 callbacks and 2,090 games.
5. There are zero array, inference, non-finite, decode, per-sample metric, or
   aggregate metric failures.

The 17 frozen Qu-v2A feature arrays are explicitly reconstructed as an
`agent.qu_v2_features.PublicFeatures` instance. Field names, shapes, dtypes,
and bytes must remain unchanged. Cached `parent_logits` have zero gating
authority and must not supply parent actions or KL terms.

Any conformance failure rejects this route before corrected metrics receive
authority. It is not permission to substitute a different parent
implementation.

## Unchanged rejection thresholds

Using the exact staged/reloaded NumPy candidate, exact deployed NumPy parent,
and production decoder:

- game-normalized `KL(parent || candidate)` must be at most `0.02`,
  inclusive;
- decoded candidate/parent disagreement must be at least `0.03`, inclusive;
- games touched by at least one disagreement must be at least `0.50`,
  inclusive.

On the fixed population these rate thresholds require at least 2,999
disagreements out of 99,946 callbacks and at least 1,045 touched games out of
2,090 games. No rounding rule may reduce either integer requirement.

The logged-action NLL is recorded only as a diagnostic. The three thresholds
above are unchanged from the completed NumPy-deployable attempt. The complete
gate is conjunctive.

If the correction fails, it is retired without threshold lowering,
post-outcome parent substitution, alternate decoder, cohort change, rerun, or
candidate retraining under this namespace. If it passes, the exact staged NPZ
may be published as a research-only bundle and only the already
preregistered direct gameplay gate may open. A pass is not itself promotion
or upload evidence.

## One-attempt and temporal rules

- The correction has a new lock, attempt marker, result, and bundle namespace.
- Exactly one official attempt is allowed; artifacts are created without
  replacement.
- The attempt marker and exact staged NPZ are written before the first
  validation callback is consumed.
- The prior failed lock, result, and human-readable record are hash-bound and
  preserved.
- The July 29 archive remains sealed. It is not opened by this correction.
- Production files, submission packages, tags, and Kaggle slots are out of
  scope.
