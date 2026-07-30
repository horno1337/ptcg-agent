# MD-v4 resource-aware PPO v1 preregistration

Status: prospective, research-only, and rejection-only. This document must be
committed together with its lock builder, runner, and focused tests before an
official lock is created or a rollout is generated. It authorizes no
production edit, package, tag, submission name, or upload.

## Prior results remain failed

This is a new optimization experiment, not a repair or reinterpretation of an
earlier result.

The fixed epoch-4 MD-v4 behavior-cloning candidate has state-dict SHA-256
`6f3ba4bdb3f34e5397a5d5988e97161a0629d3f88a5a3560a66ad31d6bf2d908`
and NumPy mapping SHA-256
`e9042f980581355bf172f6aafb9b2fd9752b9e1873157d3feb22be17e21b7e01`.
Its corrected one-shot comparison against the exact deployed NumPy MD-v3
parent remains failed:

- 99,946 callbacks and 2,090 games were evaluated;
- deployed-parent conformance passed completely;
- `KL(deployed MD-v3 || candidate) = 0.002864179944542191`;
- 2,376 callbacks disagreed, or `2.3772837332159368%`, below the unchanged
  inclusive `3%` floor of 2,999 callbacks; and
- 1,330 games were touched, above the unchanged 50% floor.

That result is immutable. This experiment does not lower, round, rescale, or
redefine its threshold. The failed PPO-v2 candidate is likewise used only as
a frozen population member; its failed direct-gameplay result is not reopened.

## Hypothesis and candidate

Behavior cloning gave the new public resource/log representation only the
objective of imitating the average logged action. Terminal-outcome PPO may use
that representation to prefer resource-preserving actions that actually win.
This is a different learning signal, not a residual multiplier.

The internal candidate identifier is `md-v4-resource-ppo-v1`. It warm-starts
the exact epoch-4 MD-v4 state above. The complete embedded MD-v3 parent,
deployed MD-v3 `ST_CARD` specialist, Qu-v2B fallback, registered deck, and all
non-`ST_MAIN` routing remain frozen. Only these existing MD-v4 actor modules
train:

- `resource1`;
- `event_embedding` and `role_embedding`;
- every `log_card_projections` member;
- `attack_embedding` and `area_embedding`;
- `log_gru`;
- `fusion`;
- `residual1`; and
- `residual2`.

A new 64-unit critic reads a detached concatenation of the frozen MD-v3 state
representation (160 values) and the current MD-v4 fusion representation (32
values). Its exact architecture is `Linear(192,64,bias=True)`, ReLU,
`Linear(64,1,bias=True)`, then tanh. Both biases initialize to zero; both
weights use orthogonal initialization, with gains `sqrt(2)` and `1`
respectively. The 192-value representation is detached and copied into each
rollout row. Critic minibatches use that immutable rollout snapshot rather
than recomputing it after actor updates. The critic is used only to estimate
PPO advantages. It has a separate optimizer group, cannot send a value
gradient into the actor, is stored only in training/recovery checkpoints, and
is absent from the deployable NumPy artifact.

There is one seed family, one run, one fixed terminal candidate, and no sweep.
Updates 1--23 are crash-recovery checkpoints and are never selection-eligible.
Only update 24 is eligible.

The sole official run path is
`tools/checkpoints/md-v4-resource-ppo-v1/training`. Before its first rollout,
the runner atomically creates a lock-bound attempt-consumption marker. A
controlled cleanliness, contract, or runtime failure creates an immutable
retirement marker and ends the experiment. A crash may resume only inside
that same output tree from its latest complete authenticated recovery, and
each recovery checkpoint is atomically consumed at most once before resumed
rollout begins. A crash before the next recovery retires that continuation;
the same checkpoint cannot be replayed. A second fresh output directory is
not permitted.

Every recovery manifest self-hashes and binds the attempt, lock, preceding
checkpoint, resume-consumption marker, checkpoint file bytes, checkpoint
payload, actor state, critic state, optimizer state, and complete update
history. Restore recomputes every identity before loading state and again
after loading it. The prospective artifact lock also binds the observation,
feature, card, corpus-index, imitation-loader, battle-binding, and gameplay
evaluation implementations that define this experiment; editing any one
invalidates the lock.

## Rollout population

Each update contains exactly 768 games, 384 paired schedules, and equal
learner physical-seat counts of 384/384. Across every update, pair mass is
exactly 50% exact-list Grimmsnarl mirror and 50% non-mirror field.

Mirror pilot mass is:

- complete frozen MD-v3: 25%;
- frozen epoch-4 MD-v4: 10%;
- retired terminal PPO-v2 policy: 5%;
- frozen MD-v1: 5%; and
- frozen Qu-v2B on the Grimmsnarl deck: 5%.

Non-mirror field mass is:

- frozen Qu-v2B pilot: 40%; and
- rules pilot: 10%.

Non-mirror deck allocation uses the already fixed July 27--28
recent-frequency snapshot, removes Grimmsnarl, and renormalizes the remaining
weights without looking at any new outcome. Every update schedule, opponent,
deck, physical seat, rollout seed, and PPO seed is included in the prospective
lock. The native engine RNG is not claimed to be seedable; the lock establishes
schedule reproducibility.

Every rollout must finish all 768 games with zero invalid games, truncations,
controller exceptions, fallbacks, legality repairs, off-deck mirror routes, or
artifact drift, and must contain at least 20,000 learner `ST_MAIN` decisions.
Any violation retires the run.

## Fixed optimization

- updates: 24;
- games per update: 768;
- total games: 18,432;
- PPO epochs per update: 2;
- minibatch size: 512;
- actor learning rate: `2e-6`;
- critic learning rate: `1e-5`;
- PPO clip: `0.10`;
- terminal reward: win `+1`, draw `0`, loss `-1`;
- discount `gamma`: `0.997`;
- semi-MDP GAE lambda: `0.95`;
- entropy coefficient: `0.002`;
- value coefficient: `0.5`;
- deployed-parent KL coefficient: `1.0`; and
- actor and critic gradient-norm ceiling: `1.0`.

The PPO ratio uses the log probability recorded by the exact policy that
generated the rollout. The rollout actor is
`md_v4_explicit_reference.TorchMDV4ExplicitFP32`; it expands the GRU equations
explicitly. The sole scientific run is CUDA FP32 with CUDA matmul and cuDNN
TF32 disabled, avoiding the fused cuDNN path that caused the retired
Torch/NumPy mismatch. CPU runs are tests only and cannot create the official
candidate. Its log probability therefore corresponds to the actual rollout
action sampler. The exported NumPy policy remains the gameplay and deployment
authority. The parent anchor on every rollout row is computed
from the exact deployed NumPy MD-v3 runtime, `agent.model.QuV2Net`, loaded from
weights SHA-256
`76420fc2e031127143f5816e64973ec49e150da22b84604308b649c9a4dc8df8`.
Cached Torch-parent logits have no training-anchor, action, KL, or offline-gate
authority.

The actor and critic use separate backward passes. The critic representation
is detached, actor and critic parameter sets are disjoint, and the embedded
MD-v3 parent must remain byte-identical after every update. One optimizer
persists across all updates and any resume.

## Terminal artifact and offline rejection screens

At update 24, export exactly one NumPy mapping and serialize it once. The
staged file is reloaded without pickle. In-memory and staged mappings must have
the same fields, shapes, dtypes, C-order bytes, and mapping hash. Two
independently loaded NumPy candidate runtimes must be bit-identical over the
complete development callback population.

The fixed-terminal Torch checkpoint is actor-only: it contains neither critic
parameters nor optimizer state. Critic and optimizer state exist solely in
updates 1--23 recovery artifacts and have no deployment or selection
authority.

The through-July-28 validation partition has already informed previous work
and is explicitly demoted to a development/integrity and behavior-sizing
population. It is not fresh evidence and cannot promote the model. It remains
fixed at 99,946 exact-list `ST_MAIN` callbacks from 2,090 games.

Over all callbacks, in canonical order:

1. convert the MD-v4 base feature record byte-for-byte to the production
   `agent.qu_v2_features.PublicFeatures` type;
2. run the exact exported/staged NumPy MD-v4 candidate;
3. run exact deployed NumPy MD-v3 through `agent.model.QuV2Net`;
4. require bit identity between the candidate's embedded NumPy parent and the
   separately loaded deployed parent for logits and value;
5. require research and production decoders to agree on both arms;
6. never read cached `sample.parent_logits`; and
7. require zero feature, inference, non-finite, decode, population, or metric
   failures.

Only then apply all of these rejection screens:

- game-normalized `KL(deployed NumPy MD-v3 || candidate) <= 0.02`;
- at least 2,999/99,946 exact full-action disagreements from deployed NumPy
  MD-v3 (`>=3%`);
- at least 1,045/2,090 games touched (`>=50%`);
- at least 1,000/99,946 exact full-action disagreements from the fixed
  epoch-4 MD-v4 initialization (`>=1%`); and
- at least 418/2,090 games touched versus that initialization (`>=20%`).

The last two screens prevent a nominal threshold-crossing change that is
effectively the same failed epoch-4 policy. All offline quantities are
rejection and behavior-sizing evidence only.

## Gameplay, temporal, and packaging order

If and only if every terminal/offline screen passes:

1. run a 20-game frozen-vs-frozen harness sanity with zero faults;
2. run one separately locked 2,560-game paired-seat exact-mirror A/B against
   complete frozen MD-v3; score is `(wins + 0.5 * draws) / 2560`, and the
   candidate's two-sided 95% Wilson lower bound must be strictly above 50%,
   with zero faults;
3. run a separately locked 1,280-game-per-arm refreshed recent-frequency field
   A/B on one identical schedule; candidate-control point delta must be
   non-negative and the conservative independent Wilson-difference lower bound
   must be strictly above -5 percentage points;
4. only after both gameplay gates pass, open the sealed July 29 cohort once
   for rejection-only temporal corroboration, requiring finite public-only
   evaluation, zero faults, KL at most `0.02`, exact-list `ST_MAIN` NLL no more
   than `0.010` above deployed MD-v3, and exact-mirror winning-seat NLL
   strictly below deployed MD-v3; and
5. require exact research-to-vendored NumPy conformance, safety tests, 200
   fault-free random smoke games, and the exact-extracted tarball audit under
   a non-owner UID.

Failure stops the sequence. No later stage can rescue an earlier failure.

## July 29 seal and naming

The July 29 archive remains sealed at SHA-256
`dbe39b021aebe75c46805105604b56c6263ebb0b4f49d7bc41025517a1f4168e`.
Its replay JSON, actions, rewards, and outcomes cannot be opened during
implementation, locking, training, offline sizing, or gameplay. It remains
the first temporal cohort; it is not relabeled as a development holdout.

`md-v4-resource-ppo-v1` is an internal experiment identifier only. If it
passes, the user still names the submission and separately approves the upload
and automatic FIFO slot retirement.
