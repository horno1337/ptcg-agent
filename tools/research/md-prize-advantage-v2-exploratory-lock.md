# MD prize-advantage v2 exploratory training lock

Date locked: 2026-08-01, after the v1 screen and before any policy update.

## Status and rationale

V1 remains a failed preregistered experiment because its auxiliary prize-head
MSE improvement was 5.58%, below the locked 10% floor.  V2 does not relabel
that result or claim prospective confirmation.  The user explicitly
authorized one exploratory candidate after observing that the failed metric
is not consumed by the policy objective.

The auxiliary head asked whether current public state alone predicts the next
prize swing before conditioning on the logged action.  Candidate supervision
instead uses the already-observed public prize delta after the logged action.
The head's prediction is therefore retained as a diagnostic but is not a
logical prerequisite for retrospective credit assignment.

## Immutable inputs

- Transition cohort SHA-256:
  `0c3bad794e634f964e8552a19419889e25d90980b899550b198230fcd38980ba`.
- Feature cache SHA-256:
  `2ed8fbc73a3f396709a2f952aa91b010da8d88a393aa5ff7cb73d03e73c81b1b`.
- Cross-fitted probe-state SHA-256:
  `d992aea5b89aa447f511a9c22ba2c47786dcc0fbe0307bc795a3705860a98efe`.
- Split-isolated policy-cache self hash:
  `4688ffe99be24943e57874ce7498a40235497e911c39395a14d6fff1a6280e41`.
- V1 screen result is immutable and continues to report `passed: false`.
- Exact deck, append-stable split, five folds, value models, gamma `0.997`,
  prize scale `0.25`, and advantage floor `0.05` remain unchanged.
- Frozen deployed MD-v3 weights SHA-256:
  `76420fc2e031127143f5816e64973ec49e150da22b84604308b649c9a4dc8df8`.
- Candidate initialization is the exact fixed epoch-4 explicit-FP32 MD-v4
  checkpoint SHA-256
  `ab1f7bd03939e9d6dba6356086a17402b232575922c5312b9150c8c8d2b01dad`.

The advantage-label export is deterministic.  Recent-train rows use only the
probe whose held-out fold contains their game.  Recent-validation rows use
the mean of all five probes.  No model is refit and no label threshold changes.

## Fixed training objective

Only exact-deck `ST_MAIN` overlay parameters are trainable.  The embedded
MD-v3 parent, ST_CARD specialist, Qu-v2B fallback, router, deck, and runtime
rules remain frozen.

Each optimizer batch contains equal recent and historical stabilization mass:

- Recent half: only rows that disagree with frozen MD-v3 and have cross-fitted
  `A >= 0.05` receive logged-action NLL authority.  Their label multiplier is
  `clip(max(A, 0) / 0.25, 0, 4)`.  Every recent prompt contributes parent KL.
- Historical half: through-July-28 exact-deck `ST_MAIN` callbacks reproduce
  frozen MD-v3's decoded action, not the human logged action, and contribute
  parent KL.  This half is stabilization only.

Loss is the equal-mass mean of weighted recent NLL, historical parent-action
NLL, and parent-to-candidate sequential KL with coefficient `1.0`.  Game-seat
groups have equal total mass within source before the 50/50 source mixture.

Train exactly four epochs with seed `2026080111`, batch size `128`, AdamW
learning rate `3e-4`, weight decay `1e-4`, and gradient clipping at `1.0`.
There is no checkpoint selection; only epoch four has authority.  No alternate
advantage floor, prize scale, seed, epoch, source mixture, or KL coefficient is
permitted.

## Rejection gates

Before gameplay, epoch four must satisfy all of:

- frozen-parent parameters byte-identical;
- finite tensors and metrics plus Torch/NumPy decoded-action parity;
- validation parent KL at most `0.02`;
- at least 3.0% decoded disagreements versus frozen MD-v3 over the fixed
  recent-validation callbacks;
- at least 50% of recent-validation seat-games touched; and
- no validation example contributes a gradient or selects a checkpoint.

Failure retires v2.  Passing authorizes only a separately locked 20-game
sanity run and 2,560-game direct Grimmsnarl-mirror A/B against frozen MD-v3.
It does not authorize packaging, naming, promotion, or upload.
