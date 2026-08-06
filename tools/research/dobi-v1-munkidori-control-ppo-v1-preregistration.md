# Dobi-v1 Munkidori-control PPO v1 preregistration

Status: locked before training or gameplay outcomes.

## Question

Can public-state potential shaping around powered Munkidori board control improve
frozen Dobi-v1 in the exact Grimmsnarl mirror without giving back its non-mirror
field strength?

This is not another behavioral-cloning run, prize-prediction model, generic
prize-delta reward, or terminal-only continuation. The actor remains the same
deployable Qu-v2A ST_MAIN network and the terminal game result remains the only
outcome authority.

## Evidence used to define the experiment

The definition is fixed from the combined byte-identical Dobi-v1 family ladder
cohort available on 2026-08-06:

- 62 resolved exact mirrors: 27 wins, 35 losses;
- first multiplicity-corrected primary divergence: own turn 4;
- powered-Munkidori differential at turn 4: win minus loss `+1.08`;
- Adrena-Brain uses at turn 4: win minus loss `+0.67`;
- damage-transfer selections at turn 4: win minus loss `+1.34`.

The locked diagnostic is
`tools/checkpoints/dobi-v1-bc-recent-v1/combined-mirror-divergence.json`.
These are observational associations, not causal estimates. Their only role is
to choose one prospective shaping signal.

## Frozen package scope

- Parent ST_MAIN: frozen Dobi-v1, weights SHA-256
  `bf93b3b7c114c0f406dea549ea8b7ac0aa783e0e80b3814fa4ea42ac412f679c`.
- Exact registered deck SHA-256:
  `c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af`.
- ST_CARD remains byte-identical to Dobi-v1.
- Qu-v2B residual routing remains byte-identical to Dobi-v1.
- Runtime features, architecture, deck, and router remain unchanged.
- Only ST_MAIN actor and critic parameters may train.

## Locked reward

At each learner ST_MAIN public observation, define:

```text
online_diff = my powered Munkidori - opponent powered Munkidori
total_diff  = my Munkidori         - opponent Munkidori
phi(s)      = clip(0.50 * online_diff + 0.25 * total_diff, -1, +1)
```

A powered Munkidori is a visible Munkidori with at least one attached Basic
Dark Energy. No hidden information, future result, hand identity, or logged
human action enters `phi`.

For a macro-transition lasting `k` engine selections:

```text
r = terminal_result_if_any
    + 0.15 * (gamma**k * phi(next_state) - phi(current_state))
```

`phi(next_state) = 0` at a terminal. `gamma = 0.997`. The implementation must
pass a telescoping-identity test and must reject any non-finite or out-of-range
potential. No other dense reward is allowed.

## Locked training population and endpoint

- Exactly 99,840 valid games: 130 updates x 768 games.
- Exact Grimmsnarl mirror only, seat-balanced within every update.
- Frozen opponent league copied from the locked Dobi-v1 mirror league:
  40% frozen Dobi-v1, 20% frozen MD-v3, 7.5% frozen MD-v5, 6.25% frozen
  BC51, 6.25% frozen PPO-v1, and four rotating learner snapshots at 5% each.
- Fresh rollout/PPO seeds must be locked before the first rollout and must not
  reuse the earlier 300k schedule.
- Actor learning rate `2.5e-6`; critic learning rate `2e-5`; GAE lambda `0.95`;
  PPO epochs `2`; minibatch `512`; clip `0.15`; entropy coefficient `0.005`;
  parent-KL coefficient `0.10`; value coefficient `0.5`.
- Frozen Dobi-v1 is the KL parent.
- Maximum terminal parent KL: `0.04`.
- Only the fixed update-130 terminal checkpoint is selection-eligible.
  Intermediate checkpoints are recovery-only and cannot be selected post hoc.

Any schedule, reward, population, numerical, parity, or engine fault invalidates
the run. It may be repaired and restarted from the frozen parent, but results
from an invalid attempt have no selection authority.

## Prospective gates

1. **Implementation gate:** unit tests for the public potential, powered-card
   identity, macro-transition reward, terminal handling, and telescoping
   identity all pass.
2. **Training gate:** all 99,840 games complete validly; terminal actor delta is
   nonzero; frozen parameters remain byte-identical; terminal KL is at most
   `0.04`; Torch and deployed NumPy outputs pass parity.
3. **Exact-mirror gameplay gate:** one fixed 10,240-game, seat-balanced direct
   A/B against frozen Dobi-v1. Advance only if the candidate score's two-sided
   Wilson 95% lower bound is strictly above 50%.
4. **Non-mirror field gate:** only after gate 3, one fixed 5,120-game paired
   recent-frequency field A/B against frozen Dobi-v1. Require the 95% CI lower
   bound for candidate-minus-control to be at least `-1.5pp`.
5. **Deployment gate:** exact extracted-tarball audit under a non-owner UID,
   `tests/test_safety.py`, and the locked 200-game random smoke all pass.

Failure of a gate retires this candidate. No threshold, population stratum,
endpoint, or cohort may be changed after an outcome is observed. A Kaggle
upload still requires a separately user-approved submission name.

