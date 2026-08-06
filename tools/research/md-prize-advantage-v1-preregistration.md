# MD prize-advantage v1 preregistration

Date locked: 2026-08-01, before this experiment opens any July 29--31 replay
outcome, action, reward, or prize-transition field.

## Question and boundary

Can public prize progression plus a cross-fitted public-state value estimate
identify a sufficiently broad set of better-than-expected logged `ST_MAIN`
transitions to justify one advantage-weighted Grimmsnarl candidate?

This is a new dense-credit experiment.  It is not another terminal-only PPO
run and it is not the failed raw winner-preference MD-v5 objective.  Frozen
MD-v3 remains the deployed parent.  The experiment grants no runtime edit,
promotion, package name, or upload authority.

## Fixed sources and split

The source contract is `md-next-recent-corpus-v1-preregistration.md` and its
self-hashed July 29--31 inventory SHA-256 is
`514490bf9a22bfd5cf83dec97a6ee66f6d24178e8be20706411c2156a9c0290a`.
Every one of the 13,110 JSON members is decoded before deck eligibility is
tested; filename, agent-name, and team-name prefilters are forbidden.

Only seats registering the exact Grimmsnarl list with canonical deck SHA-256
`c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af`
are eligible.  The existing append-stable 90% recent-train / 10%
recent-validation game split is unchanged.  Games, seats, and callbacks never
cross it.  The first complete non-overlapping official day on or after August
1 remains unopened temporal evidence.

## Fixed transition and reward definitions

A transition begins at an eligible seat's valid `ST_MAIN` callback and ends at
that same seat's next valid `ST_MAIN` callback, or at the episode terminal when
there is no next callback.  `k` is the number of that seat's valid policy
callbacks crossed, with a minimum of one.  Prize counts are the lengths of the
two public `current.players[*].prize` arrays in the acting seat's legitimate
view.  Card identities are never inspected.

For acting seat `i`, define the signed net prize swing

`p = (my_prizes_before - my_prizes_after)
     - (opponent_prizes_before - opponent_prizes_after)`.

For a terminal endpoint, missing final public counts are carried forward;
terminal win/draw/loss remains the only terminal authority.  Prize shaping is
`0.25 * p / 6`.  It can refine credit but cannot replace the terminal label.
The discount is fixed at `gamma = 0.997` per crossed valid callback, matching
the corrected PPO protocol.

The dense logged-transition advantage is fixed as

`A = terminal_reward_if_any + 0.25 * p / 6
     + (0 if terminal else gamma**k * V(next)) - V(current)`.

No damage, remaining deck, hand size, rating, agent identity, future hidden
card identity, or guide-derived bonus enters the reward.

## Fixed value and prize probes

Represent each endpoint with the detached 192-value public representation
already validated by the mirror value-signal experiment: frozen MD-v3 state
`[160]` plus frozen MD-v4 resource/history fusion `[32]`.  Neither logged
action nor future state is an input.

Use five append-stable game folds from
`SHA256("ptcg.md-prize-advantage.v1.fold\\0" + game_uid) mod 5`.  Predictions
for recent-train games are always made by a model that excluded that game.
The recent-validation split is excluded from every fit.  Each fold trains the
same fixed `Linear(192,64)-ReLU` trunk with:

- a `Linear(64,1)-tanh` terminal-outcome head trained by seat-game-normalized
  squared error;
- a `Linear(64,1)-tanh` next-transition net-prize-swing head trained on `p/6`
  by seat-game-normalized squared error;
- Adam, learning rate `1e-3`, weight decay `1e-4`, batch size `1024`, exactly
  12 epochs, seed `2026080101 + fold`.

Training-only mean and variance standardize inputs.  There is no epoch,
checkpoint, seed, architecture, or hyperparameter selection.

## Prospective screen

All metrics below are computed once on the fixed recent-validation games.
The route advances only if every condition passes:

1. terminal outcome AUC is at least `0.60` overall and at least `0.56` on the
   first half of each seat-game;
2. prize-head MSE is at least 10% below the constant-zero predictor's MSE;
3. at least 3.0% of all eligible validation `ST_MAIN` decisions both disagree
   with frozen MD-v3 and have `A >= 0.05`;
4. those actionable decisions touch at least 50% of eligible validation
   seat-games;
5. the actionable rate is at least 2.0% separately on July 29, July 30, July
   31, exact Grimmsnarl mirrors, and non-mirrors; and
6. all counts, labels, representations, predictions, and advantages are finite
   and both resolved outcomes occur in every reported date and matchup stratum.

Thresholds will not be lowered, dates or strata dropped, folds or epochs
selected, or alternative reward scales tried after results are observed.
Failure retires prize-advantage v1 before policy training.

## Fixed candidate path if and only if the screen passes

Train one candidate only, updating the exact-deck `ST_MAIN` overlay while
freezing `ST_CARD`, Qu-v2B fallback, router, deck, and all runtime rules.
Recent July 29--31 examples receive 50% of sampling mass and the locked
through-July-28 corpus receives 50%.  Recent logged-action imitation weight is
`clip(max(A, 0) / 0.25, 0, 4)`; non-actionable recent examples have zero logged
label authority.  Historical examples reproduce frozen MD-v3 behavior for
stabilization.  A KL penalty of `1.0` anchors every eligible prompt to frozen
MD-v3, and final measured parent KL must not exceed `0.02`.  Training uses one
fixed seed `2026080111`, four terminal epochs, and the existing MD-v4 resource
architecture initialized from its fixed epoch-4 checkpoint.  Only epoch four
has selection authority.

Before any ladder use, the fixed candidate must pass: array/runtime
conformance; a material behavior check of at least 3% disagreements and 50%
games touched versus MD-v3; 20 random sanity games; a locked 2,560-game direct
Grimmsnarl-mirror A/B versus frozen MD-v3 whose 95% paired interval excludes
50%; a recent-frequency field non-inferiority gate; the untouched August-1-or-
later temporal check; and the exact-tarball cross-UID, safety-test, and
200-random-game audits.  Exact gameplay cohorts and field weights must be
locked separately before their outcomes are generated.  The user names and
approves any upload.
