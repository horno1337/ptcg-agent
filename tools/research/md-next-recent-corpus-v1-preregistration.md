# MD-next recent corpus v1 preregistration

Date locked: 2026-08-01, before opening July 31 replay JSON, actions, rewards,
or outcomes.

## Purpose

Promote the complete official July 29, July 30, and July 31 daily archives
into the development pool for the next model experiment. This data contract
does not reopen the failed raw winner-preference MD-v5 route or the failed
conservative resource-PPO v4 route, and it grants no model-training,
promotion, packaging, naming, or upload authority by itself.

## Fixed sources

The self-hashed inventory is
`tools/checkpoints/md-next-recent-20260729-31/inventory.json`, inventory SHA-256
`514490bf9a22bfd5cf83dec97a6ee66f6d24178e8be20706411c2156a9c0290a`.
It contains 13,110 unique episodes with zero episode-ID overlap:

- 2026-07-29: 4,386 episodes, archive SHA-256
  `dbe39b021aebe75c46805105604b56c6263ebb0b4f49d7bc41025517a1f4168e`;
- 2026-07-30: 4,384 episodes, archive SHA-256
  `bff4c800225fab906e367390a7926f8e1077d0eb4ac0e39fa79d2cf4afcbadde`;
- 2026-07-31: 4,340 episodes, archive SHA-256
  `c19ad47fa061c8838da1375d7cb0a49efd8ed9674d53be39b6246623a578ed31`.

Every replay JSON member must be parsed. Filename/team-name grep or any other
candidate-file prefilter is forbidden. Eligibility is determined only after
decoding the registered 60-card lists from each replay.

## Eligibility and isolation

- Candidate model examples use exact-list Grimmsnarl seats only, matching
  deck SHA-256
  `c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af`.
- Game identity and content identity must be deduplicated against each other
  and against the locked through-July-28 development corpus.
- A duplicate game or duplicate replay content is included once and retains
  complete source membership.
- Splits are game-grouped. No seat or decision from one game may cross splits.
- The recent pool uses a deterministic 90% train / 10% development-validation
  assignment from the first unsigned 64 bits of
  `SHA256("ptcg.md-next.recent-corpus.v1\\0" + game_uid)`.
- Split assignment is append-stable and independent of rating, deck matchup,
  outcome, decision count, policy agreement, or any model metric.
- Existing through-July-28 split membership remains unchanged when older data
  is included by a later model lock.

## Temporal reservation

The first complete official daily dataset on or after 2026-08-01 with zero
game-UID and content overlap is reserved as untouched temporal evidence. It
cannot be used for training, hyperparameter choice, candidate selection, or
threshold changes before the next candidate is fixed and passes its direct
gameplay gates.

## Method boundary

The next training lock must separately specify its objective, baseline,
trainable scope, recent-versus-historical sampling mass, label authority,
offline rejection screens, and direct gameplay gates before it reads any
candidate outcome. Merely adding these days to the failed July-29
winner-preference objective or repeating resource-PPO with more games is not
authorized.
