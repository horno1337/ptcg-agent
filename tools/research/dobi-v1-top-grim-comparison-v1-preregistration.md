# Dobi-v1 versus top Grimmsnarl comparison v1

Status: locked before the expanded leader replay cohort is listed or downloaded.

Date: 2026-08-06 (Europe/Warsaw).

## Question

Which observable decisions and resulting public board states distinguish frozen
Dobi-v1 from the highest-rated current agent registering the same exact
Grimmsnarl list, and do those differences justify a narrowly scoped,
Dobi-anchored behavior-cloning experiment?

This is an observational screen. It cannot establish that copying a leader
action causes a win, and ladder ratings or replay win rates are not direct A/B
evidence.

## Frozen cohorts

- **Leader:** Kaggle submission `55138264`, team `Sixth Sense`, rank 2 at
  1141.0 in the first leaderboard snapshot taken for this experiment. Its deck
  was verified before this lock to be byte-for-byte the same 60-card
  registration as Dobi-v1. The cohort is the newest 300 completed episodes
  returned by the first post-lock `ListEpisodes` call, capped by replay
  availability. Only uniquely resolvable `Sixth Sense` seats with the exact
  deck are valid.
- **Dobi-v1:** the deduplicated replay union of submissions `55193892`,
  `55194498`, `55195586`, `55195620`, `55220532`, and `55305666`. Only uniquely
  resolvable `増殖するG` seats with the exact deck are valid.
- Duplicate episode IDs count once. Ambiguous same-team/exact-mirror seats are
  excluded and reported. No matchup or outcome stratum may be removed after
  results are seen.

The cohorts come from different matchmaking windows. Strength comparisons are
therefore secondary. Behavioral comparisons must be reported overall and by
common matchup, turn order, and seat outcome; sparse cells remain visible.

## Primary read-out

For own turns 1 through 6, compare the following public-state or action
quantities, using fixed unweighted replay games rather than resampled data:

- Grimmsnarl-line development, board width, Munkidori count, Dark-powered
  Munkidori count, Froslass count, energy in play, and prize differential;
- legal `ST_MAIN` options at turn opening, Munkidori ability uses, damage
  transfer selections, attacks, evolutions, attachments, retreats, and forced
  ends;
- game-touch and per-game usage for Spikemuth Gym, Petrel, Buddy-Buddy Poffin,
  Lillie's Determination, Rare Candy, Boss's Orders, Night Stretcher,
  Poké Pad, Froslass, and Munkidori;
- turns, decisions, decisions per turn, and first/second-player record.

Report effect sizes with uncertainty. Multiple turn/metric cells are
Benjamini-Hochberg corrected within the predeclared family. Outcome-conditioned
leader differences are diagnostic only and never become automatic rules.

## Training go/no-go

A targeted BC experiment may be designed only if all of the following hold:

1. At least 100 valid leader games and 30 leader wins are recovered.
2. The leader uses the exact registered Dobi-v1 deck throughout the selected
   games, or every off-list game is retained in the inventory but excluded by
   the predeclared exact-deck condition.
3. At least one model-addressable behavioral family differs by either:
   - at least 10 percentage points in game-touch rate, or
   - an absolute standardized effect of at least 0.25 on two adjacent own
     turns,
   and the direction is compatible with stronger public board/prize outcomes
   rather than merely longer games.
4. The proposed labels are confined to prompt types actually represented in
   the leader cohort. Any `ST_MAIN` and `ST_CARD` changes are separate
   experiments.

If the screen passes, a second lock must freeze the exact training sources,
weights, anchor, temporal validation, and direct gameplay gates before training
begins. Offline loss is screening-only. Frozen Dobi-v1 remains the gameplay
control and no candidate is uploaded without separate user naming and approval.

