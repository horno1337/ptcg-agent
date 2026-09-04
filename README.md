# PTCG ABC

[![CI](https://github.com/horno1337/ptcg-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/horno1337/ptcg-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A layered game-playing agent built for Kaggle's **Pokémon TCG AI Battle
Challenge** (Simulations competition `116727`) — a NumPy-only policy network,
a family of per-deck specialists behavior-cloned from top ladder play,
hand-written deterministic safety guards, and a rules-based fallback, all
wrapped in a safety layer whose only job is to guarantee the agent never
crashes, never times out, and never submits an illegal move.

The competition ran a public ladder: every submission plays real games
against other players' agents and is scored on the results. This repository
is the codebase behind the final entry, plus the training and evaluation
pipeline that produced it. **The competition's own game engine is not part of
this repository** — see [The engine boundary](#the-engine-boundary) below.

## Contents

- [Architecture](#architecture)
- [The learned policy](#the-learned-policy)
- [Deterministic safety guards](#deterministic-safety-guards)
- [Training pipeline](#training-pipeline)
- [Evaluation methodology](#evaluation-methodology)
- [Packaging and reproducibility](#packaging-and-reproducibility)
- [The engine boundary](#the-engine-boundary)
- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Running it locally](#running-it-locally)
- [What worked, what didn't](#what-worked-what-didnt)
- [License](#license)

## Architecture

Every decision the agent makes flows through the same cascade. Each stage
only has to do its own job well; if it fails for any reason, control falls
through to something simpler underneath it. The one hard rule, enforced at
the outermost layer, is that the agent **always returns a legal action,
inside the time budget** — a crash, an illegal move, or a timeout is an
automatic loss on the ladder, so nothing upstream of `safety.py` is trusted
to fail safely on its own.

```
                    observation JSON (legal options, board state, time left)
                                    │
   ┌────────────────────────────────────────────────────────────────────┐
   │ safety.py — time-budget check, blanket exception handling,         │
   │             legality repair on whatever comes back                 │
   └───────────────────────────────┬────────────────────────────────────┘
                                    ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ policy.decide()                                                    │
   │  1. exact-deck rule controller     (only when the registered deck  │
   │                                      matches one it was written for)│
   │  2. deterministic lethal override  (optional, disabled by default) │
   │  3. _model_decide()                                                │
   │       a. per-deck specialist BC heads   (hash-pinned, opt-in)      │
   │       b. general policy net             (NumPy inference)          │
   │       c. turn / legacy search           (opt-in, research-grade)   │
   │  4. decide_rules() — hand-written heuristic fallback               │
   └───────────────────────────────┬────────────────────────────────────┘
                                    ▼
                         legal action (option indices)
```

Deterministic safety guards (below) are a fifth kind of intervention on top
of this cascade, not a generic stage in it — see
[Deterministic safety guards](#deterministic-safety-guards) for how, and how
inconsistently, they're actually wired in.

Five files own everything above the guards and specialists:

| File | Role |
| --- | --- |
| `agent/safety.py` | the never-crash wrapper: time budget, exception handling, legality repair |
| `agent/policy.py` | the dispatcher plus the hand-written rules fallback |
| `agent/model.py` | NumPy-only inference for the learned policy |
| `agent/features.py` | observation → feature encoding, shared with the training code so inference and training can never drift apart |
| `agent/turn_search.py` | an opt-in, disabled-by-default belief-aware turn planner (see [What worked, what didn't](#what-worked-what-didnt)) |

`agent/obsview.py` and `agent/cards.py` are read-only helpers over the raw
observation and the card database — no decision logic lives there, so every
module downstream of them sees the same, simple view of the board.

## The learned policy

The competition's observation is public-information-only: your hand, both
players' boards, and a list of currently legal options — never the
opponent's hand or deck order. `agent/features.py` (mirrored line-for-line by
the Torch trainer) turns that into a fixed-shape feature set with **zero
hidden information**, so the exact same encoder that trains the network is
the one that runs it at inference time.

The network itself is a small pointer-style option scorer, not a
fixed-output classifier — it has to work over a *variable-length, variable
meaning* list of legal actions each turn:

- a card-embedding table, shared across every board slot;
- a state trunk over the 13 board slots (active/bench for both players plus
  the stadium) and pooled hand/discard embeddings, reduced to a state vector;
- a value head off that state vector;
- an option head that scores each legal option — including a virtual STOP
  action for multi-select prompts — by combining that option's own features,
  its card embedding, and the shared state vector.

Inference is pure NumPy; the submission never imports Torch. Weights are
exported once from the Torch trainer and loaded lazily — if the weights file
is missing or fails to load, the agent simply keeps using the rules fallback
below it, silently and safely.

On top of that general policy sit **specialist heads**, one behavior-cloned
pair (a "main decisions" head and a "card-selection" head) per meta deck the
agent was tuned against. Each specialist is scoped as narrowly as possible on
purpose:

- it is trained only from real, top-ladder replays of games where the
  learner's exact 60-card decklist matches its target;
- at runtime it re-derives a hash of both its own weights file and the
  currently registered decklist, and refuses to act unless both match
  exactly;
- any load failure, hash mismatch, or unexpected exception makes it return
  "no opinion," which falls through to the general policy net rather than
  ever guessing on a deck it wasn't trained for.

That scoping is what makes it safe to layer many specialists (Alakazam,
Dragapult, Lucario, Grimmsnarl, Froslass, an event "Festival Lead" deck) in
the same dispatcher without any of them being able to misfire on the wrong
matchup.

## Deterministic safety guards

Not every mistake is one more training data would fix. A handful of board
states are rare enough that a learned policy sees too few of them to be
reliable, but severe enough — an unforced loss, most of the time — that
they're worth handling with a few lines of provably-correct logic instead.
Four such guards were built, each one small, independently testable, and
validated against a preregistered statistical gate before being trusted:

- **suicide veto** — refuse an ability that would shuffle a player's only
  Pokémon back into the deck, emptying the board and losing on the spot;
- **lethal preservation** — refuse to spend hand cards below the exact count
  a known lethal attack needs while that attack is still available, and
  proactively draw back up to it when a safe option exists;
- **counter-tech recognition** — play a specific reactive counter-play at the
  first legal, useful opportunity, based only on what the opponent has
  publicly revealed (needed for a card that postdated most of the training
  data, so the learned policy has no real opinion on it yet);
- **deck-out guard** — decline an optional draw whose *fixed* card count
  would draw the last card from the deck, unless a visible attack provably
  wins the game that same turn.

Guards don't hijack the decision outright — they act as an inference-time
mask over the specialist head's own output (forbidding just the losing
option and re-asking the same head for its next-best answer), so the model
still picks its own best remaining move rather than following a hand-coded
order of operations.

**How they actually get wired in is worth being honest about.** Rather than
adding another generic dispatcher toggle, each guard was validated against
one specific candidate build and spliced into a *copy* of `policy.py` by a
dedicated, one-off packaging script that starts from a previously-shipped
frozen archive rather than from the live worktree — see
[Packaging and reproducibility](#packaging-and-reproducibility). That keeps
every already-proven routing decision byte-identical to whatever had already
run on the ladder, but it also means the guards are not uniformly reachable
by running this repository's own `agent/policy.py`: the counter-tech guard
(`agent/alakazam_battle_cage.py`) *is* wired into the generic dispatcher,
behind `PTCG_ALAKAZAM_BATTLE_CAGE`; the suicide-veto and lethal-preservation
guards (`agent/alakazam_lethal_guards.py`) are not called from anywhere in
the tracked dispatcher at all, and the packaging script that used to splice
them in depends on a frozen tarball that was never committed to this repo.
The guard code and its tests are real and passed a real gate — reproducing
the exact package that ran them needs more than this repository alone.

## Training pipeline

- **Corpus.** Replays are pulled from the competition leaderboard, both
  seats of every game. Indexing is content-hashed and append-stable, so
  re-running it after downloading more replays never silently moves an
  already-locked train/validation/test split. Duplicate agents under
  different submission aliases are collapsed by source membership so one
  strong player's games can't quietly dominate the corpus.
- **Base policy.** Two complementary approaches, run in a separate training
  environment (Torch + CUDA) and exported to the NumPy-only production
  format: outcome-weighted behavior cloning over the replay corpus, and an
  anchored league PPO stage (mixed rules/random/self-play opponents plus a
  KL trust region back to a frozen prior policy) for the cases pure imitation
  underfits.
- **Specialists.** Each per-deck head is fine-tuned *from* the frozen general
  policy — trunk and value head frozen, only the small option/context head
  trainable, a tight KL anchor to the parent — so a deck-specific overlay
  can improve on its matchup without drifting away from the general policy's
  overall judgment.
- **Deliberate dead ends**, kept here so they aren't repeated: vanilla PPO
  fine-tuning from scratch, mirror-only self-play, simply scaling up the
  network on the same corpus, unweighted "just imitate the winner" BC, and a
  hand-coded "always take the available knockout" override (it reliably beat
  a single opponent instance while quietly worsening overall win rate, by
  skipping board development and target selection to grab an early trade).

## Evaluation methodology

The headline lesson of this project, paid for the expensive way, is that
**a local win-rate number is a proposal, not a result** — only the live
ladder disposes. Everything else follows from taking that seriously:

- **Every gate is a paired field test** run over a shared, deterministic
  game environment against a fixed field of opponents, reporting a win-rate
  *delta* with a 95% confidence interval — never a bare percentage.
- **Sample sizes are fixed before the run**, from a noise floor that was
  measured empirically rather than assumed: running the *identical* policy
  against itself at several sample sizes showed the estimator's variance was
  sound, but that most of this project's early "the CI crossed zero, so
  reject" verdicts were simply too small a sample to say anything about a
  realistic effect size. Standard prospective sizes (thousands of games per
  arm) are chosen from that measurement, not from a result that's already
  been peeked at.
- **No promotion or rejection on a slice.** A per-matchup breakdown at low
  sample size is noise wearing a story — one candidate's supposed "only
  losing matchup" became its *best* matchup once the same test was rerun at
  four times the sample size.
- **Every promoted build is replayed from the exact packaged submission
  archive**, executed as a non-owner OS user, comparing its actions against
  the reference model action-for-action. This exists because it caught a
  real bug class: a build that passed every in-repo check but silently ran
  the dumb rules fallback on the actual competition infrastructure, because
  its weights file happened to be packaged with permissions unreadable under
  a different runtime user.

## Packaging and reproducibility

There are two different ways a submission tarball gets built here, and only
one of them is fully reproducible from a clean clone of this repository:

- **The generic path**, `tools/build_submission.py`, packages `main.py`,
  `agent/`, `data/`, and `decks/` exactly as they sit in the worktree,
  canonicalizes file permissions, and injects the engine binary from a local
  path. This is what [Running it locally](#running-it-locally) uses, and it
  reproduces byte-for-byte whatever `decks/deck.csv` and `agent/` currently
  contain.
- **Late-cycle specialist candidates** (the per-deck BC heads, the Battle
  Cage guard, the board-safety guards) were instead built by dedicated,
  named scripts (`tools/build_alakazam_august_submission.py`,
  `tools/build_alakazam_cage_submission.py`, …) that start from a specific,
  already-shipped *frozen prior archive*, pinned by its SHA-256, and apply a
  small, targeted patch — swap in a decklist, add a weights file, splice a
  new block into that archive's own copy of `agent/policy.py`. This keeps
  every other routing decision byte-identical to a package that had already
  cleared its gates, at the cost of the result depending on a local tarball
  that is deliberately gitignored (submission archives aren't checked in)
  and therefore isn't reproducible from this repository alone.

One consequence worth calling out directly: **`decks/deck.csv` in this
repository reflects whichever registration was most recently being iterated
on locally, not necessarily the decklist behind any one historical
submission.** The final competition entry shipped as an exact-deck Alakazam
specialist, assembled by the second build path above from its own pinned
decklist and weights — `decks/deck.csv` itself was last changed to a
different (Grimmsnarl-family) registration earlier in the project and was
never required to track it. `docs/research-log.md` and `CLAUDE.md` record
which archive hash and tag actually shipped when; treat `decks/deck.csv` as
"whatever's currently being worked on," not as a changelog.

## The engine boundary

The actual Pokémon TCG rules engine — legality, damage and effect
resolution, the game loop the competition environment runs — is a separate,
closed-source artifact that belongs to the competition's private engine and
sample-submission bundles, not to this project. It is **deliberately never
committed to this repository**:

- the engine's source and the official sample bundle live outside the repo
  entirely, on the machine that builds submissions;
- the compiled engine binary (`cg/libcg.so`) is pulled in only when packaging
  a submission tarball, and only from a local path (`build_submission.py`,
  overridable via `CG_LIB`) — it is injected at build time and is never
  committed or pushed;
- `engine/` is git-ignored, and the local test suite (`tests/test_safety.py`
  first and foremost) is written to run without the engine at all.

What this repository *does* depend on is the engine's public **decision
protocol**: an observation is a JSON dict describing whichever prompt the
game is currently asking (its type, its legal options, and enough board
state to reason about them), and an action is the list of option indices the
agent chooses. `agent/obsview.py` is a thin, read-only wrapper around exactly
that public shape — it carries no simulation logic of any kind.

## Repository layout

```
main.py                  entrypoint — Kaggle exec's this file directly, no __file__
agent/                    the shipped runtime
  safety.py               never-crash wrapper + per-game clock
  policy.py               dispatcher + hand-written rules fallback
  turn_search.py          belief-aware turn planner (research, opt-in)
  search_policy.py         retired search + shared engine helpers
  model.py                 NumPy inference for the general policy net
  features.py               observation -> feature encoding (shared with training)
  qu_v2_features.py         production public-relational encoder
  obsview.py / cards.py      read-only observation / card-database helpers
  alakazam_bc.py, dragapult_bc.py, lucario_bc.py, ...   per-deck specialist heads
  alakazam_lethal_guards.py, alakazam_battle_cage.py    deterministic safety guards
  *_weights.npz              trained weights for the general policy and specialists
decks/                    decklists explored over the project; deck.csv is a rolling
                          development default, not a record of what shipped — see
                          Packaging and reproducibility
data/                     generated card/attack dumps (tools/dump_cards.py — never hand-edited)
tools/                    training, packaging and evaluation
  train.py, train_vec.py, train_deck_adapter.py   BC / league PPO / specialist fine-tuning
  eval_ab.py, run_parallel_gate.py                paired field gates with fixed prospective sizes
  build_submission.py                             packages the tarball, injects the engine binary
  audit_submission_runtime.py                      exact-archive, non-owner provenance audit
  research/                                        per-experiment gates, analyses and post-mortems
  checkpoints/                                      evidence trees for shipped decisions — gitignored, local-only, NOT in the repo
tests/                    ~140 files; test_safety.py must stay green for any agent/ change
docs/research-log.md      the original, chronological research log and post-mortems
docs/submissions/         manifest records of the packaged submission builds
CLAUDE.md                 the maintainer's running cycle-by-cycle handoff notes
LICENSE                   MIT
```

The submission tarballs, the local `tools/checkpoints/` evidence trees, and
the rolling `agent/weights.npz` are gitignored and do not ship with a normal
clone; only the small, hash-pinned specialist weights the agent loads at
runtime (`agent/*_weights.npz`) are tracked. Transfer the exact archives and
checkpoints separately if byte-for-byte reproduction of a historical build is
needed.

## Environment

The project deliberately spans three environments, in increasing order of
what they require:

| Environment | Needs | Used for |
| --- | --- | --- |
| **Submission runtime** | NumPy only (Python 3.10+) | what actually runs on the ladder; never imports Torch or the local engine |
| **Local evaluation** | `pip install -r requirements.txt` (NumPy + `kaggle-environments`) plus the compiled engine (`tools/build_engine.sh`) | running games locally, packaging, regenerating `data/` |
| **Training / gates** | a separate PyTorch + CUDA virtualenv (machine- and GPU-specific, intentionally unpinned) | behavior cloning, league PPO, specialist fine-tuning |

The runtime is deliberately dependency-light so the submission can never fail
to import on the competition host. `requirements.txt` documents the local
evaluation environment; the training environment is described in `CLAUDE.md`
and is not reproducible from this repository alone (it needs the private
engine and locally-downloaded replays).

## Running it locally

```bash
pip install -r requirements.txt        # NumPy + kaggle-environments (local eval)

python tests/test_safety.py            # legality fuzz — the one check that needs NO engine build
tools/build_engine.sh                  # compile the private engine (ENGINE_SRC overrides)
python tools/eval.py 30 random         # rules-agent smoke test, needs the engine build
python tools/build_submission.py       # package submission.tar.gz (injects cg/libcg.so)
```

`tests/test_safety.py` is the portable smoke test — it runs on a clean clone
with nothing but NumPy. The broader `tests/` suite needs the private engine
(and, for a handful of historical gate harnesses, local evidence under
`tools/checkpoints/` that is intentionally not committed — see below), so it is
not expected to pass end-to-end from a bare clone.

## What worked, what didn't

The short version of a much longer story:

- **Imitation from a curated, provenance-locked replay corpus beat every RL
  variant tried from scratch.** Vanilla PPO and mirror self-play both
  regressed; the only RL that helped was anchored league PPO layered *on
  top of* a BC-trained policy.
- **Runtime search lost on the ladder even after winning locally.** A
  determinized information-set search matched or beat the reflex policy in
  local evaluation, then lost on the real ladder — the root cause was a
  strategy-fusion problem in the search itself, not a lack of compute
  budget (the ladder's time budget is generous; the agent used a small
  fraction of it).
- **A single hand-coded "always take the knockout" rule made the agent
  measurably worse**, by short-circuiting board development and target
  selection for an early trade — worth remembering any time a deterministic
  override looks like a free win.
- **Packaging bugs cost more than modeling bugs.** More than one promoted
  candidate looked correct in every local test and then silently ran the
  fallback rules on the ladder, because of file permissions, a stale import
  path, or a UID mismatch — which is why every shipped build is replayed
  from the exact packaged archive as a non-owner user before it's trusted.
- **Patching a frozen archive instead of the live dispatcher trades one
  drift risk for another.** Building late-cycle candidates from a pinned
  prior package (see [Packaging and reproducibility](#packaging-and-reproducibility))
  kept already-proven routing decisions untouched, but the text-anchor those
  scripts patch against has since moved as `agent/policy.py` kept evolving —
  so the tool that used to assemble the guarded package no longer runs
  against current source. Worth a second look before trusting either the
  worktree or a packaging script as "the" definition of shipped behavior
  without checking the other.

The full, chronological version — every experiment, gate result, and dead
end, including the ones this summary skips — is in
[`docs/research-log.md`](docs/research-log.md).

## License

Released under the [MIT License](LICENSE). The Pokémon TCG rules engine and
the competition's sample bundles are **not** part of this repository and are
not covered by this license — see [The engine boundary](#the-engine-boundary).
Pokémon and Pokémon TCG are trademarks of Nintendo / Creatures Inc. / GAME
FREAK inc.; this is an independent, non-commercial research project with no
affiliation.
