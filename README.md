# PTCG ABC — A Layered NumPy Agent for the Pokémon TCG AI Battle Challenge

[![CI](https://github.com/horno1337/ptcg-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/horno1337/ptcg-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Behavior-cloned deck specialists over a Torch-free policy network, wrapped in
statistically-gated deterministic safety guards and a never-crash safety layer —
built on the principle that the live ladder, not any local win rate, is the only
real judge.*

Built for Kaggle's **Pokémon TCG AI Battle Challenge** (Simulations competition
`116727`). The competition ran a public ladder: every submission plays real
games against other players' agents and is scored on the outcomes. This
repository is the codebase behind the final entry, plus the training and
evaluation pipeline that produced it. The competition's own game engine is
closed-source and **deliberately not included** — see
[The engine boundary](#the-engine-boundary). The full chronological research
log — every experiment, gate, and dead end — is in
[`docs/research-log.md`](docs/research-log.md).

## The design bet

Reinforcement learning from scratch lost; imitation from a curated,
provenance-locked replay corpus won. The agent is therefore not one model but a
**cascade of fallbacks**, each stage simpler and more trusted than the one above
it. The one hard rule, enforced at the outermost layer: the agent **always returns
a legal action within the time budget** — a crash, illegal move, or timeout is
an automatic ladder loss, so nothing upstream is trusted to fail safely.

## Architecture

```
observation JSON (legal options, board state, time left)
        │
  safety.py — time budget · blanket exception handling · legality repair
        │
  policy.decide()
    1. exact-deck rule controller     (only for a deck it was written for)
    2. deterministic lethal override  (optional, off by default)
    3. model:  specialist BC heads → general NumPy policy net → opt-in search
    4. rules fallback                 (hand-written heuristics)
        │
  legal action (option indices)
```

Five files own the runtime above the specialists and guards:

- `agent/safety.py` — never-crash wrapper: time budget, exception handling,
  legality repair;
- `agent/policy.py` — dispatcher plus the hand-written rules fallback;
- `agent/model.py` — Torch-free NumPy inference for the learned policy;
- `agent/features.py` — observation → features, shared with the trainer so
  inference and training can't drift apart;
- `agent/turn_search.py` — opt-in, off-by-default belief-aware turn planner.

## The learned policy

The observation is public-information-only — your hand, both boards, and a list
of legal options, never the opponent's hand or deck order. The network is a
small **pointer-style option scorer**, not a fixed-output classifier: it has to
work over a variable-length, variable-meaning list of legal actions each turn.
A shared card-embedding table feeds a state trunk over the 13 board slots plus
pooled hand/discard embeddings; a value head reads that state, and an option
head scores each legal option — including a virtual STOP action for
multi-select prompts. Inference is pure NumPy; the submission never imports
Torch, and if the weights file is missing or fails to load, the agent silently
keeps using the rules fallback beneath it.

On top of that general policy sit **behavior-cloned specialists**, one
(main-decisions + card-selection) pair per meta deck the agent was tuned against
(Alakazam, Dragapult, Lucario, Grimmsnarl, Froslass, and an event "Festival
Lead" deck). Each is scoped as narrowly as possible: it is trained only on
top-ladder replays where the learner's exact 60-card list matches its target,
and at runtime it hashes both its own weights file and the registered decklist
and **refuses to act unless both match exactly**. Any load failure, hash
mismatch, or unexpected exception makes it return "no opinion," falling through
to the general net rather than guessing on a deck it wasn't trained for. That
scoping is what makes it safe to layer many specialists in one dispatcher
without any of them misfiring on the wrong matchup.

## The deck: Alakazam draw engine

The final entry pilots an **Alakazam draw-engine combo deck**, and the
specialist and guards are all built around one game plan. Alakazam's
**Powerful Hand** attack deals 20 damage for *every card in hand*, so hand size
*is* the damage output: the deck's job is to convert its own deck into a large
hand and swing once for lethal, rather than trading prizes turn by turn.

Key cards and their roles:

- **Abra → Kadabra → Alakazam** — the win condition; this line is prioritized
  ahead of everything else in every search, promotion, and discard decision.
- **Dunsparce → Dudunsparce**, **Fezandipiti ex**, and the searchers (Dawn,
  Hilda, Poké Pad, Poffin) — the engine that turns deck into hand;
  **Enriching Energy** (draw 4 on attach) and **Telepath** double as fuel.
- **Recovery** (Lana's Aid, Night Stretcher, Sacred Ash) — ranked to be kept
  over spare energy in forced discards, so the engine never runs out of pieces.

Because every draw shortens your own clock, the deck's central risk is
**decking yourself out**. The agent encodes that discipline directly: it stops
optional draws at six cards and deck searches at three, opens the expendable
Dunsparce active (never a multi-prize ex, which would hand over easy prizes),
and the deterministic guards exist precisely to protect this plan — preserving
a lethal Powerful Hand and refusing any draw that would mill the last card.

## Deterministic safety guards

Some board states are too rare for a learned policy to handle reliably, but
severe enough — usually an unforced loss — to be worth a few lines of
provably-correct logic. Four were built, each small, independently
testable, and validated against a **preregistered statistical gate** before
being trusted: a suicide veto (refuse an ability that empties your own board), a
lethal-preservation guard (don't spend hand cards below a known lethal count), a
counter-tech recognition rule (play a reactive counter the training data
predates), and a deck-out guard (decline a fixed-count draw that would mill your
last card). Crucially, guards don't hijack the decision — they mask just the
losing option and re-ask the specialist head for its next-best move, so the
model still picks its own play. (How consistently each guard is reachable
through this repository's live dispatcher versus a frozen packaged archive is
documented honestly in the research log.)

## Training pipeline

- **Corpus.** Leaderboard replays, both seats, content-hashed and append-stable
  so re-indexing never silently moves a locked train/validation/test split;
  duplicate agents under different aliases are collapsed so no single player
  dominates the corpus.
- **Base policy.** Two complementary approaches, run in a separate Torch+CUDA
  environment and exported to the NumPy production format: outcome-weighted
  behavior cloning, and an anchored league-PPO stage (mixed rules/random/
  self-play opponents plus a KL trust region back to a frozen prior) for the
  cases pure imitation underfits.
- **Specialists.** Each per-deck head is fine-tuned *from* the frozen general
  policy — trunk and value head frozen, only the small option/context head
  trainable, a tight KL anchor to the parent — so a deck overlay improves its
  matchup without drifting from the general policy's overall judgment.

## Evaluation methodology

The headline lesson of this project, paid for the expensive way, is that
**a local win-rate number is a proposal, not a result — only the live ladder
disposes.** Everything else follows from taking that seriously:

- **Every gate is a paired field test** over a shared, deterministic game
  environment against a fixed field of opponents, reporting a win-rate *delta*
  with a 95% confidence interval — never a bare percentage.
- **Sample sizes are fixed before the run**, from a noise floor measured
  empirically rather than assumed. Running the *identical* policy against itself
  showed that most of this project's early "the CI crossed zero, so reject"
  verdicts were simply too small a sample to say anything; standard prospective
  sizes are thousands of games per arm.
- **No promotion or rejection on a slice.** A per-matchup breakdown at low
  sample size is noise wearing a story — one candidate's supposed "only losing
  matchup" became its *best* matchup once the same test was rerun at four times
  the sample size.
- **Every promoted build is replayed from its exact packaged archive**, executed
  as a non-owner OS user, action-for-action against the reference model. This
  caught a real bug class: a build that passed every in-repo check but silently
  ran the dumb rules fallback on the real competition infrastructure, because
  its weights file was packaged with permissions unreadable under a different
  runtime user.

## Results

Local paired field-gate deltas for the final Alakazam line — each measured
against the *then-current confirmed build* over a fixed opponent field, every
promotion criterion preregistered:

| Change (vs. prior confirmed build) | Win-rate Δ | 95% CI | Games/arm |
| --- | ---: | :---: | ---: |
| Exact-Alakazam BC specialist (MAIN + CARD) | +4.89 pp | [+3.57, +6.21] | 8,192 |
| + Battle Cage counter-tech guard | +6.99 pp | [+5.73, +8.25] | 8,192 |
| + board-safety guards (suicide, lethal preservation) | +3.94 pp | [+2.78, +5.10] | 8,192 |
| + deck-out guard (non-regression) | +0.71 pp | [−0.43, +1.84] | 8,192 |

The deltas are measured against successive controls and are **not additive**.
The BC gate was positive on **all eight matchup slices** — evidence the gain is
broad rather than one favorable matchup. These are local gate deltas, not
ladder scores: consistent with the methodology above, the ladder is the final
judge. The shipped entry is the exact-Alakazam BC specialist; the guarded
variants cleared the same gates and were held under one-change-per-submission
discipline.

## What worked, what didn't

- **Imitation from a curated, provenance-locked corpus beat every from-scratch
  RL variant.** Vanilla PPO and mirror self-play both regressed; the only RL
  that helped was anchored league PPO layered *on top of* a BC-trained policy.
- **Runtime search lost on the ladder even after winning locally.** A
  determinized information-set search matched the reflex policy offline, then
  lost live — a strategy-fusion problem, not a compute limit (the agent used a
  fraction of its time bank).
- **A single hand-coded "always take the knockout" rule made the agent
  measurably worse**, by short-circuiting board development and target selection
  for an early trade — worth remembering whenever a deterministic override looks
  like a free win.
- **Packaging bugs cost more than modeling bugs** — more than one candidate
  passed every local test, then silently ran the fallback live (a permission
  bit, a stale import, a UID mismatch); hence the exact-archive replay above.

## The engine boundary

The Pokémon TCG rules engine — legality, damage, effect resolution, the game
loop — is a separate, closed-source competition artifact and is **never
committed here**: the compiled binary is injected into the tarball at build time
only, `engine/` is gitignored, and the portable test suite runs without it. The
repository depends only on the engine's public **decision protocol** — an
observation is a JSON dict (prompt type, legal options, board state) and an
action is the list of chosen option indices.

## Repository layout & running it

```
main.py        Kaggle entrypoint (exec'd directly)
agent/         the shipped NumPy-only runtime: safety, policy, model, features,
               specialists, guards, and the trained weights
decks/ data/   decklists explored, and generated card/attack dumps
tools/         training, packaging, evaluation, and per-experiment research trees
tests/         ~140 files; tests/test_safety.py is the portable, engine-free check
docs/          research-log.md (the full history) and packaged-submission manifests
```

The submission runtime needs only **NumPy** (Python 3.10+); local evaluation
adds `kaggle-environments` and the compiled engine, and training runs in a
separate Torch+CUDA environment (see `requirements.txt` and `CLAUDE.md`).

```bash
pip install -r requirements.txt
python tests/test_safety.py        # portable legality fuzz — no engine needed
python tools/build_submission.py   # package the tarball (injects the engine binary)
```

## License & attribution

Released under the [MIT License](LICENSE). The rules engine and the
competition's sample bundles are **not** part of this repository and are not
covered by this license. Pokémon and the Pokémon TCG are trademarks of Nintendo
/ Creatures Inc. / GAME FREAK inc.; this is an independent, non-commercial
research project with no affiliation.
