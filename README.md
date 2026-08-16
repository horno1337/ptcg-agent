# PTCG ABC

A Kaggle Simulation agent for the Pokémon TCG AI Battle Challenge (competition
`116727`). The final agent is an **exact-deck Alakazam specialist**: a NumPy
behaviour-cloned policy over public observations, three deterministic safety
guards, and a never-crash rules fallback underneath everything.

Competition closed **2026-08-16 23:59 UTC**.

## Status at close

Both live submission slots hold the byte-identical exact-Alakazam specialist
`submission-alakazam-august-1-unsigned.tar.gz` (SHA-256 `93b462fa…c3877a`,
tag `dobi-v3-alakazam`, commit `d382a6a`) as Kaggle submissions **55545158**
and **55545162**.

**Both were still calibrating at close** — they were `PENDING` at the single
post-upload read and were deliberately not polled. Ladder ratings need roughly
100 games / 48h to settle, so no final score is recorded here, and none should
be inferred. The duplicate upload is intentional: two byte-identical archives
are a same-agent control, so any gap between their ratings is ladder variance
rather than strength.

To read them once enough games accumulate:

```bash
python tools/research/analyze_alakazam_probe_pair.py --out <report>
```

It makes exactly one `ListEpisodes` call per submission, then works offline. It
replays every logged action of our seat through the exact uploaded archive; a
match rate below ~98% means the ladder was not running the measured policy.

## Architecture

Five files carry all behaviour:

| File | Role |
| --- | --- |
| `agent/policy.py` | dispatcher + hand-written rules |
| `agent/turn_search.py` | guarded belief/turn planner (opt-in, `PTCG_TURN_SEARCH=1`) |
| `agent/search_policy.py` | retired PIMC + shared engine helpers |
| `agent/model.py` | NumPy-only inference |
| `agent/features.py` | encoding, shared with the Torch trainer |

`agent/safety.py` wraps everything: legality repair plus a per-game clock. It
stays paranoid and dumb and never depends on policy internals — a crash,
illegal action, or timeout is an instant ladder loss. Every layer fails soft
into the next: turn search → legacy search → reflex → rules → safety repair.

The shipped Alakazam stack adds, on top of that base:

- `agent/alakazam_bc.py` — separate MAIN and CARD heads, both fine-tuned from
  frozen Qu-v2B on 13,046 exact-list August games.
- `agent/alakazam_lethal_guards.py` — the Dudunsparce suicide veto and Powerful
  Hand lethal preservation.
- `agent/alakazam_battle_cage.py` — the public-state Battle Cage guard.

## What shipped, and what did not

Three deterministic guards and one fine-tune passed their preregistered gates
but were **held rather than uploaded**, because two slots could not hold both
them and the confirmed insurance agent, and FIFO eviction would have forfeited
the duplicate-sampling control before it produced a single reading.

| Candidate | Gate result (8,192 games/arm) | Disposition |
| --- | --- | --- |
| Exact-Alakazam August fine-tune | **+4.89 pp**, CI95 [+3.57, +6.21] | **Live, both slots** |
| Battle Cage guard | +6.99 pp, CI95 [+5.73, +8.25] | Held, packaged |
| Two board-safety guards | +3.94 pp, CI95 [+2.78, +5.10] | Held, packaged |
| Pilot-behaviour MAIN fine-tune | +1.42 pp, CI95 [+0.21, +2.62] | Paused, packaged |
| Optional-draw deck-out guard | +0.71 pp, CI95 [−0.43, +1.84] | Held (non-regression only) |
| Alakazam MAIN guide fine-tune | +0.84 pp, CI95 [−0.44, +2.11] | **Rejected, discarded** |
| Grimmsnarl current-meta refresh | +0.53 pp, CI95 [−0.72, +1.78] | Not promoted |

Effects measured against different controls are not additive; each row names
its own control in `CLAUDE.md`.

## Evaluation discipline

The ladder is the only real eval, so local gates propose and the ladder
disposes. Everything below was learned the expensive way:

- **Sample size was always the binding constraint.** A measured A/A null with
  byte-identical weights in both arms returns +3.03 pp at n=512 and −0.52 pp at
  n=8,192. Every historical "rejected, CI crossed zero" verdict at n ≤ 2,048 is
  uninformative about effects below ~4 pp.
- **Standard prospective sizes are fixed before running**, never chosen from an
  observed point estimate: `--size 2pp` = 8,192 games/arm, `--size 1pp` =
  32,768 games/arm.
- **Never promote or veto on a slice.** Slice CIs run ±2 to ±6 pp even at
  8,192 games/arm. One candidate's "only negative slice" became its *best*
  slice at four times the power.
- **Pairing buys only ~15% variance reduction** — the native engine RNG is
  unseedable, so arms diverge immediately. Buy power with games.
- **Test the exact extracted tarball, never repository imports.** A legal
  random smoke cannot detect a silently swallowed model exception, because the
  rules fallback is also legal. Two early packages ran the hand-written rules
  on the ladder while passing every local check; the cause was a mode-0600
  `weights.npz` unreadable under a different runtime UID.

## Layout

```
main.py                  entrypoint; survives Kaggle's exec loader (no __file__)
agent/                   the shipped runtime — policy, safety, specialists, weights
decks/                   registered decklists (deck.csv is the live one)
data/                    generated card dumps (tools/dump_cards.py — never hand-edit)
tools/                   training, packaging, evaluation
  research/              per-experiment gates, analyses and post-mortems
  baselines/             frozen controls, e.g. qu-v1-weights.npz
tests/                   142 files; test_safety.py must stay green for any agent/ change
docs/research-log.md     the archived ladder research log and post-mortems
CLAUDE.md                cycle-by-cycle handoff record and standing rules
```

Deliberately outside the repo: the competition engine source
(`~/Desktop/ptcg_engine/`), the official sample bundle
(`~/Desktop/sample_submission/`), episode logs (`~/Desktop/ptcg_episodes/`),
and local evidence trees (`tools/checkpoints/`). `cg/libcg.so` is injected at
build time only and is never committed.

## Commands

```bash
tools/build_engine.sh                       # compile engine (ENGINE_SRC overrides)
python tests/test_safety.py                 # legality fuzz, no engine needed
python tools/eval.py 30 random              # rules-agent smoke
python tools/build_submission.py            # package (injects cg/libcg.so; CG_LIB)

# fast runtime validation
~/.venvs/ptcg-rl/bin/python -m pytest -q \
  tests/test_qu_v2_deployment.py tests/test_dobi_v1_card_runtime.py \
  tests/test_festival_lead.py tests/test_safety.py \
  tests/test_alakazam_lethal_guards.py tests/test_alakazam_battle_cage.py
```

Training runs in `~/.venvs/ptcg-rl` (Torch + CUDA); the Kaggle CLI lives at
`~/.venvs/kaggle/bin/kaggle` and is not on `PATH`. Packaging must not import
Torch.

## Provenance rules

- One change per submission. Deck edits and agent edits go in separate commits.
- Tag every package on a clean tree, so a ladder result maps to exact code and
  weights.
- The maintainer names every tag and approves every upload. Training, eval and
  packaging may run autonomously; `kaggle submit` never does.
