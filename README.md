# PTCG ABC — Rule-Based Agent Infrastructure

Agent scaffold for the Pokémon TCG AI Battle Challenge (Kaggle Simulation track).

## Layout

```
main.py                  # submission entrypoint: exports `agent(obs) -> list[int]`
agent/
  safety.py              # never-crash wrapper: repair, fallback, 10-min time budget
  policy.py              # rule-based policy v0 (the only file you tune)
  obsview.py             # read-only helpers + enum constants over the raw obs dict
  cards.py               # card/attack DB lookups (from data/*.json)
data/
  cards.json             # 1,267 cards dumped from the engine (tools/dump_cards.py)
  attacks.json           # 1,556 attacks
decks/
  deck.csv               # the deck the agent registers (60 card IDs, one per line)
  sample.csv             # known-valid deck, used by the baseline agents
tools/
  build_engine.sh        # compile official engine source -> engine/libcg.so
  cabt.py                # ctypes bindings + local battle runner (kaggle-compatible)
  eval.py                # N-game win-rate eval; --save-losses dumps replays/
  run_local.py           # single game + replay.json (viewer format)
  visualizer.html        # open in a browser, pick a replay JSON -> official viewer
  dump_cards.py          # regenerate data/*.json from the engine build
  build_submission.py    # package submission.tar.gz
tests/test_safety.py     # legality fuzz over all 11 SelectTypes
engine/                  # libcg.so build output (gitignored, license)
```

## Setup

Local play runs on the official engine source (competition-use-only, so it is
NOT in this repo — keep it outside, e.g. `~/Desktop/ptcg_engine/`):

```bash
ENGINE_SRC="$HOME/Desktop/ptcg_engine/ptcgProgram 22" tools/build_engine.sh
python tests/test_safety.py
python tools/eval.py 30 random
```

Needs g++ with C++20; no Python dependencies. `kaggle-environments==1.30.1`
(requirements.txt) is only needed if you want the official Kaggle wrapper
instead — the ladder runs the same engine either way.

## Agent contract (cabt)

- `agent(obs) -> list[int]`
- `obs["select"] is None` → deck registration: return 60 card IDs.
- Otherwise return option indices from `obs["select"]["option"]`,
  between `minCount` and `maxCount` of them.
- Crash / invalid action / timeout = instant loss. `safety.py` guarantees a
  legal answer no matter what `policy.py` does.
- Time: 600 s per player per game (`remainingOverageTime`); safety.py keeps a
  30 s panic reserve.

## Reviewing games

```bash
python tools/eval.py 50 self --save-losses   # losses land in replays/*_L.json
python tools/run_local.py random             # one game -> replay.json
```

Open `tools/visualizer.html` in a browser and pick a replay file — it opens
the official viewer (note: this uploads the replay to ptcgvis.heroz.jp to
render, so it needs internet). Offline analysis should read the JSON directly.

## Where to iterate

1. **Deck** — edit `decks/deck.csv`. This is the highest-leverage knob;
   competitors report simple decks piloted cleanly outperform complex ones.
2. **MAIN priorities** — `MAIN_PRIORITY` + `_score_main_option` in policy.py.
3. **Context handlers** — `choose_card` branches in policy.py (targeting,
   discard value, search picks). Watch loss replays to find the dumb moves.
4. Only trust the real ladder: local win rates have poor rank correlation.
   Use tools/eval.py to catch regressions/crashes, not to pick decks.

Longer term: the engine also exports a `SearchBegin`/`SearchStep` API
(determinized search over hidden information) — unbound so far, but at
~0.01s/game locally it's the obvious road past hand-written rules.

## Baseline results (v0, sample deck, local engine build)

150 games, 0 errors, ~0.01s/game: 92% vs random, 60% vs first-option,
42% self-play (seat variance; expect ~50%).

## Submit

One change per submission, one commit per change; tag what you ship so ladder
results map back to exact code (see CLAUDE.md):

```bash
git tag ladder-vN                  # on a clean tree
python tools/build_submission.py
kaggle competitions submit pokemon-tcg-ai-battle -f submission.tar.gz -m "ladder-vN"
```
