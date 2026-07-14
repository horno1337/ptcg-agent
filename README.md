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
tools/
  build_engine.sh        # compile official engine source -> engine/libcg.so
  cabt.py                # ctypes bindings + local battle runner (kaggle-compatible)
  eval.py                # N-game win-rate eval vs random / first / self
  run_local.py           # single game + replay.json (viewer format)
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

## Where to iterate

1. **Deck** — edit `decks/deck.csv`. This is the highest-leverage knob;
   competitors report simple decks piloted cleanly outperform complex ones.
2. **MAIN priorities** — `MAIN_PRIORITY` + `_score_main_option` in policy.py.
3. **Context handlers** — `choose_card` branches in policy.py (targeting,
   discard value, search picks).
4. Only trust the real ladder: local win rates have poor rank correlation.
   Use tools/eval.py to catch regressions/crashes, not to pick decks.

## Baseline results (v0, sample deck)

66 local games, 0 errors: 87% vs random, 65% vs first-option, 50% self-play.

## Submit

```bash
python tools/build_submission.py
kaggle competitions submit pokemon-tcg-ai-battle -f submission.tar.gz -m "v0 rules"
```
