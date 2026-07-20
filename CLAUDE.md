# PTCG ABC agent

Kaggle Simulation agent for the Pokémon TCG AI Battle Challenge. README.md has
the architecture, tag lineage, post-mortems (research log), and workflows —
read it before proposing strategy changes; every rule below was paid for on
the ladder.

## Rules of the codebase

- Behavior lives in five files: `agent/policy.py` (dispatcher + rules),
  `agent/turn_search.py` (guarded belief/turn planner),
  `agent/search_policy.py` (retired PIMC + shared engine helpers), `agent/model.py` (inference),
  `agent/features.py` (encoding, shared with the torch trainer — never let
  train/inference features drift). `safety.py` stays paranoid and dumb; it
  must never depend on policy internals. `obsview.py`/`cards.py` are
  read-only helpers; no game logic.
- A crash, invalid action, or timeout is an instant ladder loss. Any change
  to `agent/` must keep `python tests/test_safety.py` green, and every layer
  must fail soft into the next (turn search -> legacy search -> reflex ->
  rules -> safety repair).
- Feature changes: bump `FEAT_VERSION` in features.py, append-only scalars;
  old weights load through the compat shim in model.py. Any frozen NumPy-net
  adapter must call `features.encode_options_for_net`, never the latest encoder
  directly, so v1/v2 action identities remain byte-compatible.
- `data/*.json` and `agent/meta_decks.json` are generated
  (`tools/dump_cards.py`, `tools/mine_meta_decks.py`) — never hand-edit.
- Engine source and sample bundle are competition-use-only and stay OUTSIDE
  the repo (`~/Desktop/ptcg_engine/`, `~/Desktop/sample_submission/`);
  `engine/` is gitignored; `cg/libcg.so` is injected into the tarball at
  build time only — never committed or pushed.
- Time bank is LARGE — do not treat compute as scarce. The ladder gives
  ~600s overage per game (`actTimeout=0`; thinking drains `remainingOverageTime`)
  across ~65 of our decisions, i.e. **~5–9s available per decision**, and the
  shipped reflex agent spends ~1s per *game* (<0.2% of budget). We are NOT
  time-starved: we can afford many search iterations / seconds of compute per
  decision. The v2/v3 "det starvation" was self-imposed per-decision throttling
  plus the value-net / strategy-fusion failures — NOT a total-time limit. When
  testing search, SPEND the budget (raise the per-decision budget and the
  world/iteration counts), keeping a safety reserve so the clock never reaches
  zero (instant loss). Unknown to measure empirically on the ladder: the real
  per-iteration wall-cost on the competition CPU (don't assume a fixed slowdown).
- Runtime search (PIMC) is retired (`search_policy.ENABLED = False`) — it lost on
  the ladder even after local parity (v2/v3 post-mortems), the root cause being
  strategy fusion, not lack of time.
- ACTIVE DIRECTION: **belief-aware turn search + search-policy iteration**.
  The implementation audit falsified the old `ismcts.py` prototype: it was an
  open-loop action-index tree, merged distinct information states, could issue
  illegal descendant multi-picks, and used a flat intermediate leaf score.
  `agent/turn_search.py` supersedes it with semantic complete actions,
  frequency-weighted posterior particles plus unknown mass, exact zone
  reconciliation, synchronized information-set beams, turn-boundary evaluation,
  and paired evidence/robust-margin gates. It is wired behind `PTCG_TURN_SEARCH=1` and
  remains OFF by default until `tools/eval_turn_search.py` and a one-change
  ladder A/B justify it. New learning uses `tools/selfplay_teacher.py` soft root
  targets and `tools/train_teacher.py`; the old PIMC `selfplay_search.py` is
  research history, not the next flywheel.

## Evaluation & gates

- Local win rates do not predict ladder rank; use them only for A/B between
  our own agents (150+ games, paired seats) and crash-catching. 60-game
  evals swing ±13%.
- PRIMARY ship gate for weights: candidate >= champion on
  `eval_ab.py --opp pool:8` (160 games/net). Secondary: mirror 160+,
  `--opp meta:<i>` vs a threat deck, `eval.py 200 random`. Mirror is never
  sufficient alone — it inverted on the v5 failure. Local gates propose;
  the ladder disposes.
- Corpus rule: band-representative games (our own ladder episodes, both
  seats) are the base; top-team scouting is seasoning, not foundation.
- Planner gates must additionally report root coverage, reflex disagreement,
  valid particles, fallback reasons, p50/p95/max latency, cumulative clock,
  errors, hashes and confidence intervals. Evaluate exact-known, withheld
  variants and unseen archetypes; a gain only when the exact deck is in
  `meta_decks.json` is a rejection.

## Training

- Venv: `~/.venvs/ptcg-rl` (torch+CUDA). Works: outcome-weighted BC on
  episode logs; anchored league PPO (`--league-rules 0.30 --league-random
  0.25 --bc-anchor <episodes> --lr 5e-5`). Dead ends: vanilla PPO
  fine-tuning, mirror-only self-play, bigger nets on the same corpus, pure
  top-play BC. Anti-passive: league_random seats + `gen_antipassive.py`
  demos in the anchor dir.
- New online rollouts use `tools/rl_env.py` + `tools/train_vec.py`: complete
  multi-pick actions, legal empty STOP, fixed learner deck, paired
  deck/pilot/seat schedules, explicit truncations/errors, terminal-only
  returns, and full trainer checkpoints. It refuses to write shipped weights.
  `tools/eval_ab.py` uses the same environment and scores
  `(W + 0.5D) / scheduled`; any truncation/infrastructure failure invalidates
  the gate rather than becoming a draw.
- Policy iteration: `tools/selfplay_teacher.py` keeps paired search scores and
  soft root distributions on learner-reached states against a deck/policy pool;
  `tools/train_teacher.py` uses a game-grouped holdout and the band BC anchor.
  Keep valid low-margin targets and states from lost games. Do not reduce them
  to the winning hard action. Distillation trains only the option head so the
  shared value trunk stays intact, and strict source/config hashes reject mixed
  shards. Episode downloads land in `~/Desktop/ptcg_episodes/` (user does this
  manually); refresh `meta_decks.json` after new downloads.

## Submission discipline

The ladder is the only real eval; every submission is an A/B measurement.

- One change per submission. Deck edits (`decks/deck.csv`) and agent edits
  go in separate commits — never mixed. The deck stays frozen while training
  data is deck-conditioned.
- Commit `agent/weights.npz` deliberately (`git add -f`) before packaging so
  the ladder result maps to exact code+weights. Tag every package on a
  clean tree.
- **The user names the tag and approves every upload** — training/eval/
  commit chains may run autonomously; `kaggle submit` never does.
- Ladder champion: cvkpaper-v4 (reflex-only, ft3 weights fa0fc1c) carries the
  team score at ~655. Working-tree development weights are cvkpaper-v6/ft10
  (~641), not the champion. Always name the baseline explicitly.

## Commands

```bash
tools/build_engine.sh                     # compile engine (ENGINE_SRC overrides)
python tests/test_safety.py               # legality fuzz, no engine needed
python tools/eval.py 30 random            # rules-agent smoke (engine build)
python tools/train.py --bc DIR --iters 0  # behavior cloning (RL venv)
python tools/eval_turn_search.py 40 --opp mirror --budget .5 --particles 8
python tools/selfplay_teacher.py OUT.jsonl N --opp pool:8 --worker ID
~/.venvs/ptcg-rl/bin/python tools/train_teacher.py OUT.jsonl --resume CKPT
python tools/mine_meta_decks.py           # refresh opponent-model library
python tools/eval_search.py 20 --budget 0.03 --seed N   # pre-ship: search @ ladder-like compute
python tools/build_submission.py          # package (injects cg/libcg.so; CG_LIB)
```

`[]` is a valid STOP action when `minCount == 0`; only `None` means policy
failure. Dataset, training, inference, eval and search code must preserve that
distinction.
