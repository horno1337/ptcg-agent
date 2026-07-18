# PTCG ABC — Search + RL Agent

Kaggle Simulation agent for the Pokémon TCG AI Battle Challenge. The agent
plays a three-layer decision stack, each layer falling back to the next on
any failure:

1. **Determinized search** (`agent/search_policy.py`) — for single-pick
   decisions: predict every hidden zone, build K concrete worlds via the
   engine's `SearchBegin` API, rehearse each option through our turn *and*
   the opponent's full reply (2-ply), score with the value net, pick the
   best mean. Opponent hidden cards are reconstructed by matching their
   revealed cards against decklists mined from ladder episodes
   (`agent/meta_decks.json`).
2. **Reflex policy net** (`agent/model.py` + `agent/weights.npz`) — a
   pointer-style option scorer (card embeddings + state encoder + per-option
   logits + value head), numpy-only at inference. Trained by behavior
   cloning on leaderboard episodes, then anchored league PPO
   (`tools/train.py`).
3. **Rule-based policy** (`agent/policy.py`) — the hand-written fallback;
   also the league opponent and eval baseline.

`agent/safety.py` wraps everything: legality repair, per-game time budget
(authoritative `remainingOverageTime`; `actTimeout=0` on this ladder, so
think-time drains it directly), never crash.

## Results so far (ladder = public score; local = head-to-head win rates)

| Generation | What | Ladder |
|---|---|---|
| alakazam-v2 | rules + Alakazam deck | 534 |
| cvkpaper-v0 | + reflex net (BC on 16 episodes + anchored PPO) | 666 |
| cvkpaper-v1 | + determinized 1-ply search | 867 spike → 635 |
| cvkpaper-v2 | + confidence gate, 2-ply, 51-deck library | 493 |
| cvkpaper-v3 | + evidence floor (MIN_DETS=3, adaptive ply) | 516 |
| cvkpaper-v4 | reflex-only kill switch (search retired) | ~655 — current champion |
| cvkpaper-v5 | cycle3d weights on v4 code | 497 — reverted to ft3 (fa0fc1c) |

Research log (each vs the then-champion, 100-200 game evals):

- **Behavior cloning works immediately**: 16 expert episodes ≈ 60% vs rules;
  outcome weighting (clone the *winning seat*, losses at 0.1) +10 points.
- **The corpus ceiling is real**: 4→16→69→79 episodes, loss-targeted data,
  and a 3.5× bigger net all produced statistical ties vs the shipped
  reflex champion. Reflex strength is capped by what the demonstrations
  contain.
- **Vanilla PPO fine-tuning erodes a BC policy**; anchored league PPO
  (33% games vs rules, expert-NLL mixed into every update, low lr) holds
  and slightly improves it. Mirror-only self-play collapses entropy.
- **Search converts spare clock into strength**: same net, 1-ply = 56%,
  2-ply = 64% vs reflex. We use ~135s of the 600s budget.
- **Flywheel (expert iteration), in progress**: the search agent self-plays
  (`tools/selfplay_search.py`, episode-format output), and the net retrains
  on its games — the data source now improves with the agent. Cycle 1
  failed (mirror-only data: apprentice farms weak opponents, loses
  head-to-heads); cycle 2 needs diverse generation opponents.
- **Search inverts under determinization starvation** (the cvkpaper-v2
  post-mortem): on ladder CPUs one 2-ply det fills the whole budget, and
  acting on a single sampled world scored 31.6% vs reflex (n=320 repro via
  `tools/eval_search.py --budget 0.03 --min-dets 1 --cap-mult 1e9`). Dev-box
  dets are ~0.03s, so no local eval at any realistic budget ever starved —
  always pre-ship with the throttled harness. Fix: evidence floor
  (MIN_DETS=3 complete worlds or defer to reflex), 2→1-ply downgrade when a
  det costs > budget/3, hard 2×budget per-decision cap. At starved compute
  the fixed agent is reflex-parity (50-52%, n=800).
- **Search itself was the harm, not just starvation** (the cvkpaper-v3
  post-mortem): with the evidence floor in place and no time pressure, v3
  (516) still overrode the net on 63% of contested picks (replay analysis,
  n=383) and went 1-8. Mirror A/B can't see off-distribution failure — the
  value head + meta world model degrade vs unfamiliar opponents while the BC
  policy head doesn't. Runtime search is retired
  (`search_policy.ENABLED = False`); `eval_search.py` force-enables it for
  experiments. cvkpaper-v4 is the reflex-only kill switch (~655 baseline).
- **A full local gate battery can still lie** (the cvkpaper-v5 post-mortem):
  cycle3d passed all three gate axes (mirror 59.4%!, threat deck, random)
  and shipped as v5 at 497 vs v4's 655 — same code, one weights commit, the
  cleanest A/B yet, a clean negative. The ladder band (500-700) is
  archetype *variants* piloted by mid agents; no local axis covered it, and
  a mirror edge can coexist with broad regression (style overfit to
  self-similar opponents). Root cause: pure top-play BC — top-team scouting
  is seasoning, not foundation. Weights reverted to ft3 (fa0fc1c). Local
  gates propose; the ladder disposes.
- **The pool gate retrodicts v5 where mirror inverted it**: cycle3d 61.2%
  vs ft3 65.0% on `eval_ab --opp pool:8` (160 games/net). pool:8 is now the
  primary ship gate for weights. Direction is trustworthy, ladder magnitude
  is not (rating spirals amplify).
- **Meta watch (2026-07)**: Luca took #1 swapping to Grimmsnarl/Munkidori,
  which beats our archetype 37-14 in corpus games; the mid-band pool is
  archetype variants, not top-meta lists. Corpus at this point: ~285 ladder
  episodes / 40k+ samples.

## Layout

```
main.py                    # entrypoint; survives kaggle's exec loader (no __file__)
agent/
  safety.py                # never-crash wrapper + per-game clock
  policy.py                # dispatcher: search -> reflex -> rules
  search_policy.py         # determinized 2-ply value search (self-contained lib binding)
  model.py                 # numpy inference net; shapes derive from weights.npz
  features.py              # obs -> features, shared by torch trainer and numpy inference
  obsview.py / cards.py    # read-only obs helpers / card DB lookups
  weights.npz              # shipped net (committed deliberately at each ship)
  meta_decks.json          # decklists mined from episodes (opponent modeling)
data/                      # card/attack dumps (tools/dump_cards.py — generated)
decks/deck.csv             # the deck (matches the top ladder Alakazam list)
tools/
  cabt.py                  # engine bindings + battle runner (+ search API, search_begin_input)
  train.py                 # torch twin: BC (--bc), anchored league PPO, --arch, npz export
  il_dataset.py            # episode JSONs -> (obs, action, reward); winner-seat weighting
  selfplay_search.py       # flywheel generation: search self-play in episode format
  mine_meta_decks.py       # episodes -> agent/meta_decks.json
  eval.py / run_local.py   # rule-agent eval / single game + replay
  build_submission.py      # packages submission; injects official cg/libcg.so (CG_LIB)
tests/test_safety.py       # legality fuzz — must stay green for any agent/ change
```

## Environments & data locations (this machine)

- Engine source (competition-use-only, never committed): `~/Desktop/ptcg_engine/`;
  build with `tools/build_engine.sh` -> `engine/libcg.so`.
- Official sample bundle incl. prebuilt `cg/libcg.so`: `~/Desktop/sample_submission/`.
- Training venv: `~/.venvs/ptcg-rl` (torch + CUDA). Kaggle CLI: `~/.venvs/kaggle`.
- Episode logs (downloaded from leaderboard game pages): `~/Desktop/ptcg_episodes/`.
- Search self-play output: `~/Desktop/ptcg_selfplay/`.

## Workflows

```bash
python tests/test_safety.py                       # always before committing agent/
python tools/eval.py 30 random                    # rules-agent smoke (engine build)

# imitation + RL (in the training venv)
python tools/train.py --bc ~/Desktop/ptcg_episodes --iters 0 \
    --out tools/checkpoints/bcN/weights.npz --ckpt-dir tools/checkpoints/bcN
python tools/train.py --resume .../bc_best.pt --iters 40 --lr 5e-5 \
    --league-rules 0.33 --bc-anchor ~/Desktop/ptcg_episodes ...

# flywheel generation (N parallel workers)
PTCG_SEARCH_BUDGET=0.25 python tools/selfplay_search.py ~/Desktop/ptcg_selfplay 150 w1

# refresh opponent-model library after new episode downloads
python tools/mine_meta_decks.py

# package + ship (clean tree, tag first; tag name chosen by the maintainer)
git tag <name> && python tools/build_submission.py
~/.venvs/kaggle/bin/kaggle competitions submit pokemon-tcg-ai-battle \
    -f submission.tar.gz -m "<name>"
```

## Hard-won gotchas

- Kaggle `exec`s `main.py`: **no `__file__`** — keep the try/except fallback.
- `actTimeout=0`: every second of thinking drains the 600s overage. Search
  self-limits (budget/decision, disables below 150s, safety panics at 30s).
- ST_CARD options carry no cardId — resolve via `(area, index, playerIndex)`
  and `select["deck"]` (revealed during deck searches).
- `len(prize)` is the remaining-prize count (entries are null face-down).
- The engine aborts on double `GameInitialize` — loaders share a PID-stamped
  guard.
- Local evals over many games in one process: safety's clock now trusts
  per-game `remainingOverageTime`, so this is safe — but keep evals paired
  and 150+ games; 60-game evals swing ±13%.
- Feature changes: bump `FEAT_VERSION`, append-only scalars; the numpy Net
  truncates for older weight files so shipped baselines stay evaluable.
