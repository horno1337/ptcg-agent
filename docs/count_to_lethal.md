# Design note: `count_to_lethal` — a deterministic KO/counting helper

Status: **built + first hypothesis falsified** (2026-07-19). `agent/lethal.py`
exists (13 tests green), flag-gated (`PTCG_LETHAL`, default off).

**A/B result (full agent vs rules-Lucario meta:2, N=120):** the *attack-timing*
override does not help. ST_MAIN "force the KO" **regressed −5.8** (takes the KO
before board development / Boss choice); ST_ATTACK-only "pick lethal Powerful Hand"
was **exactly neutral (+0.0)** — because features.py already feeds the net Powerful
Hand's true `20×hand` damage, so the net is *not* lethal-blind and already attacks
correctly. Attack-override disabled; flag stays off.

**Lesson:** "the reflex net can't count" was too broad. It *can* use a counting
feature we hand it (lethal recognition). What it can't do is the multi-step
*planning* — and the real Lucario bleed is prize-trade discipline (Fez ex dying for
2), which the attack-override never addressed. Any future use of `count_to_lethal`
should target **draw-discipline (don't over-draw)** or **prize-trade / Fez-exposure
rules**, not attack timing.

## Why

Three converging findings put a deterministic counting layer ahead of "more/bigger
BC" as the next lever:

1. **Alakazam's skill ceiling is arithmetic + within-turn sequencing** — the guide
   is almost entirely "count to exact lethal, draw just enough, stop, develop the
   rest" and "sequence to leave options open." (See the Lucario matchup: we play it
   at 31% but the deck wins it 61% in skilled hands.)
2. **A reflex net structurally can't do this.** [model.py](../agent/model.py) is a
   single feedforward option-scorer: no world model, no iteration, trained to
   imitate outputs not reconstruct planning. Nets approximate; they don't compute
   exact thresholds ("11 vs 12 cards for lethal"). Bigger nets / more data tied
   (research log) — the limit is architectural, not capacity.
3. **We waste our compute budget; the top of the ladder doesn't.** Measured from
   replays: Majkel/Luca burn ~0.18–0.33s per decision (~10–18s of the 600s budget
   per game); we burn ~0.013s (~1s/game, <0.2% of budget). They compute per move in
   a *bounded* way — not the 135s heavy search that starved us (v2/v3 post-mortems),
   the middle regime we never tried. A deterministic counter costs microseconds and
   leaves ~100× headroom.

**Key doctrine:** spend the budget with **deterministic** logic, not the value-net
search that failed. Search lost for two reasons — determinization starvation *and*
the value head degrading off-distribution. Bounded compute fixes the first; only
determinism (no value net, no probability) fixes the second. This helper can't be
"off-distribution wrong" because it counts, it doesn't guess.

## What it computes

`agent/lethal.py`, pure (no engine, no net, no torch):

```
count_to_lethal(view) -> {reachable_hand, need_for_ko, lethal_available, draws_needed}
```

- Powerful Hand = **20 damage × cards in hand** (2 counters/card; POWERFUL_HAND=1072).
  KO the active ⟹ `need_for_ko = ceil(opp_active_remaining_hp / 20)` cards in hand.
- `reachable_hand` = current hand + draw available *this turn*, per the guide's
  formula, counting only pieces actually present in hand / board / (unprized) deck:
  - Alakazam evolve+ability **+2**, Rare Candy→Alakazam **+1**, Kadabra **+1**
  - Dudunsparce in hand **+2**, Dudunsparce already in play **+3**
  - Enriching energy **+3**, Hilda **+1**, Dawn **+2**
- `lethal_available = reachable_hand >= need_for_ko`.

## Where it hooks in (`policy.py`, narrow + high-confidence only)

- **Draw decisions:** if lethal is reachable, draw *toward exactly* `need_for_ko`,
  then stop. (Kills the over-draw mistake — the guide's #1 counting sin.)
- **Attack decisions:** if hand ≥ `need_for_ko`, take the KO.
- **Surplus:** once at lethal, develop the board with excess cards.

## Safety principle (why this is not the retired search)

Override the net **only on arithmetic certainty** — a *guaranteed* lethal it can
prove. Otherwise fall straight through to the net. No value net, no determinized
worlds, no probabilities. Must keep `tests/test_safety.py` green, never emit an
illegal action, fail soft into net → rules → safety repair.

## Correctness-critical: protection effects

The counters are an *effect*, so several cards **prevent** them and the calculator
MUST check the opponent's active or it will claim a false lethal:
- Blocked by: **Mist energy, Rocky/Fighting-type protection energy, Articuno.**
- NOT blocked by: **Cornerstone Mask Ogerpon ex.**
- Only-on-damage-KO cards (Legacy energy, Lillie's Pearl, Hop's Trevenant) are
  irrelevant here (Powerful Hand doesn't KO with damage, it places counters).

## Open work / risks

1. Protection detection on the opponent active (the correctness gate above).
2. Spread mechanics — v1 counts only what lands on the active; multi-target lethal
   is a later refinement.
3. Obs parsing: hand/board/energy reads and the attack/draw action encodings
   ([obsview.py](../agent/obsview.py) is the schema).

## Evaluation

A/B net+helper vs net-alone on **meta:2 (Lucario)** and **pool:8**; instrument how
often the helper fires and whether it converts KOs the net previously missed.
Ship only through the normal gate battery; the ladder is the real judge.
