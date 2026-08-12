# Dragapult guide evidence — 2026-08-12

This note records project-safe conclusions from the user-provided purchased
Dragapult/Hammers guide. It is a derived rule/evidence inventory, not a copy of
the guide.

## Applicability

The guide's Turin list differs from registered deck `07bed`: it uses a
Dunsparce/Dudunsparce line, Moltres, Risky Ruins, and Special Red Card; `07bed`
instead has a second Budew, Jamming Tower, Judge, Dawn, and one more Psychic
Energy. Exact-list replay evidence takes precedence whenever the lists differ.

The broadly transferable concepts are:

- resolve deterministic setup before making a turn depend on Crushing Hammer;
- choose turn order, Budew use, and liability Pokémon by matchup;
- distribute Phantom Dive counters with future Munkidori conversions in mind;
- distinguish Prize-taking Boss lines from deliberate tempo/damage-bank lines;
- preserve backup attackers and split pre-evolution Energy where possible.

## Checked against the exact-07bed top-pilot corpus

Corpus: 162 replay files, 171 exact-list seats, 108-63, from the two highest
rated exact-list pilots captured through 2026-08-11 20:38 UTC.

### Crushing Hammer

Prompt-level rates are misleading. Across 1,214 MAIN prompts where Hammer was
offered, experts selected it immediately 478 times (39.4%) and the elite head
642 times (52.9%). At the game-turn level, however, experts eventually used a
Hammer on 379/420 eligible turns (90.2%), versus 332/420 (79.0%) for the elite
head. This is primarily a sequencing problem, not evidence for global Hammer
suppression.

Only 80/379 expert Hammer turns (21.1%) also used Unfair Stamp or Judge, while
298/379 (78.6%) ended in an attack. Therefore “allow Hammer only with hand
disruption” is rejected. A future planner should keep Hammer available, order
deterministic setup first, and avoid relying on heads for attack readiness.

Diagnostic: `tools/research/analyze_top_dragapult_hammer_intent.py`.

### Boss's Orders

Of 163 expert Boss turns, 95 had a visible same-turn direct-KO proxy, 49
attacked without reaching that proxy, and 19 did not attack. Fezandipiti ex was
the dominant non-KO target (19/49): strong pilots often bank 200 damage with
Phantom Dive instead of taking an immediate conventional Prize. The elite head
selected Boss at only 3/49 of those non-KO decision points. This explains why
the retired immediate-KO Boss guard was structurally incomplete.

The obvious replacement is still unsafe. “Boss is legal, Phantom Dive is live,
and Fezandipiti ex is benched” was true on 61 expert turns but matched the
expert Fez/Phantom line on only 12 (19.7% precision, 70.6% recall). Do not ship
that presence-only rule. The next Boss experiment must include Prize map,
existing damage, current-Active value, and follow-up counter conversion, or use
paired branches from learner-reached states.

Diagnostics:

- `tools/research/analyze_top_dragapult_boss_intent.py`
- `tools/research/probe_dragapult_boss_fez_trigger.py`

### Phantom Dive

The guide's mirror rule against leaving easy 30-damage Munkidori conversions
does not conflict with the current dead-target allocator: the allocator
preserves every live learned target and changes a selection only after the
chosen target has visibly reached zero HP. It does not force three counters or
globally target the lowest-HP Pokémon.

The allocator passed its direct exact-Grim diagnostic against MD-v1 and
Dobi-v1: aggregate +1.855 points over 1,024 paired units, with MD +3.711 points,
Dobi neutral, 1,435 interventions, and zero faults. It then failed the
separately locked public-signature field confirmation: -0.781 points overall,
CI95 [-3.589,+2.027], exactly neutral on the 102-game Grim slice. The direct
gain is opponent-policy-specific; the allocator remains disabled.

## Ordered next work

1. Build paired outcome labels for Boss and Phantom actions at learner-reached
   states; select on one root cohort and confirm on disjoint roots.
2. Keep KO, tempo/damage-bank, and setup Boss intents separate; the frozen
   occurrence classifier passed a fresh behavior screen but regressed paired
   gameplay (-0.488 points overall, Dragapult -4.264 points).
3. Test a bounded within-turn sequencing layer that delays stochastic Hammer
   behind deterministic setup while preserving eventual Hammer/attack use.
4. Add matchup-specific turn-order/Budew/Hammer priors only after each public
   signature and isolated gameplay gate is specified.
