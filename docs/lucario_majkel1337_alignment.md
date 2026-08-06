# Vincent's Lucario guide aligned to Majkel1337's Hariyama list

This document translates Vincent's Dudunsparce Lucario guide to the exact
60-card Mega Lucario/Hariyama registration used by Majkel1337. It is a
strategy and training specification, not permission to copy rules from the
guide blindly. Advice that depends on cards absent from Majkel's list is
explicitly removed or replaced.

## Exact simulator deck

The registered deck is stored in
`decks/lucario_majkel1337_hariyama.csv`.

| Count | ID | Card |
|---:|---:|---|
| 13 | 6 | Basic {F} Energy |
| 2 | 673 | Makuhita |
| 2 | 674 | Hariyama |
| 2 | 675 | Lunatone |
| 3 | 676 | Solrock |
| 3 | 677 | Riolu (80 HP, Accelerating Stab) |
| 4 | 678 | Mega Lucario ex |
| 4 | 1121 | Ultra Ball |
| 2 | 1123 | Switch |
| 4 | 1141 | Premium Power Pro |
| 4 | 1142 | Fighting Gong |
| 4 | 1152 | Poke Pad |
| 1 | 1159 | Hero's Cape |
| 2 | 1182 | Boss's Orders |
| 4 | 1213 | Judge |
| 4 | 1227 | Lillie's Determination |
| 2 | 1229 | Wally's Compassion |

The local simulator accepts this exact list. The guide's newer Chaos Rising
list is not an exact simulator target because Special Red Card is absent from
the current card pool.

## What transfers from the guide

| Vincent guide concept | Status | Majkel translation |
|---|---|---|
| Establish Riolu on turns 1-2 | Retain | Establish one Riolu and a second when the opposing deck can remove an unevolved Riolu from the Bench. |
| Lunatone + Solrock early | Retain, strengthen | This is the list's only persistent board draw engine. Preserve a pair and use Lunar Cycle to seed Basic Fighting Energy for Aura Jab. |
| Aura Jab develops the next attacker | Retain, extend | Concentrate energy on a Riolu/Lucario line for Mega Brave or on Makuhita/Hariyama for a one-prize Wild Press turn. |
| Prize-map with one-prize attackers | Retain, extend | Riolu, Lunatone, Solrock and especially Hariyama can force an extra knockout. Hariyama attacks for 210 before modifiers and gives up one prize. |
| Preserve Boss and damage modifiers | Retain | There are only two Boss. Spend the minimum PPP count that reaches the intended knockout. |
| Sometimes keep a strong hand and do nothing | Retain | Do not replace a useful large hand with Lillie merely because it is legal. Judge needs an even stricter test because it also resets the opponent. |
| Dudunsparce draw and Stamp insurance | Replace | There is no Run Away Draw. Disruption recovery is Lunatone + Solrock plus access to a Basic Fighting Energy. |
| Four Boss's Orders | Replace | Two Boss plus two banked Hariyama evolutions provide four potential gust effects. Hariyama gust does not consume the Supporter for the turn. |
| Special Red Card + Boss | Replace | Judge + Heave-Ho Catcher is the analogous disruption-plus-gust line. It requires an established Makuhita and a Hariyama in hand. |
| Maximum Belt and Gravity Mountain | Replace | PPP is the only damage modifier. Hero's Cape instead changes Lucario's survival threshold from 340 to 440 HP. |
| Wally is too situational | Override | Majkel deliberately runs two Wally with Cape. Treat Wally as a conditional prize-denial and reset line, not generic healing. |
| Rocky Energy protection | Remove | Basic Energy and Switch do not reproduce Rocky Energy's defensive effect. |
| Poffin and mixed Riolu split | Remove | All three Riolu have 80 HP and the list has no Poffin. |
| Hilda under Item lock | Remove | There is no comparable evolution tutor under Item lock. Lucario must be established proactively. |
| Stadium timing and control | Remove | The exact list has no Stadium and no Stadium removal. |

## Adapted game plan

### Opening setup

1. Establish a Riolu. Establish a second Riolu before it is needed when the
   opposing deck can snipe or place damage counters on the Bench.
2. Complete Lunatone + Solrock early. Lunar Cycle both draws cards and puts a
   Basic Fighting Energy into the discard for Aura Jab.
3. Bench Makuhita when a future gust or one-prize attacker is valuable. Do not
   fill the final Bench slot with a redundant engine body if that prevents a
   second Riolu or Makuhita.
4. Fighting Gong finds every Basic Pokemon in the deck or a Basic Energy.
   Poke Pad finds every non-rule-box Pokemon, including Hariyama. Ultra Ball is
   the direct Lucario search and can turn Basic Energy into Aura Jab fuel.
5. Preserve the last accessible copy of a Pokemon line required by the chosen
   route. The list has no Pokemon recovery.

### Building attack continuity

Aura Jab does 130 for one Energy and attaches up to three Basic Fighting
Energy from the discard to Benched Pokemon. Its development value depends on
making a real attacker available, not merely moving three Energy:

- put two Energy on Riolu/Lucario for a future Mega Brave;
- put three Energy on Makuhita/Hariyama for Wild Press when the one-prize map
  is better;
- use Lunatone or Solrock as the planned one-prize attacker only when its
  damage and prize-map value are explicit;
- avoid spreading the attachments so that no backup attacker reaches an
  attack threshold.

### Hariyama as banked gust and attacker

Hariyama's Heave-Ho Catcher switches an opposing Benched Pokemon Active when
Hariyama is played from hand to evolve Makuhita. The evolution is therefore a
resource:

- hold Hariyama when there is no useful gust target and Wild Press is not
  needed;
- evolve when the gust improves the current prize route or Hariyama must
  attack;
- prefer the Hariyama gust to Boss when the turn also needs Judge, Lillie or
  Wally;
- decline an optional switch that would make the reachable knockout or prize
  route worse;
- remember that Wild Press deals 70 damage to Hariyama after attacking.

### Hero's Cape and Wally

Hero's Cape makes Mega Lucario ex a 440-HP Pokemon while the Tool remains
attached. Wally heals all damage from a Mega Evolution Pokemon ex, then returns
all of its Energy to hand. The important line is therefore sequencing and
prize denial:

1. Use Wally only when the heal changes survival, denies a three-prize
   knockout, or materially improves the opponent's required knockout count.
2. Play Wally before the manual Energy attachment unless a measured exception
   justifies losing that attachment back to hand.
3. After Wally, a formerly powered Active Lucario normally has only the one
   manual attachment available, so Aura Jab or a Switch to an already powered
   attacker is the natural continuation; Mega Brave is not automatically
   available.
4. Do not Wally when returning the Energy destroys an available knockout and
   the heal does not change the prize map.

### Judge, Lillie and hand discipline

- Lillie is strongest at six prizes, where it draws eight, and when the
  current hand cannot complete the turn.
- Do not reset a large useful hand whose current and next-turn routes are
  already present.
- Judge is attractive when the opponent's hand is materially better than
  ours. It is unattractive when the opponent already has four or fewer cards
  and our hand contains the complete route.
- Hariyama enables gust plus Judge, Lillie or Wally in the same turn. That is
  a central advantage over spending the Supporter on Boss.

## Exact damage table

PPP adds 30 damage to the Active target for the turn and multiple copies
stack. Before Weakness and Resistance:

| Attack | Base | With 1/2/3/4 PPP |
|---|---:|---:|
| Riolu - Accelerating Stab | 30 | 60 / 90 / 120 / 150 |
| Lunatone - Power Gem | 50 | 80 / 110 / 140 / 170 |
| Solrock - Cosmic Beam | 70 | 100 / 130 / 160 / 190 |
| Lucario - Aura Jab | 130 | 160 / 190 / 220 / 250 |
| Hariyama - Wild Press | 210 | 240 / 270 / 300 / 330 |
| Lucario - Mega Brave | 270 | 300 / 330 / 360 / 390 |

Cosmic Beam does nothing without Lunatone on the Bench and ignores Weakness
and Resistance. Wild Press also deals 70 to Hariyama. Use the minimum PPP count
that reaches the target unless deliberate hand thinning before a forced reset
has separately been justified.

## What may become a guard versus a training diagnostic

The guide is not an action oracle. Only narrow, observable constraints should
be considered as fail-soft runtime guards:

- take an available game-winning knockout;
- do not choose Cosmic Beam without Lunatone on the Bench;
- do not spend extra PPP after the intended knockout threshold;
- sequence Wally before a manual attachment when the attachment would simply
  return to hand;
- when possible, make Aura Jab leave at least one intended Bench attacker at
  its required Energy threshold;
- do not discard an unrecoverable last copy required by the live route.

The following are soft labels or replay diagnostics until direct gameplay or
counterfactual evidence supports them:

- when to establish a second Riolu;
- whether to hold or evolve Hariyama;
- Lucario versus Hariyama prize mapping;
- Judge versus keeping the current hand;
- Wally versus attacking immediately;
- Aura Jab versus Mega Brave;
- which opposing target is worth gusting;
- whether Cape actually crosses the opponent's reachable damage threshold.

## Replay and training evidence

- The exact Majkel list appeared in 407 recent official seats on August 2-3,
  and those games are already contained in the downloaded Majkel replay set;
  they are not 407 additional demonstrations.
- The replay set contains 507 unique games. One exact-list mirror cannot be
  assigned to Majkel's seat, leaving 506 resolved games: 368 wins and 138
  losses (72.7% observed win rate).
- The exact list was 407 of 502 recent Mega Lucario seats (81.1%). Vincent's
  exact Dudunsparce list appeared zero times in the five-day screen.
- In Majkel's resolved sample, the largest matchups were Grimmsnarl 84-77,
  Alakazam 61-30, Mega Froslass 81-8, Mega Lopunny 45-5 and Dragapult 22-9.

These are deck-plus-pilot ladder observations under rating-based matchmaking,
not causal deck-strength estimates. The 72.7% result is evidence that the
demonstrations come from a strong working policy, not a promise that an
imitation model will reproduce it.

The action logs support the main guide translation. Across all 507 replay
files, Aura Jab appeared in 90.5% of games and Mega Brave in 72.6%. Of 1,890
logged Aura-Jab Energy placements, 843 (44.6%) went to the Hariyama line and
795 (42.1%) to the Lucario line; Solrock and Lunatone received the remainder.
The correct lesson is therefore "power the next concrete attacker," not
"always power a second Lucario." Hariyama evolved in 64.1% of games but used
Wild Press in only 20.7%, which is consistent with its evolution gust being a
primary role and its attack being conditional. Hero's Cape targeted the
Lucario line on 252 of 294 observed attachments (85.7%).

Other observed game-use rates were Judge 69.4%, Wally 35.7%, Boss 40.0%,
Hero's Cape 58.0%, Fighting Gong 89.3%, Poke Pad 89.2% and Ultra Ball 83.8%.
These frequencies size the relevant policy surface; they do not establish
that playing a card causes a win. In particular, Judge, Wally and Cape all
appeared more often in losses than wins because losses were longer and more
adverse. That correlation must not be turned into a rule against using them.

All hard action labels must be conditioned on the exact 60-card fingerprint.
Similar Lucario lists with Dudunsparce, Hilda, Stadiums, Special Red Card or
Rocky Energy encode different legal resources and should not enter the policy
loss as if they were interchangeable. Winning Majkel seats may supervise the
policy; losing seats remain useful for value/state coverage but are not proof
that their actions are correct. Newer episodes should be reserved as a
temporal gameplay check rather than silently folded into training after their
outcomes are inspected.
