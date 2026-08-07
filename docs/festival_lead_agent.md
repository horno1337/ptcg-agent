# Festival Lead agent specification

This agent is locked to the exact 60-card Majkel1337 registration in
`decks/festival_lead_majkel1337.csv`. It does not generalize the guide's Rabsca
or Shaymin branches because neither card is registered.

The deterministic layer owns the stable combo structure:

1. Open `Goldeen > Applin > Grookey` when a choice exists.
2. Fill the bench with Applin and Grookey, retaining one useful Festival Lead
   pivot rather than spending every slot indiscriminately.
3. Establish Thwackey, Festival Grounds, an energized active Dipplin, and a
   prepared replacement attacker.
4. Use Boom Boom Groove before attacking. Search for the currently missing
   combo link rather than following one static card ranking.
5. Preserve Festival Grounds when it is already in play, and prefer one-Prize
   attackers and two-Prize targets when the public board permits that trade.
6. Fail closed outside the exact registered deck.

The learned layer is intentionally narrow. Separate frozen-backbone BC heads
cover `ST_MAIN` turn sequencing and `ST_CARD` toolbox/search choices. The rule
controller remains the legality and structural fallback. Other prompt types
stay deterministic because the replay cohort shows near-complete agreement
there. PPO is deferred until the BC hybrid proves it can beat the generic
Qu-v2B policy in an exact-list paired-seat mirror.

The source cohort is Kaggle submission `55307654`: 68 public games, of which 67
have one uniquely identifiable Festival Lead seat and one is an exact mirror.
The deterministic corpus split uses seed `202608077` and fractions 70/15/15.
