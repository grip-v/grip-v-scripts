# shell_game_tracking_v2

A harder, more varied shell game. Every clip is 1280x720 at 24 fps, rendered with PIL.

## What a clip shows
- 3-8 identical cups in a row on a shelf, one cup colour per clip (6-colour palette). Small slot numbers
  (1..N from the left) are drawn under the shelf for the whole clip.
- 1-4 balls (never more than cups-1, at most one per cup). Each ball has a distinct colour and a kind:
  small ball, big ball, ring, ball with a triangle, or ball with a star. Questions name balls by colour + kind
  ("the red ring", "the green ball with a star").
- Reveal (about 1.5-2.5 s, longer with more balls): the cups hiding balls are raised and every ball sits on the
  shelf in plain view; the cups then come down. Balls are never drawn again, so later frames leak nothing.
- Then the cups swap places one swap at a time (never overlapping). One cup of each pair hops over the row
  while the other slides level. The last swap ends at least 1.0 s before the clip ends, so the final state is
  settled.

## Tiers (`scene["tier"]`, drawn from the rng)
| tier | cups | swaps | swap duration |
|---|---|---|---|
| easy | 3-4 | 4-6 | 0.75-0.95 s |
| medium | 5-6 | 7-10 | 0.55-0.75 s |
| hard | 7-8 | 11-16 | 0.42-0.55 s |

Cup width and spacing scale with the cup count so 8 cups fit in 1280 px.

## 5-rung ladder (one question per rung, >= 2 variants each, stored as `derivation_key` + `params`)
1. Observation: `ball_start_slot` (which slot hides ball X), `start_slot_ball` (what is under slot k, or
   "nothing"), `ball_count`.
2. Relational: `balls_adjacent` (do balls X and Y start in neighbouring slots), `swap_involves_ball` (does ball
   X's cup take part in swap k).
3. Measurement: `swap_count`, `ball_moves` (how many times ball X's cup changes slot).
4. Integration: `ball_final_slot` (slot holding ball X at the end), `final_slot_ball` (which ball, or
   "nothing", is under slot k at the end).
5. Counterfactual: `cf_skip_swap` (where would ball X be if swap k never happened), `cf_extra_swap` (where
   would ball X be if one more swap between two named slots followed the clip).

Options: 2-4 distinct options, shuffled; yes/no questions have 2.

## Closed loop
`answers.py` re-derives every answer from the stored scene alone (it imports neither `sim.py` nor
`questions.py`), replaying the swaps as a permutation of cups. `run.py validate` rebuilds each clip from its
seed, compares scene and questions to the annotation, re-derives every answer, checks tier/cup/ball/timing
invariants (`sim.check`) and inspects the videos.

## Run
```
python run.py generate --n 10 --seed 0 --output-dir <dir>
python run.py validate --dataset-dir <dir>
```
