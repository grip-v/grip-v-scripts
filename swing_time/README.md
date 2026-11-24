# swing_time

Video-QA sub-benchmark adapted from **Swing Time (Jane Street, Aug 2016)**: path planning with rope swings.

## What each clip shows

A particle crosses a board of squares by hooking a rope on a post (a grid corner) and swinging around it until it
lands on the centre of another square the same distance from the post. A swing

- uses a rope at most `cap` units long (squares are 1 unit),
- goes the shorter way round (either way on an exact half turn),
- must keep the particle on the board,
- may not sweep over another post (a post inside the swept wedge, or on its edge, snags the rope),
- may not land on a hatched pit square (pits scenario),
- costs its rope length.

The particle always takes the cheapest route from its start square to the checkered goal square, and every board is
kept only if that route is unique by at least `MARGIN` = 0.25 units over any other route. The rope, the arc it is
tracing and a ring around the pivot post exist **only while a swing is in progress**; every landing leaves a
persistent outline marker on its square.

The only text in the frame is the clock (top-left box, seconds to 0.1 s, truncated from the frame number). Posts are
named in questions by colour and shape ("the teal diamond post"; colours are unique per board), the start square is
where the particle begins, the goal is the checkered square, pits are the hatched squares.

## Scenarios (`scene["scenario"]`, uniform per seed)

| scenario | setup |
|---|---|
| `left_right` | 7-12 x 5-8 board, start in the left column, goal in the right column (or mirrored), 5-9 posts |
| `bottom_top` | 5-9 x 6-8 board, start in the bottom row, goal in the top row (or mirrored) |
| `corner` | start in one corner, goal in the opposite corner |
| `dense` | 8-10 posts on a 7-10 x 5-7 board, ropes capped at 3.2: many short swings |
| `pits` | like `left_right` plus 3-8 hatched pit squares the particle may not land on |

## Random knobs

Board size, post count and positions, rope cap (4.0 / 4.6 / 5.0, or 3.2 in `dense`), start/goal squares, pits; post
colours (10-colour palette, unique per board) and shapes (circle, square, triangle, diamond) and size (24-32 px);
light or dark theme, background, plain or checker board, grid line style (solid / dashed / dotted), start tint, goal
checker colours, particle fill/rim/pattern (solid, bullseye, ring) and size (26-34 px), rope colour / width (4-6 px)
/ style (solid, dashed, beaded), arc trail colour / width / style, landing marker colour / shape (ring, box,
diamond); camera framing (cell size 54-88 px and board position); timing (intro, rest between swings, outro, swing
speed). Clips are 3-6 swings, at most 10 s (240 frames at 24 fps, 1280x720).

## Timing rules

All times are integer frames. The first swing starts >= 0.5 s after the start and the last landing is >= 0.5 s before
the end; every swing lasts >= 0.5 s (12 frames) and rests between swings last >= 1/3 s (8 frames), so every event a
question uses is at least 8 frames from any other. Time answers are what the clock shows at the event frame
(`frame * 10 // 24` tenths); time options are >= 0.6 s apart.

## The 10-question ladder (`LADDER = "ten"`, common/ladder5.py)

Per clip: L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3 questions (exactly 10, split drawn per seed), ordered by level as
q01..q10, never the same `derivation_key` with the same params twice. Every question has exactly four distinct
options; answer letters are uniform.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `n_swings`, `n_posts_used`, `first_pivot`, `last_pivot`, `n_posts_total`, `n_pits` |
| L2 relational | `pivot_of_swing`, `swing_direction` (clockwise / counter-clockwise, less / more than a quarter turn; never asked within 15 deg of 90), `next_pivot`, `land_quadrant`, `earliest_of` |
| L3 measurement | `longest_rope`, `rope_of_post`, `total_rope`, `swing_duration`, `land_time`, `first_attach_time` |
| L4 integration | `pivot_sequence`, `pivot_reused`, `n_clockwise`, `n_long_ropes`, `longest_swing_post` (top two >= 0.5 s apart) |
| L5 counterfactual | `cf_remove_post`, `cf_remove_n_swings`, `cf_remove_first_pivot`, `cf_short_rope`, `cf_no_pits` |

Counterfactual questions state the rules in the question text and are answered by re-solving the changed board;
the ones that ask about the new route's shape need that route to be unique by `MARGIN` too. Length answers are given
to 2 decimals and are only asked when the value is >= 0.0015 units from a rounding boundary; options are >= 0.15
apart. L2-L4 questions all need the motion (`requires_motion=True`); counterfactuals and post/pit counts are read
from the board.

## Wrong-option labels (`option_errors`)

Every question also stores `option_errors`, a list aligned with `options`: `null` at the correct option and, at each
wrong option, the mistake that produces it (the list and one-line meanings are `answers.MISTAKES`). Examples:
`pivot_of_adjacent_swing` (the post of the swing just before or after), `left_out_one_swing` (the total without one
swing), `misread_one_rope`, `wrong_direction` / `wrong_turn_size`, `mirrored_up_down`, `cost_before_the_change`
(a what-if answered with the unchanged board's cost), `claims_no_route` / `claims_a_route`, `off_by_one`,
`time_of_another_event`. Generic tags (`other_total`, `nearby_time`, `other_post`, `other_sequence`) mark options that
match no specific mistake. The answer key is never deliberately wrong: the labels describe the distractors only.

The generator labels options from its own route; `run.py validate` re-labels every question from the rendered
frames (`answers.option_errors`, using the same rules on the frame-read route) and fails on any difference.

## min_fps

Every question stores `params["min_fps"]`: 24 / (smallest gap in frames between the events it relies on), rounded
up, at least 4 for clock readings and duration comparisons. Rests are >= 8 frames and swings >= 12, so every
question is answerable at 4 fps; at 4 fps every swing is seen in at least two sampled frames with its rope and pivot
ring, and landing markers persist to the end.

## Closed loop

`answers.py` never reads the generator's route or timing tables. It reads the swings off the rendered timeline
(`render.state(scene, frame)` for every frame: where the particle is and which post the rope is tied to), so pivots,
direction, turn size, rope length, landing squares and event frames come from what is drawn. Counterfactual routes
use an independent solver: exact integer cross products in doubled coordinates (sim.py uses float angles) and
Bellman-Ford (sim.py uses Dijkstra). `sim.check` rejects any scene whose route is not the unique cheapest by
`MARGIN`, whose timing breaks the rules above, or whose objects are too small / not unique. `python run.py validate`
fails a clip on any answer mismatch, ladder error, check failure, seed-rebuild difference, or (with video) wrong MP4
duration / resolution / motion.

## Determinism

`sim.build` and `questions.build` use only the `random.Random(seed)` passed in (no global random, numpy RNG, time,
uuid or set order). Same seed gives byte-identical JSON across processes and `PYTHONHASHSEED` values, and the
renderer is a pure function of (scene, frame), so frames are identical across renders. See `test_swing_time.py`.

## Files and commands

```
sim.py               rules, solver, scene generation, timing, consistency check
questions.py         the 10-question ladder
answers.py           independent re-derivation (validator / tests only)
render.py            PIL renderer (1280x720, 24 fps, piped to ffmpeg)
run.py               CLI (shared pipeline in common/video5.py)
test_swing_time.py   determinism, ladder, uniqueness, tamper, min_fps, hand-checked cases

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/swing_time
python run.py validate --dataset-dir ../output_videos/swing_time
python run.py generate --n 2000 --no-render --output-dir /tmp/st && python run.py validate --dataset-dir /tmp/st --skip-video
python test_swing_time.py
```
