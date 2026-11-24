# overlapping_shapes

Video-QA sub-benchmark for **counting under overlap, set relations, stacking order and order-of-events
reasoning** with flat shapes on a board. v3 runs on the shared PIL pipeline (`common/video5.py`,
`common/ladder5.py`) with the **10-question ladder** (`LADDER = "ten"`). Clips are 1280x720, 24 fps,
at most 10 s.

## Scenarios (one per seed, stored as `scene["scenario"]`)

| scenario | what happens | what a viewer has to track |
|---|---|---|
| `arrive` | 4-10 see-through shapes appear one at a time (grow, fade, spin-in or drop-in) | arrival order, which shapes each newcomer overlaps |
| `paper`  | 4-9 opaque paper cut-outs are laid down one at a time; a later sheet covers earlier ones | stacking order (which sheet is on top), arrival order |
| `depart` | 5-10 see-through shapes start on the board; 3 to n-1 of them disappear one at a time (shrink, fade, spin-out or lift-off) | removal order, which overlaps are lost |
| `glide`  | 5-8 still see-through shapes plus 1-2 extra shapes that slide in a straight line (constant speed, sometimes slowly turning) and pass over some of them | which shapes a mover passes over, in what order, when, for how long |

## Random knobs

- **Shapes:** 16 types (circle, ellipse, square, rectangle, diamond, triangle, pentagon, hexagon,
  octagon, star, cross, heart, crescent, trapezoid, semicircle, parallelogram), 2-10 types per clip,
  at most 4 of one type. Squares and diamonds stay within 20 degrees of upright so a square never
  looks like a diamond; every other shape gets any rotation. Ellipses, rectangles and parallelograms
  get an aspect ratio of 1.4-2.1.
- **Colour, fill and border:** 10 colours (8 on dark boards), fill pattern plain / stripes / crossed
  lines / dots (30% of clips are all plain), border solid / dashed / dotted / double, border width
  3-5 px, fill opacity 78-120 of 255 (paper sheets are opaque).
- **Sizes:** radius about 38-140 px (an upper bound set by the shape count, times a per-clip zoom of 0.8-1.2).
- **Layouts:** cluster, chain, ring, grid, spiral, flower, scatter, twin clusters, diagonal (glide
  uses the open ones).
- **Board and framing:** 9 themes (paper, cream, mint, sky, graph paper, dot grid, linen grain,
  slate, night grid); the board rectangle's edges move by up to ~130 px per side.
- **Timing:** intro 0.6-1.7 s, gaps between events 0.6-1.25 s, change animation 6-8 frames, hold
  0.7-1.3 s at the end; glide clips run 7-10 s with movers at 110-260 px/s.

`python run.py stats --n 5000` prints the spread of every knob, the scenario mix, unique scene
signatures, distinct (question, answer) pairs, answer-letter balance, level counts, the min_fps
distribution and retry rates.

## On-screen text

The only text in the frame is the **clock** (`12.3 s`, tenths of a second, 40 px white on a dark box
in the top-right corner, outside the board, so it never covers the action). Nothing else is written:
no titles, labels or hints. Shapes are named in questions by **colour + type** ("the red star"), and
every colour + type pair is unique in a clip, so no identifier labels are needed. Questions that refer
to a moment use the clock ("when the clock reads 3.5 s").

## Rules that keep every answer readable

- Two shapes overlap when they share at least 300 px²; scenes in which two shapes (or a larger group)
  share a sliver of 1-300 px², or in which two shapes that do not overlap come within 15 px, are
  rejected.
- Paper: every sheet keeps at least half of its area visible, and every overlap shows at least
  300 px² where one of the two sheets is visibly on top.
- Glide: a mover starts and stops clear of every shape, overlaps each shape it meets in one
  continuous stretch of at least 0.5 s with a peak of at least 1200 px², never grazes or nearly touches
  a shape it does not cross (15 px), and two movers never come within 15 px of each other.
- Every event (an arrival, departure, the start or end of a move, a mover starting or stopping to
  overlap a shape) is at least 0.5 s from the start and end of the clip and at least 8 frames
  (0.33 s) from every other event. Clock readings named in questions sit at least 8 frames from every
  change. Events leave a lasting result (a shape stays on, stays gone, or the mover keeps moving);
  nothing hinges on a single frame.

## The ladder (10 questions: 1-3 / 2-3 / 2-3 / 1-3 / 1-3 per level)

Up to three questions are built per level from the templates that apply to the scenario (templates
that need the motion are drawn three times as often as single-frame ones), then the per-level counts
are drawn at random among the splits that sum to 10. Questions are numbered q01..q10 in level order;
no two share `derivation_key` + `params`. All questions have four options.

- **L1 observation:** shapes on the board, kinds of shape, shapes of a colour / border style / fill
  pattern / type, first or last shape to appear, first to disappear, shapes left at the end, which
  shape slides, which way it moves.
- **L2 relational:** which of four shapes overlaps X, which overlaps nothing, which of two or four
  shapes appears (disappears) first, the one shape X overlapped when it arrived, which sheet lies on
  top of X, which shape disappears next after X, which shape the mover slides over, which it
  overlaps first.
- **L3 measurement:** how many shapes X overlaps, overlapping pairs, deepest stack, largest/smallest
  area ratio (to 0.5), shapes on the board at a clock reading, when X starts to appear (nearest
  second), how many shapes the mover overlaps, how many it overlaps at a clock reading.
- **L4 integration:** arrivals that overlapped nothing on the board, the arrival that overlapped most,
  how many shapes X overlapped when it arrived, overlapping pairs at a clock reading or just before the
  last arrival, X's partners just before it disappears, overlaps lost over the clip, shapes left
  alone at the end, the mover's last shape, the order it meets three shapes, the most it overlaps at
  once, which shape it overlaps at a clock reading, new overlaps between two clock readings, which
  shape it stays on longest, which of two movers meets more shapes.
- **L5 counterfactual (re-simulated):** pairs / lone shapes if X were removed, lone arrivals in
  reverse order or without the first arrival, which sheet would cover X if laid in reverse, pairs or
  lone shapes at the end if X had never disappeared, shapes the mover would meet if it stopped at a
  clock reading, travelled the trip backwards or moved at half speed.

Distractors are the plausible mistakes: neighbours that arrived later, the count before an event,
the total instead of the part, nearby integers.

## Closed loop

- Every question stores `derivation_key` + `params` (including `min_fps`).
- `answers.py` re-derives every answer **independently** of the generator: the generator uses exact
  shapely polygons and bisection for crossing times; `answers.py` rasterises the outlines the renderer
  draws into pixel masks and replays the clip frame by frame (presence from the event frames,
  mover poses at every frame, shared pixels per frame, an index image for which sheet is on top).
  Counterfactuals replay the changed trip or order the same way. A test monkeypatches the generator's
  geometry functions to raise and checks that every answer still comes out.
- `sim.check` rejects ambiguous scenes (slivers, near-touches, grazing movers, events too close).
- `python run.py validate --dataset-dir DIR [--skip-video]` rebuilds every scene and question from
  its seed, re-runs `sim.check`, the ladder checks and the independent answers.
- `python run.py tamper --dataset-dir DIR` corrupts 3 answers and reports how many get flagged.
- `python run.py framecheck --dataset-dir DIR` decodes the MP4s and checks them against the scene:
  each shape's presence read off probe pixels frame by frame (order and count answers recomputed from
  those read times), the clock at every event frame and every clock reading named in a question, the
  overlap regions (blended for see-through shapes, the later sheet on top for paper), visible gaps
  between nearby shapes that do not overlap, the mover tracked off the frames every frame (within
  9 px of its path), with first/last/total/ordered overlaps recomputed from the measured track, and
  a plain frame margin (no text but the clock).

## Determinism

All randomness comes from the `random.Random(seed)` that `common/video5.py` passes in (the linen
texture uses a numpy generator seeded from it); nothing depends on time, hashing or set order. The
same seed gives byte-identical JSON across processes with different `PYTHONHASHSEED`, and identical
raw frames (`python run.py framehash --seed S`).

## Frame sampling and min_fps

Shapes are at least ~36 px across, lines at least 3 px, the clock is ~29 px tall, and the colours stay
distinct at half size. Every question stores `params["min_fps"]`: the sampling rate needed to land a
frame inside the shortest interval the question depends on (the gap between the events it compares,
the still stretch around a clock reading, the shortest overlap it counts, the still board at the start
or end).

## Files and commands

```
sim.py        scene generation + rule checks (pure, seeded)
questions.py  the 10-question ladder (LADDER = "ten")
answers.py    independent answer re-derivation (pixel masks + frame replay)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
checks.py     stats / tamper / framecheck / framehash
run.py        CLI: generate / validate / stats / tamper / framecheck / framehash
generate.py   same as `run.py generate` (kept for generate_all_datasets.sbatch)
flatten.py    annotations.jsonl -> dataset_final.csv / .jsonl
test_overlapping_shapes.py

python run.py generate --n 50 --seed 0 --output-dir ../output_videos/overlapping_shapes
python run.py validate --dataset-dir ../output_videos/overlapping_shapes
python run.py framecheck --dataset-dir ../output_videos/overlapping_shapes
python run.py stats --n 5000
python flatten.py --dataset-dir ../output_videos/overlapping_shapes
python -m pytest overlapping_shapes/test_overlapping_shapes.py
```
