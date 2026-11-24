# sandpile_cascade

Video-QA sub-benchmark for **chip-firing / sandpile cascades**: grains are
dropped one at a time onto a board of cells; a cell holding T or more grains
**topples**: T grains leave it, one hop to each of its outgoing neighbours,
and that can make neighbours topple in turn (a cascade). Grains sent off the
board (or into a hole) are lost. Each cascade settles completely before the
next grain is dropped. v2 moves the dataset to the shared PIL pipeline with
the 10-question ladder (`LADDER = "ten"` in `questions.py`), four scenarios,
grains drawn one by one (countable, no numbers on screen) and an independent
decoder that reads only the drawn grains.

## What each clip shows

- A board of cells. Every grain is drawn separately (square, rounded square
  or disc, 22+ px, 3-4 px outline) on a fixed slot of its cell, so a cell's
  load is counted by eye.
- A dropped grain falls for F = 8-12 frames: in the side view straight down
  from above the frame, in top view from 0.7 cell pitches above its slot
  while shrinking from 2.3x to its resting size. It lands on a whole frame.
- Toppling runs in waves: D = 4-8 frames after the last change, every cell
  holding >= T grains topples at once; its grains hop to their targets in
  M = 8-12 frames (an arc in the side view, a "hop" that swells and shrinks
  in top view) and land together; the next wave starts D frames later.
- Lost grains hop off the board and fall out of the frame (row) or shrink
  away in 5 frames (top view, or into the hole).
- The next grain starts falling 4-16 frames after the cascade has settled.
  Clips are 3.75-10 s (90-240 frames at 24 fps, 1280x720).

### Scenarios (`scene["scenario"]`, 25 % each)

| scenario | board | rule | lost grains |
|---|---|---|---|
| `row` | side view, 5-9 trays in a row, grains stacked upwards | T = 2 (one grain to each side) or T = 4 (two to each side) | sent past an end of the row: fall off the bottom of the frame |
| `grid` | top view, 3x3 to 4x5 (or 3x6) square cells, 3x3 slots each | T = 4, one grain to each of the four side neighbours | sent off an edge: shrink away just outside it |
| `ring` | top view, 6-10 round cells on a ring (faint ring line), one position is a black hole | T = 2 (one each way) or T = 4 (two each way) | sent into the hole: shrink away inside it |
| `pyramid` | top view, rows of cells, each row one longer than the row above (1-3 cells on top, 3-5 rows); faint lines link each cell to the two cells below it | T = 2, both grains go DOWN to the two cells just below | sent below the bottom row: shrink away at the end of the link line |

### Random knobs

Scenario; board size (trays, rows / columns, ring size and rotation,
pyramid rows and top width); threshold T where the scenario allows two;
starting loads (a "load" share of 25-85 % of cells starts one grain short
of toppling, the rest 0..T-1); number of drops (3-9 drawn, cut to what fits
in 10 s), where they land (a focus set of 2-5 cells 60 % of the time) and
the pause before each; F / D / M timing; the pause at the end; scale (the
board fills most of the frame, 64-211 px per cell pitch) and position;
grain colour (8), grain shape (3), grain size within its slot; background
(7 palettes x flat / gradient / vignette); cell colour (5 per theme); slot
marks (none / dots / rings); outline width.

## On-screen text

- **Clock** (the only text): fixed dark panel at the top-left, "s.t s"
  truncated to 0.1 s, 36 px bold (glyphs >= 26 px), white on near-black; no
  grain or cell comes within 14 px of it (checked on every frame).
- No labels, numbers or legends. Grain counts are read by counting the
  drawn grains; cells are named by position ("the second tray from the
  left", "the cell in row 2, column 3", "the third cell clockwise from the
  hole", "the second cell from the left in the third row"), which is unique
  on every board, so no cell labels are needed. `scene["labels"]` is empty
  and the renderer draws none. The rule is not shown; the what-if questions
  state it in their text.

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total
10) among the combinations the clip can fill (a candidate pool per level
first). Different templates first; above L1 motion questions first, with at
most one single-frame question per level (it competes with them in 30 % of
levels); no (derivation_key, params) repeats; any question that would need
more than 4 fps is dropped. 4 options except yes/no, which-of-two and the
more / fewer / same question. 39 templates:

- **L1 observation:** how many grains are dropped; how many cells; grains
  in a cell at the start / at the end; grains on the board at the start;
  where the first grain lands; which cell topples first; does a cell ever
  topple; which of four cells holds the most at the end.
- **L2 relational:** which of two cells topples first; which topples more
  often; which of two dropped grains sets off more topples; which dropped
  grain sets off the cascade in which a cell first topples; a cell at the
  end vs the start (more / fewer / same); which of two cells holds more at
  the end; which cell gets a grain when a cell first topples; off which end
  / edge the first lost grain goes.
- **L3 measurement:** clock time of a cell's first topple / of a grain's
  landing; total topples; topples of one cell; grains in a cell at a clock
  reading; grains lost; grains on the board at the end; seconds from a
  landing to the last topple of its cascade.
- **L4 integration:** topples set off by a given grain; how many different
  cells topple; which of four grains sets off the most topples; grains
  dropped by the first topple; the last cell to topple; topples between two
  clock readings; grains lost in a given cascade; grains in a cell once a
  given cascade has settled.
- **L5 counterfactual (re-simulated):** a grain never dropped (total
  topples / would a cell topple / grains lost); a grain landing on another
  cell (topples it sets off); one more grain on a cell after the end
  (topples that follow); a cell starting with one grain more / fewer (total
  topples).

Over 5000 scenes 96.1 % of L2-L5 questions need the motion
(`requires_motion=True`).

## Margins (no near ties)

- Every landing, topple and arrival is a whole frame; waves are D + M >= 12
  frames apart, topples and arrivals never share a frame, and all events
  are >= 2 frames apart and >= 12 frames (0.5 s) from both ends; the first
  grain starts falling >= 12 frames in; lost grains are gone >= 5 frames
  before the end.
- Compared first topples are >= 8 frames (1/3 s) apart; compared counts
  differ by >= 1 grain / topple (>= 22 px of drawn grain); "which holds the
  most" has a unique maximum; "last cell to topple" only when one cell
  topples in the last wave; "off which edge" only when the first losses
  all go one way.
- Clock answers are the reading on the event frame, options 0.5 s apart;
  durations are given to the nearest 0.5 s and only when within 0.1 s of
  it; counts at a clock reading only where the cell does not change within
  7 frames of the frames showing it and no moving grain covers it; counts
  "once a cascade has settled" are taken before the next grain appears.
- A cell never holds more grains than it has slots (checked).

## min_fps

Every question stores `params["min_fps"]`: the smallest of 1/2/3/4/6/8/12/24
fps that puts a sample into the shortest interval the answer needs: a topple
is visible for D + M frames (the overfull cell, then the hop), a dropped
grain from when it appears until something else happens to it, a hop for M
frames, the gap between two compared topples, the calm stretch around a
named clock reading or after a cascade, and 4 fps for clock readings and
durations (+-0.25 s). Single-frame questions need 1 fps. Over 5000 scenes:
17.0 % of questions need 1 fps, 78.6 % <= 2 fps, 100 % <= 4 fps.

## Closed loop

- **Questions** store `derivation_key` + `params` (cells by position, drops
  by order, clock tenths) + `min_fps`; ground truth comes from `sim.play`
  (events, parallel waves) and `sim.counts`; every what-if is re-simulated
  with `sim.relax_waves`.
- **Independent answers** (`answers.py`) never read the drops, the timing,
  the events or `sim.play`. They read `render.track`: every grain's centre
  and scale on every frame and the cells (centre, kind, slots). A grain is
  at rest when it is drawn at full size on a slot and still there on the
  frame before or after; from that they find the starting loads, landings,
  topples (grains leaving one cell on one frame) and where each grain goes.
  The board's wiring comes from the cell positions alone (side view -> row,
  a hole -> ring, equal rows -> grid, rows growing by one -> pyramid wired
  downwards) and T from the size of the topples seen; the decoder checks
  that every topple matches the wiring, that every cascade re-simulates to
  what was seen and that the end state matches, then answers what-ifs with
  its own sequential toppling loop (toppling is order-independent, so it
  must agree with the generator's parallel waves).
- **`sim.check`** rejects scenes that are too long, overfill a cell, put an
  event too close to another or to the ends, have fewer than two topples,
  grains under 22 px, anything near the clock, or a grain leaving the frame
  in the last frames.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`frame_check.py`** decodes the MP4s: the clock (template match on every
  6th frame, every event frame and every frame a question names); grains at
  rest read from the pixels (blobs of the grain colour of resting size on a
  slot) equal to the track's count in every cell on every frame where no
  moving grain covers it; every landing, topple and arrival seen on exactly
  its event frame (and not one frame earlier); every topple read off the
  frames equal to the track's; the count and clock answers read off the
  frames equal to the stored answers. Shifting a clip by one frame makes
  every clip fail.

## Determinism

All randomness comes from the `random.Random(seed)` passed in; times are
whole frames; no global RNG, time, uuid or set order decides anything. Same
seed -> byte-identical annotations in separate processes with different
`PYTHONHASHSEED`; the same scene renders identical raw frames (hashed);
regenerating seeds into a fresh folder reproduces `annotations.jsonl`
exactly (apart from `render_meta` timing when rendering).

## Diversity (5000 dry-run seeds)

5000 / 5000 unique scenes; scenarios pyramid 1283 / row 1273 / ring 1234 /
grid 1210; T: row 2/4 = 607/666, ring 2/4 = 615/619; 5-25 cells (13 board
cell counts); 2-9 drops (947 / 1472 / 1159 / 818 / 446 / 125 / 31 / 2); 2-30+
topples per clip (median 8); 0-20 grains lost; clips 3.75-10 s (42 % over 9 s);
scale 64-211 px per pitch; 8 grain colours x 3 shapes, 7 backgrounds x 3
styles; 13667 distinct (question, answer) pairs in 50000 questions; answer
letters A/B/C/D 24.8 / 24.6 / 25.1 / 25.4 %; level counts L1 1/2/3 =
2280/1819/901, L2 2/3 = 2942/2058, L3 3010/1990, L4 1/2/3 = 2235/1863/902, L5
2262/1812/926 (22 combinations). Generation retries: 577 scenes (11.5 %)
needed more than one attempt (mean 1.13, max 5; too long 463, too few
topples 193, framing 1); no seed fails, no warnings.

## Files and commands

```
sim.py                   the rule, scene generation, kinematics, checks (pure, seeded)
questions.py             the 10-question ladder (LADDER = "ten")
answers.py               independent re-derivation from the drawn grains
render.py                PIL renderer (1280x720, 24 fps, piped to ffmpeg); render.track = what is drawn
run.py                   CLI: generate / validate (common/video5.py)
generate.py              same as `run.py generate` (generate_all_datasets.sbatch)
flatten.py               annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
frame_check.py           decoded-frame checks of rendered clips
test_sandpile_cascade.py pytest suite

python run.py generate --n 50 --seed 0 --output-dir ../output_videos/sandpile_cascade_preview
python run.py validate --dataset-dir ../output_videos/sandpile_cascade_preview
python frame_check.py --dataset-dir ../output_videos/sandpile_cascade_preview
python -m pytest test_sandpile_cascade.py -q
```
