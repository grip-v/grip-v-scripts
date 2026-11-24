# temporal_flash_counting

Video-QA sub-benchmark for **counting and relating discrete events over
time**: shapes light up ("flash") one or more times, and the questions ask
how often, how long, in what order, what follows what, and what would change
if something were different. v3 is a rework on the shared PIL + ffmpeg
pipeline (`common/video5.py`) with the 10-question ladder (`LADDER = "ten"`
in `questions.py`, `common/ladder5.py`).

## What each clip shows

- **1280x720, 24 fps, 6-10 s** (relay clips 8.5-10 s). All timing is in whole
  frames: flash k of shape i is lit on frames `[f0, f1)`. No flash starts
  before 0.5 s or ends after T - 0.5 s.
- **Shapes:** 4-30 shapes from the shared 16-type vocabulary
  (`common/shapes.py`), each in one of 11 colours. Every colour + type pair is
  unique in a clip, and two shapes of the same type have colours at least
  110 RGB apart, so "the teal crescent" always names one shape. Radius 24-72 px.
- **Looks:** at rest a shape has a dim fill (background mixed with 20-30 % of
  its colour) and a coloured outline. Lit, it shows its full colour, a light
  outline and a soft halo. On a **white flash** the fill goes near-white
  instead. About 21 % of flashes are white.
- **Flash lengths:** two tempos. `fast`: short flashes are 6-9 frames
  (0.25-0.38 s), long ones 20-30 frames (0.83-1.25 s), and the cut the
  questions name is 0.6 s. `slow`: short 12-14 frames (0.5-0.58 s), long
  27-36 frames (1.13-1.5 s), cut 0.8 s. Two flashes of one shape are always
  separated by at least 8 (fast) or 12 (slow) dark frames.
- **Decoy pulses:** a shape can grow by up to 30 % and shrink back without
  lighting up (12-20 frames). This is not a flash. Pulses stay at least 6
  frames away from the shape's own flashes. Questions that count flashes in
  a clip with pulses say so in one sentence.
- **Rhythm:** each shape flashes at a steady rhythm (an interval of about
  1.0, 1.5, 2.0 or 2.5 s, ±1 frame of jitter, onset CV <= 0.06) or at
  irregular times (CV >= 0.3).
- **Designated pair:** in about 70 % of clips one shape pair's closest
  flashes are moved to start on the same frame, to overlap with different
  starts, or to follow each other with a clear gap. The runner-up pair of
  flashes is at least 12 frames further apart.

### Scenarios (`scene["scenario"]`, about 20 % each)

| scenario | setup |
|---|---|
| `grid` | a rows x columns grid of 4-30 shapes (2x2 up to 5x6), with optional rounded or round plates under the cells |
| `scatter` | 4-9 shapes of different sizes and rotations, placed at random non-overlapping spots |
| `relay` | 5-9 shapes in a grid or scattered. A hidden cause links them: a responder flashes 0.92-1.08 s after most (90 %) of a trigger's 4-5 flashes, plus 0-2 flashes of its own. In 45 % of the clips with 6+ shapes it is a chain of three (A -> B -> C) |
| `signal` | 4-6 round lamps on a panel, slow tempo, blinking short / long patterns (named by colour: "the green lamp") |
| `drift` | 4-6 shapes gliding left-right at 30-150 px/s in their own lanes (optional rails, optional bobbing) while they flash |

### Random knobs

Scenario, clip length, tempo, background (7 dark tones), texture (plain,
vignette, dots, stripes, faint grid, noise; step and strength), outline width
(3-6 px), rest / white / halo mix strengths, shape count, types (15 % of
scenes use one type for all shapes, so colour is the only identifier),
colours, sizes, rotations, aspects, grid size, cell size and grid position,
plate style, lamp size and spacing, lane speeds, directions, rails and bob,
per-shape rhythm, interval, long-flash and white-flash rates, every onset and
length, pulse rate and timing, relay lag, chain length, and the designated
pair's relation.

Over 5000 dry-run seeds every scene signature (hash of the rounded scene
params) is unique (100 %). The scenario mix is 19-21 % each.

## On-screen text

The **timer is the only text**. It sits top-left on a dark plate (x 14-196,
y 12-80) and shows seconds truncated to 0.1 s ("4.3 s") in 40 px bold white,
so it is readable at half size. `sim.AREA` (y >= 100) keeps every shape, halo
and pulse below the plate, so nothing passes under it. No labels are needed:
shapes are named by colour and type (lamps by colour), and moments by timer
readings. Grid rows and columns are counted, not numbered.

## The 10-question ladder

Per-level counts come from `L5.ten_counts(rng)`: L1 1-3, L2 2-3, L3 2-3,
L4 1-3, L5 1-3, 10 questions in total. Questions are ordered by level with
qids q01..q10. Within a level, templates are picked round-robin over shuffled
derivation keys, and no (derivation_key, params) pair repeats. Every question
has 4 options, and the answer letters are balanced (25 % ± 0.4 over 5000
clips). A trigger question (L4) and a counterfactual that names the same
responder's trigger (L5) never share a clip, because the L5 text would give
the L4 answer away.

- **L1 observation:** `flash_count`; `n_shapes`, or `grid_rows` / `grid_cols`
  in the grid scenario (static); `first_flasher`, `last_flasher` (lead of
  at least 8 or 12 frames); `white_ever` (the one option that turns white);
  `pulse_shape` (the one option that grows and shrinks).
- **L2 relational:** `relation` (the designated pair's closest flashes:
  start at the same moment / overlap with different starts / one ends before
  the other starts / one starts as the other ends, which is never true);
  `more_by` (how many more flashes X has than Y); `earlier_of_four` (which of
  four flashes first, not the global first); `longest_flash` (single longest
  flash, margin of at least 8 frames); `lit_during` (which option lights up
  during X's longest flash; the distractors are dark around it).
- **L3 measurement:** `long_count` (flashes longer than the tempo's cut);
  `white_count`; `period` (a steady shape's interval to the nearest 0.5 s,
  at least 0.13 s from a rounding boundary); `first_time` (the timer reading
  at X's first onset; wrong readings are at least 0.8 s away);
  `count_window` (flashes of all shapes that start between two timer
  readings; no onset within the margin of either reading).
- **L4 integration:** `most_frequent`, `least_frequent` (margin of at least
  1 flash and 10 %); `steady` (one steady shape among irregular ones; its `min_fps` is found by
  re-sampling the onsets at every step and phase, as for `trigger`);
  `trigger` (which shape the responder follows, with a score margin of 2 or
  more over every option); `total_flashes` (up to 24); `total_white`;
  `busiest_window` (which quarter of the clip has the most onsets, margin 2);
  `sequence` (the short / long pattern of X's 3-5 flashes).
- **L5 counterfactual** (all re-simulated): `cf_no_trigger` (the responder's
  flashes if its trigger never flashed); `cf_chain` (C's flashes if A never
  flashed, through B); `cf_remove_most` (the new most-flashing shape);
  `cf_long_only`, `cf_white_only` (winner if only long / white flashes
  counted); `cf_no_white_most` (winner without white flashes, which must
  differ from the real winner); `cf_stop` (X's flashes if the video had
  stopped at a timer reading).

All L2-L5 questions have `requires_motion=True`. The static L1 counts do not.

## Frame-sampling rules and `min_fps`

- Every flash is at least 6 frames long, and every dark gap between two
  flashes of one shape is at least 8 frames, so no event is a single-frame
  flash. Each shape keeps its last look until the next change.
- `params["min_fps"]` = 24 / k, rounded up to 0.1, for the largest frame step
  k such that sampling every k'-th frame works for **every** k' <= k. For
  each question that means: every relied-on lit run and dark gap is at least
  k frames; onset leads, window margins and lag margins are at least k frames;
  for length questions every short flash still shows fewer than the cut, and
  every long flash more (in whole k-frame samples); `first_time` keeps its
  sampled reading closer to the answer than to any other option; `period`
  keeps the sampled mean interval on its side of the half-second boundary.
- `frame_check.py` and the tests re-derive every question from every k-th
  frame, for every phase. Over 5000 seeds there are 0 failures (state-level
  check), and over the 50 previews there are 0 failures (decoded pixels).
- Over 5000 seeds, 6.2 % of questions can be answered at 1 fps, 48.7 % at
  2 fps, 99.7 % at 4 fps and all of them at 6 fps (the highest `min_fps` is
  12, a few trigger questions with tight lags). Counting short flashes is the
  point of the task, so most questions need 2-4 fps.

## Closed loop

- **Question records:** every question stores its `derivation_key` and
  `params` (shape indices, option indices, timer readings in tenths, the
  length cut, `min_fps`).
- **Independent answers:** `answers.py` never reads `scene["flashes"]`,
  the pulses, the rhythm tags or the relay chain. It walks the renderer's
  per-frame drawing state (`render.state`: centre, radius, fill colour),
  classifies each fill as rest / lit / white by nearest colour, and builds
  flashes and pulses from runs of frames. Rows, columns and shape counts come
  from the drawn centres. Causes are inferred from timing alone (a flash
  starting 0.6-1.4 s after a trigger onset), and counterfactuals re-run that
  rule with the removed shape's flashes gone (through the middle shape for a
  chain). `sim.resimulate` does the generator's side through the stored
  cause links.
- **`sim.check`** rejects a scene for any of these reasons:
  - two shapes share a label, or same-type colours are too close;
  - a shape leaves `sim.AREA`, or two shapes come too close (on every 3rd
    frame for drift);
  - a flash has a borderline length or sits too close to the clip edges;
  - two flashes of one shape are too close;
  - a steady rhythm isn't steady, or an irregular one looks steady;
  - a pulse touches a flash;
  - the designated pair is ambiguous;
  - a response flash has an ambiguous cause, or a spontaneous one sits near
    a trigger flash;
  - timing alone suggests a reaction that is not a relay link, or hides one
    that is;
  - some level has fewer than 3 candidate questions.
  `build` retries with the same rng (deterministic). Over 5000 seeds that
  is 2.2 attempts per scene on average (4.3 for relay).
- **`python run.py validate`** rebuilds every sample from its seed and
  compares it with the stored one. It also runs `sim.check` and the ladder
  checks (counts per level, order, no repeats, options, `min_fps`),
  re-derives every answer, and checks the MP4 duration, resolution and
  motion.
- **`frame_check.py`** decodes the MP4s and reads each shape's look from the
  pixels (inside point, nearest fill colour) and its size (outline found 4 px
  past its resting edge along its tip ray). It then:
  - compares the looks with the drawing state;
  - re-derives every answer from the pixel-detected flashes;
  - runs the frame-sampling check;
  - template-matches the timer at every onset and every named moment;
  - checks shape sizes and the timer's clearance.
- **`test_temporal_flash_counting.py`** covers:
  - determinism, in-process and across `PYTHONHASHSEED` values, plus raw
    frame hashes;
  - ladder counts, options and `min_fps`;
  - scene uniqueness and `sim.check`;
  - answer agreement and the tamper test (3 corrupted answers give exactly
    3 flags);
  - frame sampling at `min_fps`;
  - the timer as the only text;
  - hand-built scenes: counts, lengths and white flashes, the three
    relations, a relay chain counterfactual, and a steady rhythm.

## Determinism

Sample `i` uses `random.Random(seed + i)` for the scene and then the
questions. Nothing else is random: there is no global RNG, no time, no uuid,
and no set or dict order decides anything (the background "noise" texture is
a fixed hash of the pixel position). The same seed gives byte-identical
scene, question and answer JSON in different processes, and identical raw
frames.

## Files and commands

```
sim.py          scenarios, layouts, flash schedules, relay, checks (pure, seeded)
questions.py    the 10-question ladder (templates per level, min_fps)
answers.py      independent re-derivation from the per-frame drawing state
render.py       PIL renderer (1280x720, 24 fps, piped to ffmpeg); render.state
run.py          CLI: generate / validate (shared pipeline in common/video5.py)
generate.py     same as `run.py generate` (kept for generate_all_datasets.sbatch)
flatten.py      annotations.jsonl -> dataset_final.csv / .jsonl
frame_check.py  pixel-level check of rendered clips
test_temporal_flash_counting.py

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/temporal_flash_counting
python run.py validate --dataset-dir ../output_videos/temporal_flash_counting
python frame_check.py --dataset-dir ../output_videos/temporal_flash_counting
python flatten.py --dataset-dir ../output_videos/temporal_flash_counting
python -m pytest test_temporal_flash_counting.py
```
