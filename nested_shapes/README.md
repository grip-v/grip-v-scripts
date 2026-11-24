# nested_shapes

Video-QA sub-benchmark for **nested-contour counting, attribute binding,
fill-mark counting and rotation patterns, made temporal**. Each clip shows a
nest of 3-7 closed outlines, each strictly inside the one around it, that
changes over time: outlines appear one by one, turn and reverse, collect
marks, change colour, or glide around a box. v2 is a port of the old Manim
generator to the shared PIL + ffmpeg pipeline (`common/video5.py`) with the
10-question ladder (`LADDER = "ten"` in `questions.py`, `common/ladder5.py`).

## What each clip shows

- **1280x720, 24 fps, 6.5-10 s.** All timing is in whole frames. Every
  question-relevant event happens at least 12 frames (0.5 s) after the start
  and before the end, and any two events are at least 8 frames (0.33 s) apart.
- **Outlines:** 12 shape types from the shared vocabulary (`common/shapes.py`):
  circle, ellipse, square, rectangle, diamond, triangle, pentagon, hexagon,
  octagon, star, cross, trapezoid. Each has a border colour (8 colours on
  light backgrounds, 9 on dark ones), a width of 5-8 px, and a line style:
  solid, dashed, dotted or double (broken lines only on outlines at least
  70 px in radius, so the shape stays readable). Some outlines carry a
  **tab**, a dot fixed to one corner that makes their turning visible.
- **Nesting:** ring k+1 fits inside ring k at any rotation. Its outer reach
  (size x max radius of its unit outline + half its line) plus its centre
  offset plus a gap stays below ring k's inner reach (size x min boundary
  distance - its inner line reach), with extra room for tabs and marks.
  The gap is 4-24 px. Deep nests use rounder shapes: the product of
  inner/outer reach ratios must clear a per-depth minimum. Nests can be
  concentric or offset (26 %), and pre-twisted.
- **Turning:** positive angles are clockwise on screen. A ring turns at
  18-130 deg/s. It can turn only when a viewer can see it: it needs a tab,
  or corners that stick out at least 7 px past its flat parts. Circles and
  tiny near-round outlines never turn unless they carry a tab. A turn per
  frame stays under 0.4 of the outline's symmetry period, and every turning
  or still stretch lasts at least 1 s.
- **Names:** an outline is named by colour + shape ("the teal hexagon").
  Every colour + shape is unique in a clip. In `recolor`, colours change, so
  shapes are unique and outlines are named by shape alone ("the hexagon").

### Scenarios (`scene["scenario"]`, about 20 % each)

| scenario | setup |
|---|---|
| `assemble` | 3-7 outlines appear one at a time (outside-in, inside-out or mixed order, 12-40 frames apart, the first after 0.5 s). In 45 % of clips with 4+ outlines, one or two disappear again. Some outlines turn steadily. |
| `spin` | 3-7 outlines, all there from the start. Each turns clockwise, counterclockwise or stays still, with 2-6 changes in total (reverse, stop, start; at most 3 per outline). 0 / 50 / 80 % of outlines carry tabs. |
| `marks` | 3-5 outlines. 4-10 filled marks (heart, star, dot, diamond, 24-32 px across, in a colour other than their outline's) are placed on outlines one at a time, and in 60 % of clips 1-3 are taken off again. Marks ride on their outline as it turns. They go only on outlines at least 2.5 mark radii wide, and never touch each other or another outline. |
| `recolor` | 3-6 outlines of distinct shapes. 3-7 colour changes spread over 2+ outlines, at least 1 s apart on one outline. Some outlines turn steadily. |
| `drift` | a small nest (3-5 outlines, 95-170 px radius) glides at 110-300 px/s inside a drawn box and bounces off its edges 1-7 times. Outlines turn; in 35 % of clips one changes how it turns. |

### Random knobs

Scenario, clip length, theme (light / dark), background (10 tones), texture
(plain, vignette, dots, stripes, grid, noise; step and strength), number of
outlines, shapes, aspect ratios (ellipse, rectangle), colours, widths, line
styles, tabs, outer size (95-284 px), how tightly rings nest, gap, centre
offset and its direction, pre-twist, nest position, turning rates,
directions and change times, appearance order and timing, disappearances,
mark count, kinds, colours, positions and times, removals, colour changes,
box size and position, glide speed and direction.

Over 5000 dry-run seeds (`python run.py stats --n 5000`):

- 5000 / 5000 scene signatures (hash of the rounded scene params) are unique.
- Scenario mix: assemble 20.7 %, drift 20.3 %, recolor 20.2 %, spin 19.5 %,
  marks 19.3 %.
- Outlines per clip: 3-7 (assemble and spin cover 3-7 evenly; marks and drift
  are mostly 3-4). Outer radius 95-284 px (median 241), innermost 24-177 px.
  Clip length 6.5-10 s (85 distinct). 1-13 events per clip. Turning rates
  18-130 deg/s. All 12 shapes, all colours, 6 textures, 10 backgrounds and
  4 line styles occur.
- 16,747 distinct (question text, answer) pairs among 50,000 questions.
- Answer letters: A 25.1 %, B 25.0 %, C 24.9 %, D 25.0 %.

## On-screen text

The **timer is the only text** (rule 2c). It sits top-left on a dark plate
(x 14-196, y 12-80) and shows seconds truncated to 0.1 s ("4.3 s") in 40 px
bold white, readable at half size. `sim.AREA` (y >= 100) keeps the nest, its
marks and the drift box below the plate, so nothing passes under it.

No other labels are needed. Outlines are named by colour + shape (or by shape
in `recolor`), and by their place in the nest ("the 2nd outline from the
outside"). Marks are counted, not labelled. Box edges are named left / right
/ top / bottom. Moments are named by timer readings. Numbering the outlines
or the marks would add text without adding information, and would risk giving
away counts.

## The 10-question ladder

Per-level counts come from `L5.ten_counts(rng)`: L1 1-3, L2 2-3, L3 2-3,
L4 1-3, L5 1-3, 10 questions in total, ordered by level, with qids q01..q10.
Within a level, templates are picked round-robin over shuffled derivation
keys. Outside `spin`, the generic turning and nesting keys usually queue
behind the scenario's own keys. No (derivation_key, params) pair repeats.
Every question has 4 options. The static nesting templates (`inside_of`,
`inside_count`, `cf_remove`) appear on L2-L5 at most once per clip, or when a
level runs out of motion templates. Over 5000 clips, 98.1 % of L2-L5
questions have `requires_motion=True`.

- **L1 observation:** `n_end`, `color_count_end`, `shape_at_depth`,
  `depth_color` (end state, static); `first_appear`, `never_turns`,
  `first_mark`, `first_wall`, `end_heading`; `start_color`, `end_color`
  (recolor).
- **L2 relational:** `inside_of`, `inside_count` (static); `pair_dir` (how two
  outlines turn at a timer reading: same way / opposite ways / only one);
  `same_dir`, `fastest` (speed margin x1.5), `first_change`;
  `appear_earliest`, `vanishes`; `lost_mark`, `more_added`, `more_at`;
  `color_at_time`, `which_recolors`; `wall_before`.
- **L3 measurement:** timer readings of events (`appear_time`, `change_time`,
  `mark_time`, `recolor_time`, `hit_time`, `hit_wall_time`; wrong options at
  least 0.8 s away); counts at a moment or in a window (`count_at`,
  `turning_at`, `marks_window`, `hits_window`); per-outline counts
  (`n_changes`, `n_recolors`, `n_hits`).
- **L4 integration:** `appear_pattern` (outside-in / inside-out / mixed),
  `appear_after`, `last_depth` (where in the finished nest the last outline
  sits), `max_on_screen`; `spin_pattern` (e.g. "turns clockwise, then stops,
  then turns counterclockwise"), `cw_end`, `most_changes`; `total_added`,
  `total_removed`, `most_added`; `color_sequence`, `total_recolors`;
  `hits_axis`.
- **L5 counterfactual** (each re-simulated on an edited copy of the scene):
  `cf_stop_count`, `cf_stop_turning`, `cf_stop_marks`, `cf_stop_color`,
  `cf_stop_hits` (the video stops at a timer reading); `cf_reverse_order`
  (same moments, reversed appearance order); `cf_remove` (one outline taken
  out of the nest); `cf_keep_dir` (no outline ever changes how it turns);
  `cf_flip` (one outline always turns the other way); `cf_no_removal` (marks
  are never taken off); `cf_no_last_recolor` (the last colour change never
  happens); `cf_no_wall` (one box edge is missing; hits before the nest
  leaves the box).

## Frame-sampling rules and `min_fps`

- Every question-relevant change persists: an outline stays on or off for at
  least 1 s, a mark stays for at least 1 s, a colour or turning stretch lasts
  at least 1 s, and events are at least 8 frames apart. Nothing is a single-
  frame flash.
- `params["min_fps"]` = 24 / k, rounded up to 0.1, for the largest frame step
  k at which sampling every k-th frame still gives the answer. For each
  question that means:
  - relied-on events land in separate samples (k <= their gaps) and before
    the last sample (k <= frames from the event to the end);
  - a state read at a timer reading has no relied-on event within k frames;
  - every relied-on turn stays unambiguous: a turn per sample is under 0.4 of
    the outline's visible symmetry period. A hexagon has a 60 deg period, a
    tabbed outline 360 deg. At higher steps the apparent direction flips
    (wagon-wheel aliasing), which a test checks;
  - every relied-on turning stretch spans at least 3 samples;
  - a timer-reading answer stays closer to the right option than to any
    other. A turning change can show up 2k - 1 frames late, so its step is
    halved;
  - wall hits keep 2 samples between them.
- `checks.sampling_failures` re-derives every question from every k-th frame,
  for every phase, through `answers.py`. Over 5000 dry-run seeds that gives
  0 failures (state level). Over the 50 previews it also gives 0 failures
  from the decoded pixels (`frame_check.py`).
- Over 5000 seeds, 32.4 % of questions can be answered at 1 fps, 56.4 % at
  2 fps, 95.2 % at 4 fps and all of them at 8 fps (median 1.9).

## Closed loop

- **Question records:** every question stores its `derivation_key` and
  `params` (outline / option indices, timer readings in tenths, colours,
  walls, `min_fps`).
- **Independent answers:** `answers.py` never reads the event lists (spin
  segments, appear / vanish frames, recolor list, mark add / remove frames,
  wall hits) or the stored appearance order. It walks the renderer's
  per-frame drawing state (`render.state`: centre, size, rotation, colour and
  visibility of every outline; position and visibility of every mark; the
  nest centre), and derives:
  - turning from frame-to-frame rotation, reduced modulo the visible
    symmetry period, grouped into runs;
  - colours by nearest palette colour;
  - which outline a mark sits on from the nearest drawn outline;
  - depth from drawn polygon areas;
  - wall hits as turning points of the nest's x / y track.
  Counterfactuals are answered from those derived events. `questions.py`
  answers them by re-simulating an edited copy of the scene through `sim.py`.
- **`sim.check`** rejects a scene when:
  - two outlines share a name, or recolor shapes repeat;
  - two outlines come within 3 px at any rotation, or the innermost is under
    24 px in radius;
  - the nest leaves its area or box;
  - an outline turns invisibly or too fast for its symmetry;
  - a turning, visibility or mark stretch is under 1 s;
  - an event is near a clip end or within 8 frames of another;
  - a wall hit falls halfway between frames;
  - marks crowd an outline, or a mark sits on an outline too small for it;
  - some level has fewer than 3 candidate questions.
  `build` retries with the same rng (deterministic). Over 5000 seeds, 53.3 %
  of seeds needed a retry, with 2.22 attempts per scene on average (max 19).
  55 % of all attempts were rejected, most for failing the nest fit or the
  question supply. There were no exceptions or warnings.
- **`python run.py validate`** rebuilds every sample from its seed and
  compares it with the stored one. It also runs `sim.check` and the ladder
  checks, re-derives every answer, and checks the MP4s. A 2000-scene dry run
  (`--no-render`, `--skip-video`) gives 0 mismatches. `python run.py tamper`
  corrupts 3 answers, and exactly 3 get flagged.
- **`frame_check.py`** decodes the MP4s and, on every frame:
  - reads each outline's visibility and colour from 48-64 points on its drawn
    path (dash or dot centres for broken lines);
  - confirms its position and rotation (no stray ink in a band just off the
    stroke; for tabs, ink at the tab tip and not half a turn away; a 4 deg
    error is caught);
  - reads each mark's presence;
  - re-derives every answer from this pixel track, also from every k-th
    frame;
  - template-matches the timer at every event and every question moment,
    and at the detected event of every timer question;
  - checks sizes, line widths and clearances.
  On the 50 previews: 40,148 outline-frames, 14,835 rotation checks and
  11,484 mark-frames with 0 disagreements, 500 questions re-derived from
  pixels, 554 timer readings matched, 0 failures.
- **`test_nested_shapes.py`** covers:
  - determinism, in-process and across `PYTHONHASHSEED` values, plus raw
    frame hashes;
  - ladder counts, options and `min_fps`;
  - uniqueness of scenes and names, and `sim.check`;
  - answer agreement and the tamper test;
  - frame sampling at `min_fps`;
  - the timer as the only text;
  - hand-built scenes: turning patterns and counterfactuals, aliasing,
    appearance order, marks, colours, and wall hits against the closed form.

## Determinism

Sample `i` uses `random.Random(seed + i)` for the scene and then the
questions. Nothing else is random: there is no global RNG, no time, no uuid,
and no set or dict order decides anything (the noise texture is a fixed hash
of the pixel position). The same seed gives byte-identical scene, question
and answer JSON across processes with different `PYTHONHASHSEED` values, and
identical raw frames (`python run.py framehash --seed S`). Regenerating the
50 previews into a fresh folder reproduces `annotations.jsonl` exactly, apart
from `render_meta` timings.

## Files and commands

```
sim.py          scenarios, nest geometry, time functions, events, checks (pure, seeded)
questions.py    the 10-question ladder (templates per level, min_fps, re-simulated counterfactuals)
answers.py      independent re-derivation from the per-frame drawing state
render.py       PIL renderer (1280x720, 24 fps, piped to ffmpeg); render.state
run.py          CLI: generate / validate (common/video5.py), stats / tamper / framehash (checks.py)
checks.py       diversity + ladder stats, frame-sampling check, tamper, frame hash
frame_check.py  pixel-level check of rendered clips
generate.py     same as `run.py generate` (kept for generate_all_datasets.sbatch)
flatten.py      annotations.jsonl -> dataset_final.csv / .jsonl
test_nested_shapes.py

python run.py generate --n 50 --seed 3 --output-dir ../output_videos/nested_shapes
python run.py validate --dataset-dir ../output_videos/nested_shapes
python frame_check.py --dataset-dir ../output_videos/nested_shapes
python run.py stats --n 5000 --jobs 8
python run.py tamper --dataset-dir ../output_videos/nested_shapes
python flatten.py --dataset-dir ../output_videos/nested_shapes
python -m pytest test_nested_shapes.py
```
