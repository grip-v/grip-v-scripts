# occluded_pattern_counting

Closed-loop 10-question video-QA benchmark on **amodal pattern counting under occlusion**. A regular pattern of
identical shapes (grid, ring, triangle or honeycomb) sits still while one or two opaque occluders rest at a position,
slide to the next, rest again, and so on. Questions ask how many objects are visible or hidden at a moment, where the
occluder moves, what the clock shows at those moments, how many objects there are in total, and what-if re-runs.

Lineage: the video-modality companion to the static-image *Occluded Pattern Counting Dataset* spec, and a scaled-up,
video-native replication of the hidden-region-only counting task from **CAPTURe: Evaluating Spatial Reasoning in
Vision Language Models via Occluded Object Counting** (Pothiraj, Stengel-Eskin, Cho, Bansal). CAPTURe found that
GPT-4o, InternVL2, Molmo and Qwen2-VL all show a larger counting error when the occluded pattern is real than when it
is simulated, and the gap is largest when the question asks only for the hidden count. Here the pattern is inferred
through the occluder and the occluder moves, so some counts need what was seen at an earlier hold.

The pipeline is PIL frames piped to ffmpeg (`common/video5.py`), with the 10-question ladder in `common/ladder5.py`
(`LADDER = "ten"`). Clips are 1280x720 at 24 fps and 4-10 s long.

## What a clip shows

- **Pattern:** one family per clip, a static set of identical objects of one outline (circle, square, triangle, star,
  diamond, hexagon) and one colour (8 colours), radius 12-28 px, spacing 52-110 px. Objects never move.
- **Occluders:** 1 or 2 opaque rectangles (flat colour from 5, texture solid / hatched / dotted / checker, light
  rim). Nothing shows through them.
- **Cover states:** `sim.states` gives 0 clear, 1 cut by an occluder edge, 2 completely hidden (disc inside the
  rectangle, each with 3 px to spare, `PAD`). At every hold no object is cut, so each object is clear or hidden;
  objects are only partly covered while an occluder slides. "Hidden" in every question means state 2.
- **Holds:** the clip is a sequence of 2-4 holds (still periods, 1.0-1.8 s) joined by slides (0.6-1.1 s), starting
  and ending in a hold, with >= 0.5 s (12 frames, `EDGE_F`) between every slide and the clip ends.
- **Every slot is seen:** each hold hides at least 2 objects and at most 60 % (1 and 30 % per occluder in
  `two_occluders`), consecutive holds hide different sets, no object is hidden at every hold, and every empty slot
  of the gaps scenario is uncovered at some hold, so the pattern can be inferred and counted.
- **Text in the frame:** only the clock (see below).

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | what happens |
|---|---|
| `slide` | 1 occluder, 2-4 stops |
| `two_occluders` | a left and a right occluder (disjoint halves), 2-3 stops each; objects are named by occluder colour |
| `gaps` | 1-3 slots of the pattern are empty (no object); the total is the visible-over-time count, not the slot count |
| `odd_one` | 1-3 objects have a different colour from the rest (same shape), asked about by colour |

Families (`scene["family"]`): `grid` 3-6 x 3-7, `ring` 8-16 objects, `triangle` 3-6 rows, `honeycomb` 3-5 x 3-6 with
offset rows.

## Random knobs (`sim._try`)

- scenario, family and its size (above); spacing `52..min(110, arena fit)` px; radius `0.26-0.36 x spacing`, capped
  at 28 px and at least 12; pattern position in the arena; object colour (8), outline (6), background (5: paper,
  mist, sage, charcoal, navy);
- gaps: 1 to `min(3, n // 5)` slots emptied; odd_one: 1-3 objects recoloured;
- stops 2-4 (2-3 for two_occluders), hold length 24-44 frames, slide length 14-26 frames, trailing hold so that the
  clip is at most 10 s;
- occluder width / height `1.2 x spacing` up to about half / 0.6 of the pattern, position per stop (resampled until no
  object is cut, the hidden set is new and in range), colour (5) and texture (4); in two_occluders the occluders keep
  to opposite halves and stay >= 12 px apart.

`sim.build` redraws (`scene["build_attempts"]`) until the scene passes `sim.check` and `questions.feasible`.

## On-screen text

- **The clock only**: a dark rounded box at the top right, above the arena (`CLOCK_BOX = (1098, 8, 1268, 58)`),
  white 34 px text, seconds to 0.1 s. Frame f shows `f * 10 // 24` tenths.

No other text is needed: objects are identical or told apart by colour ("the red object"), occluders are named by
colour ("the brown occluder", or "the occluder" when there is one), pattern families are named in the options, and
times are read from the clock. There are no titles, captions, labels, legends or scores.

## The 10-question ladder (`questions.py`)

`LADDER = "ten"`. Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total 10), qids q01..q10
ordered by level, no repeated derivation_key + params and no repeated question text.

| level | templates (derivation_key) |
|---|---|
| L1 observation | `n_visible_start` objects visible at the start, `family` arrangement, `object_shape`, `main_color`, `n_occluders` (two_occluders) |
| L2 relational | `start_vs_end` more visible at start or end, `slide_dir` which way an occluder first moves, `odd_end` / `odd_start` is the odd-coloured object visible/hidden (odd_one, one odd object), `after_first_move` more or fewer hidden after the first move, `occ_more_end` which occluder hides more at the end |
| L2 (`level2_more`) | `side_start` / `side_end` which part of the pattern is hidden at the start / end, `most_stop` at which resting position the occluder hides most, `first_move_net` does the first move uncover or cover more |
| L3 measurement | `n_visible_at` / `n_hidden_at` counts when the clock shows T (T inside a hold), `first_move_time` clock when the occluder first starts to move, `last_stop_time` clock when it stops for the last time, `first_hold_len` how long it stays still at the start (clock difference) |
| L4 integration | `n_total` objects in total including hidden ones, `n_revealed` hidden at start but visible at end, `n_covered` visible at start but hidden at end, `n_ever_hidden`, `n_gaps` (gaps), `n_odd` objects of the odd colour (odd_one), `max_hidden` most hidden at one hold |
| L5 counterfactual | occluders never moved (`stay`), stopped one move early (`stop_early`), or one removed (`remove`, two_occluders): `cf_hidden_end`, `cf_visible_end`, and for `stay` also `cf_never_seen` |

Margins (`sim.py`, `questions.py`): every slide is >= 12 frames (0.5 s) from both clip ends (`EDGE_F`), slides last
>= 14 frames and holds >= 24 frames, so every count question is asked about a hold; "when the clock shows X" questions
keep every frame showing X >= 6 frames inside a hold and >= 12 frames from the clip ends; clock options are >= 0.6 s
apart; what-ifs are asked only when the re-run passes `sim.run_errors`. Every question has four distinct options.

## Closed loop

Every question stores its `derivation_key` and `params`. `answers.py` never reads the generator's hold list or
per-hold logs: it works from the drawn objects and the per-frame occluder rectangles the renderer draws. An object is
hidden on a frame when its whole disc is inside a rectangle; still stretches are found from the track itself (no
rectangle moved since the previous frame); clock readings use the renderer's clock. What-ifs re-run the track with
`sim.track(scene, mods)` and read it the same way. `python run.py validate` fails a clip if that disagrees with the
stored answer, if rebuilding from the seed gives a different scene or questions, or if the MP4 is malformed.

`frame_check.py` decodes the MP4s and checks the clock (template match on sampled frames), the pixel at the centre of
every clear object is its colour and at the centre of every hidden object is not (sampled frames inside holds), the
number of connected object-coloured blobs at the middle of every hold equals the number of visible objects (the
counting check), the clock-based answers (`first_move_time`, `last_stop_time`, `first_hold_len`) re-derived from the
decoded clock, and the slide margins.

## Determinism

All randomness comes from the seeded `random.Random` passed to `sim.build` and `questions.build`. Same seed gives
identical scene, questions and answers JSON across processes with different `PYTHONHASHSEED`, and rendering a seed
twice gives identical raw frames (both tested).

## Frame sampling and `min_fps`

Each question stores `params["min_fps"]` = `ceil(1 / smallest relevant gap)`, capped at 24, where the gap is the
shortest hold (count questions) or slide (movement questions) in seconds. Because holds last >= 1 s, starting-state,
scenery and count questions use 1; clock and movement questions use at least 2 (`fps_for(..., floor=2)` for the
clock questions). A 1-2 fps sample always lands inside every hold.

## Results

5000-scene dry run (`run.py generate --no-render`, `run.py validate --skip-video`, then `stats.py`):

- validate: 0 mismatches, 0 ladder errors, 0 check failures, no warnings; redrawn scenes 81.8 %, mean build attempts 6.2, max 55.
- unique scene signatures: 5000 / 5000.
- scenario mix: gaps 24.7 %, odd_one 25.7 %, slide 25.5 %, two_occluders 24.2 %; family mix: grid 25.3 %, honeycomb 25.6 %, ring 24.6 %, triangle 24.5 %.
- distinct (question, answer) pairs: 5901 out of 50000 questions.
- answer letters: A 25.14 %, B 25.01 %, C 24.96 %, D 24.89 %.
- `min_fps`: 67.7 % of questions answerable at 1 fps, 100 % at 2 fps, 100 % at 4 fps.

Preview set: 50 clips (>= 8 per scenario) validated with 0 mismatches. `frame_check.py`:

| check | result |
|---|---|
| clock frames | 2979 / 2979 |
| clear centres | 34905 / 34905 |
| hidden centres | 6495 / 6495 |
| hold blob counts | 141 / 141 |
| clock answers | 72 / 72 |
| slide margins | 182 / 182 |

Tamper test: corrupting 3 stored answers flags exactly those 3 questions.

## Files and commands

```
sim.py                 scene sampler, track / cover states, ambiguity checks (run_errors, check)
questions.py           the 10-question ladder
answers.py             independent re-derivation from the drawn per-frame occluder rectangles
render.py              PIL renderer (1280x720, 24 fps)
run.py                 CLI: generate / validate (shared pipeline in common/video5.py)
generate.py            same as run.py generate, kept for generate_all_datasets.sbatch
flatten.py             annotations.jsonl -> dataset_final.csv / .jsonl (one row per question)
frame_check.py         decoded-MP4 checks
preview.py             renders the preview set (first 13 seeds per scenario, trimmed to 50) + optional poster JPEGs
stats.py               diversity / balance report
test_occluded_pattern_counting.py   determinism, ladder, uniqueness, tamper, hand-checked seeds

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/occluded_pattern_counting
python run.py validate --dataset-dir ../output_videos/occluded_pattern_counting
python run.py generate --n 5000 --no-render --output-dir /tmp/opc && python run.py validate --skip-video --dataset-dir /tmp/opc
python stats.py /tmp/opc
python preview.py && python frame_check.py --dataset-dir ../output_videos/occluded_pattern_counting_preview
python -m pytest test_occluded_pattern_counting.py
python generate.py --n 150 --output-dir <dir> && python flatten.py --dataset-dir <dir>
```
