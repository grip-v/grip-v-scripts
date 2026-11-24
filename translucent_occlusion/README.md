# translucent_occlusion

Closed-loop 10-question video-QA benchmark on **tracking shapes through occlusion**. Coloured shapes move on a dark
floor past fixed, nearly opaque translucent panels. A shape can slide behind a panel, wait, and come back out, go in
and stay, start hidden and come out, or (in `hidden_change`) change colour, shape or size while it cannot be seen.
Questions ask who went behind a panel first, where a shape comes back out, what the clock shows at those moments, how
many shapes are hidden, what a shape looked like before and after, and what-if re-runs.

The pipeline is PIL frames piped to ffmpeg (`common/video5.py`), with the 10-question ladder in `common/ladder5.py`
(`LADDER = "ten"`). Clips are 1280x720 at 24 fps and 7-10 s long (168-240 frames).

## What a clip shows

- **Shapes:** 3-6 shapes (8 outlines: circle, square, triangle, pentagon, hexagon, star, diamond, cross; 9 colours),
  radius 22-42 px. Every look (colour + shape) is unique in a clip, including the new look of a shape that changes.
- **Panels:** 1 or 2 fixed translucent panels (rectangle or circle) drawn on top of the shapes with a flat tint (4
  tints) at alpha 0.86-0.93 and a light rim. A shape behind a panel is only faintly visible.
- **Cover states:** `sim.states` gives 0 clear, 1 partly behind, 2 completely hidden (the whole disc of radius r is
  inside a panel). "Hidden" in every question means state 2.
- **Motion:** every shape follows straight keyframed segments at constant speed (about 80-330 px/s). Shapes never
  touch each other, never leave the arena and never graze a panel edge (no lingering while partly covered).
- **Text in the frame:** only the clock (see below).

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | what happens | shapes | read from the video because |
|---|---|---|---|
| `crossing` | 1 panel; hidden shapes pass through (pause inside or not) or go in and come back out the same side | 3-6 | entry and exit sides and times are visible |
| `hidden_change` | as crossing (through / back); 1-2 shapes change colour, shape or size while completely hidden | 3-5 | the look before and after differs, the change itself is never seen |
| `hide_and_stay` | one shape parks inside the panel, one starts hidden and comes out, others go through or back | 3-6 | the hidden count at the end differs from the count at the start |
| `two_panels` | a left and a right panel, shapes use either | 3-6 | "which panel" matters |

Each hidden shape has a path kind (`through`, `back`, `stay`, `emerge`); the remaining shapes are bystanders that
never touch a panel (kept 24 px clear).

## Random knobs (`sim._try`)

- scenario, then shape count 3-6 (3-5 for hidden_change) and radii 22-42 px; unique looks drawn from 9 colours x 8 shapes;
- how many shapes go behind a panel (at least 2, at least count - 2) and their path kind;
- panel kind (two rectangles to one circle), size (rect 170-380 x 170-460 px, circle R 110-210 px) and position;
  in `two_panels` one panel in each half of the arena;
- lane: horizontal (60 %) or vertical through a rectangle, or one of 4 axes through a circle, with a random offset;
  start/end 40-300 px outside the panel; pauses inside 6-50 frames; start frame; speed 150-330 px/s (bystanders
  70 % of that);
- for hidden_change: 1-2 shapes get a colour, shape or size (+-9-14 px, ratio >= 1.2) change at the middle of
  their hidden stretch (stretch >= 18 frames);
- background (5), floor texture (plain, grid, dots), tint (4), alpha 0.86-0.93, clip length 7-10 s.

`sim.build` redraws (`scene["build_attempts"]`) until the scene passes `sim.check` and `questions.feasible`.

## On-screen text

- **The clock only**: a light rounded box at the top right, above the arena (`CLOCK_BOX = (1098, 8, 1268, 58)`),
  dark 34 px text, seconds to 0.1 s. Frame f shows `f * 10 // 24` tenths.

No other text is needed: shapes are named by colour + shape ("the red triangle") and the look is unique per clip,
panels are named by position ("the left panel", "the right panel", or "the panel" when there is one), and times
are read from the clock. There are no titles, captions, labels, legends or scores.

## The 10-question ladder (`questions.py`)

`LADDER = "ten"`. Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total 10), qids q01..q10 ordered
by level, no repeated derivation_key + params and no repeated question text. Shapes that change look are only asked
about in the change questions and in counts.

| level | templates (derivation_key) |
|---|---|
| L1 observation | `n_total` shapes in total (including hidden ones), `n_clear_start` shapes completely outside the panels at the start, `color_of` a shape type, `shape_of` a colour, `panel_shape` of a panel, `n_panels` (two_panels) |
| L2 relational | `hid_first` which of two goes completely behind a panel first, `out_first` which of two starts to come out first, `exit_side` side of the panel where X comes back out, `went_behind` which shape goes behind a panel, `never_behind` which never does, `which_panel` (two_panels) behind which panel X is hidden |
| L3 measurement | `hide_time` clock when X becomes completely hidden, `reveal_time` clock when X first starts to come out, `hidden_for` duration hidden (clock difference), `n_hidden_at` hidden count when the clock shows T |
| L4 integration | `change_size` / `change_look` what happened to X while hidden (hidden_change), `before_look` what a shape that just came out looked like going in, `hide_order` / `reveal_order` order of 2-4 shapes, `n_ever_hidden`, `n_hidden_end`, `max_hidden` (most hidden at once) |
| L5 counterfactual | `nopanel` (panels removed): `cf_n_seen_end` shapes seen in full at the end; `delay` (shape X starts 1 s later): `cf_n_hidden_at` hidden at clock T, `cf_n_hidden_end`, `cf_first_out` who comes out first |

Margins (`sim.py`, `questions.py`): every hide/reveal event is >= 12 frames (0.5 s) from both clip ends (`EDGE_F`) and
>= 3 frames apart (`GAP_F`); order comparisons need >= 8 frames (0.33 s, `ORDER_F`); a shape only changes while
hidden >= 6 frames before and after the change; paused-inside points sit >= 8 px inside the edge (`INSET`); a moving
shape covers >= 4 px/frame while partly behind a panel (`MIN_STEP`); size changes need ratio >= 1.2; clock options are
>= 0.6 s apart; "when the clock shows X" questions keep every event >= 6 frames outside the frames showing X. What-ifs
are asked only when the re-run passes the same edge/gap checks. Every question has four distinct options.

## Closed loop

Every question stores its `derivation_key` and `params`. `answers.py` never reads the generator's keyframes, path
kinds or change list: it works from the per-frame track the renderer draws (centre, radius, colour, shape) and the
drawn panels. It recomputes "completely hidden" by testing each shape's disc against each panel; hide and reveal
frames are the flips of that test, covering (coverage) is recomputed on every frame, and clock readings use the
renderer's clock. What-ifs are re-run with `sim.track(scene, mods)` and read the same way.
`python run.py validate` fails a clip if that disagrees with the stored answer, if rebuilding from the seed gives a
different scene or questions, or if the MP4 is malformed.

`frame_check.py` decodes the MP4s and checks the clock (template match on every frame), the pixel at the centre of
every clear shape is its colour and at the centre of every hidden shape is not, the clock-based answers
(`hide_time`, `reveal_time`, `hidden_for`) re-derived from the decoded clock on the event frames, and the event margins.

## Determinism

All randomness comes from the seeded `random.Random` passed to `sim.build` and `questions.build`. Same seed gives
identical scene, questions and answers JSON across processes with different `PYTHONHASHSEED`, and rendering a seed
twice gives identical raw frames (both tested).

## Frame sampling and `min_fps`

Each question stores `params["min_fps"]` = `ceil(1 / smallest relevant gap)`, capped at 24, where the gaps are the
seconds between the relevant events and the neighbouring events or clip ends (all gaps for whole-clip questions).
Questions about the starting state or the scenery use 1; clock questions (`hide_time`, `reveal_time`, `hidden_for`,
`n_hidden_at`, what-if clock questions) use at least 2. Hidden counts are states that persist, not one-frame flashes.

## Results

5000-scene dry run (`run.py generate --no-render`, `run.py validate --skip-video`, then `stats.py`):

- validate: 0 mismatches, 0 ladder errors, 0 check failures, no warnings; redrawn scenes 87.9 %, mean build attempts 9.1, max 117.
- 5000 / 5000 unique scene signatures.
- Scenario mix: crossing 25.5 %, hidden_change 24.2 %, hide_and_stay 24.7 %, two_panels 25.7 %.
- 18797 distinct (question, answer) pairs out of 50000 questions.
- Answer letters: A 25.09 %, B 24.93 %, C 24.97 %, D 25.02 %.
- `min_fps`: 47.1 % of questions answerable at 1 fps, 75.2 % at 2 fps, 100 % at 4 fps.

Preview set: 50 clips (>= 8 per scenario) validated with 0 mismatches. `frame_check.py`:

| check | result |
|---|---|
| clock frames | 10315 / 10315 |
| clear centres | 11899 / 11899 |
| hidden centres | 2886 / 2886 |
| clock answers | 96 / 96 |
| event margins | 262 / 262 |

Tamper test: corrupting 3 stored answers flags exactly those 3 questions.

## Files and commands

```
sim.py                 scene sampler, track / cover states / events, ambiguity checks (run_errors, check)
questions.py           the 10-question ladder
answers.py             independent re-derivation from the drawn per-frame track
render.py              PIL renderer (1280x720, 24 fps)
run.py                 CLI: generate / validate (shared pipeline in common/video5.py)
generate.py            same as run.py generate, kept for generate_all_datasets.sbatch
flatten.py             annotations.jsonl -> dataset_final.csv / .jsonl (one row per question)
frame_check.py         decoded-MP4 checks
preview.py             renders the preview set (first 13 seeds per scenario, trimmed to 50) + optional poster JPEGs
stats.py               diversity / balance report
test_translucent_occlusion.py   determinism, ladder, uniqueness, tamper, clock text, hand-checked seeds

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/translucent_occlusion
python run.py validate --dataset-dir ../output_videos/translucent_occlusion
python run.py generate --n 5000 --no-render --output-dir /tmp/to && python run.py validate --skip-video --dataset-dir /tmp/to
python stats.py /tmp/to/annotations.jsonl
python preview.py && python frame_check.py --dataset-dir ../output_videos/translucent_occlusion_preview
python -m pytest test_translucent_occlusion.py
python generate.py --n 150 --output-dir <dir> && python flatten.py --dataset-dir <dir>
```
