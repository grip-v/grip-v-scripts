# multi_agent_chase

Closed-loop 10-question video-QA benchmark on **chasing and swallowing, read purely from motion**. Patterned balls
move in a walled box; some balls catch others, a caught ball vanishes at once and its catcher grows (areas add) and
slows down. Nobody is labelled as a hunter: who chases whom has to be read from how the balls move. Questions ask
who caught whom, when, in what order, how many balls are left, who is biggest, and what-if re-runs.

The pipeline is PIL frames piped to ffmpeg (`common/video5.py`), with the 10-question ladder in `common/ladder5.py`
(`LADDER = "ten"`). Clips are 1280x720 at 24 fps and 7-10 s long (168-240 frames).

## What a clip shows

- **The balls:** 4-9 balls inside a walled arena, all present from the first frame. Each has a unique colour (12
  available) **and** a unique pattern (9 available: solid, striped, dotted, checkered, ringed, two-tone, quartered,
  grid, starred), so "the blue striped ball" names exactly one ball. Patterns scale with the ball.
- **Catching:** a catcher that touches an eligible ball catches it. The caught ball is drawn on frame f-1 and gone
  from frame f on. The catcher's radius then grows smoothly (45 px/s) to the radius whose area is the two areas added,
  and bigger balls move slower (speed scales with sqrt(start radius / radius)). A catcher wanders for 0.9 s after a catch.
- **Smooth motion:** acceleration-limited steering at 48 steps/s (two per frame), at most 520 px/s^2. Walls, pillars
  and (for catchers) the refuge push balls away smoothly from 95 px. `sim.check` rejects any clip where a ball
  accelerates harder than 1.1x the limit.
- **Text in the frame:** only the clock (see below).

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | rules | balls | read from the video because |
|---|---|---|---|
| `open` | 1-3 hunters chase prey in an empty box; hunters catch prey only | 4-9 | the balls that move toward others and then absorb them |
| `pillars` | as open, plus 1-3 round grey pillars (radius 32-60 px); every ball steers around them; 1-2 hunters | 4-9 | pillars are drawn; paths bend around them |
| `safe_zone` | as open (1-2 hunters), plus a hatched green circle (radius 85-120 px): prey fully inside cannot be caught, hunters keep out, threatened prey run toward it | 5-9 | the circle is drawn; a prey ball inside it is never absorbed |
| `bigger` | no fixed roles: any ball absorbs a strictly smaller ball it touches, chases smaller balls and flees bigger ones; start radii differ by 16-24 % per step | 5-8 | size decides who catches whom; the catcher is always the bigger of the two |

## Random knobs (`sim._attempt`)

- ball count (by scenario above), hunters 1-3 (open; 1-2 otherwise; one or two when there are 5 balls or fewer),
  12 colours and 9 patterns drawn without repeat, so colour and pattern are both unique per clip;
- radii: hunters 24-32 px, prey 15-26 px (bigger: 13-17 px for the smallest, each next size x1.16-1.24, shuffled);
  speeds: hunters 200-260 px/s, prey 70-90 % of that (bigger: one base 165-215 px/s, +-10 %); wander headings with
  two random sine terms;
- 6 style palettes (backdrop / floor / wall), floor texture (plain, grid, dots), wall style (solid, double, thick);
- arena size 980-1220 x 500-640 px and position (always below the clock box);
- start layout: `scatter`, `split` (hunters on one side) or `ring`; hunters and others kept apart at the start;
- clip length 7-10 s.

`sim.build` redraws until the scene passes `sim.check` and at least one what-if is unambiguous.

## On-screen text

- **The clock only**: a dark rounded box at the top right, above the arena (`CLOCK_BOX = (1098, 8, 1268, 58)`, fixed;
  the arena always starts below it), white 34 px text, seconds to 0.1 s. Frame f shows `f * 10 // 24` tenths.

No other labels are needed: balls are named by colour + pattern and times are read from the clock. There are no
titles, captions, legends, scores or role markers.

## The 10-question ladder (`questions.py`)

`LADDER = "ten"`. Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total 10), qids q01..q10
ordered by level, no repeated derivation_key + params and no repeated question text.

| level | templates (derivation_key) |
|---|---|
| L1 observation | `n_start` balls at the start, `n_end` balls at the end, `n_catches`, `pattern_of` a colour, `color_of` a pattern, `biggest_start`, `n_pillars` (pillars), `zone_end` (safe_zone: balls fully inside the circle at the end) |
| L2 relational | `caught_by` (who catches X), `first_victim` of a catcher, `caught_first` of two, `bigger_end` of two (only pairs whose size order flips), `never_caught`, `is_catcher`, `first_caught` |
| L3 measurement | `catch_time` (clock when X disappears), `n_caught_by` a ball, `left_at` (balls in the box when the clock shows X), `time_between` two disappearances, `first_catch_time` |
| L4 integration | `left_after` (balls left right after X is caught), `catch_order`, `top_catcher`, `nth_victim`, `caught_between` two clock readings, `biggest_end` |
| L5 counterfactual | `cf_n_end` (balls at the end), `cf_first` (who is caught first), `cf_n_caught_by` (slow only); under one change: `nogrow` (catching does not grow / slow the catcher), `remove` a ball, `slow` a ball to half speed, `immune` (a ball cannot be caught), `nozone` (safe_zone), `nopillars` (pillars) |

Margins (`sim.py`, `questions.py`): catches >= 10 frames apart (`GAP_F`), first catch at frame >= 20 (`FIRST_F`),
last catch >= 24 frames before the end (`END_PAD_F`, growth has finished), no other ball within `CLEAR` = 20 px of
a victim when it is caught, `EAT_RATIO` 1.12 (bigger: catcher radius >= 1.12 x victim radius), `SIZE_MARGIN` 1.12
(radius ratio needed for "biggest" / "bigger" questions), clock options >= 0.6 s apart, and "when the clock shows
X" questions keep every catch >= 6 frames (`clock_safe` padding) outside the frames showing X. Every question has
four distinct options.

## Closed loop

Every question stores its `derivation_key` and `params`. `answers.py` never reads the generator's catch log or the
balls' roles: it works from the per-frame track the renderer draws (position, radius, drawn / gone) and the drawn
scenery. A catch is a ball drawn on frame f-1 and gone on frame f; its catcher is the drawn ball whose edge was
closest to it on f-1. Clock readings use the renderer's clock. What-ifs are re-simulated with `sim.run(scene, mods)`
and read the same way, and are only asked when the re-run passes `sim.run_errors` (the same ambiguity checks as the
real clip). `python run.py validate` fails a clip if that disagrees with the stored answer, if rebuilding from the
seed gives a different scene or questions, if `sim.check` finds anything borderline, or if the MP4 is malformed.

`frame_check.py` decodes the MP4s and checks the clock (template match on every frame), ball visibility on every
frame a ball is alive and its disappearance on the track's catch frame, ball centre (<= 3 px) and size (<= 2.5 px)
for isolated balls, the clock-based answers re-derived from the decoded clock and catch frames, and that nothing
but the clock is drawn outside the arena.

## Determinism

All randomness comes from the seeded `random.Random` passed to `sim.build` and `questions.build`. Same seed gives
identical scene, questions and answers JSON across processes with different `PYTHONHASHSEED`, and rendering a seed
twice gives identical raw frames (both tested).

## Frame sampling and `min_fps`

Each question stores `params["min_fps"]` = `ceil(1 / smallest relevant gap)`, capped at 24, where the gaps are the
seconds between the relevant catches and the neighbouring catches or clip ends (all gaps for whole-clip questions).
Questions about the starting state or the scenery use 1; clock questions (`catch_time`, `left_at`, `time_between`,
`first_catch_time`, `caught_between`) use at least 2. A catch is permanent (the ball stays gone), so nothing is a
one-frame flash.

## Results

5000-scene dry run (`generate --no-render`, then `stats.py`):

5000-scene dry run (`run.py generate --no-render`, `run.py validate --skip-video`): `5000 samples, 50000 questions, 0 mismatches` (0 ladder errors, 0 check failures, no warnings under `-W error`). Scene builds redraw until `sim.check` passes: 87 % of seeds redraw at least once, mean 13.5 attempts, max 308 (about 0.18 s per scene including the questions).

- 5000 / 5000 unique scene signatures.
- Scenario mix: bigger 25.7 %, open 25.5 %, pillars 24.2 %, safe_zone 24.7 %.
- 20900 distinct (question, answer) pairs out of 50000 questions.
- Answer letters: A 25.09 %, B 24.93 %, C 25.05 %, D 24.93 %.
- `min_fps`: 56.5 % of questions are answerable at 1 fps, 94.6 % at 2 fps, 100 % at 4 fps.

Preview set: 50 clips (bigger 13, safe_zone 13, pillars 12, open 12) validated with 0 mismatches. `frame_check.py`:

| check | result |
|---|---|
| clock frames | 10187 / 10187 |
| visibility, ball drawn | 44153 / 44153 |
| visibility, ball gone | 15458 / 15458 |
| catch frames | 138 / 138 |
| centre / size | 10446 / 10446 |
| clock answers | 105 / 105 |
| margins | 100 / 100 |

Tamper test: corrupting 3 stored answers flags exactly those 3 questions.

## Files and commands

```
sim.py                 scene sampler, simulation (sim.run), ambiguity checks (run_errors, check)
questions.py           the 10-question ladder
answers.py             independent re-derivation from the drawn per-frame track
render.py              PIL renderer (1280x720, 24 fps) + per-frame state
run.py                 CLI: generate / validate (shared pipeline in common/video5.py)
generate.py            same as run.py generate, kept for generate_all_datasets.sbatch
flatten_annotations.py annotations.jsonl -> dataset_final.csv / .jsonl (one row per question)
frame_check.py         decoded-MP4 checks
preview.py             renders the preview set (first 13 seeds per scenario, trimmed to 50)
stats.py               diversity / balance report
test_multi_agent_chase.py   determinism, ladder, uniqueness, tamper, clock text, hand-checked seeds

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/multi_agent_chase
python run.py validate --dataset-dir ../output_videos/multi_agent_chase
python run.py generate --n 5000 --no-render --output-dir /tmp/mac && python run.py validate --skip-video --dataset-dir /tmp/mac
python stats.py /tmp/mac/annotations.jsonl
python preview.py && python frame_check.py --dataset-dir ../output_videos/multi_agent_chase_preview
python -m pytest test_multi_agent_chase.py
python generate.py --n 150 --output-dir <dir> && python flatten_annotations.py --dataset-dir <dir>
```
