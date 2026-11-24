# moving_shadows

Closed-loop 10-question video-QA benchmark on dynamic shadow geometry (New Problem Types shortlist B5). A light (a distant sun
sweeping across the sky, or a lamp on a rail) moves while 2-5 coloured objects float in the air or stand on the
ground. Every object casts a shadow onto the ground; all shadows are the same grey and merge where they overlap, so
which object casts which shadow has to be read from how the shadows move with the light. 4-7 marker plates sit in the
ground, numbered 1..n left to right. The pipeline is PIL frames piped to ffmpeg (`common/video5.py`) with the
10-question ladder in `common/ladder5.py` (`LADDER = "ten"`). Clips are 1280x720 at 24 fps and 7-10 s long
(168-240 frames).

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | setup | readable from the video |
|---|---|---|
| `floating` | 2-5 floating shapes, a sun on an arc | the shapes hang still; only the sun moves |
| `posts` | 2-3 standing objects (post, pillar, hut) plus 0-2 floating shapes, a sun on an arc | standing objects touch the ground, so their shadows start at their foot and pivot around it |
| `drifting` | one floating shape drifts sideways at 16-45 px/s, a sun on an arc | the drifting shape changes position; its shadow moves differently from the others |
| `rising` | one small floating shape rises or sinks at 12-28 px/s, a sun on an arc | the shape visibly changes height; its shadow grows or shrinks |
| `lamp` | a lamp runs along a straight rail near the top of the frame instead of a sun; mixed floating and standing objects | the lamp, its rail and hanger are drawn |

## Random knobs (`sim._attempt`, `sim._style`)

- object count 2-5, marker count 4-7, clip length 168-240 frames;
- floating shapes: 16 outlines from `common/shapes.py` (circle, ellipse, square, rectangle, diamond, triangle,
  pentagon, hexagon, octagon, star, cross, heart, trapezoid, semicircle, parallelogram, crescent), radius 22-46 px,
  any rotation, aspect 1.4-2.1 for ellipses and rectangles, 174-330 px above the ground;
- standing objects: post (20-30 px wide, up to 250 px tall), pillar (36-66 px wide), hut (a box with a roof);
- 8 colours (unique per clip; yellow and orange are left to the light), fill patterns (solid, striped, dotted,
  checked, ringed; standing: solid, striped, banded, dotted), outline width 3-4 px;
- light path: the sun is far away, so its rays are parallel; its elevation runs linearly from 28-46 degrees to
  134-152 degrees (or back), and its disc is drawn on an elliptical sky arc (half-width 520-600 px, half-height
  260-400 px, centre x 610-670, top 50-95 px below the frame top) at that elevation. The lamp is a point light at
  70-150 px height with a random start and end of its run, and its shadows are at most 3.4 times the object's
  distance from the ground line;
- sun colour (gold, amber, lemon), radius 17-26 px, style (plain disc, rays, ring);
- sky (day, pale, dawn, dusk, haze, indoor; gradient, flat or banded), ground colour (sand, soil, grass, stone, clay,
  snow) and texture (plain, speckle, stripes, bricks), ground line height 548-610 px (camera framing), surface-strip
  depth, shadow darkness 0.40-0.56, marker plate shape (rectangle, rounded, diamond) and width 20-26 px.

## On-screen text

- **The clock** (every clip): top-left box, seconds to 0.1 s, white 44 px text (32 px digit height) on a dark box.
  The value is truncated from the frame number (`tenths = frame * 10 // 24`). `sim.check` keeps the light and every
  object at least 14 px away from it.
- **Marker numbers 1..n** (left to right), white 36 px digits (26 px tall) with a 3 px dark outline, under each
  marker plate in the ground body below the surface strip. They are identifiers, not answers: the markers are
  identical plates, and many questions are about specific markers (which one is shaded first, when marker 3 goes
  into shadow, in what order markers 2, 4 and 5 are shaded). Colour or shape coding would not do the same job
  because colours already identify the objects, and seven identical plates need a name a question can use. The
  numbers never change, sit only on the ground body (where no object, light or shadow is drawn), are at least 90 px
  apart, and are stored in `scene["labels"]` (text, anchor, size and box) which the renderer draws from and the
  checks read.

There is no other text: no titles, legends, rule text or status messages.

## The 10-question ladder (`LADDER = "ten"`)

Per clip the count per level is drawn from L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3 (total exactly 10, within what the
scene can support), ordered by level as q01..q10. Within a level the template is drawn uniformly among those with
candidates left (no template repeats until all are used), and no (derivation_key, params) pair repeats. Every
question has four distinct options.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `n_casters`, `light_dir`, `first_marker`, `n_ever` (markers shaded at some point), `never` (which of four markers is never shaded), `shaded_at_t` (which of four is shaded at a clock time), `n_shaded_at_t`, `moving_object` / `move_dir` (drifting, rising) |
| L2 relational | `caster_of` (whose shadow reaches marker j first), `who_covers_at_t`, `shadow_way` (left / right / left then back / right then back), `earliest_of` (of four markers), `last_marker`, `longest_of` (of four markers) |
| L3 measurement | `cover_time` / `uncover_time` (clock reading), `shade_dur` (total, 0.5 s), `gap_time` (between two first shadings, 0.5 s), `n_entries` |
| L4 integration | `order` (3 or 4 markers), `most_shaded` / `least_shaded`, `n_casters_for` (objects whose shadow passes a marker), `caster_markers` (markers an object's shadow passes), `total_entries`, `max_simul` |
| L5 counterfactual | `cf_remove_count` / `_first` / `_never` / `_dur` (an object taken away), `cf_reverse_first` / `_count` (light runs the other way), `cf_still_first` / `_count` (the moving shape stays put) |

Margin rules (`sim.py`, `questions.py`):

- every coverage event (a shadow edge crossing a marker) is at least 0.6 s from the start and the end, at least
  0.3 s from any other edge crossing the same marker, and every shading and every gap between shadings lasts at
  least 0.3 s; edges cross markers at 50 px/s or more; an edge that turns back stays 4 px or more from a marker; at the
  first and last frame every shadow edge is at least 8 px from every marker; no marker starts in shadow; markers stay 30 px or more from a standing object's foot;
- compared event times differ by at least 0.3 s; compared durations by at least 0.3 s and 10 %;
- clock-time questions use the 0.5 s grid and need no relevant edge within [t - 0.3, t + 0.4];
- clock answers are the reading at the first frame that shows the event; the event must sit at least 0.15 frame
  from a frame time and the shadow edge at least 1.5 px from the marker centre on both frames around it; distractors
  are 0.5 s apart;
- duration answers are rounded to 0.5 s and must be at least 0.15 s from a rounding boundary (markers with at most
  two shadings); distractors are 1 s apart;
- `max_simul` must give the same count with every shading shrunk or grown by 0.15 s;
- what-if questions are only asked if the changed scene passes the same timing rules.

## Closed loop

`sim.py` finds coverage events by projecting each outline vertex onto the ground (sun: parallel rays,
`x' = x - (G - y) cot(elevation)`; lamp: a ray through the lamp) and bisecting in
continuous time. `answers.py` does not read those events. It builds a per-frame track of what is drawn
(`render.state`: the light centre and each object's outline) and tests each marker in each frame by whether the
segment from the marker toward the light crosses an outline edge (for the sun, toward a far point in the direction
of the elevation read off the drawn disc on its arc). Runs of shaded frames give first frames, clock
readings (clock text of the frame), durations (frames / 24) and counts. Shadow movement comes from the extreme angles
of the outline seen from the light. What-ifs rebuild the track with an object removed, the light path run backwards,
or the moving shape held at its start, and re-run the same tests. `sim.check` rejects any scene that breaks a rule
above; `python run.py validate` fails a clip on any disagreement, a ladder error, a check failure, or (with videos) a
wrong duration, resolution or a static video.

`frame_check.py` decodes each MP4 and re-derives every answer from pixels: light and object centres located by
colour, marker shading read from each plate's centre pixels, the clock read by template match against all readings,
and the marker numbers read by template match against the digits 1-9. Times named in a question or given as an answer
are checked against the decoded clock at the matching frame, and marker answers against the number read under that
marker. It also checks that no object touches the clock or a number and that numbers do not overlap.

## Frame sampling and `min_fps`

Every question stores `params["min_fps"]`, the lowest sampling rate at which it is still answerable:
`ceil(1 / g)` (1-24) for the smallest gap `g` between the events it relies on (e.g. the two compared first-shading
times, or the shortest shading that must be seen at least once); clock-time questions need 1 fps at whole seconds and
2 fps at half seconds; clock-reading answers need at least 4 fps (options are 0.5 s apart); duration answers need
`ceil(k / 0.4)` for k shadings (options are 1 s apart). Shadings last at least 0.3 s, so nothing is a single-frame
flash; objects are at least 20 px, lines at least 3 px, text at least 24 px tall.

## Determinism

All randomness comes from the `random.Random` passed to `sim.build` and `questions.build`; candidate lists are built
in a fixed order and nothing depends on set or dict iteration order. Stored values are rounded before anything is
derived from them, and caches are keyed by the scene JSON. The same seed gives byte-identical JSON across
`PYTHONHASHSEED` values and identical raw frames (`test_moving_shadows.py`).

## Files and commands

```
sim.py                  scenes, shadow geometry, coverage events, consistency check
questions.py            the 10-question ladder
answers.py              answers from the drawn per-frame track (segment tests), what-if re-simulation
render.py               PIL renderer (1280x720, 24 fps)
frame_check.py          decodes MP4s, re-derives answers from pixels, reads the clock and marker numbers
preview.py              renders the first N seeds of every scenario
stats.py                diversity / balance report
run.py                  CLI (shared pipeline in common/video5.py)
test_moving_shadows.py  determinism, ladder, tamper, labels, hand-worked geometry, pinned seeds

python3 run.py generate --n 20 --seed 0 --output-dir out
python3 run.py validate --dataset-dir out [--skip-video]
python3 preview.py
python3 frame_check.py --dataset-dir ../output_videos/moving_shadows_preview
python3 stats.py out/annotations.jsonl
python3 test_moving_shadows.py
```

## Results

All numbers below are from the committed code (parallel-ray sun).

- **5000-scene dry run** (`run.py generate --no-render`, seeds 0-4999 in 3 chunks, `validate --skip-video`, both under
  `python -W error`): 0 mismatches, 0 ladder errors, 0 check failures, no warnings or exceptions. 3503 of 5000 scenes
  (70.1 %) redrew their layout at least once (mean 4.38 attempts, max 69; a redraw is cheap); no seed failed.
- **Diversity** (`stats.py`): 5000 / 5000 unique scene signatures; scenarios drifting 19.3 %, floating 20.7 %, lamp
  20.3 %, posts 19.5 %, rising 20.2 %; 2-5 objects, 4-7 markers, 7-10 s clips, sun sweep 88-123 degrees, light speed
  48-146 px/s, all 16 floating outlines and 3 standing kinds, 8 colours, 6 patterns, 6 skies x 3 sky patterns,
  6 grounds x 4 textures, 9 sun looks, both directions about equally; 13,174 distinct (question, answer) pairs over
  50,000 questions; answer letters A 25.11 %, B 25.08 %, C 25.04 %, D 24.76 %; every L2-L5 question needs the motion.
- **Level counts**: L1 8621, L2 12046, L3 12051, L4 8632, L5 8650 questions; all 22 allowed splits occur (206-245
  clips each).
- **min_fps**: 1 fps 35.8 %, 2 fps 34.3 %, 3-4 fps 25.9 %, above 4 fps 4.0 % (so 96.0 % are answerable at 4 fps).
- **Tamper**: 3 answers corrupted in each of 50 scenes -> exactly 150 questions flagged.
- **Determinism**: identical JSON and answers across `PYTHONHASHSEED` values 1 and 777; identical raw-frame hashes
  across two processes; regenerating seeds 0-49 into fresh folders reproduces `annotations.jsonl` byte for byte.
- **Previews** (`preview.py`, 10 per scenario): `run.py validate` 0 mismatches; `frame_check.py` 50 clips, 10,312
  frames, clock template match 10,312 / 10,312, marker numbers read 729 / 729, 50,026 plate readings with no
  disagreement, 500 answers re-derived from decoded frames, max light / object centre error 2.55 px (tolerance 3 px),
  0 problems. Clips were also viewed as 4 fps frames shrunk to 640x360: clock, numbers, plate shading and objects stay
  readable.
