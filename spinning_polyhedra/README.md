# spinning_polyhedra

Video-QA sub-benchmark: **Spinning Solids**, the temporal version of the image benchmark's `polyhedron` category (`Geomatric-Reasoning-Benchmark-Dataset/Dataset`). One or two wireframe solids spin in perspective, with coloured markers on some corners. Questions ask about the shape (structure from motion) and about the motion itself: direction, turns, stops, crossings, axis changes and what-ifs. It runs on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the 10-question, 5-rung ladder (`common/ladder5.py`, `LADDER = "ten"`).

## What each clip shows

- **1280x720, 24 fps, 5.0-10.0 s.** All timing is in whole frames.
- **Solids:** 21 convex solids. These are the five Platonic solids, triangular/pentagonal/hexagonal prisms, square/pentagonal/hexagonal pyramids, triangular/pentagonal/hexagonal bipyramids, square/pentagonal/hexagonal antiprisms, the cuboctahedron, the truncated tetrahedron, the truncated octahedron and the rhombic dodecahedron. Every edge is drawn, front and back, so a single frame is ambiguous and the motion shows the 3D shape.
- **Markers:** 1-3 corners carry a coloured dot, square or diamond with a contrasting rim, drawn on top. Nearer markers are drawn slightly larger and last. Marker colours are unique in a clip, and they are at least 200 RGB-L1 units from the background, the edge colours and each other.
- **The only text is the timer** (seconds to 0.1 s, 36-44 px font, 26-32 px glyph height). It sits in one corner per clip (top-left, top-right, bottom-left or bottom-right), and its ink contrasts with the background. `sim.check` keeps every solid's bounding circle at least 12 px clear of the timer box. `scene["labels"]` is empty. No label is needed: the solids are named by position ("the solid on the left"), the markers by colour and shape, and moments by the timer. Nothing in the frame names a value a question asks for.

## Scenarios (`scene["scenario"]`)

| scenario | motion | what it adds |
|---|---|---|
| `steady` | one solid, constant spin about the upright axis (period 2.4-9.0 s, exact tenths) | turns, period, crossings, timing |
| `reverse` | one solid; it spins, stops for 0.5-1.0 s, spins back; 1 or 2 reversals; speeds equal or differing by 1.6-2.4x | stops/restarts, before/after speed, net direction |
| `speed_change` | one solid about the upright axis, speeding up, slowing down, or speeding up then slowing down (2.6-3.6x); never stops | speed trend, halves |
| `pair` | two different solids side by side, each with constant spin, periods differing by at least 1.5x, independent directions | comparisons across solids |
| `axis_switch` | one solid spins about the upright axis, stops for 0.5-1.0 s, then spins about a left-right axis (tumbling) or a front-back axis (like a clock hand) | axis identification, axis order |

## Random knobs (all from the seeded `random.Random`)

- **Solid:** one of 21, or two different ones in the pair scenario.
- **Size and position:** the bounding radius is 175-290 px, or 140-225 px in a pair. The centre is jittered by up to ±120 px horizontally (±45 px in a pair) and ±25-30 px vertically.
- **Starting orientation:** a tilt about two axes (±0.55 rad each) and a phase of 0-360°.
- **Camera:** elevation 6-28°, perspective distance 3.2-6.0 body radii.
- **Speeds and timings:** as listed in the scenario table. Clip length is 6-10 s for the steady, pair and speed-change scenarios.
- **Theme:** dark (60 %) or light, 7 dark and 6 light backgrounds, drawn flat, as a vertical gradient or as a vignette.
- **Edge colour:** 6 per theme.
- **Edge width:** 3-5 px base.
- **Depth cue:** `shade` (nearer edges brighter and thicker), `dashed_back` (edges behind the centre dashed and dim) or `flat` (depth comes only from the motion).
- **Extras:** translucent faces (35 % of clips) and small corner dots (40 %).
- **Markers:** 1-3, with colour from 7 and shape from 3. The marked corner is at least 0.35 radii from the upright axis, and two markers on one solid are at least 50° apart around the axis.
- **Timer:** corner and font size.

Measured over 5000 seeds (`generate --no-render`):

- **Uniqueness:** 5000/5000 scene signatures are unique (hash of the rounded scene params).
- **Scenario mix:** pair 1196, axis_switch 1093, reverse 1028, steady 929, speed_change 754.
- **Solids:** all 21 appear, 268-325 times each.
- **Spread of key parameters:**
  - duration 5.0-10.0 s, median 7.6;
  - top speed 40-194 deg/s, median 95;
  - bounding radius 140-290 px;
  - camera elevation 6-28°;
  - 0/1/2 pauses: 2879/1642/479;
  - 1/2/3 markers: 1026/2466/1508.
- **Styles:** the three depth cues, three background styles and four timer corners each take about a third or a quarter of clips.
- **Questions:** 3748 distinct (question, answer) pairs and 51 derivation keys.
- **Answer letters:** A 25.3 %, B 24.8 %, C 25.1 %, D 24.9 %.
- **Retries:** 974 of 5000 scenes needed at least one retry (mean 1.24 tries per scene, at most 7). There were no warnings and no exceptions, with Python run as `-W error`.

## The 10-question ladder

Per-level counts come from `L5.ten_counts(rng)`: L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3, 10 questions in total. Questions are ordered by level with qids q01..q10. Within a level, templates are picked round-robin over shuffled derivation keys, and no (derivation_key, params) pair repeats. Every question has 4 options, and the correct letter is uniform over A-D. On L2-L5 at most one question per level is a pure shape-count question (`requires_motion=False`); all the others need the motion.

- **L1 observation:** `front_dir` (which way the nearest edges move at the start or end: left / right / up / down), `n_stops`, `spin_axis` (upright / left-right / front-back / slanted, at the start or end), `mark_side` (where a marker is at the very start or end: left or right of the spin axis, near or far side).
- **L2 relational:** `more_fv` (faces vs corners), `face_kinds`, `common_face`, `speed_trend`, `halves` (turns further in the first or second half), `start_end_dir` (end direction vs start), `first_front` (which marker crosses in front of the axis first), `wider_orbit`, `speed_compare` (speed before vs after the first stop), `faster`, `dir_pair`, `vf_pair` (pair).
- **L3 measurement:** `n_faces`, `n_corners`, `n_edges`, `n_turns`, `turns_by_time` (complete turns by a timer reading), `period`, `stop_time`, `resume_time`, `count_passes` (times a marker crosses in front of the axis), `first_pass_time`, `turns_before_stop`.
- **L4 integration:** `name` (which solid), `mark_degree` (edges meeting at a marked corner), `pass_time` (2nd or 3rd crossing), `passes_before_stop`, `extra_turn` (how much it turns after its last complete turn, in quarters), `side_at_time` (marker position at a timer reading), `mark_at_stop`, `pair_sum`, `turn_diff`, `turns_total` (complete turns per stretch, summed), `net_dir`, `axis_seq`.
- **L5 counterfactual** (every one re-simulated): `cf_truncate_faces`, `cf_truncate_corners` (every corner sliced off), `cf_dual_corners` (the solid built from face centres), `cf_more_sides` (prism / antiprism / pyramid / bipyramid with one more side), `cf_speed` (twice or half as fast), `cf_speed_first` (twice or half as fast until the first stop), `cf_other_way` and `cf_stop_side` (where a marker would be if it had spun the other way), `cf_no_stop`, `cf_no_switch`, `cf_swap` (the left solid at the right one's speed), `cf_passes_double`.

A "crossing in front of the spin axis" is a marker passing through the middle of its solid on the side nearest the viewer (azimuth 0 about the upright axis). "Complete turns" are whole multiples of 360° turned.

## Margins (no ties or near-ties)

A template is offered only when its answer is clear of every boundary:

- **Turn counts:** the fractional part of the turn count is 0.2-0.8 (at least 72° from a whole turn). `extra_turn` is at least 0.06 turn from any quarter.
- **Crossings:** no crossing within 14 frames (0.5 s + 2 frames) of either end. At the start, the end and every stop, the marker is at least 30° from straight in front.
- **Times:** all time options are whole multiples of a 0.8-2.0 s step apart. The answer is the timer reading at the event frame. Stops and restarts are at least 14 frames from either end.
- **Sides:** a marker named as left or right is at least 25 px from the axis line, and named near or far only when its depth is at least 0.3 of its orbit radius. The label holds for at least 0.5 s around the moment asked about, and for at least 3 frames on either side of a named timer reading. A crossing counted before a stop is at least 0.5 s before it.
- **Comparisons:**
  - speed comparisons need a factor of at least 1.5 (or the speeds are exactly equal);
  - first crossings must be at least 12 frames apart;
  - orbit sizes must differ by at least 1.3x;
  - the halves must differ by at least 1.4x (or the speed is constant);
  - corner and face counts must differ in `vf_pair`.

## Frame-sampling rules and `min_fps`

- Every question-relevant event leaves a persistent result: pauses last at least 12 frames, and positions hold for a window. There are no flashes. Markers are at least 22 px across, and lines are at least 3 px.
- `params["min_fps"]` is the larger of two rates, rounded up to 0.1:
  - the rate that keeps the solid from turning more than 90° between samples, at the fastest speed in the window the question uses;
  - 2 / (the smallest gap in seconds between the events it relies on): the pause length, the time-option step, the gap between first crossings, the gap between crossings, or the length of the side window.
- Over 5000 seeds, 33.4 % of questions are answerable at 1 fps, 84.3 % at 2 fps and 100 % at 4 fps. The maximum `min_fps` is 4.0. 62.8 % of L2-L5 questions have `requires_motion=True`.

## Closed loop

- Every question stores its `derivation_key` and `params`. The generator (`questions.py`) answers from sim's closed-form motion: segment angle integrals, analytic marker azimuths and crossings, and the `SOLIDS` count table.
- `answers.py` re-derives every answer independently from what the renderer draws:
  - **Motion:** the per-frame vertex positions from `render.camera_coords`, taken back to world space, with a Kabsch (SVD) fit of the rotation between consecutive frames. This gives the turning axis, the signed angle and the still frames.
  - **Markers:** crossings are sign changes of the drawn corner's x on the near side, and sides are read from its screen offset and depth.
  - **Edge directions:** the screen motion of the near-side vertices.
  - **Shape counts:** brute force on the drawn vertex coordinates. Every plane through three vertices with all others on one side is a face, an edge is two vertices that share two faces, and the name is the unique table entry with the same signature.
  - **Counterfactuals:** re-simulated. The changed motion is re-tracked through the renderer. Truncated solids (cut by planes inside each corner's normal cone), duals and one-more-side solids are actually built and counted.
- `sim.check` rejects a scene in any of these cases:
  - the geometry disagrees with the table;
  - the motion segments do not tile the clip, or a pause is shorter than 0.5 s;
  - a stop or restart is within 14 frames of either end;
  - a solid can leave the frame, touch the timer, or overlap the other solid;
  - markers are too close to the axis, to each other, or in colour;
  - some level has fewer than 3 usable candidate questions.

  `build` then retries with the same rng, which keeps the result deterministic.
- `python run.py validate` rebuilds every sample from its seed and compares the result. It runs `sim.check` and the ladder checks, re-derives every answer, and checks the MP4 duration, resolution and motion.
- `frame_check.py` decodes the MP4s and checks:
  - the timer pixels at every 12th frame and at every frame a question relies on;
  - each marker's colour blob within 3 px of its corner, on every 2nd frame;
  - marker crossings read from pixels against the ground truth (±1 frame);
  - stops and restarts read from pixel change per solid (±1 frame);
  - that time answers are on the timer at an event frame;
  - side answers against the blob position;
  - ink along every edge;
  - that nothing is drawn outside the solids and the timer.
- `test_spinning_polyhedra.py` covers:
  - determinism, both in-process and across `PYTHONHASHSEED` values, and frame-hash determinism;
  - the ladder, `min_fps`, option counts and the static-question cap;
  - scene uniqueness, `check` and the duration limit;
  - answer agreement;
  - the tamper test (3 corrupted answers give exactly 3 flags);
  - the geometry table (counts, truncation, dual, one more side, unique signatures);
  - that the timer is the only text;
  - hand-checked scenes: a steady cube with known crossing frames, turns and period, a reverse with known stop and restart times, an axis switch, and the speed profiles.

## Determinism

Sample `i` uses `random.Random(seed + i)` for the scene and then the questions. Nothing else is random: there is no global RNG, no time and no uuid, and no set or dict iteration order decides anything. Lists are sorted where it matters, and caches are keyed by content. The same seed gives byte-identical scene, question and answer JSON in different processes (tested with `PYTHONHASHSEED` 0, 1 and 12345), and identical raw frames.

## Files and commands

```
sim.py            solids, scenarios, motion segments, closed-form events, check (pure, seeded)
questions.py      the 10-question ladder (templates per level, phrasing)
answers.py        independent re-derivation from the renderer's per-frame geometry (validator only)
render.py         PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py            CLI: generate / validate (shared pipeline in common/video5.py)
frame_check.py    pixel-level check of rendered clips
test_spinning_polyhedra.py

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/spinning_polyhedra
python run.py validate --dataset-dir ../output_videos/spinning_polyhedra
python frame_check.py --dataset-dir ../output_videos/spinning_polyhedra
python -m pytest test_spinning_polyhedra.py
```
