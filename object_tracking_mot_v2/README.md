# object_tracking_mot_v2

Closed-loop 10-question video-QA benchmark on multiple-object tracking. 3-8 moving objects travel inside a walled
rectangular arena; the questions ask where they are, what they bump into, when, and what would change if the scene
were different. The pipeline is PIL frames piped to ffmpeg (`common/video5.py`) with the 10-question ladder in
`common/ladder5.py` (`LADDER = "ten"`). Clips are 1280x720 at 24 fps and 6-10 s long (144-240 frames).

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | colours / targets / occluders / orbit | what is drawn so it is readable |
|---|---|---|
| `colours` | 3-7 unique objects bouncing off the walls and off each other | every object has its own colour, usually its own shape and a pattern (solid, striped, dotted, ringed); questions name it "the teal hexagon" |
| `targets` | 5-8 identical objects (same colour, shape and size, solid); 1 to min(3, n-2) are targets | the targets wear a thick ring (radius r + 7, 4 px) while everything stands still; the rings vanish and all objects start moving at `start_t` (1.0-1.6 s, a frame instant) |
| `occluders` | 3-7 unique objects and 1-2 opaque screens | a screen is a filled rectangle with a hatched face, drawn over the objects; an object counts as hidden when its drawn shape overlaps a screen by more than 3 px; nothing starts behind a screen |
| `orbit` | 3-7 unique objects, no collisions | each object runs on a smooth Lissajous-type curve inside the arena; paths may cross and later objects are drawn on top |

## Physics

Bouncing motion is exact and event driven (`sim.simulate`): equal-mass elastic discs, walls reflect the normal
velocity, a disc-disc contact exchanges the normal velocity components. Each object is a disc for the physics, and the
drawn shape (circle, square, hexagon, diamond, triangle) is inscribed in that disc. Frame f is shown at t = f / 24.
Orbit objects follow `cx + ax sin(wx t + px)`, `cy + ay sin(wy t + py)`.

## Random knobs (`sim.build`, `sim._style`, `sim._arena`)

- scenario; object count (5-8 for `targets`, 3-7 otherwise); radius 14-30 px (18-30 for `targets`, one shared radius);
  speed 90-260 px/s in a random direction; clip length 144-240 frames;
- arena 560-1000 px wide and 420-670 px high, placed at random inside the play area; objects start apart and inside;
- theme (light twice as likely as dark), background, arena and wall colours, wall width 4-8 px, floor pattern
  (plain, grid, dots, step 40-80 px), centre-line style (solid, dashed, dotted);
- `colours` / `occluders` / `orbit`: distinct colours from nine, shape and pattern per object; `targets`: one shared
  shape and one shared colour from the theme;
- `occluders`: 1-2 screens (4 colours), vertical (70-120 px wide) or horizontal (70-110 px high), long across the
  arena, kept 30 px apart;
- `targets`: the number of targets and `start_t`; `orbit`: amplitudes, angular speeds 0.5-1.6 rad/s and phases.

`sim.build` redraws until `sim.check` and `sim._ladder_ok` pass (see Closed loop).

## On-screen text

The clock and nothing else (the test counts one `d.text(` call in `render.py`). It is a box at the top-left with the
time in seconds to 0.1 s, white 44 px text on a dark box; the value is truncated from the frame number
(`tenths = frame * 10 // 24`). There are no labels: positions are named by the four quadrants (top-left, top-right,
bottom-left, bottom-right) split by faint unnumbered centre lines through the arena middle, and objects by colour and
shape ("the ringed objects" in `targets`).

## The 10-question ladder (`LADDER = "ten"`)

Per clip the count per level is drawn from L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3 (total exactly 10), ordered by level
as q01..q10. Each question has four distinct options. `questions.build` retries up to 20 times to find a level split
the candidates can fill, else the scene is rejected.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `n_objects`, `n_targets` (ringed at the start), `shape_of`, `pattern_of`, `largest` |
| L2 relational | `quadrant_at_t`, `leftmost_at_t`, `first_partner` (first object bumped), `first_wall`, `closest_at_end`, `targets_in_quad_at_t` |
| L3 measurement | `untargeted_in_quad_at_t`, `n_walls`, `n_hits` (in a time window), `time_first_hit`, `time_first_wall`, `time_hidden` (occluders), `quad_changes`, `total_hits` |
| L4 integration | `order_end` (left to right, three objects), `hidden_at_end` (occluders), `quadrant_end`, `targets_in_quad_end` |
| L5 counterfactual | `cf_remove_quad` (an object removed), `cf_speed_quad` (0.5x / 1.5x, orbit also 2x), `cf_remove_hits`, `cf_delay_targets` (start 0.5 / 1.0 / 1.5 s later) |

Margin rules (`questions.py`, `sim.py`):

- `EV` = 0.3 s: an object's own events (wall bounces, collisions) are at least this far apart (`sim.check`, not for
  `targets`, which never asks about events); question times use a 0.5 s grid (`Ctx.grid`, from `start_t + 0.5` to
  1 s before the end); windowed counts keep events 0.25 s from the window ends.
- No two events (any objects) within 2 frames of each other, none within 2 frames of the clip ends or of `start_t`.
- No near-miss: every local minimum of a pair's gap is either a collision or at least 30 px (`GAP_PX`).
- Queried positions are at least 20 px (`QM`) from the centre lines in every frame that shows the named tenth.
- Leftmost, closest and ordering answers lead the runner-up by at least 15 px; the answer object is not hidden.
- Clock readings of an event (`event_ok`): the event falls 0.25-0.75 of the way through a frame interval and both
  frames around it show the same tenth; clock-reading options are at least 0.5 s apart.
- Nothing happens behind a screen: no bounce or collision of an object that is hidden then (`sim.check`).
- Candidates are picked with distinct keys first (`_pick`).

## Closed loop

`answers.py` does not read `sim.py`'s events. Factual answers are measured from what `render.state` draws, frame by
frame (the visible centres):

- unique objects are followed by their look; identical objects (`targets`) by matching each point to the predicted
  position (previous position plus last step), with identities taken from the first, ringed frame;
- a hidden stretch is bridged in a straight line from the last seen velocity;
- a wall bounce is a contact-range minimum of the distance to a wall where the velocity component really flips; a
  collision is a minimum of a pair's gap within 30 px where both velocities change, timed by extrapolating the
  incoming motion to contact;
- what-ifs (`cf_*`) use a separate fixed-step (1 ms) integrator in `answers.resim` with bisection for the contact time
  inside a step, not `sim.simulate`; `test_whatif_integrator_agrees` compares them. `sim._ladder_ok` rejects scenes
  that cannot fill the ladder (a trial `questions.build` with a fixed rng).

`frame_check.py` decodes the MP4s and checks: the clock box on every frame against templates of all readings
0.0-10.0 s; objects located in the pixels (by colour blob) against `render.state` (tolerance 3 px, same seen / hidden
status); frame 0 of `targets`: the rings found equal the stored targets. It then re-derives every answer except
`shape_of` / `pattern_of` from the decoded centres (jitter band 2 px) and checks the clock at the frame of every
question that names a time.

## Frame sampling and `min_fps`

Every question stores `params["min_fps"]`, the lowest sampling rate at which it is still answerable:

- static looks and final-frame questions (`n_objects`, `shape_of`, `pattern_of`, `largest`, `closest_at_end`,
  `order_end`, `quadrant_end`, `targets_in_quad_end`) need 1 fps; `hidden_at_end` needs 2;
- questions about a named clock time need 1 fps for whole seconds and 2 fps for half seconds, or
  `fps_for(gaps)` = ceil(1 / smallest gap in seconds to an event) if larger, at most 24;
- clock readings (`time_first_hit`, `time_first_wall`, `time_hidden`) and every `cf_*` question need 4.

Share of questions by `min_fps` (5000 dry-run scenes, 50,000 questions):

| min_fps | share | cumulative |
|---|---:|---:|
| 1 | 50.8 % | 50.8 % |
| 2 | 12.1 % | 62.9 % |
| 3-4 | 37.1 % | 100 % |

All L2-L5 questions need the motion (requires_motion = True).

## Determinism

All randomness comes from the `random.Random` passed to `sim.build` and `questions.build`. Clip length and `start_t`
are whole frames, and stored parameters are rounded before anything is derived from them. Caches are keyed by the
whole scene: `render._static` and `answers._CACHE` by the scene JSON, and `sim.motion` by the identity of the scene
object (`_MEMO`, which holds the scene so its id cannot be reused; a copy of the scene simply simulates again). The
same seed gives byte-identical JSON across `PYTHONHASHSEED` values and identical raw frames
(`test_object_tracking_mot_v2.py`).

## Files and commands

```
sim.py                              scenes, exact event solver, consistency check
questions.py                        the 10-question ladder
answers.py                          answers from the drawn tracks, separate what-if integrator
render.py                           PIL renderer (1280x720, 24 fps)
frame_check.py                      decodes MP4s, re-derives answers from pixels
preview.py                          renders the first N seeds of every scenario
stats.py                            diversity / balance report
run.py                              CLI (shared pipeline in common/video5.py)
test_object_tracking_mot_v2.py      determinism, ladder, tamper, clock only text, physics, what-if, tracker, pinned seeds

python3 run.py generate --n 20 --seed 0 --output-dir out
python3 run.py validate --dataset-dir out
python3 preview.py [--per 10]
python3 frame_check.py --dataset-dir ../output_videos/object_tracking_mot_v2_preview
python3 stats.py out/annotations.jsonl
python3 test_object_tracking_mot_v2.py
```

## Results

- 5000-scene dry run (`--no-render`, validate `--skip-video`, 4 chunks): 0 mismatches, 0 ladder errors, 0 check
  failures, no warnings (`python -W error`). 3806 of 5000 scenes (76 %) were redrawn at least once (mean 149
  attempts, max 1895; mostly event spacing, glancing hits, wall grazes and, in the targets scenario, a ringed object
  bumping an un-ringed one); no seed failed.
- Diversity: 5000 / 5000 unique scene signatures; scenarios 24.2-25.7 % each; 13,084 distinct (question, answer)
  pairs; answer letters A 24.8 %, B 25.4 %, C 24.8 %, D 25.0 %.
- Tamper: 3 answers corrupted in each of 100 scenes -> exactly 300 questions flagged (plus one "rebuilt questions
  differ" line per tampered scene).
- Determinism: identical JSON across `PYTHONHASHSEED` values; identical raw frames on re-render; regenerating seeds
  0-49 into a fresh folder reproduces `annotations.jsonl` byte for byte.
- Previews (`preview.py --total 50`, 12-13 per scenario): `run.py validate` 0 mismatches; `frame_check.py` 50 clips,
  9236 frames, clock template match 9236 / 9236, target rings found exactly, 454 answers re-derived from decoded
  frames, max centre error 2.56 px, 0 problems.
