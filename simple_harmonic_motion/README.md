# simple_harmonic_motion

Closed-loop 10-question video-QA benchmark on simple harmonic motion. Each clip shows 3-6 oscillators side by side
(masses on springs, or pendulums), each with its own unnumbered tick ruler. The pipeline is PIL frames piped to
ffmpeg (`common/video5.py`) with the 10-question ladder in `common/ladder5.py` (`LADDER = "ten"`). Clips are
1280x720 at 24 fps and 6-10 s long (144-240 frames). Questions are answered from what is drawn on screen.

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | setup | readable from the video |
|---|---|---|
| `horizontal` | 3-6 blocks on springs fixed to a wall, undamped, random start phase | motion along a horizontal ruler per block |
| `vertical` | 3-6 masses hanging from a ceiling on springs, undamped, random start phase | motion along a vertical ruler per mass |
| `pendulum` | 3-5 small-swing pendulums of different string lengths hanging from a bar | sideways offset from the rest mark; string length is drawn |
| `damped` | springs (horizontal or vertical orientation), every mass starts at rest, pulled out, at t = 0 | a dashpot is drawn beside each spring; the swing dies away |
| `staggered` | springs (horizontal or vertical orientation); each mass is held out, then released at its own time | a small stop post is drawn while a mass is held and vanishes at its release |

Each ruler has unnumbered ticks; the longer tick is the rest mark (where the mass is at u = 0).

## Random knobs (`sim._style`, `sim.build`)

- object count (3-6, pendulum 3-5), colours (9 in the palette, unique per clip), shapes (circle, square, hexagon,
  diamond, triangle), patterns (solid, striped, dotted, ringed), sizes (36-64 px springs, 32-52 px pendulums);
- periods 0.8-3.0 s for springs, pairwise at least 10% apart; pendulum string lengths 200-560 px with the same 10%
  period gap; pendulum `g` scaled so a 380 px string has a 1.2-2.0 s period;
- amplitudes within 0.15 tick of a whole tick count; tick spacing 24-44 px (pendulum 22-34 px);
- theme (light or dark), background colour and pattern (plain, gradient, grid, dots), spring style (zigzag, coil,
  wave), coil count, spring width and line width, fixture style (plain or hatched), layout positions;
- clip length, start phases (undamped), release times (staggered), damping decay rate 0.10-0.32 per second (damped);
- speed cap: amplitude x omega <= 1000 px/s, so nothing moves more than about 42 px per frame.

`sim.check` also requires every object to stay in the frame, away from the clock box and from the other objects.

## On-screen text

The only text in the frame is the clock: a box at the top-left with the time in seconds to 0.1 s, white 44 px text on
a dark box. The value is truncated from the frame number (`tenths = frame * 10 // 24`). No object can reach the box
(`sim.check`).

There are no other labels. The rulers carry **unnumbered** tick marks with a longer rest-mark tick. Numbers are not
needed because questions ask distances in tick spacings of the mass's own ruler, which are counted from the rest
mark, and colour plus shape identify each object ("the teal hexagon"), so labels are not needed either.

## The 10-question ladder (`LADDER = "ten"`)

Per clip the count per level is drawn from L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3 (total exactly 10), ordered by level
as q01..q10. Each question has four distinct options; a thin template pool redraws the parameters (up to 20 times).

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `n_objects` (count), `shape_of`, `pattern_of`, `widest_swing` (farthest from rest mark), `largest_object`, `longest_string` (pendulum), `first_released` (staggered), `n_right_at_t` (how many right of their rest mark at a clock time), `n_held_at_t` (staggered) |
| L2 relational | `direction_at_t` (moving direction and side at a clock time), `farthest_at_t`, `first_to_far_end` (first to reach the end of its swing after a clock time), `n_faster_than`, `n_moving_far_at_t`, `released_before_count` (staggered) |
| L3 measurement | `period_of`, `amplitude_ticks` (undamped), `peak_ticks` (damped, a turning point at a clock time), `crossing_time` (clock reading of the next rest-mark pass), `crossings_count`, `far_end_count`, `release_time` (staggered) |
| L4 integration | `period_difference`, `order_by_period`, `total_crossings`, `period_of_extreme` (widest, narrowest, largest, smallest, longest/shortest string, first/last released), `return_time` (staggered), `decay_count` (damped) |
| L5 counterfactual | `cf_period` (changed mass, spring stiffness, amplitude, string length or bob), `cf_crossings`, `cf_no_damping` (damped), `cf_release_later` (staggered) |

Margin rules (`questions.py`):

- `EV` = 0.3 s: events a question relies on stay this far from anything that could change its answer.
- Clock-time questions use the 0.5 s grid (`Ctx.grid`).
- Seconds answers are at least 0.015 s from a 0.1 s rounding boundary (`safe1`).
- Clock readings (`crossing_time`, `cf_release_later`) require the event to sit well inside a frame interval and both
  frames around it to show the same tenth (`event_frame`).
- Tick answers: the amplitude is within 0.25 tick of a whole number (by construction within 0.15 for undamped
  amplitudes; `peak_ticks` checks 0.25).
- Option spacing: durations at least max(0.3 s, 15%) apart (`period_difference` 0.3 s); clock readings at least 0.5 s
  apart.
- Comparisons (widest, largest, farthest) need at least 10% and 15 px between first and second.
- Period questions need at least three visible crossings (half a tick or more of swing).
- Counterfactuals are answered by re-simulating the changed system (see Closed loop).

## Closed loop

`answers.py` does not read `sim.py`'s event lists. It builds a per-frame track of what is drawn (`render.state`: mass
centre and whether its stop post is still drawn) and measures it:

- rest crossings are sign changes of the offset from the rest mark, interpolated between frames;
- turning points come from a zigzag detector on the per-frame offset, refined with a parabola through three frames;
- periods are twice the mean gap between crossings (taken at a clear speed); amplitudes are the largest refined offset;
- the release is the last frame drawn with the stop post; clock values are the clock text of the relevant frame.

Counterfactual answers re-simulate the changed system with fixed-step RK4 (dt = 1/1200 s) and read the answer off
the numerical trajectory, while `questions.py` uses the closed form. `sim.check` rejects bad scenes (shared colours,
periods within 10%, amplitude not near a whole tick, too fast, overlaps, release times too close). `frame_check.py`
runs the same derivation on centres located in the decoded MP4 (jitter band h = 2 px) and matches the clock box
against templates of all possible readings; every question except `shape_of` / `pattern_of` must reproduce the
stored answer.

## Frame sampling and `min_fps`

Every question stores `params["min_fps"]`, the lowest sampling rate at which it is still answerable:

- `fps_for(gaps)` = ceil(1 / smallest gap in seconds between the events the question relies on), at least 1 and at
  most 24;
- questions that need a period or a clock-time reading have a floor of 4;
- clock-grid times need 1 fps for whole seconds and 2 fps for half seconds (`align`);
- amplitude / widest questions on swinging masses use `fps_sampling`: the smallest rate (from 1, 2, 3, 4, 6, 8, 12,
  24) at which the largest sampled offset of a swing is within the allowed slack of the true peak.

Share of questions by `min_fps` (5000 dry-run scenes, 50,000 questions):

| min_fps | share | cumulative |
|---|---:|---:|
| 1 | 21.6 % | 21.6 % |
| 2 | 17.5 % | 39.1 % |
| 3-4 | 58.8 % | 98.0 % |
| > 4 | 2.0 % | 100 % |

## Determinism

All randomness comes from the `random.Random` passed to `sim.build` and `questions.build`. Release times are kept as
exact `frame / 24` (`t_rel`), and stored parameters are rounded before anything is derived from them. Caches in
`render.py` and `answers.py` are keyed by the full scene JSON. The same seed gives byte-identical JSON across
`PYTHONHASHSEED` values and identical raw frames (`test_simple_harmonic_motion.py`).

## Files and commands

```
sim.py                          scenes, closed-form motion, consistency check
questions.py                    the 10-question ladder
answers.py                      answers from the drawn track, RK4 counterfactuals
render.py                       PIL renderer (1280x720, 24 fps)
frame_check.py                  decodes MP4s, re-derives answers from pixels
preview.py                      renders the first N seeds of every scenario
stats.py                        diversity / balance report
run.py                          CLI (shared pipeline in common/video5.py)
test_simple_harmonic_motion.py  determinism, ladder, tamper, closed form vs numeric, pinned seeds

python3 run.py generate --n 20 --seed 0 --output-dir out
python3 run.py validate --dataset-dir out [--skip-video]
python3 preview.py
python3 frame_check.py --dataset-dir ../output_videos/simple_harmonic_motion_preview
python3 stats.py out/annotations.jsonl
python3 test_simple_harmonic_motion.py
```

## Results

- 5000-scene dry run (`--no-render`, validate `--skip-video`, 4 chunks): 0 mismatches, 0 ladder errors, 0 check
  failures, no warnings (`python -W error`). 1215 of 5000 scenes (24 %) redrew their layout at least once (max 15
  attempts); none failed.
- Diversity: 5000 / 5000 unique scene signatures; scenarios 19.3-20.7 % each; 23,651 distinct (question, answer)
  pairs; answer letters A 24.9 %, B 25.2 %, C 24.9 %, D 25.0 %; all L2-L5 questions need motion.
- Tamper: 3 answers corrupted in each of 100 scenes -> exactly 300 questions flagged (plus one "rebuilt questions
  differ" line per tampered scene).
- Determinism: identical JSON across `PYTHONHASHSEED` values; identical raw frames on re-render; regenerating seeds
  0-49 into a fresh folder reproduces `annotations.jsonl` byte for byte.
- Previews (`preview.py`, 10 per scenario): `run.py validate` 0 mismatches; `frame_check.py` 50 clips, 9758 frames,
  clock template match 9758 / 9758, 456 answers re-derived from decoded frames, max blob-centre error 2.81 px,
  0 problems.
