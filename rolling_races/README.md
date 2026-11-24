# rolling_races

Closed-loop 10-question video-QA benchmark on **constrained motion and projectile jumps on curved tracks** (New
Problem Types shortlist B7). 2-10 frictionless balls get the same push and race down their own colour-coded tracks
(straight, cycloid, sag, bend, slow start, steep start, wavy) from one start to one finish hole. About a third of the
races have no gaps. In the rest, some tracks have a gap with a take-off lip, and a ball either flies across and lands
(keeping only its along-track speed) or falls through and comes to rest on the pit floor. Questions cover the winner,
falls, finish times and places, equal final speeds (energy conservation), and re-simulated what-ifs: a track with no
gap, a straight ramp, a push three times as strong.

The pipeline is PIL frames piped to ffmpeg (`common/video5.py`), with the 10-question ladder in `common/ladder5.py`
(`LADDER = "ten"`). Clips are 1280x720 at 24 fps and at most 10 s long (72-240 frames).

## What a clip shows

- **The start**: a ledge at the top-left or top-right (the layout is mirrored in half the clips) and a short bar, the
  **start gate**, just downhill of it. All balls wait at the start behind the gate; the gate is gone from the release
  frame on, and every ball sets off with the same push.
- **The tracks**: each drawn in its ball's colour from the start to the shared finish hole, lower and on the other
  side. A gap is a missing stretch, with a short take-off lip at its near edge.
- **The finish rack**: beside the hole, one ring slot per ball in the ball's colour, in a fixed random order (so the
  rack never shows the finishing order). A ball that reaches the finish drops into the hole and fills its slot.
- **The pit floor**: a hatched band at the bottom. A ball that misses a jump falls and comes to rest there.
- **Depth**: each ball rides only its own track. A ball in the air (jumping a gap or falling) is drawn behind the
  other colours' tracks, so when it crosses one it visibly passes behind it instead of looking as if it should land
  on it (about 88 % of falling and 68 % of jumping balls cross another track). Rolling balls are drawn on top.
- **The clock** (see below). The race plays 1.4-2.6 times slower than real time.

Every question-relevant event leaves a persistent result: a filled rack slot, a ball on the pit floor, the gate gone.
Leaving the track at a lip is a continuous motion, never a one-frame flash.

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | weight | setup | what makes it readable from the video |
|---|---:|---|---|
| `plain` | 34 % | no gaps at all: a pure race of shapes | every track is unbroken |
| `jumps` | 18 % | 1 to about half of the tracks have one gap | gaps are visible breaks with a lip |
| `double` | 16 % | 1-3 tracks have two gaps one after the other (each such ball clears the first); other tracks may have one | a track broken in two places |
| `all_gaps` | 16 % | every track has a gap | every track is broken |
| `steep` | 16 % | a tall, narrow arena (W 5.5-7 m, D 4-5 m instead of 8.5-11 m by 2.6-4 m), 1 to about half of the tracks gapped | the steep, narrow layout |

## Random knobs (`sim.build`, `sim._track`, `sim._gap_list`, `sim._style`)

- arena: width W and drop D (above); pit-floor depth 1.25-1.45 D + 0.5 m;
- balls: 2-10; push v0 0.6-1.5 m/s (the same for every ball); release 0.6-1.2 s into the clip, on a frame instant;
  slow-motion factor 1.4-2.6 (capped so the clip fits in 10 s); tail 0.8-1.4 s after the last event;
- tracks: family (straight, cycloid, sag 0.15-0.75, slow start u^1.3-2.3, steep start u^0.4-0.8, wavy 0.08-0.22 with
  2-3 humps, bend -0.4..0.9 peaking early or late); no two tracks closer than 0.17 m over more than 20 % of the middle
  stretch;
- gaps: one gap starting at 24-60 % of W and 11-30 % of W long, or two gaps (18-36 % + 9-16 % of W, then after 12-22 %
  of W, 8-16 % long); lip 8-24 degrees;
- look: theme (light twice as likely as dark), 5 backgrounds per theme, background pattern (plain, faint grid,
  dots) and grid step; ball colours from 10 per theme (every pair at least 135 apart in summed RGB); ball pattern
  (solid, dot, ring, cross); ball radius 11-16 px; track width 3-6 px and style (solid, sleepers, cased);
- camera framing: scale (84-100 % of the largest that fits), position of the arena, mirrored or not; finish-rack
  offset and spacing.

`sim.build` redraws a track that breaks a rule (too close to another, a borderline jump, a fallen ball resting
outside the pit or on top of another, a two-gap track whose ball never reaches the second gap), and redraws the whole
race if it fails `sim.check` or cannot fill the ladder (`sim._ladder_ok`).

## On-screen text

**The clock is the only text in the frame**: a dark box with white 44 px digits, in the top corner opposite the
start, showing clip time to 0.1 s (`tenths = frame * 10 // 24`). Tracks never rise above the start line, which is
below the clock box, and `sim.check` verifies on every second frame that no ball comes near the box.

No other text is needed. Balls are told apart by colour (each ball and its track share one of 10 well-separated
colours), so questions name them as "the red ball". The race needs no scale: no question asks for a distance. The
original renderer's ball letters and the race-time readout are gone. There are no titles, legends or status lines.

## The 10-question ladder (`LADDER = "ten"`)

Per clip the count per level is drawn from L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3 (total exactly 10), uniformly
among the splits the candidates can fill, ordered by level as q01..q10. Every question has four distinct options;
most templates have two or three phrasings.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `n_balls`, `n_gapped`, `n_fell`, `n_finish`, `winner`, `lowest_track`, `straight_track`, `two_gap_track` |
| L2 relational | `earlier` (which of two finishes first), `fell_first` (which of two fallers leaves its track first), `which_falls`, `which_lands` (lands across at least one gap), `leader_at` (who is farthest along at a clock second, asked only while no ball is falling), `same_speed` (fastest at the finish: all the same, by energy conservation), `before_x`, `after_x` |
| L3 measurement | `finish_clock`, `leave_clock` (reaching a gap's edge), `floor_clock` (landing on the pit floor), `release_clock`, `race_time` (release to finish), `finish_gap` (between two finishers) |
| L4 integration | `finish_order`, `place_of`, `last_finisher`, `next_after`, `n_after_x`, `n_racing_at_finish`, `fell_before_first`, `n_done_at_rest`, `total_landings`, `lead_changes` |
| L5 counterfactual | `cf_no_gap` / `cf_straight` (place if this track had no gap / were a straight ramp), `cf_no_gap_clock` / `cf_straight_clock` / `cf_push_clock` (clock time at the finish under the change), `cf_push_falls`, `cf_push_winner`, `cf_push_lands`, `cf_no_gaps_winner` |

Distractors are plausible: the next finishers for a winner, the neighbouring places, a swapped neighbouring pair or
a faller slipped into a finishing order, other events' clock readings and readings 0.5-2 s away for times, the
actual count for a what-if count, and nearby integers. With fewer than four balls, colour questions are padded with
"two balls tie for first" / "they arrive at the same moment" / "neither of them reaches the finish".

Pick-one questions (`which_falls`, `which_lands`, `before_x`, `after_x`, `cf_push_lands`) are checked strictly:
`answers.py` fails the question unless exactly one option has the property.

Counterfactual answers are re-simulated under the changed rule (no gap, straight ramp, three times the push), never
guessed.

Margin rules (`questions.py`, `sim.py`):

- anything compared in time is at least 0.3 s apart on the clock (7 frames); `finish_gap` pairs at least 0.5 s;
- clock readings (`event_ok`): the event falls 0.3-0.7 of the way through a frame interval and both frames around it
  show the same tenth; options at least 0.5 s apart; a gap-edge crossing is only asked about where the ball moves at
  least 2.5 px per frame;
- "farthest along" (`leader_at`, `lead_changes`): the leader is at least 15 px ahead for the whole second around the
  asked time, with no ball finishing, landing on the floor or leaving the screen in that window;
- track comparisons: the lowest track is at least 0.25 m (>= 20 px) lower than the next;
- every jump clears or misses its far edge by at least 0.15 m at the real push and at three times the push;
- every event (release, leave, land, floor, finish) is at least 0.55 s after the start and 0.75 s before the end.

`requires_motion` is false for the static counts and track questions, `n_fell` / `n_finish` / `which_falls` (the rack
and the pit show them at the end) and `same_speed`; true otherwise.

## Closed loop

`answers.py` never reads `sim.py`'s runs. Factual answers are measured from what the renderer draws, frame by frame
(`render.state`, or the same read off decoded MP4 frames by `frame_check.py`): each ball's centre, which rack slots
are filled and whether the gate is there. A ball has finished from the first frame its slot is filled; it fell if it
ends up resting on the pit floor; it leaves its track at a gap when its centre crosses the gap's near edge (fitted to
the frames around the crossing); the release is the first frame without the gate. Orders, places, counts, clock
readings and the leader at a moment all follow from these.

What-ifs and the final-speed question use a separate integrator: on each straight piece of the drawn polyline the
acceleration is constant, so speed and time are exact (v_b^2 = v_a^2 + 2 g dy, dt = 2 L / (v_a + v_b)), and a flight
is the parabola from the lip intersected with each polyline piece in closed form. `sim.py` integrates ds / v with
Simpson's rule and finds landings by bisection. The two agree to 2e-7 s (`test_integrators_agree`).

`sim.check` rejects: tracks too close, borderline or degenerate jumps (at both pushes, also with no gap and as a
straight ramp), tracks dipping near the pit floor, no finisher, scenario features missing, events too near the clip
edges, a release off a frame instant, fallen balls resting outside the pit or on top of each other, balls leaving the
screen or touching the clock box, the rack leaving its area, objects too small.

`frame_check.py` decodes the MP4s and checks: the clock on every frame against templates of all readings 0.0-10.0 s;
the gate on every frame; the rack slots on every frame; every ball located in the pixels (candidates from a box
filter on its colour, refined by masked template matching against its own sprite) against `render.state` (3 px
tolerance) wherever it is isolated; then it re-derives every answer from the decoded frames and checks the decoded
clock at the event frame of every clock-reading answer.

## Frame sampling and `min_fps`

Every question stores `params["min_fps"]`:

- 1 fps for counts, track questions, what-if counts and winners, and `leader_at` (the leader holds for a whole second
  around the asked time);
- comparisons and orders: `ceil(1 / smallest gap)` between the events compared (1-4 fps);
- clock readings, durations and what-if clock times: 4 fps (options 0.5 s apart);
- `lead_changes`: at least 2 fps.

5000 dry-run scenes (50,000 questions): 47.4 % answerable at 1 fps, 59.9 % at 2 fps, 100 % at 4 fps.

At half size (640x360) the clock digits are 22 px tall, balls 11-16 px across and tracks 1.5-3 px wide.

## Determinism

All randomness comes from the `random.Random` passed to `sim.build` and `questions.build` (no global RNG, time, uuid
or set-order decisions). Each level's candidate set is the same for any rng, so the ladder check in `sim.build`
holds for the real build. The release is a frame instant, clip length is a whole number of frames, and parameters
are rounded before anything is derived from them. Caches are keyed by the full track shape or the scene JSON and are
bounded. The same seed gives byte-identical JSON across `PYTHONHASHSEED` values and identical raw frames
(`test_rolling_races.py`).

## Files and commands

```
sim.py                 scenes, physics, layout, consistency check
questions.py           the 10-question ladder
answers.py             answers from the drawn frames; own integrator for what-ifs
render.py              PIL renderer (1280x720, 24 fps)
frame_check.py         decodes MP4s, checks clock / gate / rack / balls, re-derives answers
preview.py             renders the first N seeds of every scenario
stats.py               diversity / balance report
run.py                 CLI (shared pipeline in common/video5.py)
test_rolling_races.py  determinism, ladder, tamper, clock-only text, integrators agree, hand-checked physics

python3 run.py generate --n 20 --seed 0 --output-dir out
python3 run.py validate --dataset-dir out [--skip-video]
python3 preview.py --per 10
python3 frame_check.py --dataset-dir ../output_videos/rolling_races_preview
python3 stats.py out/annotations.jsonl
python3 test_rolling_races.py
```

## Results

- 5000-scene dry run (`--no-render`, validate `--skip-video`, 4 chunks, `python -W error`): 0 mismatches, 0 ladder
  errors, 0 check failures, no warnings or exceptions (the 2000-scene requirement is the first 2000 of these).
  2277 of 5000 seeds (45.5 %) redrew their race at least once (mean 1.85 attempts, max 14); no seed failed.
- Diversity: 5000 / 5000 unique scene signatures; scenarios plain 34.3 %, jumps 18.5 %, double 16.7 %, all_gaps
  15.3 %, steep 15.3 % (34.3 % of races have no gap); 2-10 balls (median 6); 7014 distinct (question, answer) pairs;
  answer letters A 25.13 %, B 24.75 %, C 25.27 %, D 24.85 %; 41 templates; all 22 level splits used;
  `requires_motion` on 95.7 % of L2-L5 questions.
- Level counts over 50,000 questions: L1 9131, L2 12050, L3 12071, L4 8847, L5 7901.
- min_fps: 47.4 % of questions answerable at 1 fps, 59.9 % at 2 fps, 100 % at 4 fps.
- Tamper: 3 answers corrupted in each of 100 scenes -> exactly those 300 questions flagged.
- Determinism: identical JSON across `PYTHONHASHSEED` values; identical raw frames on re-render; regenerating seeds
  0-49 into a fresh folder reproduces `annotations.jsonl` byte for byte (and matches the 5000-run's first 50 lines).
- Previews (`preview.py`, 10 per scenario): `run.py validate` 0 mismatches including the MP4 checks.
  `frame_check.py` on the decoded MP4s: 50 clips, 8963 frames, clock template match 8963 / 8963, gate
  8963 / 8963, rack slots 8963 / 8963, 18,862 isolated ball positions within 1.45 px of `render.state` (155
  airborne balls more than 35 % hidden behind other tracks not position-checked), all 500 answers re-derived from
  the decoded frames, clock checked at every clock-answer event frame, 0 problems.
