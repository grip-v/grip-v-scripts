# traffic_light_queue

Closed-loop 10-question video-QA benchmark on vehicles queuing at traffic lights. 4-8 vehicles enter a road through a
gate, drive at constant speed, stop dead at a red stop line and behind the vehicle ahead, and pull away when they can.
The questions ask which vehicle does what, when the lights change, how many vehicles have passed, and what would change
if the scene were different. The pipeline is PIL frames piped to ffmpeg (`common/video5.py`) with the 10-question
ladder in `common/ladder5.py` (`LADDER = "ten"`). Clips are 1280x720 at 24 fps and 7-10 s long (168-240 frames).

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | roads / lights | what is drawn so it is readable |
|---|---|---|
| `single` | one lane, one light | a stop line across the lane, a lamp box beside it showing red or green; the light is "the traffic light" |
| `two_lanes` | two parallel lanes, each with its own light, cycle and phase | lights are named "the light on the upper / lower road" |
| `crossing` | a left-to-right and a top-to-bottom road crossing; the lights are opposite (one green only while the other is red, with a 0.5-0.8 s all-red gap at both ends; stop lines 100 px before the junction so lamps sit clear of the other road) | lights are named "the light on the left-to-right / top-to-bottom road"; cars of both roads are never inside the junction together |
| `mixed` | one lane, one light, vehicles with clearly different speeds (0.7x, 0.85x, 1.0x, 1.3x of a base speed 130-210 px/s) | fast vehicles close up on slow ones; at least two distinct speeds |
| `two_lights` | one lane with two lights in a row (at 30-42 % and 66-80 % of the lane) | lights are named "the first / second light" |

## What is drawn / on-screen text

Road rectangles (with a dashed, solid or no edge line, optional kerb), a dark gate at the lane start, a stop line and
a lamp box per light (red or green lamp), day or night theme (day twice as likely). Vehicles are named by colour and
body ("the teal van"). The colour is unique per clip, drawn from ten (red, orange, yellow, green, teal, blue, purple,
pink, brown, lime; no white or grey, which would look like road markings). The body is shown by its length
(car 1.0, van 1.25, truck 1.6, bus 1.9 times a base 38-52 px) and by details:

| body | details |
|---|---|
| car | windscreen at the front and a rear window |
| van | windscreen and one long side window |
| truck | a cab separated from the cargo box by a dark gap |
| bus | a row of side windows |

The only text is the clock (the test counts one `d.text(` call in `render.py`): a box at the top-left with the time in
seconds to 0.1 s, white 44 px text on a dark box; the value is truncated from the frame number
(`tenths = frame * 10 // 24`). Frame f is shown at t = f / 24. There are no labels or numbers on vehicles or lights.

## Simulation (`sim.simulate`)

Fixed-step (1/240 s), frame-aligned. A vehicle enters at its `t_in` when there is room behind the previous vehicle on its
lane, drives at its own speed, stops dead at the stop line of a light that is red if its front has not crossed the line
yet, stops behind the vehicle ahead keeping a fixed gap (10-22 px), and moves on at full speed as soon as it can.
Events: `enter`, `stop`, `go`, `cross` (front passes a stop line), `leave`. A light has `red` and `green` durations
(1.6-3.4 s each) and a `phase`.

## Random knobs (`sim.build`, `sim._style`, `sim._lanes`, `sim._lights`)

- scenario; vehicle count 4-8; colours sampled without repeat; body per vehicle; lane per vehicle; entry times 0.0-0.4 s
  for the first, then 0.4-1.4 s apart; clip length 168-240 frames;
- lane width 54-74 px, lane position, lamp size 22-30 px, theme, background and road colours, edge line, verge, pole side;
- light cycles, phases, stop-line positions.

`sim.build` redraws until `sim.check` and `sim._ladder_ok` (a trial `questions.build` with a fixed rng) pass.

## The 10-question ladder (`LADDER = "ten"`)

Per clip the count per level is drawn from L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3 (total exactly 10), ordered by level
as q01..q10. Each question has four distinct options. `questions.build` retries up to 20 times to find a level split
the candidates can fill, else the scene is rejected. 21 derivation keys:

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `n_lights`, `n_on_road_end`, `light_end` (colour in the last frame), `longest_end` |
| L2 relational | `light_at_t`, `first_enter`, `first_stop`, `front_of_queue_at_t` |
| L3 measurement | `n_stops`, `time_first_stop`, `time_first_green`, `n_crossed_by_t`, `n_still_at_t`, `n_turn_red` |
| L4 integration | `order_cross`, `n_crossed_end`, `never_cross`, `n_still_end` |
| L5 counterfactual | `cf_always_green` (a light stays green; not in `crossing`), `cf_remove_stops` (a vehicle never came), `cf_late` (every vehicle arrives 0.5 / 1.0 / 1.5 s later) |

Question templates (one per key):

| level | key | template |
|---|---|---|
| L1 | `n_lights` | How many traffic lights are there? |
| L1 | `n_on_road_end` | In the last frame of the clip, how many vehicles are on the road? |
| L1 | `light_end` | In the last frame of the clip, what colour is {light} showing? |
| L1 | `longest_end` | In the last frame of the clip, which vehicle on the road is the longest? |
| L2 | `light_at_t` | At {t} s on the clock, what colour is {light} showing? |
| L2 | `first_enter` | Which vehicle comes onto the road first? |
| L2 | `first_stop` | Which vehicle is the first to come to a complete stop? |
| L2 | `front_of_queue_at_t` | At {t} s on the clock, which vehicle is at the front of the queue at {light}? |
| L3 | `n_stops` | How many times does {vehicle} come to a complete stop during the clip? |
| L3 | `time_first_stop` | When does {vehicle} first come to a complete stop? Give the clock reading. |
| L3 | `time_first_green` | When does {light} first turn green? Give the clock reading. |
| L3 | `n_crossed_by_t` | By {t} s on the clock, how many vehicles have driven past the stop line of {light}? |
| L3 | `n_still_at_t` | At {t} s on the clock, how many vehicles are standing still? |
| L3 | `n_turn_red` | How many times does {light} change from green to red during the clip? |
| L4 | `order_cross` | In what order do these vehicles drive past the stop line of {light}? |
| L4 | `n_crossed_end` | During the whole clip, how many vehicles drive past the stop line of {light}? |
| L4 | `never_cross` | Which of these vehicles never drives past the stop line of {light}? |
| L4 | `n_still_end` | At the end of the clip, how many vehicles are standing still? |
| L5 | `cf_always_green` | Suppose {light} had stayed green the whole time. How many vehicles would have driven past its stop line by the end of the clip? |
| L5 | `cf_remove_stops` | Suppose {vehicle A} had never come along. How many times would {vehicle B} have come to a complete stop during the clip? |
| L5 | `cf_late` | Suppose every vehicle had arrived {d} s later than it did (the lights unchanged). How many vehicles would have driven past the stop line of {light} by the end of the clip? |

(Wording is abridged; exact strings are in `questions.py`.)

### Ambiguity guards

`sim.check` rejects a scene when:

- a vehicle stops at a line or crosses a line within 0.15 s of its own light changing (a car that waited at the line and
  pulls away on green is fine);
- one vehicle's stop / go events are less than 0.3 s apart (`EV`);
- any event falls within 0.5 s of the clip ends (entries at the start are allowed);
- in `crossing`, vehicles of both roads are inside the junction box in the same frame;
- `mixed` has only one speed; fewer than 4 vehicles enter or fewer than 2 stops happen; two vehicles share a colour.

Question-level rules (`questions.py`): question times lie on a 0.5 s grid from 1 s to 1 s before the end; a queried
state is not within 0.3 s of the vehicle's own event; clock-reading answers need the event 0.25-0.75 of the way through
a frame interval with the same tenth on both sides, and clock options are at least 0.5 s apart. What-if questions
additionally need the answer unchanged with all speeds x0.97 and x1.03, simulated 1 s past the end of the clip.

## Closed loop

`answers.py` does not read `sim.py`'s events. Factual answers are derived from what `render.state` draws, frame by
frame: each vehicle's box (its front read from the box and the known body length) and each lamp's colour. A vehicle is
still over a frame interval when its front does not move; a stop is the first still interval after moving; a vehicle
passes a stop line in the first frame its front is beyond the line. The what-ifs (`cf_*`) re-simulate with a separate
fixed-step (1/480 s) integrator in `answers.resim`, not with `sim.simulate`.

`python run.py validate` fails a clip if any stored answer disagrees with the re-derived one, if rebuilding the
questions from the seed differs, or if the MP4's duration, resolution or motion is off.

`frame_check.py` decodes the MP4s and checks: the clock box on every frame against templates of all readings
0.0-10.0 s; each lamp's colour against the drawn light; each vehicle located in the pixels (by colour, within its
lane) against where it is drawn, and absent when not drawn. It then rebuilds the tracks from the decoded boxes and lamp
colours, re-derives every answer with a jitter band, and checks the clock at the frame of every question that names a
time.

## Frame sampling and `min_fps`

Every question stores `params["min_fps"]`, the lowest sampling rate at which it is still answerable: single-frame
questions (last-frame looks) need 1 fps; questions at a named clock time need 1 fps for whole seconds and 2 fps for half
seconds, or `fps_for(gaps)` = ceil(1 / smallest gap in seconds to an event) if larger, at most 24; clock readings and
every `cf_*` question need 4. All L2-L5 questions need the motion (`requires_motion = True`).

## Determinism

All randomness comes from the `random.Random` passed to `sim.build` and `questions.build`. Clip length is a whole number
of frames and stored parameters are rounded. Caches are keyed by the whole scene (`render._static` by the scene JSON).
The same seed gives byte-identical JSON across `PYTHONHASHSEED` values and identical raw frames
(`test_traffic_light_queue.py`).

## Files and commands

```
sim.py                              scenes, fixed-step simulation, consistency check (sim.check)
questions.py                        the 10-question ladder
answers.py                          answers from the drawn frames, separate what-if integrator
render.py                           PIL renderer (1280x720, 24 fps)
frame_check.py                      decodes MP4s, re-derives answers from pixels
preview.py                          renders a preview set spread over the scenarios
stats.py                            diversity / balance report
run.py                              CLI (shared pipeline in common/video5.py)
test_traffic_light_queue.py         determinism, frames, ladder, tamper, clock only text, cars inside frame

python3 run.py generate --n 20 --seed 0 --output-dir out
python3 run.py validate --dataset-dir out
python3 preview.py [--per 10] [--total 50]
python3 frame_check.py --dataset-dir ../output_videos/traffic_light_queue_preview
python3 stats.py out/annotations.jsonl
python3 test_traffic_light_queue.py
sbatch generate_all_datasets.sbatch   # entry "traffic_light_queue:run.py::none:none"
```

## Results

- 5000-scene dry run (`--no-render`): 5000 / 5000 unique scene signatures; 0 warnings. Validation of all 5000 scenes
  (50,000 questions): 0 mismatches.
- Scenario mix: crossing 19.3 %, mixed 20.2 %, single 20.7 %, two_lanes 19.5 %, two_lights 20.3 %.
- Answer letters: A 24.60 %, B 25.03 %, C 24.98 %, D 25.39 %; 8744 distinct (question, answer) pairs.
- Questions per level: L1 8678, L2 12474, L3 12304, L4 9167, L5 7377.
- `min_fps`: <=1 29.4 %, <=2 20.8 %, <=4 49.8 %, >4 one question. `requires_motion` on L2-L5: 100 %.
- Tamper: 3 answers corrupted in each of 100 scenes -> exactly those 300 questions flagged (plus each tampered scene
  flagged as differing from its seed rebuild).
- Determinism: regenerating 50 seeds under two `PYTHONHASHSEED` values gives byte-identical output, identical to the dry
  run.
- Preview: 50 clips in `output_videos/traffic_light_queue_preview` (10 per scenario).
- Frame check of the 50 previews: 10328 frames, clock template match 10328/10328, every lamp colour and vehicle
  front (max error 3.05 px) agrees with the drawn state, 500 questions re-derived from the decoded frames, 0 problems.
