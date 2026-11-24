# robot_archery

Closed-loop 10-question video-QA benchmark on **running-record reasoning**, adapted from Jane Street's *Robot
Archery* puzzle (Dec 2021). Robots wait in a queue and take turns shooting one arrow at a ten-ring target. A shot
strictly closer to the centre than every earlier shot is a **new record**; what happens to the shooter after each
shot depends on the scenario's rule set. The last robot left wins. Questions ask about records, whose record was
beaten, the elimination order, ring readings and times, and what-if replays.

The pipeline is PIL frames piped to ffmpeg (`common/video5.py`), with the 10-question ladder in `common/ladder5.py`
(`LADDER = "ten"`). Clips are 1280x720 at 24 fps and 4-10 s long. Every game finishes inside the clip.

## What a clip shows

- The **target** on one side: ten rings (ring 1 at the centre, ring 10 at the edge), ring numbers 2, 4, 6, 8, 10 on
  the ray straight down from the centre.
- The **robots** on the other side: a queue along the bottom, the shooting spot (a shadow next to the target) and a
  bench at the top. Each robot has a unique colour, a body shape (circle, square, triangle, diamond, hexagon,
  pentagon) and a body pattern (plain, stripes, dots, band).
- Each **turn**: the front robot steps onto the spot (others move up), an arrow flies and lands, and the arrow stays
  on the target as a marker in the shooter's colour and body shape with a black dot at the exact landing point.
- A **new record** moves the black-and-white dashed **record circle** to the new arrow's distance. An **eliminated**
  robot turns pale, closes its eyes and walks to the bench; otherwise the shooter walks to the back of the queue. Walks are L-shaped (across first, then up / down into the slot), so a walking robot never crosses robots standing in the queue or on the bench.

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | rules | robots | readable from the video because |
|---|---|---|---|
| `classic` | record -> back of the queue; anything else -> eliminated | 3-6 | the robot walks to the queue or to the bench |
| `two_lives` | two pips each; a non-record shot costs a pip (hollow) and sends the robot to the back; the last pip lost eliminates it | 3-4 | the pips over every robot's head |
| `stay_on` | a record setter stays on the spot and shoots again; anything else -> eliminated | 3-6 | the same robot stays on the spot and shoots again |
| `big_field` | classic rules, faster pace (0.55-0.68 s turns) | 5-7 | as classic |

## Random knobs (`sim._draw`, `sim._draws`, `sim._layout`, `sim._angles`)

- robot count (3-7 by scenario), queue order, colours (8, unique per clip), body shapes, body patterns, body size
  44-54 px;
- shot distances: Rayleigh-distributed with spread sigma 2.0-3.6 rings, plus a per-clip 15-45 % chance per shot of
  aiming inside the current record; every shot is at least 0.6 ring (>= 16 px) from the record it is judged
  against; landing angle anywhere except the label ray;
- pace: turn length 0.55-0.85 s, intro 0.5-0.9 s, outro 0.7-1.3 s;
- framing: target side (left / right), centre position, 28-31 px per ring, queue height, queue / bench spacing;
- look: target palette (classic, mono, pastel, sand), background colour (6) and pattern (plain, grass, tiles,
  gradient).

`sim.build` redraws until the scene passes `sim.check` and can fill the ladder.

## On-screen text

- **The clock**: a box at the top-left with the time in seconds to 0.1 s (white 44 px on a dark box), the value is
  `frame * 10 // 24` tenths. The target, robots, queue and bench never reach it (`sim.check`).
- **Ring numbers** 2, 4, 6, 8, 10: 34 px bold digits (>= 24 px tall), black on a white tag, centred in their ring on
  the ray straight down from the centre; stored in `scene["labels"]` (`sim.labels`), fixed for the clip.

Why the ring numbers are needed: ring questions ask "in which ring did X land" and "in which ring is the record"
on a 1-10 scale. Counting up to ten thin rings is error-prone at reduced resolution; numbering every second ring
keeps the text to five short numbers. They are a measuring aid only: they never name a robot, a shot, an event or
an answer, and no arrow ever lands on the label ray (arrows are kept clear of every label box). Robots are named by
colour, so they need no names. There are no titles, captions, legends, scores, "record" / "out" banners, queue
headings or rule texts; the old renderer's title, record readout, turn counter, robot names, distance readouts
and "NEW RECORD" / "OUT" verdicts are gone.

## The 10-question ladder (`questions.py`)

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total 10), qids q01..q10 ordered by level, no
repeated derivation_key + params. A shot is named "the red robot's second shot" (or "only shot").

| level | templates (derivation_key) |
|---|---|
| L1 observation | `n_arrows` (on the target at the end), `shots_by` a robot, `first_shooter`, `winner` |
| L2 relational | `record_beaten` (whose record a record shot beat), `shooter_after` a shot, `out_after` (next robot eliminated), `holder_at_out` (record holder when X goes out), `closest_of` four shots |
| L3 measurement | `ring_of` a shot, `record_gain` (rings the record moved in), `out_time` / `last_record_time` (clock readings), `arrows_at` and `record_ring_at` when the clock shows X |
| L4 integration | `elim_order`, `records_by` a robot, `most_records`, `n_records`, `arrows_at_out` (arrows shot when X goes out), `full_pips_at` (two_lives) |
| L5 counterfactual | `cf_shot_winner` / `cf_shot_arrows` (one shot lands in ring K instead), `cf_swap_winner` (two robots swap places in the starting queue), `cf_rule_winner` / `cf_rule_first_out` (another rule set) |

Every what-if is replayed with every turn's arrow landing where it did, and is asked only when the replay finishes
within the arrows seen in the clip; "landed in ring K" is offered only when no other arrow is in ring K, so every
spot in that ring gives the same game.

Margins: ring readings sit >= 0.2 ring inside their ring; records and compared shots differ by >= 0.6 ring
(>= 16 px); record gains are within 0.3 of a whole ring; clock options are >= 0.6 s apart; "when the clock shows X"
questions keep every landing >= 0.25 s outside the frames showing X. Distractors are other robots of the clip, other
orders of the same robots (or with the winner swapped in), nearby counts and rings, and other clock times of
neighbouring landings.

## Closed loop

Every question stores its `derivation_key` and `params`. `answers.py` never reads the generator's game log, draws,
angles or landing times: it builds a per-frame track of what the renderer draws (`render.state`: landed arrow
markers with colour and centre, each robot's position, out flag and pips), reads distances from the marker centres
(px / px per ring), the starting queue from the robots' frame-0 positions and the landing / elimination frames from
the first frames that show them, and replays the game (and every what-if) with its own deque-based replay.
`python run.py validate` fails a clip if that disagrees with the stored answer, if rebuilding from the seed gives a
different scene or questions, if `sim.check` finds anything borderline, if the ladder is malformed, or if the MP4's
duration, resolution or motion is off.

`frame_check.py` decodes the MP4s and re-reads the clock (template match on every frame), the ring numbers
(template match against every text "1".."10", plus "never covered" on sampled frames), the arrow markers (colour
blobs inside the target, centre within 4.5 px, below the 0.2-ring answer margin, appearing on the landing frame), the robots' out shading and pips,
then re-derives every answer from that decoded track.

## Determinism

Sample `i` uses `random.Random(seed + i)`; the scene and the questions come from the same rng stream. No global
random, numpy RNG, time, uuid, or set / dict iteration order decides anything. Same seed -> byte-identical scene,
questions and answers in separate processes with different `PYTHONHASHSEED`; rendering a seed twice gives identical
raw frames (tested); regenerating a range of seeds reproduces `annotations.jsonl` byte for byte.

## Frame sampling and `min_fps`

The landing is the only event; its result (arrow marker, record circle, pale robot walking to the bench, hollow pip)
stays on screen, so nothing is a one-frame flash. Landings are 0.55-0.85 s apart, at least 0.6 s inside the clip and
0.25-0.75 of the way through a frame interval. Objects are >= 20 px (markers 24 px, pips 20 px, robots 44-54 px),
lines >= 3 px. Each question stores `params["min_fps"]`: 1 for what the last frame shows, `ceil(1 / turn length)`
(usually 2) for questions that need to see who shot when, `ceil(1 / gap)` for "when the clock shows X" questions,
and 4 for clock-time questions (options 0.6 s apart).

## Files and commands

```
sim.py            scene sampler, rules (sim.play), consistency check
questions.py      the 10-question ladder
answers.py        independent re-derivation from the drawn per-frame track
render.py         PIL renderer (1280x720, 24 fps) + per-frame state
run.py            CLI: generate / validate (shared pipeline in common/video5.py)
frame_check.py    decoded-MP4 checks + answers from the decoded track
preview.py        renders the preview set (>= 12 clips per scenario, 50 in total)
stats.py          diversity / balance report
test_robot_archery.py   determinism, ladder, uniqueness, tamper, labels, hand-checked games

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/robot_archery
python run.py validate --dataset-dir ../output_videos/robot_archery
python run.py generate --n 5000 --no-render --output-dir /tmp/ra && python run.py validate --skip-video --dataset-dir /tmp/ra
python stats.py /tmp/ra/annotations.jsonl
python preview.py && python frame_check.py
python test_robot_archery.py
```
