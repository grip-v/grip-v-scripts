# gear_train

Video-QA sub-benchmark for **mechanical causal-chain reasoning with gear
trains**: 3-8 meshed spur gears driven from one driver gear, plus sometimes a
loose gear that touches nothing. Direction and speed pass through the train,
and the motion changes in a different way in each scenario (reversals,
disengaging gears, a brake, a motor speed change). Rendered with PIL and
piped to ffmpeg through the shared pipeline (`common/video5.py`). Each clip
gets the 10-question ladder (`LADDER = "ten"` in `questions.py`).

## What each clip shows

- **Gears.** Real involute-profile spur gears (20 degree pressure angle, a
  little backlash), 10-36 teeth, with one module (tooth size, 8-14 px) per
  clip so every pair can mesh. A gear's pitch radius is module x teeth / 2.
  Meshing gears sit exactly r_a + r_b apart and their teeth are phase-aligned
  along the mesh tree from the driver, so the teeth interlock and keep
  interlocking (`sim.check` checks the tooth phase every third frame).
- **Physics** (kept from the Manim version, commit e2e24ee): meshing gears turn
  in opposite directions, and |omega| x teeth = K is the same for every gear in
  the train. So a gear with half the teeth turns twice as fast, whatever idlers
  sit between them.
- **No false meshes.** Gears that do not mesh always keep a tip-to-tip gap of
  at least 2.8 modules (>= 22 px). The loose gear keeps at least 3.2 modules.
  So nothing looks meshed when it is not, and no gear overlaps another.
- **Readable motion.** Every gear carries a white dot with a black rim (>= 21
  px across) on its face, so turn counts and directions can be read even from
  sampled frames. The teeth and face pattern turn with the gear. The fastest
  gear turns at <= 2.0 rad/s at the start and <= 2.45 rad/s at any time, which
  is <= 35 degrees between frames at 4 fps. The slowest train gear turns at
  >= 0.3 rad/s at the start (>= 0.2 rad/s after a slow-down).
- **The driver** has a large black hub with a white rim. Every other gear has a
  small grey axle.
- **Size and duration.** 1280x720, 24 fps, 5-10 s. Every gear's body
  (root circle) is >= 72 px across. Lines (outlines, spokes, brake arm) are
  >= 3 px.

### Scenarios (`scene["scenario"]`, about 20% each)

| scenario | what happens | how it is visible |
|---|---|---|
| `steady` | the driver turns at a constant speed for the whole clip | constant motion |
| `reversal` | the driver reverses once or twice (0.6 s ramp through zero, reversals >= 2.4 s apart); the whole train reverses with it | every marker slows, stops and turns back |
| `disengage` | one non-driver gear slides straight out of mesh; it and every gear it drives spin down, stand completely still for 1.85-3.15 s, spin up and slide back in. The stop is a whole number of tooth periods, so the teeth re-mesh exactly. Gears upstream keep turning | a >= 24 px gap opens between the two gears' teeth |
| `brake` | a grey brake block presses on one gear's teeth; the whole connected train (driver included, since meshed gears are rigidly linked) slows to a stop over 0.35 s, stands still for 1.85-3.15 s, and restarts when the block lifts | the block slides in, sits on the teeth while everything is still, then slides away |
| `speed_change` | the driver's motor changes speed (x2, x3, x1/2 or x1/3, 0.4 s ramp); every gear changes by the same factor | all markers speed up or slow down together |

The old **skip/fault** event (a meshed gear slipping a few teeth while its
centre stayed put) is gone, because its teeth passed through the
neighbouring gear's teeth. The physically clean versions of "part of the train
loses drive" are `disengage` (visible gap) and `brake` (visible cause). The
stall's 4-6 s stop is now 1.85-3.15 s so that the clip fits the 10 s limit.

Every event starts >= 1.5 s after the clip starts and ends >= 1.2 s before
it ends, so "at the start" and "at the end" always mean steady motion.
Event moments and stop lengths lie within 0.15 s of a whole second, so
"about which second" answers are >= 0.35 s from a rounding boundary.

### Random knobs

Scenario, train size 3-8 (4-8 for disengage/brake), loose gear (~24%),
topology (chain / hub with 2-3 branches / random tree with <= 3 children at
the driver and <= 2 elsewhere), bend angle of chains (25-75 degrees), branch
spread (100-200 degrees), teeth per gear (free 10-36, or a "family" of
multiples of 4, 5 or 6 that gives clean ratios), module 8-14 px, framing
(fill 62-98% of the frame, random offset), the driver's direction and speed
(fastest gear 0.7-2.0 rad/s), colour per gear (9 colour names, never repeated,
+-14 shade jitter), face pattern per gear (plain, 3-6 holes, 3-6 spokes, ring),
outline width, 11 background themes (6 dark, 5 light) x 7 textures (plain,
gradient, grid, dots, pegboard, hatch, vignette), event gear, event timing,
stop length, number of reversals, speed factor, brake position (24 angles
around the gear, wherever the block stays clear of the other gears), clip
length.

## On-screen text

The clock in the top-right box (seconds to 0.1 s, 42 px bold white on black,
about 30 px cap height) is the **only text** in the frame. No gear stays
within 22 px of its box. Gears are named in questions by colour alone ("the
red gear"). The 9 colour names are unique within a clip, so no label is
needed. Tooth counts are never printed: the teeth-count question is only
asked for gears with <= 18 teeth at a module >= 9 px, so the teeth can be
counted off the drawing. The driver is named by its black hub.

## The 10-question ladder

`L5.ten_counts(rng)` draws the per-level counts (L1 1-3, L2 2-3, L3 2-3, L4 1-3,
L5 1-3; total 10). Templates per level are picked round-robin in random order,
and no `derivation_key` + params pair repeats. Every question has 4 options.

- **L1 observation**: number of gears; teeth on a gear; which way two gears turn at the
  start (4 combinations); which gear never turns; largest / smallest gear;
  which gear slides away / which gear the brake presses; how many times the
  driver changes direction; what the speed change does
- **L2 relational**: which gear turns the same way as X; fastest / slowest of
  three; which gear meshes with X; which gear keeps turning while X is stopped
  ("none of them" under the brake); which gear X slides away from; how X turns at
  the end compared with the start (faster / slower / same, same / opposite
  direction)
- **L3 measurement** (the clock and the marker): full turns of X's dot between two
  clock readings; turns of B per turn of A (a ratio like 3/2); seconds per turn;
  clock reading of the reversal / speed change / slide-out / brake contact;
  seconds X stands still; speed factor after the change
- **L4 integration**: which way two gears turn at the end (after 1-2 reversals);
  which other gears stop with the disengaged gear; how many gears stand still at
  clock T; order three gears by speed; which of three gears makes the most turns
  over the whole clip (stops included); which gear turns X directly; how many gears
  turn clockwise at the start / end
- **L5 counterfactual** (re-simulated): directions if the driver had turned the
  other way; which gears would not turn if X were taken out; how the loose gear
  would turn if it meshed with X (direction, faster or slower than X); which gears
  would stop if Y had slid out instead; how many gears would stop if the brake
  pressed Y instead; directions at the end without the reversal; full turns
  without the event (no slide-out / no brake / no speed change); how fast Y
  would turn if X had twice the teeth (with the motor turning the driver as
  before)

About 85% of L2-L5 questions need the motion (`requires_motion=True`). The
rest are structural: meshes-with, drives-directly, remove-a-gear, double the teeth.

## Margins and frame sampling

- Speed comparisons need >= 15% between the gears compared. Sizes need >= 10%.
  "Most turns" needs >= 10% and >= 0.3 turn of lead.
- Turn counts are asked only when the true count is >= 0.2 turn from a whole number.
  Clock answers are >= 0.35 s from a rounding boundary. "How many still at T" uses
  T >= 0.3 s from any start / stop. Reversal moments are avoided by 0.6 s.
- Every event leaves a persistent result: stops last >= 1.85 s, the disengage
  gap stays open the whole stop, the brake block sits on the teeth the whole stop,
  and reversals and speed changes persist to the end.
- `params["min_fps"]` is the sampling rate needed to see the answer: for motion it is
  max(|omega| / 120 degrees, 2 / window length), so a marker never moves more
  than a third of a turn between samples and the window gets at least two
  samples. Timing and counting questions use 1 / (margin to the nearest
  boundary). Static questions use 0.5. Over 5000 scenes, 77.9% of questions can be answered at 1 fps, 90.2% at 2 fps and 100% at 4 fps (max 4.0).

## Closed loop

Every question stores `derivation_key` + `params`. `answers.py` never reads
`scene["log"]` or sim's closed-form motion. It rebuilds the clip from what is
drawn: gear centres, tooth counts, module, the slide path of the moving gears,
the brake block's path and the driver's motor programme. It then integrates
the motion step by step (8 sub-steps per frame). At every step it decides from
the geometry which gears touch (their tip circles overlap) and propagates
from the driver. A group cut off from the driver coasts down on the spin
profile, and a group the brake block touches stops. Counterfactuals re-run
the same integration with a gear removed, the driver flipped, the event
removed, the loose gear meshed, a different gear cut or braked, or the teeth
doubled. `sim.check` rejects scenes with:

- gears that look meshed but are not, or meshing gears not at the exact distance;
- teeth out of phase while engaged (including after re-meshing);
- a stop that is not a whole number of teeth;
- a disengage gap < 24 px;
- a brake block near another gear;
- gears near the clock;
- speeds outside the limits;
- events too close to each other or to the clip ends;
- event times near a rounding boundary;
- a log that differs from a replay.

`sim.build` retries until the scene passes and every level has >= 3 candidate
questions. Over 5000 seeds this takes 8.0 attempts per scene on average (max 108; 87.6% of attempts are rejected, mostly layouts that do not fit the frame with clear gaps). No seed fails, and there are no warnings or exceptions.

## Determinism

All randomness comes from the `random.Random(seed)` that is passed in. There is
no global RNG, time, uuid or set order in any decision. Sprites and backgrounds
are cached by scene content. The same seed gives byte-identical JSON across
processes with different `PYTHONHASHSEED` values, and identical raw frames.

## Files and commands

```
sim.py        scene generation, closed-form gear motion, check (pure, seeded)
questions.py  the 10-question ladder (candidates + selection, min_fps)
answers.py    independent step-by-step re-simulation from the drawn geometry (validator only)
render.py     PIL renderer (involute sprites, affine-rotated per frame; 1280x720, 24 fps)
checks.py     stats / tamper / framecheck / framehash
run.py        CLI (generate / validate from common/video5.py, plus the checks)
test_gear_train.py

python run.py generate --n 50 --seed 200 --output-dir ../output_videos/gear_train_preview
python run.py validate --dataset-dir ../output_videos/gear_train_preview [--skip-video]
python run.py stats --n 5000
python run.py tamper --dataset-dir DIR       # corrupts 3 answers -> exactly 3 flagged
python run.py framecheck --dataset-dir DIR   # marker angle of every gear at 4 fps vs theta(t), gear colours,
                                             # teeth counted on the pitch circle, sizes by pixel area, turn
                                             # counts / directions / stops from tracked markers, event times
                                             # read off the frames, disengage gap, brake block, clock crops
python -m pytest test_gear_train.py
```

## Diversity (5000 dry-run seeds, `python run.py stats --n 5000`)

- 5000/5000 unique scene signatures.
- Scenario mix: brake 1012, disengage 966, reversal 973, speed_change 1014,
  steady 1035.
- Ranges: 3-9 gears (3-8 in the train), module 8-14 px (61 values), 10-36
  teeth, fastest gear 0.7-2.0 rad/s, 5-10 s, fill 62-98%.
- Topology: chain 1513, branching 1775, tree 1712.
- 9628 distinct (question, answer) pairs out of 50000 questions.
- Answer letters are 24.9-25.1% each.
- 0 ladder errors, 0 check failures, 0 answer mismatches.
