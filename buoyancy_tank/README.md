# buoyancy_tank

Video-QA sub-benchmark: **Buoyancy (shortlist B13)**, from `ideas/New Problem Types Shortlist.html`. It runs on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the 10-question ladder (`LADDER = "ten"` in `common/ladder5.py`).

## What each clip shows

A side view of a water tank (2-D, everything per unit depth, lengths in cm). 3-6 solid objects with hidden densities sit in or above the water. Anything lighter than water floats, displacing exactly (density x its own area) of water; anything heavier rests on the floor. The water level is (water area + displaced area) / tank width, so every object that goes in raises it.

Motion is one coupled ODE for all scenarios: gravity, buoyancy from the current submerged area, linear water drag, and the water level re-solved from the displaced area on every 1/240 s sub-step (Newton step). The renderer samples every 10th sub-step (24 fps).

### Scenarios (`scene["scenario"]`, picked uniformly per seed)

| scenario | setup | events questions use |
|---|---|---|
| `drop` | objects hang on strings from a rail above the water; strings vanish one at a time | let go, touches the water, comes to rest on the floor |
| `rise` | objects strapped to the floor of a deep tank; straps vanish one at a time | let go, pokes out above the surface |
| `fill` | objects stand in a shallow tank; a tap pours for 2.75-5.8 s | floaters lift off the floor; tap turned off |
| `drain` | objects float or rest in a deep tank; a drain pipe runs for 2.75-5.8 s | floaters come to rest on the floor; drain stops |

### Random knobs

- Objects: 3-6; six shapes (square, wide block, tall block, ball, hexagon, triangle) with random sizes and aspect ratios; 8 colours (unique within a clip); 4 patterns (plain, stripes, spots, checks); density 0.2-0.8 for floaters (40% near 1/4, 1/2, 3/4), 1.5-3.5 for sinkers; at least two floaters.
- Layout: random left-to-right order and gaps; random release order and gaps (0.5-1.4 s); random hanging heights.
- Tank: width 70-110 cm, height 46-62 cm, starting water level, tap/drain start, duration and rate.
- Look and framing: px/cm scale (about 7.4-12.4), tank position, floor height, 7 backgrounds, 5 wall colours, 5 back-wall tints, 4 water tints and opacities, grid spacing (5/6/8/10 cm, always >= 55 px), solid or dashed grid lines, wall thickness.

### On screen

The only text is a clock (`s.s s`, top-left box, never overlapping the tank). Objects are named in questions by colour + pattern + shape ("the red striped ball"); colours are unique within a clip. Faint horizontal lines on the back wall (no numbers) serve as the height scale; a dashed mark outside the right wall shows the starting water level. Objects are >= 26 px, lines >= 3 px. Clips are 1280x720, 24 fps, 4-8 s (never more than 10 s); everything has been still for >= 1 s before the end.

## The 10-question ladder

`L5.ten_counts` draws per-level counts (L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3, total 10). Every template is first enumerated into a candidate pool per level (only candidates with a clear visible margin get in); the build picks different templates first, then fills with other parameters. No two questions in a clip share derivation_key + params. Every question has exactly four distinct options.

- **L1 observation**: number of objects; number floating / resting on the floor at the end; where an object ends up (4 states); shape of the X-coloured object; pattern on an object.
- **L2 relational**: which of four objects does an event first; which of two does an event first and by more or less than 1 s; which of two ends with the larger share of its height under water; whose top ends higher.
- **L3 measurement**: clock reading when an event happens (options 1 s apart); share of a block's height under water (about 1/4, 1/2, 3/4); which back-wall line an object's top / the water surface is closest to (start or end).
- **L4 integration**: highest top / lowest top / highest bottom at the end; least dense material (from how deep things float); objects on the floor at the moment of an event or when the tap/drain stops; which object is the k-th to do an event; how many line spacings the water moves overall.
- **L5 counterfactual** (re-simulated): if an object had never been in the tank; if the water had started one line spacing higher/lower; if the tap/drain had stopped 1 s earlier/later (fill/drain); asking the final water line or the number of objects on the floor; if a floating object were 1/2, 2, 3 or 4 times as heavy, where it would end up. Every counterfactual is re-run through the ODE on the changed scene and must settle onto the closed-form state before it is used.

Counterfactuals are asked only when the answer follows from what the video shows: a density is known exactly when the object is seen floating at rest, otherwise only a lower bound (from the deepest water it was seen resting in); a counterfactual is skipped if a bounded object might float under it.

### Margins (sim.check and the pools)

- Floaters end >= 15 px above the floor and >= 12 px out of the water; grounded floaters need >= 0.6 cm more water to lift; a grounded object's top is >= 12 px from the surface.
- Compared heights differ by >= 15 px; shares under water by >= 0.12; line answers are within 0.2 spacing of the line (>= 16 px from the rounding boundary); densities by >= 0.12.
- Compared events are >= 10 frames (0.42 s) apart; "by more/less than 1 s" needs |gap - 1 s| >= 0.3 s; every event a question uses is >= 0.5 s from both ends of the clip and is a clean, persistent crossing between two drawn frames; "how many on the floor at the moment ..." needs no floor contact change within 0.5 s.

### Frame sampling and `min_fps`

Every question stores `params["min_fps"]`: the lowest of 1, 2, 3, 4, 6, 8, 12, 24 fps that is sure to put a frame inside the smallest window the answer relies on (the gap between compared events, the gap to the nearest floor-contact change, or 0.4 s for clock readings with options 1 s apart). End-state questions use the >= 1 s still hold at the end (1 fps). Every event leaves a persistent result (a string or strap stays gone, objects stay on the floor / afloat / out of the water). On preview data, about 80% of questions are answerable at 1 fps, about 87% at 2 fps and 100% at 4 fps.

## Closed loop

Every question stores its `derivation_key` and `params`. `answers.py` does not read the generator's events, densities or shape formulas. It uses only what the renderer draws: `render.track` (per-frame water level and object bottoms), `render.outline_cm` (the drawn outlines), `render.attached` and `render.flow_on`. Events come from scanning frames; submerged areas from shapely polygon clipping; densities from displaced area / own area at rest; counterfactuals from a force-balance bisection on the water level. `python run.py validate` fails a clip if any re-derived answer disagrees, if rebuilding from the seed gives a different scene or questions, if `sim.check` rejects the scene, if the ladder is malformed (counts, order, repeated key + params, options, `min_fps`), or if the MP4 duration, resolution or motion is off.

## Determinism

`sim.build(rng)` and `questions.build(scene, rng)` use only the seeded `random.Random`; no global random, numpy RNG, time, uuid or set/dict ordering decides anything. The same seed gives byte-identical JSON across processes and `PYTHONHASHSEED` values, and the same frames when rendered twice. `test_buoyancy_tank.py` covers determinism, ladder counts, unique naming, unique scenes, the tamper test, `min_fps`, and hand-checked geometry and a hand-built scene.

## Files and commands

```
sim.py        scenes, ODE, events, checks (pure, seeded)
questions.py  candidate pools + the 10-question ladder
answers.py    independent answer re-derivation from the drawn geometry (validator only)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)
test_buoyancy_tank.py

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/buoyancy_tank
python run.py validate --dataset-dir ../output_videos/buoyancy_tank
python -m pytest test_buoyancy_tank.py
```
