# turntable_voxels

Video-QA sub-benchmark: **Turntable voxels**. It is the temporal version of the image benchmark's
`cube_structure + orthographic` categories. A structure of unit cubes stands on a turntable. It holds still
in a START VIEW where some cubes are hidden, then turns so every side can be seen. In some scenarios it stops
partway, reverses, has cubes dropped onto it, or has cubes lifted off. The questions ask about counting,
hidden cubes, visibility, front views, which side faces you and when things happen.

The dataset runs on the shared PIL + ffmpeg pipeline (`common/video5.py`). It uses the 10-question ladder
(`LADDER = "ten"`). Clips are 1280x720, 24 fps, 5.7-10 s.

## What each clip shows

- **Structure.** A GX x GY grid (3-5 squares each way). Each square holds a stack of 0-5 cubes, with no
  overhangs. There are 6-36 cubes and at least 4 stacks. 1-3 cubes are painted a strong colour (red, yellow,
  green, purple, orange, blue, pink; never two the same). All other cubes share one pale base colour.
- **Camera.** Orthographic, seen from the front, 22-40 degrees above. Every still view is square on (a
  multiple of a quarter turn), so only top and front faces show.
- **Turntable.** A disc with a visible rim and a grid of squares. A notch in a contrasting colour (black or
  white) sits on the rim. It starts at the front and turns with the disc, so you can count turns.
- **Motion.** Every turn eases in and out (smoothstep). Its average speed is 50-100 deg/s (peak <= 150 deg/s).
  The clip always starts still (the start view, 0.9-2.1 s) and ends still (the end view, 1.0-1.8 s).
  Positive angles are counter-clockwise seen from above: the near edge moves right on screen.

### Scenarios (`scene["scenario"]`, about 20% each)

| scenario | what happens |
|---|---|
| `spin` | one smooth turn of 1, 1 1/4, 1 1/2 or 1 3/4 turns, then it stands still |
| `stops` | 2-3 turns of 1/4 to 1 turn each, all the same way, standing still 1 or 2 s between them (total >= 1 turn) |
| `reverse` | 1-1 1/2 turns one way, then 1/4-3/4 turn back the other way, with no pause |
| `build` | one smooth turn while 2-4 base-colour cubes drop from 2 cubes above onto stacks or empty squares |
| `remove` | one smooth turn while 2-3 top cubes rise 2 cubes and shrink away (sometimes a coloured one) |

The structure is always seen from every side during the clip.

### Random knobs

Scenario; grid size (9 shapes, 3x3 to 5x5); layout style (random, pyramid, stairs, towers, terrace, ring) with
fill and maximum height; the coloured cubes' count, colours and positions (about 45% of them are picked among
cubes hidden at the start); elevation (22-40 degrees); cube size (54-160 px) and framing (position in the
frame); base colour (6); background (9 colours x 4 styles: plain, vertical gradient, vignette, floor);
turntable colour (7); turning direction, amounts, speeds and timing; stop lengths; event count, squares and
timing; clip length.

## On-screen text

The clock in the top-right box (seconds to 0.1 s, 42 px bold white on black, about 30 px cap height) is the
**only text** in the frame. The structure and turntable never come within 18 px of its box (`sim.check`
tests this at every 0.2 s), and the frame check confirms that nothing else is drawn next to it.

No labels are needed. Coloured cubes are named by colour, and the colours are unique in a clip and never
close to the base colour (for example, no yellow or orange on sand cubes). Other cubes are only counted.
Grid squares, stacks and turns are counted from the drawing. The notch replaces any angle text. Questions
say "the side that was on your left at the start" instead of naming sides with letters.

## The 10-question ladder

`L5.ten_counts(rng)` draws the per-level counts (L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3; total 10).
Templates within a level are picked round-robin in random order. No `derivation_key` + params pair repeats.
Every question has 4 options.

- **L1 observation**: squares in the grid; the tallest stack at the start; which of four colours is on a
  cube; which way the near edge moves whenever it turns (always right / always left / right then left /
  left then right); how many cubes drop or rise away
- **L2 relational**: whether a coloured cube is visible at the start and whether it is the top of its stack
  (4 combinations); cubes on top of / under a coloured cube; which coloured cubes you can see in the start /
  end view; which of three coloured cubes sits highest; which side faces you at the first stop
- **L3 measurement**: squares with cubes; cubes visible in the start / end view / first stop; total cubes;
  clock reading when it starts turning, first stops, changes direction, or when the i-th cube lands / starts
  to rise; how long the first stop lasts; how far and which way it turns before it first stops; total
  quarter turns; how many layers higher one coloured cube is than another
- **L4 integration**: cubes completely hidden at the start / at the end; which side faces you at the end;
  the front view (tallest stack per column, left to right) at the start and of the side facing you at the
  end; total cubes and tallest stack at the end (build / remove)
- **L5 counterfactual** (all re-simulated): cubes after adding one on every stack; the front view and the
  hidden count without a coloured cube's stack; which side would face you if every turn went the other way;
  cubes visible if the last turn had stopped a quarter turn sooner; cubes visible in the end view if no cubes
  had been dropped / lifted

About 80% of L2-L5 questions need the motion (`requires_motion=True`). Start-view questions, front views
and the stack-removal front view are the static ones.

## Margins and frame sampling

- **Visibility is never borderline.** In every still view (start, stops, end) each cube is either clearly
  visible, with >= max(0.15 S^2, 14 S) pixels (a band >= 14 px deep across a face), or not visible at all.
  Re-simulated views that a counterfactual asks about meet the same rule, or the question is not asked.
- **Clock answers** ("about which clock reading", whole seconds): every such moment is within 0.13 s of a
  whole second, so it is >= 0.37 s from a rounding boundary. Stop lengths are 1 or 2 s +- 0.1 s.
- **Event spacing.** Events start >= 0.9 s after the clip starts and end >= 1.0 s before it ends. Drops and
  lifts happen >= 0.4 s inside a turn and >= 0.15 s (3.6 frames) apart. Landing / lift-off moments are >= 0.76 s apart.
- **Visible events.** A dropped or lifted cube is clearly visible (>= the visibility threshold) in at least 3
  of 5 frames spread over its 0.6 s motion. Its result persists to the end. Every stack top and every coloured
  cube is clearly seen at some sampled moment of the clip.
- **Sizes.** Cubes are >= 54 px, outlines and grid lines are 3 px, and the notch is >= 22 px wide.
- `params["min_fps"]` is the rate needed to see what a question relies on. Start view: 1 / start-hold
  length. End view: 1 / end-hold length. Seeing every side: peak speed / 90 deg. Directions and
  quarter-turn counts: peak speed / 120 deg (and 2 / length of the turn back). Clock answers: 1 / margin to
  the rounding boundary. Event counts: 1 / smallest gap between events. Over 5000 scenes, 36.7% of
  questions can be answered at 1 fps, 94.4% at 2 fps and 100% at 4 fps (max 2.7).

## Closed loop

Every question stores `derivation_key` + `params`. `answers.py` does not import `sim.py`. It works from what
is drawn:

- it replays the turntable angle frame by frame at 24 fps, and reads the start, stops, reversal, directions
  and quarter-turn counts off that track;
- it rebuilds the stack heights at any frame from the drawn drops and lifts, and reads event moments off
  the per-frame cube positions;
- it decides visibility by **ray casting** through the heightmap. The generator uses an ID-buffer render
  (pixel counts per cube). In a square-on view a ray from a cube face towards the camera stays in its
  column and climbs over the rows in front, so each top and front face is sampled at 16 points. A cube is
  visible if any ray clears every stack it passes.

Counterfactuals re-run the same ray casting on the changed structure or angle.

`sim.check` rejects scenes with:

- a partly visible cube in a still view;
- no hidden cube at the start;
- a stack top or coloured cube that is never seen;
- a hidden drop / lift;
- events outside a turn or too close together;
- key moments near a rounding boundary or the clip ends;
- turns that are too fast / slow or don't end on a quarter turn;
- the structure leaving the frame or nearing the clock;
- cubes < 40 px;
- a clip longer than 10 s.

`sim.build` retries until the scene passes and every level has >= 3 candidates from >= 2 templates. Over
5000 seeds this takes 2.08 attempts per scene on average (max 13; 51.9% of attempts are rejected). No seed
fails, and there are no warnings or exceptions.

`python run.py framecheck` decodes the MP4s. It:

- reads the turntable angle from the notch in every frame (an opening filter removes cube outlines) and
  compares it with the programme at 4 fps;
- derives directions, quarter counts, the first turn and the facing side from the notch track;
- checks that the notch's screen x moves the way the answer says;
- checks each cube's colour in the middle of its visible band in every still view, using a dense band
  search that must agree with `answers.py`;
- checks that coloured-cube blobs are present exactly when ray casting says they are visible;
- checks that the moving cube of every drop / lift shows its colour;
- checks the clock crop at every key moment and still view;
- checks that nothing is drawn beside the clock.

## Determinism

All randomness comes from the `random.Random(seed)` that is passed in. There is no global RNG, time, uuid or
set / dict order in any decision (sets are only tested for membership or sorted). The same seed gives
byte-identical JSON across processes with different `PYTHONHASHSEED` values and identical raw frames
(`python run.py framehash --seed S`).

## Files and commands

```
sim.py        scene generation, motion, ID-buffer visibility, check (pure, seeded)
questions.py  the 10-question ladder (candidates + selection, min_fps)
answers.py    independent re-derivation: per-frame replay + ray casting (validator only)
render.py     PIL renderer (1280x720, 24 fps)
checks.py     stats / tamper / framecheck / framehash
run.py        CLI (generate / validate from common/video5.py, plus the checks)
test_turntable_voxels.py

python run.py generate --n 50 --seed 46 --output-dir ../output_videos/turntable_voxels_preview
python run.py validate --dataset-dir DIR [--skip-video]
python run.py stats --n 5000 [--workers 8]
python run.py tamper --dataset-dir DIR        # corrupts 3 answers -> exactly 3 flagged
python run.py framecheck --dataset-dir DIR
python run.py framehash --seed S
python -m pytest test_turntable_voxels.py
```

## Diversity (5000 dry-run seeds, `python run.py stats --n 5000`)

- 5000/5000 unique scene signatures.
- Scenario mix: build 1012, remove 1014, reverse 966, spin 1035, stops 973.
- Ranges:
  - grids 3x3-5x5 (9 shapes);
  - 6-36 cubes (31 distinct counts);
  - 1-22 hidden at the start;
  - cube size 54-160 px;
  - elevation 22-40 degrees;
  - 5.7-10 s;
  - peak speed 75-150 deg/s;
  - 4-8 quarter turns in total.
- Layout: towers 1334, terrace 1038, random 733, ring 713, pyramid 596, stairs 586.
- 63 different sets of coloured-cube colours.
- 3596 distinct (question, answer) pairs out of 50000 questions.
- Answer letters are 25.0-25.1% each.
- 0 ladder errors, 0 check failures, 0 answer mismatches.
