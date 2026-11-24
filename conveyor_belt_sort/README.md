# conveyor_belt_sort

Video-QA sub-benchmark for **rule induction from partial observation**:
objects come out of a tunnel onto a moving belt, pass a sorting gate at the
end of the belt and slide down a chute into a tray ("bin"), where each one
glides to the farthest free slot and stops. The machine follows a hidden
rule that has to be worked out from the objects it has already sorted; the
clip ends while the last one or two objects are still riding the belt.
v2 moves the dataset to the 10-question ladder (`LADDER = "ten"` in
`questions.py`), adds five scenarios, removes the object numbers and the
caption, and keeps clips to 10 s.

## What each clip shows

- A tunnel at one edge of the frame (left or right), a belt 220-560 px long
  with cleats moving at the belt speed, the gate (a bar across the belt's end)
  and two or three chutes into horizontal trays with slot ticks; each tray
  fills from its far end, so objects never pass each other.
- 5-9 objects (4-7 sorted before the end, 1-2 still on the belt), each a
  circle, square, triangle or star, red / blue / green / yellow / purple /
  orange, plain, striped or "dotted" (one round spot in the middle, white, dark
  on yellow); one, two or
  three sizes (radius 22-25 px, x1.25, x1.5). Colour + pattern + shape is
  unique in each clip.
- Belt speed 144-240 px/s (9 steps); every path is a whole number of frames
  long, so every event (out of the tunnel, at the gate, at rest) falls on a
  frame. Seven background palettes, flat / gradient / vignette, three belt and
  three metal shades per palette.

### Scenarios (`scene["scenario"]`, 20 % each)

| scenario | rule | bins |
|---|---|---|
| `size` | small objects go to one bin, large to the other | upper / lower |
| `colour` | each of 2-3 colours always goes to the same bin | upper / lower |
| `shape` | each of 2-3 shapes always goes to the same bin | upper / lower |
| `three_way` | one property with three values (colour, shape or size) picks one of three bins | upper / middle / lower |
| `alternate` | the gate sends objects to the two bins in turn, whatever they look like | upper / lower |

In every clip each other property (colour, shape, pattern, size) is
contradicted by the objects already sorted (two objects that share it end up
in different bins), and so is taking turns (except in `alternate`): exactly
one simple rule fits what is shown, and every value a pending object has was
already sorted. `sim.check` and `answers.py` both test this.

### Random knobs

Scenario; which property and which mapping; number of sorted and pending
objects; colour / shape / pattern pools (1-4 each) and sizes; base radius;
belt speed; belt length; tunnel side; belt height (300-420 px); chute rise
and run; tray capacity (count or one spare slot); slot spacing and the
run-in to the first slot; entry gaps (at least one slot plus 0-1.5 slots,
wider where a steep chute turns); clip length; background, belt, metal, tray
tint, cleat spacing, rail and outline width.

## On-screen text

- **Clock** (only text): fixed dark panel at the top-left, "s.t s" truncated
  to 0.1 s, 36 px bold (glyphs >= 26 px), white on near-black; nothing in the
  scene comes within 12 px of it.
- No other text: objects are named by look ("the striped red star"), bins by
  position ("the upper bin"), moments by the clock. `scene["labels"]` is
  empty and the renderer draws none. The old object numbers and the caption
  are gone (they named the entry order and explained the task).

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total
10) among the combinations the clip can fill (a candidate pool per level
first). Different templates first; above L1 motion questions first, with at
most one single-frame question per level (it competes with them in 30 % of
levels); no (derivation_key, params) repeats. 4 options except yes/no, the
bins (2 or 3) and "which first" (3).

- **L1 observation:** how many objects; how many red objects / stars; which
  object is resting in a bin at the end; how many rest in a bin at the end;
  which object is still on the belt at the end; which object comes out first.
- **L2 relational:** same bin or not; which of two comes to rest first; which
  object comes out right after X; which bin the next one goes to; which bin
  a size / colour / shape goes to; where Y is when X comes to rest (tunnel /
  belt / chute / bin); which bin gets the first object to rest.
- **L3 measurement:** clock time X comes to rest; clock time X reaches the
  end of the belt; objects resting in a bin at a clock reading; objects on
  the belt at a clock reading; seconds along the belt; seconds between two
  objects coming out; seconds from the end of the belt to rest.
- **L4 integration:** which bin a pending object will go to; final count of
  a bin once everything is sorted; clock time a pending object would come to
  rest; how many come to rest between two clock readings; the k-th object to
  come to rest; which bin gets the last object to rest.
- **L5 counterfactual (re-simulated):** every object sent to the other bin
  (count at the end / clock time X would rest); everything twice as fast
  (objects resting at the end); X never there (clock time Y would rest /
  which bin Y would go to); one more object (described) after the last; X
  coming out 0.5 / 1 s later (clock time it would rest); a pending object
  of another colour / shape / size.

Over 5000 scenes 93 % of L2-L5 questions need the motion
(`requires_motion=True`).

## Margins (no near ties)

- First object starts coming out >= 0.5 s after the start; every sorted
  object rests >= 0.5 s before the end; pending objects are fully out >= 0.5 s
  before the end and still >= 40 px before the gate on the last frame; no
  object is ever in a chute at the end.
- Objects are >= 6 px apart on every frame (checked on every frame; entry
  gaps widen with the chute angle).
- Compared moments >= 8 frames (1/3 s) apart; clock readings that a count
  question names have no change of that count within 7 frames either side;
  "where is Y" only when Y's state holds >= 6 frames either side and away
  from the blurred moments (coming out of the tunnel, crossing the gate).
- Clock answers are the reading on the event frame, options 0.5 s apart;
  durations are given to the nearest 0.5 s and only when the true value is
  within 0.1 s of it; what-if counts only when no what-if rest lands within
  8 frames of the end.

## min_fps

Every question stores `params["min_fps"]`: the smallest of 1/2/3/4/6/8/12/24
fps that puts a sample into the shortest interval the answer needs: the gap
between the two compared events (entries, rests, gate passes), the calm
window around a named clock reading, the state window in "where is Y", the
gap between gate passes for taking turns, the what-if's distance from the
end, and 4 fps for clock readings and durations (+-0.25 s). Single-frame
questions need 1 fps. Over 5000 scenes: 44 % of questions need 1 fps, 64 %
<= 2 fps, 100 % <= 4 fps.

## Closed loop

- **Questions** store `derivation_key` + `params` (object names, bin, clock
  tenths, attribute value, delay) + `min_fps`; ground truth comes from the
  exact frames in `sim.py` and `sim.simulate` (re-simulation for every
  what-if).
- **Independent answers** (`answers.py`) never read the rule, the bins, the
  slots, the frame counts or the keyframes. They read `render.track` (each
  object's drawn look and centre on every frame, and the drawn geometry):
  out-of-tunnel, gate and rest frames from the centres; the bin from where an
  object rests; the belt speed from the motion; lane times from the drawn
  lengths (checked against every object that rests); slots filled in order;
  the rule by trying every simple rule on the sorted objects (exactly one must
  fit); its own re-simulation for pending objects and every what-if.
- **`sim.check`** rejects scenes that don't re-simulate, look-alike objects,
  objects closer than 6 px, early / late events, objects in transit at the end,
  pending objects near the gate, more or fewer than one fitting rule, unseen
  pending values, and geometry near the clock or outside the frame.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`frame_check.py`** decodes the MP4s: the clock (template match on every
  6th frame, every event frame and every frame a question names); every
  object found by its colour where the track puts it on every 3rd frame and
  every event frame (pixel count and centroid against its own sprite, so
  striped triangles and stars are not mistaken for off-centre); hidden objects
  invisible in the tunnel; at rest on its slot on the rest frame and still
  moving the frame before; clock answers read on the event frame; counts in a
  bin read as occupied slots (an object sitting still over 3 frames) on every
  frame showing the named reading; objects on the belt counted as colour
  blobs; end-of-clip bins and belt; the order of coming to rest read off the
  frames.

## Determinism

All randomness comes from the `random.Random(seed)` passed in; times are
whole frames or exact fractions; no global RNG, time, uuid or set order
decides anything. Same seed -> byte-identical annotations in separate
processes with different `PYTHONHASHSEED`; the same scene renders identical
raw frames (hashed); regenerating seeds into a fresh folder reproduces
`annotations.jsonl` exactly (apart from `render_meta` timing when rendering).

## Diversity (5000 dry-run seeds)

5000 / 5000 unique scenes; scenarios size 1035 / alternate 1014 / three_way
1012 / colour 973 / shape 966 (three-way by shape 449, colour 343, size 220);
5-9 objects (802 / 1426 / 1614 / 949 / 209), 1 or 2 pending (3309 / 1691);
9 belt speeds 144-240 px/s; belt 24-60 frames; clips 132-240 frames (median
215); tunnel left 2545 / right 2455; 24291 distinct (question, answer) pairs;
answer letters A/B/C/D 25.2 / 25.1 / 24.9 / 24.9 %; mean 4.5 attempts per
scene (62 % retried at least once, max 99, no failures).

## Files and commands

```
sim.py                     scene generation, kinematics, re-simulation, checks (pure, seeded)
questions.py               the 10-question ladder (LADDER = "ten")
answers.py                 independent re-derivation from the drawn track
render.py                  PIL renderer (1280x720, 24 fps, piped to ffmpeg); render.track = what is drawn
run.py                     CLI: generate / validate (common/video5.py)
generate.py                same as `run.py generate` (generate_all_datasets.sbatch)
flatten.py                 annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
frame_check.py             decoded-frame checks of rendered clips
test_conveyor_belt_sort.py pytest suite

python run.py generate --n 50 --seed 46 --output-dir ../output_videos/conveyor_belt_sort_preview
python run.py validate --dataset-dir ../output_videos/conveyor_belt_sort_preview
python frame_check.py --dataset-dir ../output_videos/conveyor_belt_sort_preview
python -m pytest test_conveyor_belt_sort.py -q
```
