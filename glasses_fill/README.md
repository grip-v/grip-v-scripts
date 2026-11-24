# glasses_fill

Reading a measurement scale and tracking liquid volumes over time (the
planned "Video jug" dataset). Patterned jugs on a shelf pour, one at a time,
into a row of numbered 100 ml glasses with a ml scale. A jug delivers what is
asked or runs dry part-way; liquids stack as layers and never mix; a glass
can crack and drain down to the crack. Questions ask what was poured where,
in which order, how much is in a glass (read off the scale), when a pour
started (read off the clock), and what would change under a different pour.

Shared PIL pipeline (`common/video5.py` + `common/ladder5.py`, ten-question
mode). Files: `sim.py` (scene builder, state at time t, `check`),
`questions.py` (10-question ladder), `answers.py` (independent
re-derivation), `render.py` (PIL frames), `run.py` (CLI),
`frame_check.py` (pixel checks on rendered clips), `test_glasses_fill.py`.

```
python run.py generate --n 20 --seed 0 --output-dir ../output_videos/glasses_fill
python run.py validate --dataset-dir ../output_videos/glasses_fill          # add --skip-video for dry runs
python run.py generate --n 5000 --seed 0 --no-render --output-dir <dir>     # dry run, no video
python frame_check.py --dataset-dir ../output_videos/glasses_fill           # needs rendered clips
python test_glasses_fill.py
```

## What a clip shows

- A row of 4-10 identical 100 ml glasses on a table, numbered 1..N under
  them. Each glass has a tick every 10 ml (a longer one at 50 ml); a ruler on
  the left with faint guide lines across the row carries the numbers
  0, 20, ..., 100.
- 3-4 jugs on a shelf above. Each jug has its own **body pattern** (striped,
  banded, dotted, checkered, zigzag, wavy, diagonal, crosshatch) and its own
  **liquid colour** (red, blue, green, orange, purple, yellow, teal, brown),
  seen through a clear window in the jug.
- One jug at a time moves over a glass, tilts, pours a stream into it and
  goes back. A pour delivers min(asked, what is left in the jug): an empty jug
  tilts and nothing comes out; a jug that runs dry stops streaming while it
  is still tilted (and stays tilted >= 0.45 s before turning back), while a
  jug that delivers everything turns back up as its stream ends.
- Each pour sits on top as a new layer (same colour on top of same colour
  merges). A cracked glass shows a dark hole on its wall; liquid above the
  hole streams out onto the table until the level reaches the hole.
- Every amount is a multiple of 10 ml, so every level sits on a tick.
- 24 fps, 1280x720, 4.7-10 s (median 7.8 s).

## On-screen text (and why each is there)

| text | why it is needed |
|---|---|
| clock, top-right, seconds to 0.1 s (`floor(10 f / 24) / 10`) | every clip has a timer; clock questions read it |
| ruler numbers 0, 20, 40, 60, 80, 100 | the measurement questions read ml off the scale; unnumbered ticks alone would leave the viewer counting 10 ticks from the bottom |
| glass numbers 1..N | the glasses are identical, so colour/shape can't name them |

Nothing else is written: no titles, no "ml" unit, no captions or status
text. Labels are stored in `scene["labels"]` (text, x, y, anchor, kind) and
`scene["label_px"]`; the renderer draws exactly that list, and the checks
use it. All labels are the same font (bold, 34 px, digits >= 25 px tall),
dark on a light background, fixed for the whole clip. The glasses start at
x = 250 so a jug tilting over glass 1 never reaches the ruler numbers
(checked pixel-by-pixel on every 2nd frame of 41 worst-case scenes, and on
every 6th frame of the 50 previews).

## Scenarios (`scene["scenario"]`, drawn once per seed, about 25% each)

| scenario | setup |
|---|---|
| `row_pours` | 5-10 glasses; every pour goes into a different, empty glass |
| `layer_stack` | 4-6 glasses; after the first pour, 80% of pours land on a glass that already holds a different colour (layers) |
| `dry_jug` | 4-9 glasses; one jug starts empty and tilts with nothing coming out, another (20-40 ml) is asked for more than it holds and runs dry mid-pour |
| `cracked_glass` | 4-9 glasses; 1-2 glasses crack 0.35-0.8 s after their last pour, at a mark >= 20 ml below their level, and drain at 25-40 ml/s |

## Random knobs

Glass count, glass width (0.5-0.68 of its slot), jug count, jug patterns and
colours (sampled without repeats), jug body colour (3), background and guide
colour (4), table colour (4), start amounts (40-150 ml), number of pours
(2-4; 3-4 in `dry_jug`), which jug pours into which glass, the amount asked
(half a glass / a top-up / a random amount), move speed (0.34-0.46 s per
move), tilt speed (0.18-0.26 s), stream rate (55-90 ml/s, slower for small
pours so every stream lasts >= 0.5 s), pauses, crack mark, crack delay and
drain rate. Over 5000 seeds: 5000/5000 distinct scenes; glasses 4..10, jugs
3/4 about evenly, 2/3/4 pours = 1704/2369/927, each pattern on 10-11% of
jugs, each colour on 10-11%, each background on 24-26% of scenes.

## The 10-question ladder

Counts per level are drawn per seed inside 1-3 / 2-3 / 2-3 / 1-3 / 1-3
(total 10; 22 different splits occur), from 38 templates. Glasses are named
by their number, jugs by pattern, liquids by colour.

| level | templates |
|---|---|
| L1 observation | glass count, jug count, colour in a jug, jug holding a colour, first jug to tilt over a glass, the jug that tilts but pours nothing, the jug that never leaves the shelf, the cracked glass, number of cracked glasses |
| L2 relational | glass a jug is held over on its first tilt, which jug tilts next after another, the jug that poured into a glass, first of several glasses to receive liquid, last glass to receive liquid, jug that pours the most, glass holding the most at the end, bottom / top layer colour, the jug that runs dry mid-pour |
| L3 measurement | level of a glass at the end, ml of one colour in a glass, ml a jug pours into a glass, the crack's ml mark, level of a glass when the clock shows a given reading, clock time when a jug's stream starts, number of tilts, number of empty glasses at the end |
| L4 integration | total liquid at the end, total a jug pours, ml leaked from a cracked glass, difference between two glasses, number of glasses that ever receive liquid, level just before the last pour into a glass |
| L5 counterfactual | level without the crack, total without cracks, level if one pour had not happened, level of another glass if a pour had gone there instead, bottom colour if two pours into a glass had been swapped |

Every question has four distinct options; letters are balanced
(A/B/C/D = 25.1/25.0/24.8/25.1% over 50 000 questions). Wrong ml options
are >= 20 ml from the answer and from each other (other real readings from
the clip are preferred); wrong jugs/colours are the other jugs/colours in the
clip first; clock options are >= 0.6 s apart. 78.6% of L2-L5 questions need
the motion (`requires_motion`). 5651 distinct (question, answer) pairs.

## Closed loop

Each question stores `derivation_key` + `params`. `answers.py` re-derives the
answer without questions.py and without sim's state functions:

- jug poses come from `render.jug_pose` (the geometry the frames are drawn
  with); a tilt is a run of frames with the jug turned >= 70 degrees, its
  glass is the one under the spout tip, the jugs' order is the tilts' order;
- glass levels come from its own per-frame formula; what a tilt delivered is
  the level change of its glass across the tilt; layers are rebuilt by
  stacking those amounts and cutting at the crack;
- clock answers come from the first frame on which the level rises, run
  through the clock formula; level-at-clock answers read all frames that
  show the asked reading (and fail if they disagree);
- counterfactuals re-run the jug bookkeeping (each pour gets
  min(asked, left)) with the pour removed or redirected, then apply the
  cracks.

`sim.check` rejects scenes with overflowing glasses, amounts off the 10 ml
grid, pours < 2 frames apart, a stream < 0.5 s, a dry-out visible < 0.4 s, a
full pour that stays tilted, a crack < 0.3 s after its glass's last pour,
a crack losing < 20 ml, or any event within 0.5 s of either end of the clip.
Over 5000 seeds 1582 scenes needed a retry (2601 extra draws in all, at most
12 draws for one scene); retries use the same seeded rng, so they are
deterministic.

## Margins (no ties, no borderline readings)

- "Most" comparisons need >= 20 ml and >= 10% between first and second.
- Order questions compare separate pours, which are always >= 0.8 s apart.
- Every amount sits on a tick (never between two ticks).
- Clock options are >= 0.6 s apart; a level-at-clock question uses a reading
  that lies >= 0.5 s inside a >= 1.2 s stretch where the level doesn't
  change, and a level different from the glass's end level.

## Determinism

All randomness comes from the `random.Random(seed)` that video5 passes in
(no global random, numpy RNG, time, or set order). Same seed -> byte-identical
scene + questions JSON across processes with different `PYTHONHASHSEED`
(tested); regenerating 50 seeds into a fresh folder reproduces
annotations.jsonl byte for byte; rendering a seed twice gives identical raw
frame hashes (tested).

## Frame sampling and `min_fps`

Each question stores `params["min_fps"]` = ceil(1 / g), clamped to 1..24,
where g is the shortest time a cue it needs is on screen or the smallest gap
between the events it needs (tilts and the gaps between them for order and
count questions, the stream for who-poured-what, the stretch before and after
a pour for amounts, the settled end state for end questions, the start for
shelf questions). Clock-reading questions need 4. Over 50 000 questions:
38.8% are answerable at 1 fps, 93.3% at 2 fps and 100% at 4 fps.
Every stream lasts >= 0.5 s, every result persists (levels, layers, holes,
puddles), objects are >= 48 px wide, lines >= 3 px, and the 8 liquid colours
and 8 patterns stay apart at 640x360.

## Checks run

- `run.py validate`: 2000 and 5000-scene dry runs, 0 mismatches, 0 ladder
  errors, 0 check failures, no warnings (`python -W error`); tamper test
  (3 answers corrupted) flags exactly those 3 questions.
- `frame_check.py` on the 50 previews (seeds 0-49: 14 row_pours, 12 of each
  other scenario): 1826 clock readings template-matched, 18 788 label crops
  equal to the static render, 334 end levels + layer lists, 134 pours
  (delivered amount, stream colour / no stream), shelf order, cracks (none
  before, stream mid-drain, hole after) and 139 answers measured from the
  decoded frames: 0 failures.
- `test_glasses_fill.py`: 13 tests (determinism, ladder counts, min_fps,
  uniqueness, tamper, labels vs renderer, three hand-checked scenes).
