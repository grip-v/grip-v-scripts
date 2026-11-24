# symmetry_pattern_video

Video-QA sub-benchmark for **seeing symmetry as a picture is built, moved
or flipped**, and spotting the one shape that breaks it. v3 ports the old
Manim version to the shared PIL + ffmpeg pipeline (`common/video5.py`) with
the 10-question ladder (`LADDER = "ten"` in `questions.py`).

## What each clip shows

- **Pattern:** one symmetry group about a centre point: rotational 2/3/4/5/6/8-fold,
  mirror left-right, mirror up-down, both mirrors, or dihedral 3/4 (that many
  turns plus as many mirror lines). The pattern is 2-5 *groups* ("orbits"):
  one base shape carried to every position by every element of the group, so
  shapes turn like a pinwheel under rotations and appear as true mirror
  images under reflections. 6-16 shapes.
- **Shapes:** 10 outlines (triangle, square, diamond, pentagon, hexagon,
  star, circle, kite, chevron, lightning bolt; the last three are chiral, so
  mirror images differ from turned copies), 10 colours (each ≥ 200 RGB-sum
  from the background), 4 fills (outline, solid, outline + centre dot, double
  outline), solid or dashed line, size 20-58 px circumradius scaled with the
  pattern, stroke 3-5 px. No two groups share colour *and* outline, so "the
  red kites" names one group.
- **Framing:** pattern radius 200-320 px, centre anywhere in the frame that
  keeps the pattern clear of the clock panel; six backgrounds (dark and
  light) drawn flat, as a gradient or as a vignette.
- **Breaks:** about 45 % of clips have exactly one shape that differs from
  its partners: different fill, colour, outline, size (×0.58-0.74 or the
  inverse), turn (20-32°), place (shifted 0.6-0.95 of its size + 10 px), or,
  in waves clips, traced the other way round or never finished (stops at
  50-80 % of its outline). Colour / outline breaks are not used in 2-copy
  groups (they would leave a "group" of one).
- Clips are 4-10 s at 24 fps (median 5.4 s).

### Scenarios (`scene["scenario"]`, 25 % each)

| scenario | what happens |
|---|---|
| `waves` | groups are traced one after another (0.6-1.2 s each, 0.3-0.7 s pauses); each outline grows from a start corner with a dot riding its tip, all turning the same way; fills fade in when a wave is done |
| `reveal` | the shapes pop in one at a time in random order (0.3-0.6 s apart), so the picture's symmetry changes as it fills |
| `assemble` | every group glides in from scattered spots, turning as it goes (0.75-1.7 s glides); groups start and settle at different times |
| `turns` | the finished pattern sits on a disc with a notch on its rim; the disc makes 2-4 moves: turns of 45-180° either way (75-160°/s) and left-right or upside-down flips, with pauses |

### Random knobs

Scenario, group, number of groups, outlines (25 % of clips use one outline
for every group), colours, fills, dashes, sizes, base positions and turns,
trace direction, break kind / shape / amount, pattern radius and centre,
background colours and style, stroke and dot size, every timing (intro,
trace length, pauses, pop order and gaps, glide starts and lengths, start
spots and turns), the disc's moves, angles, speed and notch angle.

## On-screen text

- **Clock** (always, the only text): fixed dark panel at the top-left,
  "s.t s" truncated to 0.1 s, 36 px bold (glyph height ≥ 26 px, white on
  near-black); the pattern circle (and the disc) never reaches it.
- **No labels.** Groups are named by colour + outline, the odd shape by its
  role ("the shape that breaks the symmetry"), places by compass direction
  from the centre, moments by the clock. The disc's notch is a plain
  triangle, not text. `scene["labels"]` is `[]`.

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total
10) among the combinations the clip can fill (a candidate pool is built per
level first; a scene is only kept if every level has at least 3 distinct
candidates). Different templates first, motion questions first, at most one
single-frame question per level while others remain; no (derivation_key,
params) repeats; every question has 4 options. Wrong group names are other
groups or plausible mixes of a colour and an outline that are on screen but
never together.

- **L1 observation:** shapes at the end; shapes of one outline; colours;
  number of waves; which group is drawn / appears / starts moving first;
  number of disc moves or flips.
- **L2 relational:** what kind of break (if any) the picture has; which wave
  a group is drawn in; which group is last / first to be complete or
  settled; of four listed groups which starts first; what the k-th disc
  move is; in which direction from the centre the odd shape is.
- **L3 measurement:** smallest self-mapping turn; number of mirror lines;
  clock when a group starts being drawn; clock when the last wave / shape /
  group / move is done; shapes on screen, shapes fully traced or groups
  settled when the clock shows t; total degrees turned; where the notch
  points after move k; size ratio of the odd shape.
- **L4 integration:** which wave holds the odd shape; shapes finished before
  a wave starts; shapes drawn after a wave; shapes traced the majority way
  round; shapes on screen the first time the picture is half-turn / mirror
  symmetric; group of the k-th shape to appear; shapes on screen when a
  group completes; k-th group to settle; groups moving at t; after which
  move / after how many moves the shapes look as at the start; where the
  odd shape is after move k; degrees turned by move k.
- **L5 counterfactual (re-derived):** which of four turns / flips leaves the
  picture unchanged; mirror lines if the odd shape were removed; shapes
  after one more wave; clock at the end of the last wave if every trace took
  twice as long; mirror lines / smallest turn if the last shape had never
  appeared; which group settles last if one group started 1 s later; where
  the notch would point if move k were skipped.

86 % of L2-L5 questions set `requires_motion=True`; the rest (break kind
outside waves, symmetry counts of the finished picture, turn/flip what-ifs,
odd-shape removal) are read off one frame and are marked so.

## Margins (no near ties)

- Timed events sit on whole frames; any two compared or neighbouring events
  are ≥ 7 frames (0.29 s) apart and ≥ 12 frames from either end of the clip.
- Clock moments in questions are ≥ 7 frames from every event that could
  change the answer; clock answers are offered in 0.5 s steps.
- The unbroken pattern measures exactly its group even with a loose 6 px
  tolerance (no near-extra symmetry); a break destroys it even at 4 px.
- Compass answers are ≥ 10° from a sector boundary and ≥ 40 px from the
  centre; odd-shape questions are only asked for groups of ≥ 3 copies (with
  two copies it is ambiguous which one is "odd").
- Size ratios are asked only away from the rounding boundary.

## min_fps

Every question stores `params["min_fps"]`: the smallest of 1/2/3/4/6/8/12/24
fps that puts a sample in the shortest interval the answer needs: the whole
clip for single-frame questions; the pause between waves; the gap between
compared appearances, starts or settles; the interval around a clock moment
between the nearest events; half a trace (to see which way the tip turns);
half a move; ±0.25 s for clock answers (4 fps). Over 5000 scenes: 33 % of
questions need 1 fps, 63 % ≤ 2 fps, 100 % ≤ 4 fps.

## Closed loop

- **Questions** store `derivation_key` + `params` (group as colour + outline,
  move number, clock tenths, listed options) + `min_fps`.
- **Generation** (`sim.py`) measures symmetry on shape descriptors
  (outline corners from type, size, turn, mirror flag).
- **Independent answers** (`answers.py`) never read the timeline, the orbit
  list or the break record. They read `render.track` (every shape's drawn
  outline in pixels, how much is drawn, the tip dot, the fill fade, the
  disc and notch): symmetry with a separate matcher (every corner of a
  moved outline must lie on the other outline's boundary); appearances,
  traces and settles from first / last frames of the track; turning
  direction from the area swept by the tip dot; the odd shape as the one no
  symmetry of the rest maps onto a like shape; break kinds from what
  differs between like shapes; disc moves by fitting a 2x2 map to all
  corners on every frame (angles summed frame by frame); what-ifs replayed
  from those.
- **`sim.check`** rejects scenes with too many / too small shapes, colours
  close to the background, two groups with the same name, overlapping
  shapes, a pattern leaving the frame or reaching the clock, near-extra
  symmetry, breaks that leave (near-)symmetry, events too close together or
  to the ends.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`frame_check.py`** decodes the MP4s: template-matches the clock (all 100
  readings) on every 6th frame, every event frame and every frame a question
  names; counts and locates each colour's blobs (pixels classified to the
  nearest scene colour) on calm frames; checks every wave / pop is absent on
  the frame before and present on its first frame, and every glide and disc
  move is still from its end frame on; checks each half-traced outline's
  pixels follow the stored direction rather than the other; finds the notch
  where the track puts it; reads the clock on each clock answer's event
  frame; checks the clock panel is outside the pattern.

## Determinism

All randomness comes from the `random.Random(seed)` passed in; no global
RNG, time, uuid or set order decides anything (sets are sorted before use).
Same seed → byte-identical annotations in separate processes with different
`PYTHONHASHSEED`; the same scene renders identical raw frames (hashed);
regenerating seeds into a fresh folder reproduces `annotations.jsonl`
exactly (apart from `render_meta` timing when rendering).

## Diversity (5000 dry-run seeds)

5000 / 5000 unique scenes; scenarios turns 1283 / waves 1273 / assemble 1234
/ reveal 1210; all 11 groups (dihedral_4 136 … rotational_4 562); 6-16
shapes; 2721 unbroken, 2279 broken over 8 kinds; 6476 distinct (question,
answer) pairs; answer letters A/B/C/D 25.0 / 25.1 / 25.0 / 24.9 %; mean 2.1
attempts per scene (no failures).

## Files and commands

```
sim.py                         scene generation, group algebra, symmetry measurement, timelines, checks (pure, seeded)
questions.py                   the 10-question ladder (LADDER = "ten")
answers.py                     independent re-derivation from the rendered track
render.py                      PIL renderer (1280x720, 24 fps, piped to ffmpeg); render.track = what is drawn
run.py                         CLI: generate / validate (common/video5.py)
generate.py                    same as `run.py generate` (generate_all_datasets.sbatch)
flatten.py                     annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
frame_check.py                 decoded-frame checks of rendered clips
test_symmetry_pattern_video.py pytest suite

python run.py generate --n 50 --seed 0 --output-dir ../output_videos/symmetry_pattern_video_preview
python run.py validate --dataset-dir ../output_videos/symmetry_pattern_video_preview
python frame_check.py --dataset-dir ../output_videos/symmetry_pattern_video_preview
python -m pytest test_symmetry_pattern_video.py -q
```
