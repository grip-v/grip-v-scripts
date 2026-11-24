# stacking_collapse

Video-QA sub-benchmark for **stability and toppling**: blocks are stacked
(or pulled out of a stack) one at a time; after every change each resting
surface is checked, and when the combined centre of mass of everything
above a surface lies outside the contact span under it, everything above
the lowest failing surface turns over the contact edge as one rigid piece
and falls out of the frame. The clip goes on after a collapse (later blocks
land on what is left). v2 moves the dataset to the 10-question ladder
(`LADDER = "ten"` in `questions.py`), adds four scenarios, a measuring grid,
multiple collapses per clip and an independent decoder that reads only the
drawn outlines.

## What each clip shows

- Grey pillars rising from the bottom edge (pedestals / a support), an
  unnumbered background grid of 1-unit squares (lines or dots), and
  rectangular blocks 1-6 units wide and 0.75 / 1 / 1.25 units tall
  (mass = area). Each block is one of 8 colours and plain, striped or dotted;
  colour + pattern is unique in a clip, and blocks that ever touch differ in
  colour.
- A dropped block starts at rest above the frame and falls under gravity
  (1.6-2.4 px/frame^2) straight down, landing on a whole frame. A pulled
  block slides out sideways at 18-30 px/frame; once it is clear the part
  above drops onto the block below. A collapse waits 3-8 frames, turns about
  the contact edge (2-3.5 deg/frame, speeding up) and flies off outward
  until it has left the frame.
- Scale 28-46 px per unit, fitted so the whole build fills the frame, with a
  random horizontal position; clips 2.2-9.9 s.

### Scenarios (`scene["scenario"]`, 25 % each)

| scenario | setup | collapses per clip |
|---|---|---|
| `tower` | one pedestal (2-5 units wide); 4-9 blocks drop onto the top of the stack | 0 / 1 / 2 (30 / 45 / 25 %) |
| `twin` | two pedestals (2-4 wide, tops at different heights); 5-10 blocks, each dropped onto one of them | 0 / 1 / 2 (30 / 45 / 25 %) |
| `pull_out` | a standing tower of 5-8 blocks; 2-4 blocks are pulled out sideways one at a time, the part above drops onto the block below | 0 / 1 (40 / 60 %) |
| `plank` | a 10-16 unit plank lying on a 1-3 unit support; 3-6 small blocks (1-3 wide) dropped onto it side by side; plank + blocks tip together | 0 / 1 (45 / 55 %) |

The target number of collapses and the number of blocks are drawn first,
then the build is searched for (placements that lean towards the edge
before a planned collapse, a "weak spot" block in `pull_out`).

### Random knobs

Scenario; collapse count; block count; pedestal widths, heights and
spacing; plank length, offset and support width; block widths (3-5 of the
sizes 2-6) and heights (one to three of 0.75 / 1 / 1.25); every drop
position (0.5-unit steps) and order; which blocks are pulled and to which
side; pause before a collapse; tipping speed and fly-off speed; gravity;
pull speed; scale and framing; colour pool (2-8 colours) and pattern pool
(1-3 patterns); background (7 palettes, flat / gradient / vignette), grid
style and contrast, pillar and plank colour, outline width, stripe width.

## On-screen text and the grid

- **Clock** (the only text): fixed dark panel at the top-left, "s.t s"
  truncated to 0.1 s, 36 px bold (glyphs >= 26 px), white on near-black;
  nothing comes within 12 px of it.
- **Grid** (no numbers): 1-unit squares, lines 3 px wide or dots 7 px
  across, low contrast behind everything. It is the measuring aid for the
  width questions ("how many grid squares wide") and for the what-ifs that
  move a block "one grid square". An unnumbered grid does this job, so no
  ruler numbers are drawn.
- No labels: blocks are named by pattern and colour ("the striped red
  block"), pedestals by side, moments by the clock. `scene["labels"]` is
  empty and the renderer draws none.

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total
10) among the combinations the clip can fill (a candidate pool per level
first). Different templates first; above L1 motion questions first, with at
most one single-frame question per level (it competes with them in 30 % of
levels); no (derivation_key, params) repeats; any question that would need
more than 4 fps is dropped. 4 options except yes/no, which of two, left /
right and the three-way pedestal questions. 43 templates:

- **L1 observation:** how many blocks are dropped / start in the tower /
  are pulled out; how many times blocks fall; does the plank tip; how many
  blocks are left at the end; which block is on top at the end; which block
  lands / is pulled first; how many striped (plain, dotted) blocks appear.
- **L2 relational:** which of two lands (is pulled) first; what a block
  lands on; what the part above comes down onto after a pull; which way the
  falling part tips; whose landing (pull) makes blocks fall; which of two
  blocks is wider; onto which pedestal a block drops; which pedestal loses
  blocks first; which pedestal holds more at the end; on which side of the
  support a block lands; which block lands directly on another.
- **L3 measurement:** clock time a block lands / starts to slide out;
  blocks stacked at a clock reading (per pedestal in `twin`); how many
  blocks fall in a collapse; seconds between two landings (pulls); width in
  grid squares; the most blocks stacked at one moment.
- **L4 integration:** blocks fallen over the whole clip; what the falling
  part was resting on just before it tipped; the k-th block to land (be
  pulled); blocks stacked right after a given block lands; blocks that fall
  right after a given pull; landings (pulls) between two clock readings;
  which of four blocks falls at some point.
- **L5 counterfactual (re-simulated):** a block never dropped (would any
  fall / how many left at the end); a block dropped one grid square to the
  left / right (would blocks fall when it lands / how many); two
  consecutive drops swapped; a different block pulled out, or a pull left
  out; a pedestal (support) one square wider on each side; a plank block
  landed on the other side of the support.

Over 5000 scenes 92.5 % of L2-L5 questions need the motion
(`requires_motion=True`).

## Margins (no near ties)

- Every centre of mass the rule compares is >= 0.25 units from both contact
  edges; touching layers overlap by >= 1 unit; plank blocks sit 0.5 units in
  from the plank's ends and 0.5 units apart. What-ifs are only asked when
  the re-simulated sequence keeps the same margins.
- Landings / pulls are >= 8 frames (1/3 s) apart; compared events >= 8
  frames; every event >= 2 frames from any other; the first block starts
  falling and every event happens >= 0.5 s after the start, and every event
  (including the end of a turn) >= 0.5 s before the end; a falling piece
  has left the frame >= 4 frames before the end.
- Things that do not touch stay >= 8 px apart on every frame (checked on
  every frame against the drawn outlines; touching pairs may not overlap by
  more than 1 px).
- Clock answers are the reading on the event frame, options 0.5 s apart;
  durations are given to the nearest 0.5 s and only when within 0.1 s of
  it; counts at a clock reading only where nothing changes within 7 frames
  of the frames showing it (a landing, the slide and drop of a pull, the
  first turn of a collapse); widths differ by >= 1 unit (>= 16 %) in "which
  is wider".

## min_fps

Every question stores `params["min_fps"]`: the smallest of 1/2/3/4/6/8/12/24
fps that puts a sample into the shortest interval the answer needs: the gap
between the compared landings / pulls, the calm stretch around a named
clock reading, how long a block is wholly in view (for counting), how long
a falling piece is in view (for its side and size), how long a state lasts,
and 4 fps for clock readings and durations (+-0.25 s). Single-frame
questions need 1 fps; questions that would need more than 4 fps are not
asked. Over 5000 scenes: 42 % of questions need 1 fps, 80 % <= 2 fps,
100 % <= 4 fps.

## Closed loop

- **Questions** store `derivation_key` + `params` (block names, pedestal
  index, clock tenths, collapse index, shift) + `min_fps`; ground truth
  comes from `sim.timeline` (event frames) and `sim.play` (the rule;
  re-simulation for every what-if).
- **Independent answers** (`answers.py`) never read the actions, the event
  list or `sim.play`. They read `render.track`: the drawn outline of every
  block and the plank on every frame, the pillars and the grid unit. From
  the outlines they find landings (first frame at rest), pulls (sideways
  motion), drops after a pull, collapses (first turned frame) and their
  side, what each resting block stands on (bottom edge on a top edge), the
  width in grid squares; for the what-ifs they rebuild the sequence of
  drops / pulls and re-simulate it in px with torques about the contact
  edges (sum of area x lever arm), and they first check that this rule
  reproduces every collapse seen in the clip.
- **`sim.check`** rejects scenes whose outcome is not clear-cut, look-alike
  blocks, touching blocks of one colour, blocks under 20 px, events too
  close to each other or to the ends, a falling piece still in view at the
  end, things closer than 8 px or overlapping, and geometry near the clock.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`frame_check.py`** decodes the MP4s: the clock (template match on every
  6th frame, every event frame and every frame a question names); every
  block found inside its drawn outline (colour pixels and their centroid)
  on every 3rd frame and every event frame; each landing at rest on its
  landing frame and the landing frame read off the video equal to the event
  frame (so the landing order); pulls moving by the slide speed; collapses
  moving towards their side; the grid spacing measured in the frame equal
  to the unit, and widths read off the frame in grid squares equal to the
  answers; counts of blocks at rest on the frames showing a named clock
  reading and at the end; the top block at the end.

## Determinism

All randomness comes from the `random.Random(seed)` passed in; positions
are multiples of 0.5 units, times whole frames; no global RNG, time, uuid
or set order decides anything. Same seed -> byte-identical annotations in
separate processes with different `PYTHONHASHSEED`; the same scene renders
identical raw frames (hashed); regenerating seeds into a fresh folder
reproduces `annotations.jsonl` exactly (apart from `render_meta` timing
when rendering).

## Diversity (5000 dry-run seeds)

5000 / 5000 unique scenes; scenarios plank 1283 / tower 1273 / pull_out 1234
/ twin 1210; collapses per clip tower 0/1/2 = 382/565/326, twin 367/557/286,
pull_out 0/1 = 459/775, plank 549/734; blocks fallen per clip 0-9; 3-10
blocks (308 / 476 / 935 / 1106 / 740 / 759 / 458 / 218) and 2-10 drops or
pulls; scale 28-46 px per unit (19 values); 5 gravities; clips 54-237 frames
(median 138); 34 width sets, 7 height sets, 1-8 colours and 1-3 patterns per
clip; 14 background colours x 3 styles, lines / dots grid 2501 / 2499;
13476 distinct (question, answer) pairs in 50000 questions; answer letters
A/B/C/D 24.9 / 24.9 / 24.9 / 25.4 %; level counts L1 1/2/3 = 2244/1767/989,
L2 2/3 = 2936/2064, L3 2942/2058, L4 1/2/3 = 2382/1767/851, L5
2253/1830/917 (22 combinations). Generation retries: 2773 scenes needed
more than one attempt (mean 6.3, max 311, plank 11.4 on average); no seed
fails, no warnings.

## Files and commands

```
sim.py                    the rule, scene generation, kinematics, checks (pure, seeded)
questions.py              the 10-question ladder (LADDER = "ten")
answers.py                independent re-derivation from the drawn outlines
render.py                 PIL renderer (1280x720, 24 fps, piped to ffmpeg); render.track = what is drawn
run.py                    CLI: generate / validate (common/video5.py)
generate.py               same as `run.py generate` (generate_all_datasets.sbatch)
flatten.py                annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
frame_check.py            decoded-frame checks of rendered clips
test_stacking_collapse.py pytest suite

python run.py generate --n 50 --seed 0 --output-dir ../output_videos/stacking_collapse_preview
python run.py validate --dataset-dir ../output_videos/stacking_collapse_preview
python frame_check.py --dataset-dir ../output_videos/stacking_collapse_preview
python -m pytest test_stacking_collapse.py -q
```
