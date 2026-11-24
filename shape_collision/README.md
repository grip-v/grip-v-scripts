# shape_collision

Video-QA sub-benchmark: **patterned shapes colliding, trading or passing on
patterns, bouncing off posts, falling into holes and riding rails**. v3 runs
on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the
10-question ladder (`LADDER = "ten"` in `questions.py`).

## What each clip shows

- **Box:** a walled box of random size (860-1240 × 470-610 px) and place,
  always below the clock panel. Nine floor colours (dark and light), five
  faint floor textures (plain, grid, dots, tiles, planks) with two strengths
  and four spacings, six wall colours, wall thickness 5-12 px, six plain or
  gradient backgrounds.
- **Shapes:** 5-8 shapes. Each is a disc for the physics (mass ∝ r²,
  radius 28-58 px, scaled with the box) drawn as one of 11 outlines (circle,
  square, triangle, pentagon, hexagon, octagon, star, cross, heart, diamond,
  trapezoid; thin outlines are left out so patterns stay readable at half
  size) inside a thin ring (solid or dashed, 3 px) that marks the disc, so
  every contact is visible ring to ring. Every shape has its own colour (10
  colours, each ≥ 200 RGB-sum away from the floor), so "the blue star" is
  always unique; outline width 3 or 4 px.
- **Patterns:** 10 fills: solid, hollow, horizontal / vertical / diagonal
  stripes, dots, checks, grid, rings, two-tone. The second colour of a
  pattern is a pale tint of dark colours and a deep shade of light ones;
  stripe and dot spacing scale with the shape.
- **Motion:** straight lines at 40-500 px/s (typically 80-190), elastic
  bounces off walls, posts and each other. Clips are 5.5-10 s at 24 fps
  (median 9.9 s). There are no trails, glow or impact flashes: the only cues
  are the contact itself and the pattern change.

### Scenarios (`scene["scenario"]`, 20 % each)

| scenario | setup |
|---|---|
| `swap` | open box; every shape has its own pattern; two shapes that collide **swap** patterns |
| `infect` | open box; all shapes solid except one wearing the scene's tag pattern (one of 7); a collision between a tagged and an untagged shape tags the other one too; tags never go away |
| `posts` | swap rule, plus 2-5 round grey posts (r 22-40) the shapes bounce off |
| `holes` | swap rule, plus 2-3 dark holes in the floor; a shape whose centre passes over a hole falls in and is gone from that frame on |
| `rails` | swap rule, but the shapes ride 2-4 horizontal rails (2-3 shapes each) and only meet shapes on their own rail: head-on 1-D collisions with unequal masses |

### Random knobs

Scenario, shape count, outlines, colours, radii, starting patterns, tag
pattern and first tagged shape, speeds and directions (start state is cut
from a 40 s run, see below), box size and position, floor colour / texture /
strength / spacing, wall colour and width, background, ring style, outline
width, post / hole / rail count, sizes and positions, post and hole-rim
colours, clip length.

## On-screen text

- **Clock** (always, the only text): dark panel fixed at the top-left
  (14-196 × 14-70 px), "s.t s" truncated to 0.1 s, 36 px bold (glyph height
  ≥ 26 px, white on near-black). The box always starts below the panel, so
  nothing ever passes under it.
- **No labels.** Shapes are named by colour + outline, patterns by name,
  moments by the clock, so no measurement text is needed: no titles, legends,
  counters, rule hints or captions. `scene["labels"]` is `[]`.

## Clean events

The physics is solved exactly, event by event (next wall, post, hole or pair
contact from closed-form straight-line motion). A random start is run for
40 s and the clip is the longest window of it (≤ 10 s) in which every event is
clean; the shapes' state at the window start becomes the scene. A window is
kept only if:

- nothing happens in its first 0.55 s, and no collision or fall happens in
  its last 0.55 s or the 0.3 s after it ends;
- 3-12 collisions between shapes happen (infect: ≥ 2 tags passed; posts:
  ≥ 2 post bounces; holes: 1 to n-3 shapes fall);
- any two collisions / falls are ≥ 0.2 s apart, two collisions of one shape
  ≥ 0.3 s apart, any two events of one shape ≥ 0.15 s apart;
- no collision is glancing (closing direction within 78° of the line of
  centres); no pair or shape-post comes within 15 px of touching without
  touching; no shape's centre comes within R + 0.5 r of a hole without
  falling in, and a falling shape's path runs within 0.8 R of the centre;
- every shape moves at ≥ 40 px/s.

`sim.check` re-simulates the stored scene and repeats every rule. 5000 seeds:
mean 10.5 attempts per scene (holes 31, posts 12, swap 3.7, infect 4.2, rails 1.8; 76 % of seeds needed more than one, none failed);
attempts are drawn from the same seeded rng, so retries are deterministic.

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total
10, `L5.ten_counts`); templates are spread so the same template is used at
most once per level before any repeats, at most one single-frame L1 question
is used when others are available, and no (derivation_key, params) repeats.
Every question has 4 options.

- **L1 observation:** number of shapes; outline of the X-coloured shape;
  colour of the only Y; number of collisions; first pair to collide; start /
  end pattern of X (swap); number of posts / holes / rails; which shape starts
  tagged, how many are tagged at the end (infect); how many are left in the
  box at the end (holes).
- **L2 relational:** which of four shapes X ever collides with; X's first
  partner; which of four collides most; which of four pairs collides first;
  which of four shapes first bounces off a wall; which of four moves fastest
  at the start (≥ 15 % margin); who passes the tag to X; which of four is
  tagged first (infect); which of four falls first (holes); which of four
  first bounces off a post (posts).
- **L3 measurement:** X's collision count; X's wall bounces (only shapes whose
  bounces are clean and ≥ 0.3 s apart); X's post bounces; clock reading when
  X and Y collide / when X falls (options 0.5 s apart); time between the
  first and last collision; collisions so far / X's pattern / tagged count /
  shapes left when the clock shows t (t ≥ 0.3 s from every relevant event).
- **L4 integration:** which shape collides most; which pair makes the k-th
  collision; how many different shapes X collides with; patterns X wears;
  shapes that wear pattern P; who ends with X's starting pattern; how many
  keep their pattern (swap); how many X tags; who is the k-th tagged
  (infect); who falls last (holes); who bounces off posts most (posts).
- **L5 counterfactual (re-simulated):** had X and Y not traded / passed on
  at their collision: X's final pattern, or how many keep their pattern /
  end tagged; had the video stopped before the last collision: X's pattern;
  had Y started tagged instead; had X not been in the box: how many
  collisions, which pair collides first, how many fall into holes; had the
  posts been removed / holes covered: how many collisions. Physical
  what-ifs are only asked when the changed world is as clean as a real clip.

All L2, L3 and L5 questions and all L4 questions except "who ends with X's
starting pattern" and "how many keep their pattern" (first and last frame
suffice) set `requires_motion=True`: 95 % of L2-L5 questions.

## Margins (no near ties)

Compared events ≥ 0.3 s apart; clock moments ≥ 0.3 s from every event that
could change the answer; clock answers taken on the event's first frame
(events sitting within 0.1 frame of a frame boundary are not asked) and
offered in 0.5 s steps; speed comparisons ≥ 15 %; "which of four" has
exactly one qualifying option (answers.py returns no answer on a tie).

## min_fps

Every question stores `params["min_fps"]`: the smallest of 1/2/3/4/6/8/12/24
fps that puts a sample in the shortest interval the answer needs: the whole
clip for single-frame questions; the gaps between consecutive collisions of
the same shape (counts and pattern tracking; different pairs touching at the
same moment are visible in one frame); the gaps between compared events
(orders); the interval around a clock moment between the nearest relevant
events; ±0.25 s for clock answers (4 fps); half the time to the first event
for start-speed and remove-a-shape what-ifs. Over 5000 scenes: 49 % of
questions need 1 fps, 78 % ≤ 2 fps, 100 % ≤ 4 fps.

## Closed loop

- **Questions** store `derivation_key` + `params` (shape ids, pairs,
  collision numbers, clock tenths, option ids) + `min_fps`.
- **Generation** (`sim.py`) uses the exact event-driven solver.
- **Independent answers** (`answers.py`) never read the event list. They
  decode `render.track` (each shape's centre and pattern on every frame,
  None once fallen): where a shape's frame-to-frame displacement changes,
  its previous motion is extended from the last clean frame and the cause is
  the wall, post, or other shape (changing motion in the same frame step)
  that the extended path touches inside that step, which also gives the
  contact time; a shape that stops being drawn fell into the hole its path
  enters. Patterns are replayed from the frame-0 patterns with the rule.
  Physical counterfactuals are re-simulated with a fixed 2 ms step from the
  frame-0 positions and first-step velocities, rewinding each contact to
  the touch inside its step.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`frame_check.py`** decodes the MP4s: template-matches the clock against
  all 100 readings on every 6th frame, every event frame and every frame a
  question names; finds each shape as its colour blob (pixels classified to
  the nearest of all scene colours) every 6th frame and compares the centre
  with the track, checks fallen shapes are gone and the blob count; classifies
  every shape's pattern on every frame against its sprite in all 10 patterns
  and requires the decoded pattern changes to be exactly the
  pattern-changing collisions on exactly their frames; checks each fall
  frame; reads the clock on the decoded collision / fall frame of every
  clock question; checks the timer panel is outside the box.

## Determinism

All randomness comes from the `random.Random(seed)` passed in; no global
RNG, time, uuid or set order decides anything. Same seed → byte-identical
annotations in separate processes with different `PYTHONHASHSEED`; the same
scene renders identical raw frames (hashed); regenerating seeds into a fresh
folder reproduces `annotations.jsonl` exactly (apart from `render_meta`
timing when rendering).

## Diversity (5000 dry-run seeds)

5000 / 5000 unique scenes (hash of the scene rounded to 0.1); scenarios
swap 1035 / rails 1014 / holes 1012 / infect 973 / posts 966; 5-8 shapes;
3-12 collisions; 25 600 distinct (question, answer) pairs out of 50 000;
answer letters A/B/C/D 25.0 / 25.1 / 24.9 / 25.0 %.

## Files and commands

```
sim.py                  scene generation, exact physics, rules, clean-event checks (pure, seeded)
questions.py            the 10-question ladder (LADDER = "ten")
answers.py              independent re-derivation from the rendered track + fixed-step re-simulation
render.py               PIL renderer (1280x720, 24 fps, piped to ffmpeg); render.track = what is drawn
run.py                  CLI: generate / validate (common/video5.py)
generate.py             same as `run.py generate` (generate_all_datasets.sbatch)
flatten_annotations.py  annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
frame_check.py          decoded-frame checks of rendered clips
test_shape_collision.py pytest suite

python run.py generate --n 50 --seed 200 --output-dir ../output_videos/shape_collision_preview
python run.py validate --dataset-dir ../output_videos/shape_collision_preview
python frame_check.py --dataset-dir ../output_videos/shape_collision_preview
python -m pytest test_shape_collision.py -q
```
