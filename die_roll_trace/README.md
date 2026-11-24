# die_roll_trace

Video-VQA sub-benchmark testing **3D spatial-orientation tracking, physical
rotation-rule verification, face-history reasoning, and counterfactual
re-rolling** — a die with a CUSTOM (non-standard) set of face values rolls
across a 2D grid, tipping over one edge per step in one of four cardinal
directions, and the video may show a physically CORRECT sequence of
orientations, or one containing a SINGLE roll that violates real
die-rotation physics.

## 1. What each clip shows

- An N=8-11 grid of decorative, colored, numbered cells (colors and cell
  values carry **no mechanical meaning** — see §8).
- A **custom die**, folded from a randomized cross-shaped net (`die.py`
  §4), positioned on its current cell, rendered as a simple pseudo-3D cube
  icon (front + right-side + top-diamond faces) with its TOP face's value
  always clearly legible — the single most important piece of visual
  information for every question in this dataset.
- The die **tips one step per roll** in a random cardinal direction
  (revisits allowed), animated as smooth interpolated motion between cell
  centers, with the top-face value updating the instant it lands.
- A **persistent trail line** (with dot markers at each earlier stop) shows
  every visited cell, so revisit patterns are visible at a glance.
- **Two content modes**: `valid_rolls` (~55% target) — every roll follows
  the die's fixed rotation physics exactly; `broken_rolls` (~45% target) —
  every roll is correct except exactly one, whose displayed resulting
  orientation is a genuine roll result **for a different direction than
  the one actually shown** (see §4) — always still one of the die's 6
  real values, in a real cube orientation, just physically inconsistent
  with the movement shown; all following rolls then apply correct physics
  from that new, wrong state onward.
- **Final hold** (~18% of duration) on the finished position and full
  trail.

## 2. Why this needs video, not a single frame

The final frame only shows where the die ended up and its final trail —
it never shows the *order* faces appeared in, which single roll (if any)
broke physics, or what the path looked like at an intermediate point.
Level 4c (top value after N rolls), Level 4a (locating the broken roll),
and Level 5a/5b (full downstream/alternate-start re-simulation) all
require replaying — or re-deriving from — the actual roll-by-roll
sequence, not just reading the held final state.

## 3. The 5-question ladder

| qid | level | difficulty | asks |
|---|---|---:|---|
| q01 | Image Description | 1 | total number of rolls |
| q02 | Basic Relational | 2 | (a) final top-face value, or (b) was any cell revisited |
| q03 | Comparative | 3 | (a) total unique cells visited, (b) first roll direction, or (c) how many times a named value appeared on top |
| q04 | Compound | 4 | (a) locate the broken roll + correct top value there, else confirm all correct, (b) opposite of the starting top value, or (c) top value after exactly N rolls |
| q05 | Extended Temporal | 5 | (a) [broken_rolls] counterfactual final top value had the break been correct, (b) counterfactual final top value from a different starting top face, or (c) is the current bottom value determinable, and what is it |

## 4. Die identity, rolling physics, and the invalid-roll construction

**`die.py` is the single source of truth** every other piece of ground
truth depends on, and has exactly two responsibilities:

- `fold_net(values_by_label)` folds a randomized **cross/plus hexomino**
  net (`U` above `F`; `L F R B` in a row; `D` below `F` — a fixed shape,
  only the 6 randomly-sampled 1-9 values assigned to its squares vary per
  video) into a starting `{top,bottom,north,south,east,west}` orientation,
  using the standard "wrap L-F-R-B around, U up, D down" fold: `F`→north,
  `B` (opposite end of the strip, hence opposite `F` once folded)→south,
  `L`→west, `R`→east, `U`→top, `D`→bottom.
- `roll(orientation, direction)` is the exact physical tipping rule (e.g.
  east: `new_top=old_west, new_bottom=old_east, new_east=old_top,
  new_west=old_bottom`, north/south unchanged; the other three directions
  are the analogous cyclic rule). This ONE function builds every roll
  sequence in the dataset — nothing downstream re-derives physics
  independently.

`build_adjacency(initial_orientation)` reads the die's permanent identity
directly off the three opposite pairs present in ANY orientation
(top/bottom, north/south, east/west never change *which* two values they
relate, only which spatial slot each currently occupies) — every other
value-pair is mutually adjacent, since a cube face touches exactly the 4
faces it isn't opposite to. **No "opposite faces sum to 7" shortcut
applies** — this is a custom die, and its adjacency must be read from the
values actually shown in that video's net.

**Injecting the one broken roll**: rather than fabricating an arbitrary
inconsistent state, `puzzle.build_scene` computes the correct orientation
for the broken step, then computes what `roll()` would have produced had
a *different* cardinal direction been applied from the same prior
orientation, and displays that instead (retrying decoy directions until
it's confirmed to differ from the correct result). This guarantees the
injected error is always a **genuinely reachable cube orientation** —
never an impossible 7th value or nonsensical face combination — that is
simply wrong *for the direction actually shown*, exactly matching the
README's stated limitation that errors are only catchable by
rotation-rule verification, never by spotting an invalid number. All
rolls after the break then apply correct physics starting from that wrong
state, so the sequence diverges exactly once and stays internally
consistent afterward.

**Level 5b's counterfactual** ("what if the die had started with a
different top face Y") uses `die.enumerate_orientations`, a BFS over all
24 reachable rotations of the die from its initial orientation, to
deterministically pick a canonical starting state with `top=Y` — then
re-applies the *exact same directions* used in the video, via `roll()`,
from that alternate start. **Level 5a's counterfactual** ("what if the
broken roll had been correct") needs no extra machinery: `true_roll_sequence`
is already the always-fully-correct sequence computed independent of what
was displayed, so its final top value directly answers the question.

## 5. Files

```
config.py       ranges, canvas size, palette (decorative), broken-roll probability
die.py          fold_net() + roll() + build_adjacency() + enumerate_orientations() -- the physics core, zero rendering imports
puzzle.py       grid + custom die identity + random roll path + invalid-roll injection, zero rendering imports
questions.py    builds the fixed 5-question ladder from a SceneData
render.py       PIL + OpenCV per-frame raster renderer (no Manim) -- grid, pseudo-3D die icon, trail, tipping animation
generate.py     params -> annotations.jsonl (+ raster render + final-frame PNG extraction)
flatten_annotations.py   annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
validate_die_roll_trace_dataset.py   independent roll()-reimplementation cross-check + net/adjacency re-derivation + question re-derivation + video checks
annotations.jsonl, videos/, frames/, dataset_final.{csv,jsonl}, validation_report.txt
```

## 6. Generation and reproducibility

Python 3, PIL (Pillow), OpenCV, NumPy — no other dependencies. Sample `i`
uses ONE `random.Random(i)` instance, continued (not re-seeded) across
`puzzle.build_scene` and then `questions.build_questions` — the validator
replays this exact consumption order.

```
python generate.py --n 20 --output-dir . --seed 0
python flatten_annotations.py --dataset-dir .
python validate_die_roll_trace_dataset.py --dataset-dir . --report validation_report.txt
```

`--sample` for 5 dev examples, `--no-render` for a param-only dry run,
`--append` to resume a batched/interrupted run.

## 7. Validation

`validate_die_roll_trace_dataset.py` first **independently re-implements
`roll()` from scratch** — a second, separately-written function using 3D
vector rotations about the die's own axes, rather than `die.py`'s direct
field-swapping — and cross-checks the two implementations agree across
400 randomized roll sequences (this is the single highest-value check: a
bug shared between generation and a naive self-check would otherwise
silently corrupt the whole dataset identically in both places). It then
independently re-derives `die_face_adjacency` and `initial_orientation`
straight from the stored `die_net_layout`, re-simulates the entire
`true_roll_sequence` using the independent `roll()` implementation,
confirms `displayed_roll_sequence` matches `true_roll_sequence` exactly up
to (and diverges exactly at) `broken_step_index` for `broken_rolls`
videos and matches it entirely for `valid_rolls` videos, recomputes
`top_face_value_history`/`unique_cells_visited`/`total_rolls` directly,
rebuilds the scene from a fresh `Random(seed)` to re-derive all 5
questions (including full Level 5a/5b re-simulation), and decodes every
rendered MP4 with OpenCV to check frame counts. A `--sample` batch of 5
was rendered and visually inspected: the die's pseudo-3D icon and top-face
label are legible in both mid-roll and final-hold frames, the trail line
and visit-order dot markers read clearly, and at least one `broken_rolls`
sample's injected orientation was hand-confirmed against the die's true
adjacency to be a genuinely reachable-but-wrong-for-that-direction state.

Ends with a machine-greppable `RESULT: <n_samples> samples, <n_questions>
questions, <n_mismatches> mismatches` line. Current status: **20 samples,
100 questions, 0 mismatches**; roll() cross-check 0/400 disagreements;
`content_mode` 60% `valid_rolls` / 40% `broken_rolls` against a 55/45
target (expected sampling noise at n=20); `N` spread 5/6/4/5 across 8-11.

## 8. Known limitations

- **Cell colors and cell values in the grid are purely decorative** and
  carry no mechanical meaning (no scoring, no color-triggered rules) — a
  deliberate simplification, since any scoring mechanic implied by the
  reference puzzle's example score could not be confidently
  reverse-engineered from the image alone; this dataset isolates and
  tests ONLY die-rolling physics/orientation-tracking, not any scoring
  system.
- **The die uses a CUSTOM, randomized face-value net**, not a standard
  1-6 die — face-adjacency differs per video and must be freshly
  determined from that video's net; no fixed "opposite faces sum to 7"
  shortcut applies, by design, to prevent shortcut-memorization.
- **Only one canonical cross/plus hexomino net shape is used** (only its
  6 values are randomized) — a real cube has 11 distinct valid net
  foldings; supporting all of them is a deliberate scope reduction, not
  a claim that this is the only valid net.
- **Path length is capped (15-40 rolls)** to bound video duration; no
  attempt is made to guarantee "interesting" or maximal-length paths.
- **The single injected orientation error always shows one of the die's 6
  valid values, in a genuinely reachable cube orientation** — it is
  simply the result of a different (decoy) roll direction than the one
  actually shown, never an impossible 7th value — so errors are only
  detectable by rotation-rule verification, never by spotting an
  obviously invalid number.
- **Grid size and path length are randomized independently**; this
  dataset is not an optimization or path-planning task.

## 9. Reasoning skills tested

- **3D spatial-orientation tracking** under sequential physical
  transformations.
- **Physical rotation-rule verification** — is this the physically
  correct resulting face given a real die's rolling behavior.
- **Fixed-identity inference** — deducing the always-true
  opposite/adjacent relationships of a die's faces from its net,
  independent of any specific roll.
- **Precise anomaly localization** within a physically-simulated
  sequence.
- **Full downstream re-simulation** under a hypothetical
  starting-state or correction change (Level 5a, 5b) — highly sensitive
  to a single change, requiring genuine re-derivation rather than local
  patching.
- **Certainty/inference calibration** (Level 5c) — recognizing that an
  unseen face's value is nonetheless fully determined, not unknown.
