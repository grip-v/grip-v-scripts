# laser_mirror_trace

Video-VQA sub-benchmark testing **ray-tracing/reflection-physics
tracking, path-length and direction-change counting, physics-violation
anomaly detection, and counterfactual re-tracing** — a laser beam enters
an N×N grid of diagonal ("/" or "\") mirrors and travels cell-by-cell,
reflecting at every mirror it hits, until it exits the grid (possibly
hitting a marked goal arrow) or — in principle — loops forever.

## 1. What each clip shows

- An 8×8 to 14×14 grid with ALL mirrors visible from frame 1 (unlike this
  suite's "reveal" datasets — the puzzle here is tracing the beam through
  an already-visible layout, matching how the reference puzzle is
  genuinely solved) and 3-6 goal arrows fixed along the edges, unchanged
  throughout.
- The beam animates one cell-step at a time (~0.15-0.25s/step) as a
  bright dot with a persistent trail, flashing white for a frame at
  every mirror it reflects off so direction changes are unambiguous.
- **Three content modes**: `valid_path_reaches_goal` (~55%) — the true,
  100%-correct simulated path exits exactly at a marked goal, which
  flashes green; `valid_path_no_goal_or_loop` (~25%) — the true path is
  rendered correctly but exits at a non-goal edge position (see §4 for
  why the "loop" half of this mode's name is never actually realized);
  `broken_path` (~20%) — the path is correct up to one specific mirror
  encounter, where the reflection is displayed as if that mirror were
  the OTHER orientation, and correct physics resumes from that wrong new
  direction for the rest of the video.
- **Final hold** (~18% of duration) on the completed trail.

## 2. Why this needs video, not a single frame

The final frame shows the whole trail at once — it never shows *which*
reflection was the broken one, or what direction the beam was travelling
at any intermediate point. Level 4a (pinpointing the exact reflection
that violated its mirror's true orientation) and every Level 5
counterfactual (which require re-deriving an entire downstream path from
a hypothetical change) both need the ordered sequence of reflections, not
just the held final trail.

## 3. The 5-question ladder

| qid | level | difficulty | asks |
|---|---|---:|---|
| q01 | Image Description | 1 | how many mirror cells the laser passed through |
| q02 | Basic Relational | 2 | (a) did it reach a goal, or (b) did it get stuck in a loop |
| q03 | Comparative | 3 | (a) direction after the last reflection, (b) total direction changes, or (c) entry edge |
| q04 | Compound | 4 | (a) [broken_path] which reflection was wrong + correct direction, else confirm all correct, (b) goal reached / non-goal exit / loop description, or (c) loop cycle length (or "not applicable") |
| q05 | Extended Temporal | 5 | (a) re-trace with one on-path mirror flipped, (b) confirm an off-path mirror's removal has NO effect (attention check), or (c) [broken_path] would the fully-corrected path reach the same goal |

## 4. Ground truth / the ray-tracing engine, and a real mathematical finding

`raytrace.py`'s `simulate()` is pure Python, zero rendering imports, and
is the ONLY place beam physics is implemented — reused identically for
the true path, the displayed (possibly broken) path, and every Level 5
counterfactual re-simulation. Loop detection uses the standard condition
(revisiting a `(cell, direction)` pair already seen earlier in the same
path), which is guaranteed to terminate since the state space
(cell × direction) is finite.

**A real finding from development, not assumed from the spec**: a beam
that enters from OUTSIDE the grid boundary can never actually loop
forever under this exact reflection rule. Reflection here is reversible
— if entering a cell in direction *d* leaves in direction *d′*, then
entering in the reverse of *d′* always leaves in the reverse of *d* —
which makes the full transition function (over every cell+direction
state, plus the boundary "outside" states every entry/exit connects to)
a bijection built from disjoint orbits. Boundary entries and exits sit
in short entry↔exit orbits; genuine closed cycles (confirmed to exist —
e.g. a 2×2 "pinwheel" `{(0,0):"/",(0,1):"\\",(1,0):"\\",(1,1):"/"}`
started internally at (0,0) facing "left" closes a length-4 cycle
immediately) are necessarily a SEPARATE orbit consisting only of
interior states, structurally unreachable from any boundary-entry orbit.
This was verified empirically before being trusted, not just reasoned
through: an exhaustive search over all 512 mirror configurations of a
3×3 grid, from every boundary entry point, found zero loops; 200,000
further random trials up to 7×7 at up to 100% density found zero more.
**Consequence**: `valid_path_no_goal_or_loop` videos in this dataset are
always, in practice, `exited_non_goal` — `outcome_type ==
"infinite_loop"` never actually occurs from this dataset's real
(boundary-entry) generation. `simulate()` still fully implements and
correctly detects loops (needed for the engine to be honest and
future-proof, e.g. if a later variant ever started a beam from an
interior cell), and every loop-related question (Level 2b, 4c) is kept
in the ladder as a **calibration/sanity check** — always answering
"no"/"not applicable" — rather than silently removed, exactly like
other always-true calibration questions elsewhere in this suite.

**`broken_path` construction**: `error_at_reflection_index` in
`simulate()` forces one specific reflection (counting only mirror
encounters, 0-indexed) to use the OTHER valid orientation; every
reflection before and after still follows correct physics relative to
whatever direction the beam actually has at that point — a single clean
divergence, not accumulated inconsistencies.

## 5. Files

```
config.py       colors, canvas/grid-size/duration ranges, mirror-density/goal-count ranges, content-mode weights
raytrace.py     the beam-physics engine (simulate, reflect, classify_result, loop detection), zero rendering/generation-policy imports
puzzle.py       two-pass scene assembly: picks content_mode, retries random layouts until one satisfies it, injects the broken_path error
questions.py    builds the fixed 5-question ladder, including every Level 5 counterfactual re-simulation
render.py       PIL + OpenCV per-frame raster renderer (no Manim) -- static mirrors/goals + animated beam trail
generate.py     params -> annotations.jsonl (+ raster render + final-frame PNG extraction)
flatten_annotations.py   annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
validate_laser_trace_dataset.py   independent re-simulation of true/displayed paths + every counterfactual + question re-derivation + video checks
annotations.jsonl, videos/, frames/, dataset_final.{csv,jsonl}, validation_report.txt
```

## 6. Generation and reproducibility

Python 3, PIL (Pillow), OpenCV, NumPy. Sample `i` uses ONE
`random.Random(i)` instance, continued (not re-seeded) across
`puzzle.build_scene` and then `questions.build_questions` — the
validator replays this exact consumption order.

```
python generate.py --n 20 --output-dir . --seed 0
python flatten_annotations.py --dataset-dir .
python validate_laser_trace_dataset.py --dataset-dir . --report validation_report.txt
```

`--sample` for 5 dev examples, `--no-render` for a param-only dry run,
`--append` to resume a batched/interrupted run.

## 7. Validation

`validate_laser_trace_dataset.py` independently re-simulates the true
path from `entry_position`/`entry_direction` through `mirror_layout` via
a fresh `raytrace.simulate()` call and confirms an exact match to stored
`true_path`; for `broken_path` videos, confirms `displayed_path` matches
`true_path` exactly up to `broken_reflection_index` and diverges at
EXACTLY that one reflection; recomputes `outcome_type`/
`reached_goal_position`/`loop_cycle_length` directly from
`displayed_path` (an early version of this check mis-derived a beam's
final exit direction from the last path entry's stored INCOMING
direction rather than its true post-reflection outgoing direction when
the beam reflects on its very last cell before exiting — caught by 8
false mismatches on the first full run, fixed by storing `exit_direction`
explicitly in the annotation schema rather than trying to re-derive it
from `displayed_path` alone); and rebuilds every Level 5 counterfactual
via a fresh, independent re-simulation, confirming exact matches. `--sample`
renders 5 videos for a manual watch-and-confirm pass: mirror orientations
read unambiguously, the reflection flash makes every direction change
clear, and goal arrows flash green exactly when reached.

Ends with a machine-greppable `RESULT: <n_samples> samples, <n_questions>
questions, <n_mismatches> mismatches` line. Current status: **20
samples, 100 questions, 0 mismatches** — `content_mode` 65%
`valid_path_reaches_goal`, 20% `valid_path_no_goal_or_loop`, 15%
`broken_path` against a 55/25/20 target (expected sampling noise at
n=20); `outcome_type` 13 `reached_goal` / 7 `exited_non_goal` / 0
`infinite_loop` (expected — see §4).

## 8. Known limitations

- **Only a single laser beam is traced per video**, not the full
  multi-laser board of the reference puzzle — this dataset tests
  single-beam ray-tracing and reflection tracking, not the original
  puzzle's global mirror-placement optimization problem.
- **The reference puzzle's actual scoring objective (maximize points
  across ALL lasers, accounting for mirror installation cost) is NOT
  reproduced** — this is a simulation-TRACKING task (given a fixed
  layout, trace one beam) rather than an optimization task (design the
  best layout); a meaningful narrowing of the original puzzle's scope.
- **Genuine infinite loops never occur in this dataset's actual output**
  (see §4) — a beam entering from the grid boundary provably cannot
  close a cycle under standard reversible 45° reflection; loop-related
  questions remain in the ladder as calibration checks rather than being
  removed, always answering "no"/"not applicable".
- **The single injected reflection error in `broken_path` videos always
  uses the OTHER valid mirror orientation** (a "/" shown as "\\" or vice
  versa), never an arbitrary wrong direction — a deliberate, bounded
  difficulty choice.
- **Mirror layouts are randomly generated**, not hand-designed like the
  reference puzzle's layout, and carry no guarantee of being
  "interesting" or "efficient" in any optimization sense — they exist
  purely to create traceable, goal-reaching or non-goal-exiting paths for
  this dataset's tracking task.

## 9. Reasoning skills tested

- **Ray-tracing / reflection-physics tracking** — spatial simulation
  over discrete steps.
- **Direction-change counting and reflection-event enumeration.**
- **Loop detection and cycle-length identification** (engine-complete;
  see §4 for why it's exercised as a calibration check in this dataset
  rather than a discriminative one).
- **Precise anomaly localization** within a physically-simulated
  sequence — which specific reflection violated the mirror's true
  orientation.
- **Counterfactual re-simulation under a hypothetical rule/layout
  change** (Level 5a, 5c) — requires re-deriving an entire downstream
  path, not just a local correction, since ray-tracing is highly
  sensitive to any single change.
- **Distractor/attention-check reasoning** (Level 5b) — confirming an
  irrelevant change has no effect, resisting the assumption that any
  mentioned mirror must matter.
