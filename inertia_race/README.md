# inertia_race

Video-QA dataset on **rotational inertia**: bodies of different shape released on a
common inclined ramp, racing to a finish line.

A body rolling without slipping down a ramp obeys

```
a = g sin(theta) / (1 + k)        k = I / (m R^2)
```

so how fast it gets down depends on how its mass is distributed and on nothing
else — **mass and radius cancel exactly**. A thin ring (k = 1) always loses to a
solid ball (k = 2/5), and two discs of different size always dead-heat. That is
the thing this category tests, and it is the thing physical intuition most often
gets backwards.

Clips are 1280x720, 24 fps and 9.5 s, rendered with PIL on the shared pipeline
(`common/video5.py`). Each clip has 10 questions on the 5-rung ladder
(`common/ladder5.py`, `LADDER = "ten"`) plus one separate `L6_consistency` item.

## What is on screen

- **Ramp:** one lane per body, drawn as parallel inclined slabs at a shared angle
  of 15-27 degrees, with a shaded under-edge so they read as separate ramps.
- **Ruler:** along the lowest lane, marked in fractions of the ramp. Gridlines
  every 0.05 (darker every 0.25) cross every lane and meet the ruler.
- **Finish line:** a red line at 1.00, across every lane.
- **Start marks:** in `handicap` and `pursuit`, a white block at each body's start.
- **Bodies:** a sliding block, a solid ball, a solid disc, a hollow shell and a
  thin ring (whose middle is open, so the lane shows through). Each carries white
  radial spokes and a dark rim pip, so its rotation can be counted frame by frame.
  The block never spins.

## Scenarios (`scene["scenario"]`)

| scenario | what happens | racers |
|---|---|---|
| `fair_start` | all released together from the top, so finishing order is shape order alone | 2-3 |
| `handicap` | released together from different start marks, so order is not shape order | 2-5 |
| `stagger` | released from the top at different, timer-aligned moments | 2-5 |
| `pursuit` | the slower shapes start further down, so at least one overtake happens mid-clip | 2-4 |
| `twin` | two bodies of the **same shape** at clearly different radii, which dead-heat | 3-4 |

`twin` is the direct probe that radius does not matter. When its planted violation
lands on one of the twins it is always `slip`, which leaves the dead heat intact
while the spin counts diverge — one clip carrying two independent probes.

## Separability

Finishing times scale as `tau = sqrt(1 + k)`: 1.000 / 1.183 / 1.225 / 1.291 / 1.414.
A solid ball and a solid disc are 0.042 apart in tau, so for them to finish 0.7 s
apart the race would have to run 16.9 s — past the clip cap. Requiring **every**
finisher to be separated therefore admits no 4- or 5-body field at all.

So separation is enforced **per pair, not globally**, as `robot_long_jump` does:
`sim.check` requires a clear winner and at least one separated pair, and
`questions.py` filters each pair at candidate time against `ORDER_GAP_F`. Start
offsets and staggered releases then decouple finishing order from shape entirely,
which is also what stops "read the answer off the shapes" from working.
`FEASIBLE_SETS` still enumerates the shape sets that separate on their own; only
`fair_start` and `twin` draw from it.

## Random knobs (all drawn per seed)

- **Bodies:** a unique colour from 7 (never red), a pattern, and a drawn radius
  from a set that narrows as the field grows (40-64 px at two bodies, 20-28 at five).
- **Ramp:** angle from 15/18/21/24/27 degrees; the on-screen run and the lane
  spacing are then solved against the canvas budget rather than drawn, so the
  widest lane stack that still leaves 520 px of ramp is always used.
- **Race:** `T0` (the time a frictionless block would take from the top), start
  offsets, release frames.
- **Violations:** `wrong_inertia` (a body accelerates as if it had another shape's
  inertia) or `slip` (it turns through 0.55 of the angle rolling would give).

Over 2000 seeds (`--no-render`):

- **Uniqueness:** every scene signature is unique (2000/2000); 1992/2000 line-ups.
- **Scenario mix:** 388 fair_start, 475 handicap, 457 stagger, 348 pursuit, 332 twin.
- **Questions:** 8457 distinct (question, answer) pairs.
- **Retries:** 27.1 per scene on average, 72% of seeds needed at least one.
- **Violations:** 49.4% of clips (533 wrong_inertia, 455 slip).

## On-screen text

Besides the timer, the only text is the five ruler numbers **0.00, 0.25, 0.50,
0.75, 1.00**.

- **Why they are needed:** the measurement questions ask for positions and gaps in
  ruler units. Unnumbered gridlines give no absolute scale.
- **They never give an answer:** no body is labelled, nothing names a shape, and
  there is no title or caption.
- **Storage:** labels live in `scene["labels"]` (text, x, y, size). The renderer
  draws them from there and `answers.py` converts pixels back to ruler units
  through them — projecting onto the axis the 0.00 and 1.00 labels define, which
  cancels the lane offset, so one conversion serves every lane.
- **Timer:** a dark box top-left, inside a reserved 84 px strip no body ever enters.

## The 10-question ladder

- **Counts:** from `L5.ten_counts`: L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3, ten in all.
- **Naming:** bodies are named by **colour only** ("the teal body"), never by shape,
  so no prompt leaks the answer to a shape question.
- **Mid-race questions:** every one names a clock instant, and that same instant is
  the one used to score it. Named instants are always timer-aligned and at least
  0.5 s from both ends.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `shape_of`, `n_racers`, `not_finished`, `widest`, `starts_lowest` |
| L2 relational | `first_of_pair`, `same_finish`, `ahead_at_t`, `more_turns_by_t`, `closing_at_t`, `passes_before_t` |
| L3 measurement | `pos_at_t`, `gap_at_t`, `finish_time`, `release_time`, `turns_by_t` |
| L4 integration | `finish_order`, `place_of`, `win_margin`, `leader_at_t`, `crossed_by_t` |
| L5 counterfactual | `cf_gravity_time`, `cf_steeper_time`, `cf_from_top`, `cf_as_shape`, `cf_all_together` |

Four of the six L2 templates are pairwise **and** time-grounded (`ahead_at_t`,
`more_turns_by_t`, `closing_at_t`, `passes_before_t`), so one clip with five
bodies offers hundreds of L2 candidates rather than a handful.

- **Counterfactuals** are all numeric and clip-specific, and never target a body
  that is itself breaking the rules — such a question is ill-posed, and could not
  be re-derived from its drawn shape either.
- **Motion share:** 83% of all questions and 100% of L2-L5 carry
  `requires_motion`. 22% are answerable at 1 fps, 59% at 3 fps, 100% at 4 fps.

### Balancing the candidate pool

Several of these questions have an answer distribution that is lopsided by physics
rather than by design: a quicker body nearly always draws away, a body has nearly
always turned once by mid-race, finishers nearly always arrive apart. No
precondition fixes that, and a blind model that always answers the common case
scores well above chance.

So the pool is **balanced before the draw**: for each template, `_balance` keeps the
same number of candidates for every distinct answer that template can give in this
clip (`questions._sig` computes that answer cheaply; a test asserts it partitions
candidates exactly as the groundtruth does). Templates that offer one candidate per
clip cannot be balanced this way and were dropped rather than kept at a known bias.

Measured over 2000 clips / 20000 questions:

| adversary | accuracy |
|---|---|
| chance (1 / options) | 28.45% |
| always the first option | 28.34% |
| always the longest option | 26.75% |
| always the shortest option | 32.02% |
| **template-majority prior** (no video at all) | **33.92%** |

No single template's majority answer beats its own chance rate by more than 11
points. Before balancing the same measurement read 47.65%.

### The L6 consistency item

One identical extra question per clip, outside the ten:

> Watch the whole clip. Does every body move the way a real object of its shape
> would when rolling down a ramp, or is at least one of them breaking the rules of
> rolling motion?

It is deliberately **not** on the L1-L5 ladder: deciding it needs shape, motion,
spin and finishing order together, and `common/ladder5.py` requires the graded
ladder to be exactly ten. It lives at `scene["l6"]` and is answered through the
same closed loop (`answers.answer("l6_consistency", ...)`). Realised split over
2000 clips: **50.6% consistent / 49.4% violated**, so chance is 50%.

`answers.py` decides it with two checks, both read off the drawing: the turns a
rolling body makes must match the ground it covered, and every pair's fitted
accelerations must stand in the ratio their drawn shapes imply.

## Frame rules

- **Duration:** 228 frames (9.5 s) exactly.
- **Edges:** the first finish is no earlier than 2.2 s; no finish is within 0.5 s
  of either end.
- **Timer alignment:** the first finish is always on a timer-aligned frame; release
  frames are in `stagger` and `pursuit`. Clock-reading questions are asked only on
  aligned frames.
- **Event spacing:** any two finishes compared in time are at least 8 frames apart.
  A lead change counts only once the new leader is clearly ahead (0.02 of the ramp)
  and has held it for 3 frames — without that, two bodies running abreast register
  phantom overtakes, and the renderer's measurement of the same moment disagrees.
- **Near misses:** a body that ends within 0.03 of the line is rejected, because it
  cannot be told apart from one that crossed.
- **No overlaps:** lane pitch is at least the drawn diameter plus 14 px, and
  `sim.check` asserts the whole drawing stays inside the canvas and clear of the
  timer strip, on every scene.

## Closed loop

`answers.py` never calls `sim.py` and never reads the generator's per-body facts
(`t0`, `ramp`, `shape`, `k_shape`, `start`, `release_f`, `finish_f`, `violates`).
A test enforces this by parsing the file: the only scene keys it may touch are
`labels` and `n_frames`. Everything else comes from replaying `render.layout` —
the same per-frame geometry `render.frame` draws:

- position, by projecting a body's contact point onto the ramp axis through the
  ruler labels, then snapping to a 0.05 gridline;
- release and crossing frames, from where that position starts and stops changing
  (a finished body is drawn at the line, so its position stops moving exactly then);
- turn count, by unwrapping the drawn rotation angle — so a body that slips is
  measured as drawn, not as rolling would imply (max 33 degrees per frame, well
  inside Nyquist);
- acceleration, fitted from the observed trajectory, which is what the
  counterfactuals are re-solved from.

The renderer likewise keeps its own replay of the kinematics and never imports
`sim`, so generator and renderer agreeing is a test result, not a tautology: over
40 scenes and every frame, positions agree to 1.1e-6 of the ramp and turn counts
exactly.

## Validation

- `python run.py validate` rebuilds every clip from its seed, diffs scene and
  questions, runs `sim.check` and the ten-question ladder check, re-derives every
  answer independently and, without `--skip-video`, decodes the MP4.
  **2000 clips / 20000 questions: 0 mismatches.** With video over 20 clips: 0.
- The closed form is checked against numerical integration in metres: positions
  agree to 2.9e-5 of the ramp and finishing frames exactly.
- `test_inertia_race.py` covers determinism across `PYTHONHASHSEED`, the ladder,
  uniqueness, the closed loop, the forbidden-read rule, labels against the
  renderer, canvas bounds, rotation aliasing, the radius-cancellation property,
  the balancing signature, and tamper tests on both a scene and an answer.

## Cost

25 s per clip end to end (19 s rendering, the rest encoding), single-threaded;
`common/video5.encode` has no worker pool. A 50-clip preview is about 21 minutes;
3000 clips is about 21 hours on one core. MP4s are 142-300 KB.

## Files and commands

```
sim.py           scene generation, the ramp physics, checks (pure, seeded)
questions.py     the 10-question ladder, with pool balancing
answers.py       independent answer re-derivation from the renderer's geometry
render.py        PIL renderer + per-frame layout (1280x720, 24 fps, piped to ffmpeg)
run.py           CLI: generate / validate (shared pipeline in common/video5.py)
test_inertia_race.py

python run.py generate --n 50 --seed 0 --output-dir ../output_videos/inertia_race
python run.py validate --dataset-dir ../output_videos/inertia_race
python -m pytest -q test_inertia_race.py
```