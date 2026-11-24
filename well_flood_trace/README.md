# well_flood_trace

Fluid levelling on a grid: water poured into one section of a grid of
sections with different depths fills it, then overflows to its lower
neighbours. Questions ask about fill order, how many sections the water
reaches, when a given section starts (or finishes) filling, and spotting a
physically impossible flow.

Shared PIL pipeline (`common/video5.py` + `common/ladder5.py`, ten-question
mode). Files: `sim.py` (scene builder, flood engine, `check`), `questions.py`
(10-question ladder), `answers.py` (independent re-derivation), `render.py`
(PIL frames), `run.py` (CLI), `test_well_flood_trace.py`.

```
python run.py generate --n 20 --seed 0 --output-dir ../output_videos/well_flood_trace
python run.py validate --dataset-dir ../output_videos/well_flood_trace          # add --skip-video for dry runs
python run.py generate --n 5000 --seed 0 --no-render --output-dir <dir>         # dry run, no video
python test_well_flood_trace.py
```

## What a clip shows (the original look)

- An N x N grid of sections (N = 5..8) on an off-white board. Every section
  shows its **depth value**, all different (a shuffle of 1..N*N), so a
  section is named by its number ("the section with depth value 17").
- Water rises from the bottom of a section as a blue overlay: **light blue
  while the section is filling, dark blue once it is completely full** (and
  passing water on). Numbers are white on full sections.
- The entry section is outlined in red. A header strip above the grid shows
  the **clock** (seconds to 0.1 s; the reading at frame f is floor(10 f / 24)
  tenths). The clock and the depth values are the only text.
- 1280x720, 24 fps, 5.6-10 s per clip.

Changes from the original renderer: the clock replaces the "simulated minute"
counter (real seconds, no accelerated playback), clips are at most 10 s, the
numbers are larger and readable over water, grid lines are 3 px, and a few
cues were added for the new scenarios (purple second entry, grey / thick
outline when the pour changes, thick wall bars). The first water in a section
always shows at least 4 px above anything covering the bottom of the cell.

## The flood rule (the original convention)

A section of depth d holds d units of water. It fills at the rate water
reaches it; the moment it is full, its whole inflow is split evenly among
**all** its neighbouring sections (up, down, left, right, not across a wall)
with a **lower** depth value. A full section with no lower neighbour stays
full and the extra water soaks away. (The original drew such a section as
filling forever, capped at 90%, which never matched its stored state; it now
fills and stays full.)

## Scenarios (`scene["scenario"]`, drawn once per seed, about 20% each)

| scenario | what happens |
|---|---|
| `valid_flow` | one entry, constant pour, everything physically correct |
| `two_entries` | a second entry, outlined in purple from the moment it starts, begins pouring 1-4 s after the first |
| `pour_change` | the pour stops (its outline turns grey) or doubles (its outline gets twice as thick); variant drawn once per seed, about half each |
| `walled` | 2-5 thick dark walls between neighbours block the spill there (at least one blocks a spill that would otherwise happen) |
| `broken_flow` | one physically impossible event, the original two kinds (about half each): `premature_fill`, where a section starts filling before any full higher neighbour could feed it (the water does reach it later in the correct flood), and `uphill_flow`, where a section receives water from a full neighbour with a lower depth value. Everything downstream follows the rule from that moment on |

## Random knobs

Grid size N (5-8); depth terrain (`random` shuffle, `cone` around a peak,
`tilt` along a random direction, with noise, ranked into 1..N*N); entry
section (one whose downhill basin holds >= 6 sections), start frame
(0.52-1.27 s) and pour rate (the entry fills in 1.2-3.2 s, rates to 0.25
units/s); second entry and its start; pour change kind and time; wall count
and placement; impossible-event kind, section, time and rate; clip length
(the longest cut <= a random 7-10 s cap with no event in the last 0.5 s);
board size (640-700 px) and horizontal position; three close variants of the
original palette.

## The 10-question ladder

Ten multiple-choice questions per clip (four distinct options each), ordered
by level, `q01..q10`. Per-level counts are drawn per seed: L1 1-3, L2 2-3, L3
2-3, L4 1-3, L5 1-3 (total 10), uniformly over the valid splits the clip can
support. Every template first builds its candidate question (or declines when
the clip can't support it with clear margins); each level's questions are
then drawn by weight without replacement, so no derivation key repeats in a
clip. Scenario-specific templates carry extra weight. Wording follows the
original questions where the original had that question.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `n_wet_end` how many distinct sections had water reach them by the end (original) · `n_full_end` how many are completely full at the end · `entry_depth` depth value of the entry · `reached_one_of_four` / `dry_one_of_four` which of four sections water reached / never reached (original "reached by end") |
| L2 relational | `earliest_of_four` / `latest_of_four` which of four sections water reached first / last · `fed_from` from which neighbour water first flowed into X · `full_when_start` which was already full when X began to rise · `first_full_of_four` which of four became full first (original "first_complete") |
| L3 measurement | `start_clock` / `full_clock` clock time X begins to rise / becomes full (original "when does X begin to rise") · `count_at_clock` / `n_full_at_clock` sections reached / full when the clock shows T (original "count at a time") · `fill_duration` seconds from X's first water to full · `start_gap` / `full_gap` seconds between two starts / two fills · `first_depth` depth value of the first section reached after the entry (original) |
| L4 integration | `order_three` order in which three sections were reached · `kth_wet` which section water reached k-th · `impossible_which` which section received water in a way that breaks the rule, or none (original "violation or confirm") · `violation_kind` what kind of impossible event, if any (original) · `wet_before_full` sections wet just before X became full · `wet_at_entry2` sections wet when the second entry started · `new_after_change` sections first wet after the pour changed · `walls_blocking` walls holding back water at the end |
| L5 counterfactual | `cf_rate` double / half pour rate (original "pour rate doubling") · `cf_move_entry` pouring into Y instead · `cf_block` X sealed · `cf_x2_which` which dry section double pour would reach · `cf_no_violation` without the impossible event · `cf_correction` would X have started earlier, later, at the same time, or not at all without the impossible event (original "counterfactual correction") · `cf_wall` one wall removed · `cf_unchanged` the pour not changing · `cf_drop_entry` one of two entries never pouring |

`impossible_which` and `violation_kind` appear in every scenario (heavily
weighted in `broken_flow`), so "none" / "nothing" is a real answer, not a
giveaway. All L2-L5 questions have `requires_motion = True`.

## Closed loop

- Every question stores its `derivation_key` and `params` (section indices,
  clock tenths, multipliers and so on, plus `min_fps`).
- `answers.py` re-derives every answer **without** importing `sim.py`,
  `questions.py` or `render.py`, and without reading the generator's stored
  track. It rebuilds the flood from the setup (grid, depths, walls, entries,
  impossible event, clip length) with **exact rational arithmetic**: sections
  are visited in a depth-first topological order of the spill graph, each
  section's inflow is a piecewise-constant function (pour pieces plus an even
  share of each upstream section's outflow), and integrating it gives the exact
  full time. `sim.py` instead uses an event-stepped float engine that
  recomputes all rates at every event. Counterfactuals are re-simulated under
  the changed setup in both.
- Answers that pick one listed section return `None` (a validator failure)
  unless exactly one option qualifies, and margin rules are re-checked there.
- `sim.check` rejects any scene with: depths that are not a shuffle of
  1..N*N; a spill cycle; a stored track that differs from a fresh simulation;
  distinct event instants closer than 3 frames, within 0.5 s of either end, or
  within 0.03 frame of a frame boundary; fewer than 5 sections reached or 2
  full; cells under 60 px; an impossible event that isn't clearly shown; a pour
  change or second entry too late to see.
- `sim.build` also dry-runs the question builder on a copy of the rng state and
  retries the scene if it can't carry a full 10-question ladder.
- `run.py validate` rebuilds every sample from its seed (scene and questions
  must match byte for byte), runs `sim.check`, the ladder check and the
  independent answers, and (without `--skip-video`) checks each MP4's size,
  duration and motion.

## Margins (no ties, no borderline readings)

- Order comparisons need >= 0.3 s between the events compared; sections that
  start at the same instant are never ordered.
- Clock readings: options >= 0.6 s apart, and only events whose first frame and
  the frame after it show the same reading.
- "When the clock shows T" counts: no relevant event within 0.3 s of any frame
  showing T.
- Full / not-full questions: every section asked about as not full is at most
  90% full at that moment (and full ones are dark blue).
- Counterfactual worlds are used only if no section starts within 0.3 s of the
  end.

## Determinism

- All randomness comes from the `random.Random(seed)` passed in; scene and
  questions come from the same stream. No global RNG, time, uuid or
  set/dict-order decisions.
- Same seed gives byte-identical scene + questions JSON across processes and
  `PYTHONHASHSEED` values (tested). Rendering is a pure function of the scene,
  so the same seed gives identical raw frames.
- Pour events sit at frame + 1/2 and all events are kept away from frame
  boundaries, so "wet at frame f" never depends on float rounding.

## Frame sampling and `min_fps`

Videos are stored at 1280x720, 24 fps, but models may see ~4 fps frames shrunk
to ~640x360, so:

- the clock is >= 40 px tall and the depth numbers >= 24 px at full size;
  both stay readable at half size;
- every question-relevant result persists: water stays, full sections stay dark
  blue, outlines keep their changed colour / thickness;
- cells are >= 65 px, lines >= 3 px, and the two blues differ clearly from each
  other and from the dry cells;
- each question stores `params["min_fps"]` = ceil(1 / smallest relevant gap),
  clamped to 1..24: the gap between the compared events, the time a needed cue
  lasts, or 0.3 s for clock readings (options 0.6 s apart). All questions are
  answerable at 4 fps; about 66% at 1 fps and 76% at 2 fps.

## Checks run

- `test_well_flood_trace.py`: determinism (in-process and across processes),
  ladder counts and `min_fps`, scene uniqueness and `sim.check`, independent
  answers agree, a tamper test (3 corrupted answers flagged exactly), three
  hand-computed 3x3 scenes (basic flood, wall + premature fill, uphill flow),
  clock edges, the rendered water appearing on its exact frame, render
  determinism, no warnings.
- 5000-seed dry run for diversity and balance, 2000-seed `validate` dry run,
  and a frame-level check of the 50 preview clips: water levels measured from
  the pixels on every frame against the exact model, first-wet and first-full
  frames, entry outlines, walls, and clock readings.
