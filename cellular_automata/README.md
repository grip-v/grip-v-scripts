# cellular_automata

Video-QA sub-benchmark: **Cellular automata (shortlist B11)**, from `ideas/New Problem Types Shortlist (2).html`. A grid of cells evolves under one hidden birth/survival rule, and the questions ask what changes, when, and why. It runs on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the 10-question, 5-rung ladder (`common/ladder5.py`, `LADDER = "ten"`).

## What each clip shows

- **1280x720, 24 fps, at most 10 s** (4.9-10 s). Generation 0 holds for 0.83-1.42 s, each later generation fades in over 2-5 frames and holds for 0.58-1.08 s in total, and the last one holds for 1.08-1.67 s.
- **The rule is hidden.** An empty cell fills when its number of filled neighbours is in the rule's B set, and a filled cell stays filled when its count is in the S set.
- **The only text is the timer** (top-left, seconds to 0.1 s, 40 px; readable at half size). The grid always sits inside the box x 250-1250, y 30-700, so it never touches the timer. Nothing is labelled. Questions name things by colour ("navy blue cells"), by marker ("the cell with the red ring", "the column under the orange triangle") and by timer reading ("at 3.4 s").

## Scenarios (`scene["scenario"]`, about 20 % each)

| scenario | neighbourhood / states | rule pool | what you see |
|---|---|---|---|
| `life_like` | 8 neighbours, cells beyond the edge are empty, 2 states | 15 Life-like rules (Life, HighLife, Seeds, Day & Night, Maze, 2x2, Life without Death, Diamoeba, Morley, DryLife, Pedestrian, Coral, Amoeba, 34 Life, Long Life) | the grid changes in place |
| `torus` | 8 neighbours on a wrap-around grid | same 15 | dashed border. The start pattern sits on an edge, and `check` requires the wrap to change the outcome |
| `von_neumann` | 4 side neighbours, edges empty | 14 rules over counts 0-4 | the grid changes in place |
| `generations` | 8 neighbours, 3 states | 8 rules (Brian's Brain, Star Wars-like, ...) | a filled cell that fails to survive turns a second colour for one step, then empties |
| `elementary` | 1-D row, the 2 cells beside it in the row above | 24 B/S rules over counts 0-2 | a space-time diagram: each generation is drawn as a new row under the previous ones, with marker shapes above the marked columns |

## Random knobs (all from the seeded `random.Random`)

These knobs are set per scene:

- **Rule:** drawn from the scenario's pool.
- **Grid size:** 9-15 rows by 11-22 columns, or 21-41 columns in the elementary scenario.
- **Generations:** 6-12.
- **Start layout:** patch, two patches, mirror-symmetric patch or scatter. The elementary scenario uses single seed, a few seeds, random row or block. Density also varies.
- **Timing:** start, generation, fade and end lengths, all counted in whole frames.
- **Theme:** light or dark, with 7 backgrounds and 6 dead-cell tones.
- **Colours:** 11 colours for filled cells, plus 9 for fading cells in the generations scenario.
- **Cell shape:** square, rounded or circle.
- **Cell size and framing:** cell size from 24 to 51 px, and the grid's position within the grid box.
- **Markers:** 2-3 marked cells, each with a distinct colour and a distinct shape. The 2-D shapes are ring, square frame and diamond frame. The elementary shapes are triangle, dot and square.

Marker colours are at least 110 RGB units away from the cell, fading and background colours, and no marker shares its colour name with them.

Over 5000 seeds, the scene signatures (scene params without the retry counter) are 100 % unique. The run covers 67 scenario/rule pairs and 7 start layouts.

## The 10-question ladder

Per-level counts come from `L5.ten_counts(rng)`: L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3, 10 questions in total. Questions are ordered by level with qids q01..q10. Within a level, templates are picked round-robin over shuffled derivation keys, so the 10 questions vary from clip to clip. No (derivation_key, params) pair repeats. Every question has 4 options, and the answer letters are balanced (about 25 % each).

- **L1 observation:** `count_live` (filled cells at the start or end), `markers_filled` (which of two marked cells is filled at the start or end; both / neither / only X), `count_fading` (generations scenario).
- **L2 relational:** `more_live` (which of two moments has more filled cells, with a clear margin), `cell_change` (what happens to a marked cell between two consecutive moments), `trend` (start vs end), `marker_longer` (which marked cell is filled in more generations), `marker_first` (which marked cell changes colour first).
- **L3 measurement:** `pop_at` (count at a timer reading), `n_born` / `n_lost` (cells that fill or stop being filled between two consecutive moments), `peak_moment` (which of 4 moments has the most filled cells, with a clear margin), `marker_count` (in how many generations a marked cell is filled), `first_change_time` (timer reading when a marked cell first changes).
- **L4 integration:** `rule` (which birth/survival rule; distractors favour rules that share B or S), `always_filled`, `ever_filled`, `total_born`, `empty_time`, `biggest_change_time` (the step that changes the most cells, with a clear margin).
- **L5 counterfactual** (all re-simulated): `cf_next` (count 1 or 2 steps after the clip), `cf_markers_next` (which marked cells would be filled one step later), `cf_rule` (count after 1-2 steps from the same start under a different described rule; never equal to the real count), `cf_toggle` (count after 1-2 steps if one marked cell had started flipped; never equal to the real count).

Counts asked about are at most 40. Comparisons need a margin of at least 2 cells and at least 10 %. All L2-L5 questions except `cf_rule` have `requires_motion=True` (94 %).

## Frame-sampling rules and `min_fps`

- Moments in questions are timer readings whose every frame shows one still generation, at least 2 frames from any change and at least 0.5 s from both ends of the clip (`sim.time_ref`). The first change comes at least 0.83 s in, and the last at least 1.08 s before the end. Changes are at least 14 frames (0.58 s) apart.
- Every change leaves a persistent result: a generation stays on screen until the next one, and elementary rows stay drawn. There are no single-frame flashes. Every 2-D step changes the grid, so no generation is invisible. Cells are at least 24 px, marker lines 4 px, and the timer font 40 px.
- `params["min_fps"]` = 24 / (shortest still window, in frames, among the generations the question relies on), rounded up to 0.1. Over 5000 seeds, about 26 % of questions are answerable at 1 fps, about 96 % at 2 fps and 100 % at 4 fps.

## Closed loop

- Every question stores its `derivation_key` and `params`. Moments are stored as tokens: `"start"`, `"end"` or `"3.4"`.
- `answers.py` re-derives every answer independently. It does not read `scene["history"]` or call `sim.step`. It replays the generations with a sparse dict / Counter method, where `sim.py` uses dense numpy shifts. It maps moments to generations through the renderer's per-frame state (`render.gen_at_frame`) and the timer text it draws. It answers the rule question by fitting every rule in the pool.
- `sim.check` rejects a scene in any of these cases:
  - the stored history differs from a replay;
  - another rule in the pool fits every transition;
  - a 2-D step leaves the grid unchanged;
  - the start is too small, or the pattern dies at once;
  - more than 60 cells are filled;
  - a torus scene never uses the wrap;
  - the timing is too tight, or some generation has no clean time reference;
  - markers are ambiguous, too close together, or never change;
  - some level has fewer than 3 distinct candidate questions.
  - `build` retries with the same rng (about 5 tries per scene on average; this is deterministic).
- `python run.py validate` rebuilds every sample from its seed and compares the result. It also runs `sim.check` and the ladder checks (counts per level, order, no repeated key + params, options, `min_fps`), re-derives every answer, and checks the MP4 duration, resolution and motion.
- `frame_check.py` decodes the MP4s and checks them against the sparse replay:
  - it reads every cell's colour in every generation, plus the frames just before and after each change;
  - every frame named by a timer reading must show one still generation;
  - timer pixels must match the stated time, including the frames of time-valued answers;
  - each marker must be visible in its colour.
- `test_cellular_automata.py` covers:
  - determinism, both in-process and across `PYTHONHASHSEED` values, and frame-hash determinism;
  - the ladder, `min_fps` and option counts;
  - scene uniqueness;
  - answer agreement;
  - the tamper test (3 corrupted answers give exactly 3 flags);
  - hand-checked scenes: a blinker in Life, a torus blinker that wraps, a Brian's Brain pair, and an elementary XOR (rule 90) diagram whose rule `check` correctly rejects as ambiguous.

## Determinism

Sample `i` uses `random.Random(seed + i)` for the scene and then the questions. Nothing else is random: there is no global RNG, no time and no uuid, and no set or dict iteration order decides anything. All timing is in integer frames. The same seed gives byte-identical scene, question and answer JSON in different processes, and identical raw frames.

## Files and commands

```
sim.py            scenarios, rule pools, dense dynamics, timeline, check (pure, seeded)
questions.py      the 10-question ladder (templates per level, phrasing helpers)
answers.py        independent sparse re-derivation (used only by the validator)
render.py         PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py            CLI: generate / validate (shared pipeline in common/video5.py)
frame_check.py    pixel-level check of rendered clips against the sparse replay
test_cellular_automata.py

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/cellular_automata
python run.py validate --dataset-dir ../output_videos/cellular_automata
python frame_check.py --dataset-dir ../output_videos/cellular_automata
python -m pytest test_cellular_automata.py
```
