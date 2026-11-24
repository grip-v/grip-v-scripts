# die_roll_trace - changes

Requests:

> Add dice outside the board, make it roll, add same stars.
> 2-3 objects associated with single shape or different shape.

Follow-up (call with the lead):

> The die should move to the numbered cell. The number on top causes it. If
> the number is not in the neighbourhood, it does not move.
> The die on the board should be a plain blue die, no numbers.

Second follow-up:

> The die outside the board should display a number/digit, natural and in
> line with the rest of the design. Keep the die on the board as it is.
> All questions follow the five closed-loop questions: a shared ladder from
> direct observation to relational reasoning, measurement, multi-step
> integration, and counterfactual reasoning.

Third follow-up:

> Remove the die outside the board. In its place show just the digit (the
> number that came up), nothing else.

Fourth follow-up:

> Keep this version (the step-count variant that was tried afterwards is not
> used). The digit on the right should show the number before the die lands
> on it. Videos should be about 10-15 seconds, 10 preferred.

This dataset is used to evaluate vision-language models (VLMs), so everything
an answer depends on must be readable from the video frames.

`README.md` is unchanged. This file lists every change, where it is in the
code, how it was checked, and where the videos now differ from the README.

In the code, every changed spot is marked with a comment starting with
`CHANGE:`. To list them all:

```
grep -n "CHANGE:" config.py puzzle.py generate.py render.py questions.py validate_die_roll_trace_dataset.py
```

## 1. The number rule

The die always shows the number of the cell it stands on.

- **Start.** The die starts on a cell whose number equals its top face.
- **Each roll.** Of its side neighbours (up, down, left, right), exactly one
  is a cell where rolling that way brings that cell's number to the top. The
  die rolls there, so it again shows the number of the cell it stands on.
  Diagonal cells never count: a die can only roll over an edge.
- **Never straight back.** The cell the die just came from does not count.
  Rolling straight back undoes the last roll, so it would always bring that
  cell's number up again: without this condition there would always be two
  valid moves and the die could never stop.
- **Dead end.** When no direction works, the die stops. The path ends there
  and the video ends after the usual final hold.
- **Exactly one choice.** The grid is built so that at every step exactly one
  direction works. The move is never ambiguous.
- **No loops.** The die never comes back to a cell with the same orientation
  (it would then repeat the same moves forever).
- **Path length.** Paths are 8-10 rolls (was 15-40), so videos are about
  10-13 s long (9.8-13.4 s, mean 11.5 s; was 18-54 s). The path and the grid
  numbers are planned together (`puzzle._plan_number_path()`).
- **Broken rolls.** About 45% of videos contain one physically wrong roll. On
  it, the die lands showing a wrong face, so on that one landing the top does
  not match the cell number (the cell number is the physically correct top).
  After it, the die carries on from the wrong state, following the rule.
- **Grid numbers now matter.** Before, the numbers were decoration (README
  section 8). Now they decide the path.

## 2. What changed in the videos

- **Top number shown as a large digit.** The frame is 1280x720. The board is
  on the left (720x720); the right-hand panel (560x720) shows one large blue
  digit: the number on the die's top face. While the die sits still it shows
  the current top; as soon as the die starts tipping it switches to the number
  the die will land with, so the number is shown before the die lands on that
  cell. Nothing else is in the panel.
- **The die rolls.** Each roll tips the die 90 degrees over its leading bottom
  edge, like a real die. Before, it slid from cell to cell.
- **Small board die: plain blue.** No digits and no shapes. It only shows
  where the die is and how it rolls. The top value is read from the digit panel.
- **Bigger grid numbers.** Each number is about 2.6 times taller than before,
  and sits in the top-left corner of its cell, clear of the trail line. They
  stay readable when frames are shrunk to the sizes VLMs typically use (for
  example 448x252).
- **0.5 s pause before every roll.** The die sits still for 0.5 s showing its
  top face (at the start, and after every landing), then tips over in 0.5-0.6 s.
  Before, each roll took 0.30-0.40 s with no pause.
- **Board die off-centre.** The small die sits 0.17 of a cell right of the
  cell centre, so the cell's own number (top-left corner) stays visible next
  to it.
- **Faces do not jump on landing.** The die's full 3D rotation is carried from
  roll to roll, so each face keeps its pattern and angle when the die lands.
- **Colours.** Blue die and blue digit (the old die was cream).
- **Board position.** The board moved from the centre of the frame to the
  left, to make room for the digit panel. Grid size and colours are unchanged.

## 3. Questions: the 5-step ladder

The five questions form one closed-loop ladder, from direct observation to
relational reasoning, measurement, multi-step integration and counterfactual
reasoning. Each video gets one question per level, in this order. Each level
picks one of its variants per video. Every answer is closed-form: the
validator re-derives it from the stored scene, and `verify_ladder.py` works it
out again independently.

| qid | Level | Variants (one per video) |
|---|---|---|
| q01 | 1 Direct Observation | top number at the start; top number at the end |
| q02 | 2 Relational Reasoning | direction of the first roll; did the die return to a cell; number opposite the start top |
| q03 | 3 Measurement | number of rolls; number of different cells; how often number X landed on top |
| q04 | 4 Multi-step Integration | find the impossible roll and the correct top (or confirm all rolls are correct); top number after N rolls; number on the bottom at the end |
| q05 | 5 Counterfactual Reasoning | one extra tip before the start; one roll in a different direction; the broken roll done correctly (broken-roll videos only) |

Level names in the output (`task` field) are now `Direct Observation`,
`Relational Reasoning`, `Measurement`, `Multi-step Integration`,
`Counterfactual Reasoning` (were `Image Description`, `Basic Relational
Reasoning`, `Comparative Reasoning`, `Compound Reasoning`, `Extended Temporal
Reasoning`).

Every question can be answered from the video. The video shows only the top
number, so the side and bottom faces are never seen directly; they have to be
worked out from how the top number changes as the die rolls. For each video
the question builder lists every die that explains what is shown (for a
broken-roll video: every die plus one broken roll), and asks a question about
hidden faces only when all of them give the same answer. Without this, about
1 question in 120 had more than one possible answer (mostly "which roll was
the impossible one").

Fixes to individual questions:

- **Single-option questions.** The broken-roll question and the bottom-number
  question offered only ONE option, so the answer was always "A". Both now
  have 4 options. Broken roll: the value actually shown (the natural mistake)
  and nearby roll numbers. Bottom number: "no, it cannot be determined", the
  top number (the usual mix-up) and one side number.
- **"Started with a different face on top" had no single answer.** It only
  named the new top number, but a die with that number on top can sit four
  ways, and in 68% of cases those gave different answers. It now states the
  start exactly: one extra tip in a named direction before the first roll.
- **New counterfactual variant:** "suppose roll #K had gone a different
  direction". Videos without a broken roll now have two counterfactual
  variants instead of one.
- **Moved to fit the ladder:** "number of rolls" is now Measurement (was
  Image Description); "number on the bottom at the end" is now Multi-step
  Integration (was in the last level, but it is not a counterfactual).

## 4. What did not change

- **Physics:** `die.py` is unchanged. The die moves exactly one cell per roll,
  and `roll()` decides the new top face.
- **Question format:** still 5 multiple-choice questions per video, q01-q05,
  with the same fields. `flatten_annotations.py` is unchanged.
- **Output files:** `annotations.jsonl`, `dataset_final.csv` and
  `dataset_final.jsonl` have the same fields and columns as before.

Because the path now comes from the number rule, a given seed produces a
different grid, path and answers than the original dataset did. The original
samples cannot be compared one-to-one with new ones.

## 5. Changes by file

### config.py

| Setting | Change |
|---|---|
| `STEP_DURATION_S_RANGE` | (0.30, 0.40) -> (1.00, 1.10) seconds per roll (pause + tipping) |
| `PATH_LEN_RANGE` | (15, 40) -> (8, 10) rolls, for videos of about 10-13 s |
| `REST_S` | New. The die sits still this long before every roll (0.5 s) |
| `MAX_PATH_ATTEMPTS` | New. Retries allowed when planning a path that follows the number rule |
| `DIE_FACE_HEX`, `DIE_INSET_HEX`, `DIE_EDGE_HEX` | New blue die colours. Replace `DIE_BODY_HEX`, `DIE_TOP_HEX`, `DIE_SIDE_HEX`, `DIE_LABEL_HEX` |
| `TOP_DIGIT_HEX`, `TOP_DIGIT_PX` | New. Colour and size (300 px) of the top-number digit in the right-hand panel |
| `TASK_NAME` | The five level names now follow the ladder (section 3) |

### puzzle.py

- The random-walk path is replaced by the number rule (section 1): the die
  always shows its cell's number, never rolls straight back, and never loops.
  New function `_plan_number_path()` builds the path and the grid numbers
  together, retrying if an attempt gets stuck. Results are deterministic per
  seed.
- `content_mode` and the broken roll are now chosen before the path, because
  the wrong top after a broken roll decides where the die goes next. The wrong
  orientation is picked the same way as before (the result of rolling a
  different direction).

### render.py

Rewritten. The grid, trail, frame timing and final-frame export work as
before. New parts:

| Part | What it does |
|---|---|
| `BOARD_PX`, `PANEL_W` | Frame split: board left, top-number panel right |
| `NUMBER_FONT_FRAC`, `NUMBER_OFFSET` | Bigger grid numbers, top-left of each cell |
| `BOARD_DIE_OFFSET` | Board die sits 0.17 cell right of centre so its cell's number stays visible |
| `_roll_pose()` | Tips the die over its bottom edge (replaces sliding) |
| `_state_rotation()` | Works out the die's full 3D rotation from which value is on which side. Faces keep their angle from roll to roll, so nothing jumps on landing |
| `_CubeSprite` | Draws the board die as a plain blue 3D cube |
| `_Painter` | Paints the grid once per clip; each frame only adds the trail, the die and the top-number digit |
| `render_clip()` | Each roll: the die sits still for `REST_S` (0.5 s), then tips over |

### questions.py

- `build_questions()` rewritten as the 5-step ladder (section 3).
- New helpers `_side_rng()` and `_broken_roll_distractors()` for the
  multi-option fixes.

### generate.py

- New option `--workers N` renders several videos at once (default: all CPU
  cores). Samples are written in the same seed order and are byte-identical to
  a one-at-a-time run.

### validate_die_roll_trace_dataset.py

- New check `number_rule_errors()`: the start cell's number is the start top;
  for every roll exactly one direction (not straight back) brings its cell's
  number up and the die went there; after every landing except the broken roll
  the die shows its cell's number; the die stops only where no direction works;
  the die never loops. Every validator run now checks this.

### New scripts

| Script | What it checks |
|---|---|
| `verify_number_rule.py` | The number rule and no loops, over many seeds (default 3000) or on a generated dataset (`--dataset <dir>`). Uses only the saved grid, path and tops, not the path planner |
| `verify_answerable.py` | Every question about hidden faces has exactly one answer given only what the video shows (top numbers + moves), and it is the stated answer. Own die-rolling table and own search |
| `verify_ladder.py` | The 5-step ladder: levels, names and variants, and every answer worked out independently (own die-rolling table); exactly one option correct. Over many seeds (default 3000) or on a generated dataset |
| `verify_questions.py` | The q04/q05 fix over many seeds (default 5000): 4 different options, exactly one correct, and it is the stated answer. The correct answer is worked out independently. `--original <dir>` also compares with an old `questions.py` |

## 6. Where the videos differ from README.md

| README says | Now |
|---|---|
| Section 1: grid cells are "decorative"; section 8: "cell colors and cell values in the grid are purely decorative" | Cell colours are still decorative. Cell **values** decide the path (section 1 above) |
| Section 1: the die tips "in a random cardinal direction (revisits allowed)" | The direction follows the number rule (section 1 above); revisits can still happen |
| Section 8: "Grid size and path length are randomized independently" | Still true for size and length; the grid numbers are planned together with the path |
| Section 1: small die is a "pseudo-3D cube icon ... with its TOP face's value always clearly legible" | Small die is a plain blue 3D cube. The top value is shown as a large digit in the right-hand panel |
| Section 1: die moves as "smooth interpolated motion between cell centers" | Die tips over its edge from cell to cell |
| Sections 1 and 5: one die, on the board | Still one die; a right-hand panel shows its top number as a digit |
| Section 5: `render.py` draws a "pseudo-3D die icon" | `render.py` draws both dice with a 3D cube renderer |
| Section 6: generation commands | Same commands still work; `--workers N` added |
| Section 7: validation steps | Validator also checks the number rule |

## 7. Validation

All results below are from the final code (digit panel, number rule, ladder).

### Validation run: 20 videos, seeds 0-19 (188 rolls, 10.1-13.4 s each)

| Check | Result |
|---|---|
| Validator (`validate_die_roll_trace_dataset.py`), including the number-rule check | 20 samples, 100 questions, 0 mismatches, 0 errors |
| Number rule on the saved files (`verify_number_rule.py --dataset`) | Pass: the die always shows its cell's number (except the broken roll), each roll is the only valid move, every video ends on a dead end, no loops |
| 5-step ladder (`verify_ladder.py --dataset`) | Pass: levels, names and variants correct; every answer re-derived independently |
| Answerable from the video (`verify_answerable.py --dataset`) | Pass: every question about hidden faces has exactly one answer given only the top numbers and moves |
| Board die never covers its cell's number | Pass |
| 3D math: die fully tipped equals die as it lands | Pass, 188 rolls |
| Drawn images: last rolling image vs landed image (board die) | 0 pixels change on every roll |
| Decoded video: panel digit read from the video in the last pause frame before every roll (current top), in the first tipping frame and mid-roll (already the new top), and on landing (new top); board die moves one cell in the rolled direction; die still during every 0.5 s pause | Pass, 188 rolls |
| Frames sampled at 2 per second, shrunk to 448x252 (typical VLM input) | Every landing caught at rest (188/188); panel digit read correctly in 188/188 |
| Output fields and CSV columns vs original dataset | Identical |

### Many-seed checks (no rendering needed)

| Check | Result |
|---|---|
| Number rule and no loops, 3000 seeds | Pass. Paths 8-10 rolls (mean 9.0); videos 9.8-13.4 s (mean 11.5 s); 45% broken_rolls; 0 loops |
| 5-step ladder, 1000 seeds | Pass. 5000 questions; every variant appears; correct letters A 1374, B 1347, C 1127, D 1152 (C/D lower because yes/no and true/false questions have only A/B) |
| Answerable from the video, 300 seeds | Pass. 595 hidden-face questions |
| Multi-option fixes, 5000 seeds | Pass. Every broken-roll and bottom-number question has 4 different options, exactly one correct |

### Checks of the checks

Each check was also run against a deliberately wrong input and correctly
failed:

| Wrong input | Result |
|---|---|
| Wrong expected top number (the old one) at every landing | Video check fails at every roll (376 failures: landing and tipping) |
| Older videos where the digit switched at landing | Video check fails: the digit is not yet the new number while the die tips |
| Questions built without the answerable filter (600 videos) | `verify_answerable.py` flags the 10 that have more than one possible answer |
| Questions in the old (pre-ladder) format | `verify_ladder.py` fails all 100 |
| Old random-walk videos, checked for the number rule | `verify_number_rule.py` rejects all of them |
| Videos made before the loop fix | Both rule checks flag the 2 videos (of 20) that loop |
| Board die drawn centred (old position) | Number-cover check fails at all 600 positions |
| Original single-option questions | `verify_questions.py` fails on all 181 fixed questions in 300 seeds |
| Old renderer (faces jump on landing) | Drawn-image check fails |
| Reversed roll direction | Video check fails on all rolls |

### By eye

- Landing frames from the 20-video run: the panel digit matches the number of
  the cell the die lands on; on the broken roll it does not (the visible
  impossible moment). The cell's number stays visible next to the die.

### Code comments

Where `CHANGE:` comments were added to code that was already finished, the
code was parsed before and after and is identical (apart from one added line
in the usage text of `generate.py`). The number rule, plain board die, bigger
numbers and validator check were written with their comments from the start.

### Not checked

- The digits drawn on the grid are not read back from the video (the drawing
  uses the saved grid values directly; the rule is checked on those values).
- No full-size dataset has been generated yet.
- Readability for a specific VLM has not been tested by running a model (the
  448x252 check above measures the pixels, not a model's answers).

## 8. Open question

The first request asked for stars and "2-3 objects associated with single
shape or different shape". Later requests replaced this: the die outside
the board was removed and only its top number is shown, so stars and shape
kinds are no longer used.

## 9. Running the code

The scripts need OpenCV, Pillow, NumPy and imageio-ffmpeg. On Windows they
also need a font path set (the font finder only knows macOS and Linux paths):

```
python -m pip install opencv-python-headless pillow numpy imageio-ffmpeg
set GEOMETRIC_VIDEO_DATASET_FONT_PATH=C:\Windows\Fonts\arialbd.ttf
python generate.py --n 20 --output-dir . --seed 0
python flatten_annotations.py --dataset-dir .
python validate_die_roll_trace_dataset.py --dataset-dir . --report validation_report.txt
python verify_number_rule.py
python verify_questions.py
python verify_ladder.py
python verify_answerable.py
```
