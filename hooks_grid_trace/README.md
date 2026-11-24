# hooks_grid_trace

Video-VQA sub-benchmark testing **nested-structure construction tracking,
running-sum arithmetic reasoning over time, and precise error
localization within a sequentially built numeric grid** — built on the
"Hooks" puzzle concept: an N×N grid divided into N nested L-shaped
"hooks" radiating from one corner, hook k filled entirely with digit k,
with every row/column sum required to match a border target.

## 1. What each clip shows

- A 6×6 to 9×9 grid, divided into N nested hooks anchored at the
  top-left corner: hook k is every cell at Chebyshev distance k-1 from
  that corner (an L-shaped shell of 2k-1 cells), and a correct fill
  assigns every cell in hook k the digit k. **Bold nested-square hook
  boundaries are visible from frame 1** (hook k's outer edge is exactly
  the boundary of the k×k sub-block from the corner — see §4), well
  before any digit is filled in, against a faint ordinary cell grid.
- **Row-sum and column-sum targets** shown along the left and top edges
  in a warm accent color, visible and unchanged from the first frame.
- A **static tan/white shading pattern**, independent of hook structure
  (each cell independently shaded with 50% probability), purely a visual
  reference the final question can ask about.
- Cells are filled **one hook at a time** — every cell in a hook fades in
  together, not cell-by-cell, since a hook is this puzzle's atomic unit —
  in a per-clip **`fill_direction`**: `outward` (hook 1, the corner,
  first, hook N last) or `inward` (hook N first, hook 1 last).
- **~50% of clips show a brief running-sum overlay** after each hook
  completes, for every row/column that hook just touched, fading out
  again after ~0.5s — a deliberate difficulty-modulating variable (see
  §8): clips without it require mentally tracking partial sums for the
  Level-5 questions, clips with it make that a read-off task.
- **~50% of clips inject exactly one error**: one cell in one hook
  (never hook 1, the single corner cell) is filled with `correct_digit ±
  1` instead of the right value — thrown off by exactly one, not an
  arbitrary wrong number (see §8) — which necessarily mismatches exactly
  the one row and one column that cell belongs to.
- **Final hold**: the complete grid, plus every row/column's actual sum
  displayed next to its target (matching color if correct, accent-red if
  mismatched), held for the last ~20% of the clip.

## 2. Why this needs video, not a single frame

The final frame shows the completed grid and every sum at once — it
never shows *when* a row's sum locked into its final value relative to
the rest of the construction, or how many cells had been placed at any
given moment. Level 5's questions specifically require replaying the
fill sequence: whether more or fewer than half the grid was filled by
the midpoint hook, and whether a given row's sum was already final well
before the last hook (a genuinely non-obvious, hook-structure-dependent
fact — see §4).

## 3. The 5-question ladder

| qid | level | difficulty | asks |
|---|---|---:|---|
| q01 | Image Description | 1 | how many hooks make up the grid |
| q02 | Basic Relational | 2 | (a) fill direction, or (b) did every row/column sum match its target |
| q03 | Comparative | 3 | (a) outermost hook's digit (always N — a calibration check), (b) [outward-fill only] which hook was filled right after hook 1, or (c) which row mismatched (or "none") |
| q04 | Compound | 4 | (a) [if broken] which hook the error was in and whether it was one more/less than correct (or "confirm true" if not broken), (b) sum of the shaded cells, or (c) total row+column mismatch count (always 0 or 2) |
| q05 | Extended Temporal | 5 | (a) more or fewer than half the grid filled by the midpoint hook, (b) [if broken] would fixing the one error resolve every mismatch, or (c) did a named row's sum lock in before the last hook, or only on it |

## 4. Ground truth / hook geometry

`hooks.py` is pure Python, zero rendering imports. `build_hooks` and
`correct_grid` are re-derivable from `grid_size` alone — the single
source of truth used identically by generation and validation.

**Two structural facts fall out of the corner-anchored hook construction
for free, not by independent design choice:**
- **Row r's target always equals column r's target**, since a cell's
  hook index (`max(r, c) + 1`) is symmetric in r and c — so `row_targets
  == column_targets` for every clip, verified directly rather than
  assumed.
- **Row r is only ever touched by hooks `(r+1)` through `N`** — hooks
  `1..r` never place a digit in row r. This is exactly what makes the
  Level-5c "did this row lock in early" question genuinely interesting
  for `inward` fills (a row typically locks in one hook before the
  video ends, since hook `r+1` — not hook 1 — is the last of its
  contributing hooks to be filled) and correctly, if less interestingly,
  always "only on the last hook" for `outward` fills (hook N, filled
  last in that direction, touches every row) — both are real, computed
  answers, not hardcoded per direction.

**Error injection** picks a hook (never hook 1), a cell within it, and a
`correct_digit ± 1` replacement, entirely in `hooks.build_scene` before
any rendering; `final_grid` is `correct_grid` with that single cell
overwritten. Every downstream statistic (row/column sums, match flags,
shaded-cell sum) is computed by scanning the actual `final_grid`, not by
separately reasoning about "what the error should imply" — so a
construction bug would show up as an internally inconsistent
`final_grid` vs. derived stats, not just a wrong label.

**Timing.** `generate.py`'s `_timing()` solves for the exact
`step_duration_s`/`total_duration_s` pair so that `scene.py`'s actual
elapsed time — N hook-fill animations, plus the running-sums overlay's
fixed 0.5s/hook overhead when shown, plus the final all-sums fade-in —
lands precisely on the target duration once the hold is added; the
overlay overhead was missing from the very first version of this
calculation, causing every `running_sums_displayed=true` clip to render
noticeably longer than its stored `duration_seconds`/`frame_count` —
caught immediately by the validator's frame-count check and fixed by
having `scene.py` track real elapsed time and size the final hold off
of that, not off of the fill animations alone.

## 5. Files

```
config.py       colors, grid-size/duration ranges, running-sums/broken/shade probabilities
hooks.py        two-pass hook geometry + fill order + error injection + every derived statistic, zero rendering imports
questions.py    builds the fixed 5-question ladder from a SceneData
scene.py        Manim scene (reads params from $HGT_PARAMS_FILE); nested hook squares + per-hook fade-in fills
generate.py     params -> annotations.jsonl (+ manim render + final-frame PNG extraction)
flatten_annotations.py   annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
validate_hooks_grid_dataset.py   independent hook/sum/error re-derivation + fill-sequence replay + question re-derivation + video checks
annotations.jsonl, videos/, frames/, dataset_final.{csv,jsonl}, validation_report.txt
```

## 6. Generation and reproducibility

Python 3, Manim, OpenCV (validator only). Sample `i` uses ONE
`random.Random(i)` instance, continued (not re-seeded) across
`hooks.build_scene` and then `questions.build_questions` — the validator
replays this exact consumption order.

```
python generate.py --n 20 --output-dir . --seed 0 --quality low
python flatten_annotations.py --dataset-dir .
python validate_hooks_grid_dataset.py --dataset-dir . --report validation_report.txt
```

`--sample` for 5 dev examples at `-qh`, `--no-render` for a param-only
dry run, `--append` to resume a batched/interrupted run.

## 7. Validation

`validate_hooks_grid_dataset.py` independently reconstructs the hook
structure and fill order from `grid_size`/`fill_direction` alone and
diffs against storage; recomputes `final_row_sums`/`final_column_sums`
directly from `final_grid` and confirms exact match, and confirms
`row_sum_matches`/`column_sum_matches` are correctly derived from
comparing to `row_targets`/`column_targets`; for broken clips, confirms
`final_grid` differs from the theoretically correct grid at EXACTLY one
cell, off by exactly 1, and that the resulting mismatches are exactly
the stored row/column; for unbroken clips, confirms zero deviation and
that every sum matches; recomputes `shaded_cells_sum` directly from
`final_grid`; and — the most important check in this dataset — rebuilds
the entire scene from a fresh `Random(seed)` and re-derives all 5
questions (Level 5a/5c in particular require correct fill-sequence
replay, not just final-state inspection), diffing
`groundtruth`/`answer`/`options` against storage; and decodes every
rendered MP4 with OpenCV, checking frame counts. A `--sample` batch of 5
was rendered at high quality and visually inspected: hook boundaries
read clearly as nested squares against the faint ordinary grid, digits
are legible up to 9×9, and the single-digit error (shown in accent red)
is clearly distinguishable from its correct neighbors, with the
resulting row/column mismatch visibly flagged in the same accent color
at the final hold.

Ends with a machine-greppable `RESULT: <n_samples> samples, <n_questions>
questions, <n_mismatches> mismatches` line. Current status: **20
samples, 100 questions, 0 mismatches** — `is_broken` 13/7, `grid_size`
5/6/4/5 across 6-9, `fill_direction` 12/8 outward/inward (a 2000-seed
dry run gives a closer 50/50 and near-even split, expected at this
20-clip sample size).

## 8. Known limitations

- **The injected error is always exactly ±1 from the correct digit**,
  never an arbitrary wrong value — a deliberate difficulty choice (an
  off-by-one digit next to its correct neighbors is genuinely harder to
  spot on sight than an arbitrarily wrong one) documented explicitly,
  not an oversight; a "wrong by an arbitrary amount" variant would be a
  natural harder follow-up.
- **Row/column targets are generated to be consistent with a valid
  hook-filling by construction**, not independently puzzle-authored —
  this dataset tests whether a model can track and VERIFY a shown
  construction process (and localize a deliberate deviation from it),
  which is a different reasoning task than SOLVING an arbitrary Hooks
  puzzle from scratch.
- **`running_sums_displayed` clips are meaningfully easier for Level 5**
  since partial-sum information is handed to the viewer directly rather
  than requiring mental tracking — realized roughly 50/50 in this
  checkout (see §7); downstream users should stratify results by this
  field rather than pooling scores across both modes.
- **The shaded-cell pattern is independent of hook structure** — a
  purely static visual reference from the original puzzle format that
  never affects fill logic.
- **Hooks fill instantaneously as a whole shell, not cell-by-cell** — a
  cell-by-cell variant within each hook was considered and rejected as
  adding animation time without adding reasoning difficulty, since the
  hook is this puzzle's semantically atomic unit.
- **The hook anchor corner is always fixed at top-left** — the four-way
  choice of anchor corner was not randomized, a simplification not
  called out in the original spec but worth stating plainly.

## 9. Reasoning skills tested

- **Nested-structure construction tracking** — nested L-shaped shells
  built inward or outward from a fixed corner.
- **Running-sum arithmetic reasoning across a sequential fill process.**
- **Precise single-cell error localization** within a larger correct
  structure.
- **Distinguishing a "locked-in" partial state from an incomplete one**
  — recognizing when a row/column sum reached its final value well
  before the construction actually finished.
- **Static arithmetic aggregation (shaded-cell sum) combined with
  construction-tracking** — holding both concerns simultaneously.
- **Counterfactual correction reasoning** — would fixing the one error
  resolve every resulting sum mismatch.
