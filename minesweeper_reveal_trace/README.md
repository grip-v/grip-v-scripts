# minesweeper_reveal_trace

Video-QA sub-benchmark for **sequential logical deduction and
consistency checking** on a minesweeper board whose tiles are uncovered
one at a time. v2 moves the dataset onto the shared PIL + ffmpeg
pipeline (`common/video5.py`) with the 5-rung ladder.

## What each clip shows

- **The board:** a **4×4 to 12×12** board starts fully covered, with
  **row numbers on the left and column numbers on top (1..n)**. Every
  question names tiles as "row r, column c" using those numbers.
- **Counters:** above the board are the total mine count ("Mines: k") and
  a reveal-step counter ("Step: s"). There is no other text in the frame.
- **Reveals:**
  - One tile is uncovered every 0.6 s and outlined in yellow for its step.
  - Uncovering works outward from a starting tile the way a player would,
    with occasional fresh guesses, up to 40 reveals.
  - About 12% of clips end by uncovering a mine.
- **Wrong numbers:** in about 40% of clips exactly one uncovered number
  is wrong. Nothing marks it. It is chosen so the numbers become provably
  impossible (no mine layout fits them and the mine total) at a later
  step, not necessarily at the step of the wrong tile itself.
- **Covered tiles:** every tile a question calls covered is still covered
  in the final frame. v1's bug of asking about an "unrevealed" tile that
  the video later uncovers is gone.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1 observation: tiles uncovered / the number on a given tile / was a mine uncovered
- L2 relational: which of two tiles was uncovered first / which of two shows the larger number
- L3 measurement: how many tiles show a given number / on which step a tile was uncovered / covered tiles touching a given tile at the end
- L4 integration: which of four covered tiles must be a mine, or is certainly safe, from all the numbers and the mine total; in clips with a wrong number, the step after which the numbers first stop fitting any mine layout
- L5 counterfactual: would a certain tile still be certain if one uncovered tile had stayed covered / could the numbers still fit if one tile showed one more / would the numbers still be impossible if one reveal step were skipped

## Closed loop

- **Generator:** logical facts come from `probability.py`'s exact engine, which splits the board into independent constraint regions and counts every mine layout. It is fed only the numbers a viewer has seen.
- **Independent answers:** `answers.py` never counts. It runs its own backtracking search that only asks whether *any* mine layout fits (optionally with one tile forced to be a mine or safe), and derives certain-mine, certain-safe and consistent answers from those yes/no searches.
- **Validator:** `python run.py validate` fails a clip if the two disagree, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds an inconsistent scene (wrong stored truth, repeated tile, stored contradiction step, a wrong number that doesn't cause a contradiction), or if the MP4 is off.

This cross-check caught two real bugs in the v1 engine, both fixed:
1. An uncovered mine was treated as an unknown covered tile instead of a known mine.
2. A number with no covered neighbours left was skipped, so a wrong number there was never flagged.

## Files and commands

```
sim.py                 scene generation + consistency check (pure, seeded)
probability.py         exact constraint-counting engine (used by sim.py)
questions.py           the 5-rung ladder
answers.py             independent yes/no layout search (used only by the validator)
render.py              PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py                 CLI: generate / validate (shared pipeline in common/video5.py)
generate.py            same as `run.py generate` (kept for generate_all_datasets.sbatch)
flatten_annotations.py annotations.jsonl -> dataset_final.csv / .jsonl

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/minesweeper_reveal_trace
python run.py validate --dataset-dir ../output_videos/minesweeper_reveal_trace
python flatten_annotations.py --dataset-dir ../output_videos/minesweeper_reveal_trace
```
