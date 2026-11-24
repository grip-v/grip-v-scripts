# fold_punch_unfold

Video-QA sub-benchmark: **Fold, punch, unfold (shortlist B23)**, from `ideas/New Problem Types Shortlist (2).html`. It runs on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the 5-rung ladder.

## What each clip shows

- **The sheet:** an 8 x 8 gridded square, columns A-H and rows 1-8 labelled around its original position. Front face blue, back face cream, so you can see which layers were flipped.
- **Folds:** 1-4 folds, each along a grid line (vertical or horizontal) or a diagonal through grid points (about 30% of folds). Each fold first shows the crease as a red dashed line, then the moving part swings over on top of the rest.
- **Punch:** the folded stack is punched once or twice at cell centres; every layer underneath gets a hole.
- **The clip stops folded.** It never unfolds, so the hole pattern on the opened sheet has to be worked out from the folds.
- **Rules:** each fold moves and keeps at least 3 cells of paper, its flap lands mostly on paper, and the folded stack stays inside the original square. Punches sit at least 0.3 cells from every paper edge and crease. The opened sheet has at most 16 holes, all at distinct cell centres.
- **Text in the frame:** only the grid labels and the "Fold k" counter.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1 observation: how many folds; which way the paper moves in fold k
- L2 relational: which of two cells has a hole after unfolding; which of two punches goes through more layers
- L3 measurement: how many layers a punch goes through; how many holes the opened sheet has
- L4 integration: the full set of hole cells (up to 6 holes); how many holes are in a given row or column
- L5 counterfactual: hole count if fold k had folded the other side over, or had been skipped; which cell would have a hole if fold k were skipped; hole count if the punch had been at another cell

## Closed loop

Every question stores its `derivation_key` and `params`. `python run.py validate` fails a clip if the independent answer disagrees with the stored one, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds a broken rule, or if the MP4's duration, resolution or motion is off. The generator tracks the paper as clipped polygon layers (shapely). `answers.py` uses no polygons: it traces each punch point back through the folds one at a time, keeping the point if it lies on the side that stayed and was paper, and adding its mirror image if that lies on the side that moved and was paper, recursively down to the flat sheet.

## Files and commands

```
sim.py        scene generation + rule checks (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/fold_punch_unfold
python run.py validate --dataset-dir ../output_videos/fold_punch_unfold
```
