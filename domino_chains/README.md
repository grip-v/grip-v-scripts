# domino_chains

Video-QA sub-benchmark: **Branching domino chains (shortlist B9)**, from `ideas/New Problem Types Shortlist.html`.

## What each clip shows

A top-down board of lettered junction dominoes joined by straight lanes of plain dominoes. One push knocks over A; lanes fork, merge (the first chain to arrive knocks the junction over) and stop at dud gaps (one missing domino). Every link takes distance / speed, so fall times are an exact shortest-path computation. A clock is on screen, and fallen dominoes stay down, so the final frame doesn't show the order.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1 observation: lettered dominoes knocked over / still standing
- L2 relational: which of two falls first; the next lettered domino to fall after X
- L3 measurement: the clock time when X falls
- L4 integration: which lane actually knocks over a merge junction; the lettered path the chain took to X
- L5 counterfactual: lettered dominoes that fall if X were removed, or if a gap were filled

## Closed loop

Every question stores its `derivation_key` and `params`. `python run.py validate` fails a clip if the independent answer disagrees with the stored one, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds an inconsistent scene, or if the MP4's duration, resolution or motion is off. `answers.py` rebuilds every lane from the domino coordinates and finds fall times by Bellman-Ford relaxation (the generator uses Dijkstra). `check` confirms the stored times and that every gap is clearly within reach or clearly out of reach.

## Files and commands

```
sim.py        scene generation + consistency check (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/domino_chains
python run.py validate --dataset-dir ../output_videos/domino_chains
```
