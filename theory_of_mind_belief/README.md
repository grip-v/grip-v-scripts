# theory_of_mind_belief

Video-QA sub-benchmark for **belief-state tracking** (Sally-Anne style; not a claim of genuine theory of mind).

## What each clip shows

Two or three tracked shapes sit in a row (a number that ticks up/down, a shape that morphs, or a colour that swaps). A "watcher" (an eye icon at the lower left) looks away for one or two windows; a clock shows the time. Changes that happen while the watcher is away are never seen by it, so its belief about a shape stays frozen at the last state it saw. The camera always shows the truth.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1: number of shapes / total changes over the clip
- L2: is the watcher watching at time t / when a given change happens
- L3: changes that happen while the watcher is away (all shapes / one shape)
- L4: what the watcher believes about a shape at t / does it hold an outdated belief
- L5: watcher's belief if one change had never happened / if every away window were shifted earlier or later

## Closed loop

Every question stores its `derivation_key` and `params`, and `answers.py` re-derives each answer from the stored scene only (counterfactuals re-simulate with the change). `python run.py validate` fails a clip if that disagrees with the stored answer, if rebuilding from the seed gives a different scene or questions, if `sim.check` fails, or if the MP4's duration, resolution or motion is off.

## Files and commands

```
sim.py        scene generation + consistency check (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/theory_of_mind_belief
python run.py validate --dataset-dir ../output_videos/theory_of_mind_belief
```
