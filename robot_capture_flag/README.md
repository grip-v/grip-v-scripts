# robot_capture_flag

Video-QA sub-benchmark adapted from **Robot Capture-the-Flag (Jane Street, Apr 2024), multi-round**.

## What each clip shows

Each round a flag appears in a unit-circle arena. Aaron knows only how far it is and runs that far in a random direction; Erin knows only its direction and always runs the same hidden distance d. Closer robot wins. No tally is shown.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1: rounds played / flag inside or outside the half-way circle
- L2: winner of round k
- L3: estimate Erin's hidden distance d
- L4: Erin's total wins / Aaron's longest streak
- L5: Erin's wins if she had always run a different distance

## Closed loop

Every question stores its `derivation_key` and `params`, and `answers.py` re-derives each answer from the stored scene only. `python run.py validate` fails a clip if that disagrees with the stored answer, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds an inconsistent scene, or if the MP4's duration, resolution or motion is off. Rounds closer than 0.08 are resampled so every winner is clear by eye. `answers.py` uses complex arithmetic.

## Files and commands

```
sim.py        scene generation + consistency check (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/robot_capture_flag
python run.py validate --dataset-dir ../output_videos/robot_capture_flag
```
