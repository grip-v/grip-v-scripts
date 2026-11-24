# robot_road_trip

Video-QA sub-benchmark adapted from **Robot Road Trip (Jane Street, Jul 2025), overtaking version**.

## What each clip shows

4–6 cars with different speeds enter a one-lane road with a shoulder. Faster cars never slow down: when one catches a slower car, the slower car pulls onto the shoulder until the lane is clear. A clock and unit markers are on screen.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1: pull-overs / cars
- L2: which car a pull-over made way for / first to finish
- L3: shoulder waiting time / a car's speed
- L4: finish order / most delayed car
- L5: total pull-overs if one car had never entered

## Closed loop

Every question stores its `derivation_key` and `params`, and `answers.py` re-derives each answer from the stored scene only. `python run.py validate` fails a clip if that disagrees with the stored answer, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds an inconsistent scene, or if the MP4's duration, resolution or motion is off. Cars only drive at their own speed or stand still, so each trip is an exact event log; `check` confirms speeds and that no two cars overlap in the lane.

## Files and commands

```
sim.py        scene generation + consistency check (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/robot_road_trip
python run.py validate --dataset-dir ../output_videos/robot_road_trip
```
