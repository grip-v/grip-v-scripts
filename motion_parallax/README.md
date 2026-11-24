# motion_parallax

Video-QA sub-benchmark: **Motion parallax**, the temporal version of the image benchmark's `depth_height` category (`Geomatric-Reasoning-Benchmark-Dataset/Dataset`). It runs on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the 5-rung ladder.

## What each clip shows

- **The scene:** a camera slides sideways at a steady speed past 3-6 flat coloured balls (no shading) hanging at different distances against a plain dusk sky. A dashed line marks the centre of the screen.
- **The cue:** a ball at depth z drifts across the screen at f·v/z, so nearer balls slide faster.
- **Misleading stills:** each ball's on-screen size and height are chosen independently of its depth, so a single frame points the wrong way (every clip has at least one farther ball that looks bigger).
- **Depths:** a base depth times distinct multipliers (1, 1.5, 2, 2.5, 3, 4), so depth ratios are tidy numbers.
- **Rules:** every ball is on screen for at least 3 s, and balls leave the screen at least 0.6 s apart.
- **Text in the frame:** only the clock.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1 observation: the first ball to leave the screen; the ball that slides fastest; how many balls
- L2 relational: which of two balls is closer to the camera (preferring pairs where the farther one looks bigger)
- L3 measurement: how many times farther one ball is than another; when a ball crosses the centre line (clock, 0.1 s)
- L4 integration: nearest-to-farthest order; which ball is actually the largest in real size
- L5 counterfactual: when a ball would cross the centre line if the camera moved twice as fast; which ball would leave first if the camera slid the other way; which of two balls would slide faster if one were twice as far away

## Closed loop

Every question stores its `derivation_key` and `params`. `python run.py validate` fails a clip if the independent answer disagrees with the stored one, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds a broken rule, or if the MP4's duration, resolution or motion is off. `answers.py` never reads the stored depths: it measures each ball's drift from its screen position at two times, takes depth as the inverse of that drift and real size as on-screen radius divided by drift, and finds crossing times by bisection on the measured track.

## Files and commands

```
sim.py        scene generation + rule checks (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/motion_parallax
python run.py validate --dataset-dir ../output_videos/motion_parallax
```
