# moving_points

Video-QA sub-benchmark: **Moving points on a coordinate plane**, the temporal version of the image benchmark's `coordinate_geometry` category (`Geomatric-Reasoning-Benchmark-Dataset/Dataset`). It runs on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the 5-rung ladder.

## What each clip shows

- **The plane:** -10 to 10 on both axes, unit gridlines, bold axes, numbers every 2 units.
- **The points:** 3-4 lettered points (A-D) move in straight lines at constant velocity. Velocity components are -1, -0.5, 0, 0.5 or 1 units per second and starting coordinates are whole numbers, so every point is on whole-number coordinates at every even second.
- **No trails:** paths have to be read from the motion.
- **Rules:** points stay inside the plane, never overlap, and all have different speeds.
- **Text in the frame:** only the axis numbers, the point letters and the clock.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1 observation: which point moves fastest; which quarter of the plane a point ends in
- L2 relational: which of two points ends closer to the origin; which of two points crosses the vertical axis first
- L3 measurement: a point's coordinates at an even second; the distance between two points at the end (nearest unit)
- L4 integration: when two points are closest (0.5 s); how many times a point crosses an axis (only asked when no crossing is borderline)
- L5 counterfactual: where a point would end at half speed; when it would then be closest to another point

## Closed loop

Every question stores its `derivation_key` and `params`. `python run.py validate` fails a clip if the independent answer disagrees with the stored one, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds a broken rule, or if the MP4's duration, resolution or motion is off. The generator uses closed forms (closest approach at t = -(dp·dv)/|dv|²). `answers.py` measures each velocity from two sampled positions, minimises distances with a fine time scan plus golden-section refinement, and finds axis crossings by scanning for sign changes.

## Files and commands

```
sim.py        scene generation + rule checks (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/moving_points
python run.py validate --dataset-dir ../output_videos/moving_points
```
