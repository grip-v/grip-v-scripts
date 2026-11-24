# rotating_rays

Video-QA sub-benchmark: **Rotating rays**, the temporal version of the image benchmark's `angle_estimation` category (`Geomatric-Reasoning-Benchmark-Dataset/Dataset`). It runs on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the 5-rung ladder.

## What each clip shows

- **The dial:** a round dial with a tick every 15 degrees (longer ticks every 90).
- **The rays:** a long red ray and a shorter blue ray turn about the centre at steady but different speeds (15-90 degrees per second), possibly in opposite directions. Start angles are multiples of 15 degrees.
- **Events:** the rays line up when they point the same way and are at a right angle when 90 degrees apart.
- **Rules:** every event is at least 0.4 s from the start, the end and every other event; the clip never starts or ends exactly on an event; turn counts are never close to a whole number.
- **Text in the frame:** only the clock.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1 observation: which ray turns faster; which way the red ray turns; how many complete turns a ray makes
- L2 relational: whether the rays turn the same way; whether the final angle between them is acute, right or obtuse
- L3 measurement: the final angle between them (15-degree steps); when they first line up (clock, 0.1 s)
- L4 integration: how many times they line up; how many times they are at a right angle; the time between line-ups
- L5 counterfactual: how many line-ups if the blue ray turned the other way, or stayed still; when they would first line up if the red ray turned twice as fast

## Closed loop

Every question stores its `derivation_key` and `params`. `python run.py validate` fails a clip if the independent answer disagrees with the stored one, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds a broken rule, or if the MP4's duration, resolution or motion is off. The generator solves events in closed form from the relative angle. `answers.py` samples both ray directions every millisecond, unwraps the relative angle into a continuous signal, and counts crossings of multiples of 360 (line-ups) or of 90 + 180k (right angles).

## Files and commands

```
sim.py        scene generation + rule checks (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/rotating_rays
python run.py validate --dataset-dir ../output_videos/rotating_rays
```
