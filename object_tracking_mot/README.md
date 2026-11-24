# object_tracking_mot

Video-QA dataset on the classic **multiple-object-tracking (MOT)** paradigm (Pylyshyn & Storm).

## What each clip shows

Four identical dots move on independent closed-form Lissajous-style paths inside a square arena with a faint centre cross marking the quadrants. The dots are numbered 1-4 for the first ~0.8 s, then the labels fade and never return, so every later question about a numbered dot needs continuous tracking through the whole clip.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1: dot count / quadrant of dot k at the start
- L2: did two named dots ever pass close by each other / which dot is closest to dot k at the end
- L3: total close-encounter count / how often dot k crosses the vertical centre line
- L4: quadrant of dot k at the end of the video
- L5: quadrant at the end if dot k had moved at a different speed / had frozen at time t (re-simulated)

## Closed loop

Every question stores its `derivation_key` and `params`, and `answers.py` re-derives each answer from the stored scene only (counterfactuals re-evaluate the modified path). `python run.py validate` fails a clip if that disagrees with the stored answer, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds an inconsistent scene, or if the MP4's duration, resolution or motion is off.

## Files and commands

```
sim.py        scene generation + consistency check (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/object_tracking_mot
python run.py validate --dataset-dir ../output_videos/object_tracking_mot
```
