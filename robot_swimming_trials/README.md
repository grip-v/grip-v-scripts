# robot_swimming_trials

Video-QA sub-benchmark adapted from **Robot Swimming Trials (Jane Street, Oct 2021)**.

## What each clip shows

6 or 9 robots split one unit of fuel into bids for 2–3 races. Bids resolve highest first: a bid wins if its race is open and its robot hasn't qualified, otherwise it is crossed out, until every race has a winner.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1: bids crossed out / robots bidding
- L2: robot whose bid was skipped because it had already qualified / race winner
- L3: winning bid of a race / largest crossed-out bid
- L4: order the races were decided / bids checked before race k was decided
- L5: race winner if one bid had been different (half raise a loser above the winner, half lower the winner)

## Closed loop

Every question stores its `derivation_key` and `params`, and `answers.py` re-derives each answer from the stored scene only. `python run.py validate` fails a clip if that disagrees with the stored answer, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds an inconsistent scene, or if the MP4's duration, resolution or motion is off. No ties among resolved bids. `answers.py` resolves by repeatedly taking the largest remaining bid.

## Files and commands

```
sim.py        scene generation + consistency check (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/robot_swimming_trials
python run.py validate --dataset-dir ../output_videos/robot_swimming_trials
```
