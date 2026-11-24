# phantom_traffic_jams

Video-QA sub-benchmark: **Phantom traffic jams (shortlist B18)**, from `ideas/New Problem Types Shortlist (2).html`. It runs on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the 5-rung ladder.

## What each clip shows

- **The road:** a ring road seen from above with a white start line at the top. 10-19 cars drive clockwise and never overtake.
- **The rule:** a 60-cell Nagel-Schreckenberg cellular automaton with no randomness. Every 0.3 s step, each car speeds up by 1 (top speed 3 cells per step), slows to the number of empty cells in front of it, and then all cars move together. The renderer slides each car smoothly between steps.
- **Brake taps:** four cars are coloured (red, blue, green, gold) and the rest are grey. 1-3 coloured cars each brake to a stop by themselves once, for 2-3 steps, with bright brake lights.
- **The phenomenon:** a tap can start a queue of stopped cars that drifts backward around the ring while every car keeps moving forward, catching cars far behind the one that braked. Above about one car per four cells the jam never clears.
- **Rules:** every tap changes which cars stop; at least one coloured car stops; the jam spreads beyond the car that braked.
- **Text in the frame:** only the clock.

## The 5-rung question ladder

Every clip has exactly five multiple-choice questions, one per rung (`common/ladder5.py`):

- L1 observation: how many brake taps there are; whether a coloured car ever stops; how many coloured cars stop
- L2 relational: which of two coloured cars stops first; which travels farther
- L3 measurement: how many times a car crosses the start line; how many separate times it stops; how long it stands still in total
- L4 integration: how many cars (grey included) ever stop; the order in which coloured cars first stop; which coloured car stands still longest
- L5 counterfactual: with one brake tap removed, whether a given car would ever stop, how many coloured cars would stop, or how many cars in total would stop

## Closed loop

Every question stores its `derivation_key` and `params`. `python run.py validate` fails a clip if the independent answer disagrees with the stored one, if rebuilding from the seed gives a different scene or questions, if `sim.check` finds a broken rule, or if the MP4's duration, resolution or motion is off. The generator steps an occupancy array of the road. `answers.py` replays the same rule by tracking each car's distance driven and its gap to the fixed car ahead (cars never overtake), and reads stops, laps and distances off that run.

## Files and commands

```
sim.py        scene generation + rule checks (pure, seeded)
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/phantom_traffic_jams
python run.py validate --dataset-dir ../output_videos/phantom_traffic_jams
```
