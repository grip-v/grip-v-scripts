# blinking_networks

Video-QA sub-benchmark: **Blinking signal networks (shortlist B15)**, from `ideas/New Problem Types Shortlist.html`.

## What each clip shows

Coloured, patterned nodes (circle / square / triangle / diamond / star; solid, striped, dotted or ringed) joined by non-crossing straight links. A starting node flashes; a node that lights up sends a white (dark on light themes) pulse down each of its links except those that already brought it a pulse. A pulse reaching a dark node lights it; a pulse reaching a lit node fizzles (short fading ring). Every lit node keeps a persistent halo ring, so the final frame shows *which* nodes lit but not the order: the order has to be watched.

The clock in the top-right box (seconds, 0.1 s) is the **only text** in the frame. Nodes are named in questions by colour + pattern + shape ("the striped red square"); no two nodes share a colour + shape, so every description is unique.

1280x720, 24 fps, at most 10 s. Every event a question depends on lies between 0.5 s and (end - 0.5 s).

## Scenarios (`scene["scenario"]`, uniform per seed)

| scenario | rule | how it is visible |
|---|---|---|
| `hidden_slow` | one link carries pulses at half speed | drawn like every other link; only its slow pulse gives it away |
| `marked_speeds` | 2-4 links carry pulses at half speed | slow links are dashed, fast links solid |
| `one_way` | 60-100 % of links carry pulses only one way | arrowhead in the middle of the link; some nodes may never light |
| `two_sources` | a second node flashes on its own, 0.5-0.8 of the way to when the wave would reach it | it flashes with no pulse arriving |
| `double_trigger` | 2-4 nodes need two pulses before they light | double outline; after the first pulse a half halo stays on |

## Random knobs

Node count 6-11 (fixed per seed), node size R 20-30 px, layout style (scatter / jittered grid / ring / 2-3 clusters), framing box (60-100 % of the play area, random offset), node spacing, 1..n/2 extra links on top of the Euclidean MST, colour / shape / pattern per node (shape pool of 2-5 shapes; 30 % of scenes all solid), 9 themes (6 dark, 3 light) x 6 background textures (plain, gradient, grid, dots, hatch, vignette), link colour, link width 3-6 px, solid or twin-rail links, pulse speed (85-420 px/s, set so the cascade takes 3-7.4 s), start time 0.6-1.3 s, tail 0.7-1.2 s.

## The 10-question ladder (`LADDER = "ten"`)

Per clip, `L5.ten_counts(rng)` draws the per-level counts (L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3; total 10). Templates per level are picked round-robin in random order, never repeating a `derivation_key` + params. All questions have 4 options.

- **L1 observation**: first node to flash; last to light; number of nodes / links; number of dashed links, arrowed links or double-outlined nodes; nodes still dark at the end; links touching X
- **L2 relational**: first / last of three named nodes; which neighbour of X lights first; the k-th to light; which lights right after X
- **L3 measurement** (the clock): clock reading when X lights (nearest second); travel time of the pulse that lit X (nearest second); how many nodes are lit when the clock reads T; first flash to last light-up (nearest second)
- **L4 integration**: which node's pulse lit X; hops in X's lighting chain; how many nodes X lit directly; which link is the hidden slow one; which node flashed on its own; which self-starting node started X's chain; the one node that never lights; first double-outlined node to light
- **L5 counterfactual** (re-simulated): last node if link e were removed; whether X would light later / at the same time / earlier / never without link e; last node if Y had started; last node if slow links were full speed; how many nodes would light if an arrow were reversed; last node / node count if every node lit on one pulse; effect of the second node not flashing on its own

## Margins and frame sampling

- No two nodes ever light within 0.125 s (3 frames); every question only uses light-ups at least 0.3 s apart from whatever they are compared with. Pulses reaching the same node are always at least 0.3 s apart, so who lit whom is never borderline.
- Clock-reading answers are at least 0.25 s from a rounding boundary (durations at least 0.3 s); "count at T" picks T at least 0.3 s from any light-up.
- Counterfactual winners lead by at least 0.3 s in the re-simulated world; "same time" means exactly the same.
- Every result is persistent (halo, half halo), never a one-frame flash. Pulses are 20 px dots and stay on a link at least 0.4 s. Links are at least 3 px, nodes at least 40 px across, and the clock is still readable at 640x360.
- `params["min_fps"]` = 1 / (smallest gap between the events the question uses); for rounding and duration questions the gap is the distance to the rounding boundary (halved for durations). Static questions use 0.5. Over 5000 scenes, 55 % of questions are answerable at 1 fps, 76 % at 2 fps and 91 % at 4 fps (max 6.7).

## Closed loop

Every question stores `derivation_key` + `params`. `answers.py` never reads the generator log (`fire`, `parent`, `arrivals`, `half`, `pulses`). It re-runs the cascade from the drawn world (positions, links, dash / arrow / double-outline markings, hidden slow link, speed, starting flashes) with a fixed-step (1/240 s) simulation that moves every pulse along its link in pixels, and re-runs counterfactuals the same way. `sim.simulate` (the generator) is an exact priority-queue replay. `sim.check` replays the log and rejects any scene with crossing links, links that pass near a node, links leaving a node at less than 25 degrees, nodes too close, pulses faster than 0.4 s per link, a pre-empted start, fewer than 5 lit nodes, events outside [0.5, end-0.5], or near-simultaneous events. `sim.build` retries until the scene passes and every level has at least 3 candidate questions (about 8 attempts per scene on average).

## Determinism

All randomness comes from the `random.Random(seed)` passed in. There is no global RNG, time, uuid or set order in any decision. The render cache is keyed by the scene content. The same seed gives byte-identical JSON across processes with different `PYTHONHASHSEED` values, and identical raw frames.

## Files and commands

```
sim.py        scene generation, exact event simulation, check (pure, seeded)
questions.py  the 10-question ladder (candidates + selection, min_fps)
answers.py    independent fixed-step re-simulation (used only by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
checks.py     stats / tamper / framecheck / framehash
run.py        CLI (generate / validate from common/video5.py, plus the checks)
test_blinking_networks.py

python run.py generate --n 50 --seed 0 --output-dir ../output_videos/blinking_networks
python run.py validate --dataset-dir ../output_videos/blinking_networks [--skip-video]
python run.py stats --n 5000
python run.py tamper --dataset-dir DIR       # corrupts 3 answers -> exactly 3 flagged
python run.py framecheck --dataset-dir DIR   # node colours, halo state 2 frames either side of every light-up,
                                             # clock crop vs the expected reading, final lit count, pulse positions
python -m pytest test_blinking_networks.py
```
