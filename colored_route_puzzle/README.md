# colored_route_puzzle (Colour Route)

Video-QA sub-benchmark for **subway-map route reasoning over time**. v3 is a
rework on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the
10-question ladder (`LADDER = "ten"` in `questions.py`). It replaces the
Manim pipeline (config.py, scene.py, geometry.py, generate.py, flatten.py,
validate_colored_route_puzzle.py and the committed annotations / dataset_final
/ validation_report files), keeping its core idea: grid-routed,
node-disjoint coloured routes that never cross or overlap, drawn over time so
that the draw order exists only in the video.

## What each clip shows

- **Stops:** 4-6 dark shape markers (circle, square, triangle, diamond, star,
  cross; each shape at most once) on the nodes of a routing grid, at least 4
  nodes apart and 2 nodes from the grid edge. Every stop is visible from the
  first frame.
- **Routes:** 4-8 coloured lines (11-colour palette, every colour once per
  clip, any two colours >= 85 RGB apart and far from the background, the stop
  colour and the ring). Each route is a grid path found by a randomized
  Dijkstra search (random node costs, random turn penalty) over nodes no
  earlier route uses, so routes are node-disjoint: they never cross, touch or
  run along each other. Two routes only meet at a stop, each through its own
  side (up / right / down / left). Line styles: solid, dashed or dotted.
- **Over time:** every route grows from one of its two ends (random) at a
  constant rate until complete. Nothing in the last frame tells which route
  came first, which end it grew from, or when.
- Clips are 3.4-9.9 s at 24 fps, 1280x720. The first event is >= 12 frames
  (0.5 s) after the start, the last >= 12 frames before the end, and any two
  events (route start / finish / erase start / gone, ring departure /
  arrival) are >= 2 frames apart.

### Scenarios (`scene["scenario"]`, about 20% each)

| scenario | setup |
|---|---|
| `sequential` | 5-8 routes drawn one at a time (10-24 frames each, 4-11 frame pauses), random order |
| `overlap` | 4-7 routes; a new one starts every 8-20 frames and takes 18-44 frames, so 2-4 grow at once and the finish order differs from the start order |
| `dead_ends` | 3-6 normal routes plus 1-2 dead ends: a route that leaves a stop and stops short at a crossbar (it may grow from the crossbar towards the stop) |
| `erase` | 5-7 routes drawn one at a time; 1-2 of them later shrink back to their start and vanish (after >= 3 routes have been drawn), so the last frame no longer shows every route |
| `traveller` | 4-7 routes drawn quickly, then a black ring that sits on a stop from frame 0 rides 2-4 routes from stop to stop (each route at most once, smooth start / stop, 6-12 frame rests at stops) |

### Random knobs

Scenario; number of stops and routes (retries keep the drawn counts, so large
maps are not filtered out); grid step (44-62 px), grid size and position
(framing); stop placement; 0-2 "parks" (pale rounded blocks routes avoid);
route end pairs (every stop gets >= 1 route, <= 4, <= 2 routes per pair);
routing noise and turn penalty (0-16 corners per route, 4-70 steps long);
colours; line styles (all solid in 30% of clips); line width (7-12 px);
sharp or rounded corners; marker size (32-44 px) and colour (4 dark inks);
marker shadows; background colour (6) and pattern (plain / grid lines / node
dots / gradient); park tint; growth end; draw speeds, pauses, intro and
hold; erase targets and timing; ring trail and ride speeds.

## On-screen text

- **Clock** (the only text): fixed dark panel top-left, "s.t s" truncated to
  0.1 s, 36 px bold (cap height 26 px), white on near-black. The whole map
  lives below y = 120 px; `sim.check` rejects any scene whose drawing (routes,
  markers, parks, the ring's sweep) comes within 8 px of the clock panel.
- **No labels.** The old version labelled stops with letters; v3 names stops
  by **shape** and routes by **colour**, which do the same job without text
  (at most 6 stops, 6 distinct shapes; every colour once). `scene["labels"]`
  is therefore empty and `scene["timer"]` holds the only text spec. There are
  no titles, legends, counters, hints or arrows.

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total
10, `L5.ten_counts`); questions are spread over as many templates as
possible (the scenario's own templates usually first; on L2-L5 at most one
question per level that a single frame could answer), ordered by level,
numbered q01..q10, and every question has 4 options. 43 templates:

- **L1 observation:** routes drawn (in `erase` this needs the video); stops;
  routes still on screen at the end (erase); routes of a line style; first /
  last route to start being drawn; which of four routes disappears (erase);
  dead ends (dead_ends); where the ring starts / ends (traveller).
- **L2 relational:** which two stops a route connects; which end a route
  grows from; which of four routes starts first; the next route to start
  after X; which of four routes is completed first (overlap; never the first
  starter); routes ending at a stop; routes joining two stops; the route of
  the ring's k-th trip; the stop it reaches on its k-th trip.
- **L3 measurement:** longest route (>= 15% margin); corners of a route; clock
  when a route becomes complete / has completely disappeared / the ring
  reaches a stop (options 0.5 s apart); which of four routes is growing when
  the clock shows t; complete routes on screen at t; routes growing at once at
  t (overlap); slowest-drawn route (>= 8 frames margin).
- **L4 integration:** which route, once complete, first lets you travel
  between two stops (changing routes only at stops); complete routes when the
  first route of a style starts; first-started route at a stop; the k-th
  route to start; stops reachable from a stop at clock t; stops the ring
  visits; corners of all ridden routes; the most complete routes on screen at
  once (erase).
- **L5 counterfactual (re-simulated):** first connecting route had route X
  never been drawn ("it would never become possible" is an option); first
  connecting route had the routes been drawn in reverse order; stops
  reachable without route X; the k-th route to start had X been drawn first;
  stops reachable at the end had an erased route stayed (erase); fewest rides
  for the ring without one of its routes ("it could not get there" is an
  option); stops left with no route if one line style were removed.

Distractors are plausible by construction: routes adjacent in the draw
order or touching the same stops, the other end of the route, the real
(non-counterfactual) answer, counts with or without the growing / erased
routes, ±1 counts, clock readings ±0.5 / ±1.0 s.

## Margins (no near ties)

- Compared start or finish times differ by >= 8 frames (0.33 s); in
  `overlap` every start is >= 8 frames after the previous one.
- Clock moments: every frame showing that reading is >= 2 frames from any
  event (and, for "growing" questions, past the first quarter of the
  growth, so the partial line is clearly visible beyond its marker).
- Connection questions need the deciding completion >= 8 frames from every
  other completion / erase.
- Routes are node-disjoint and >= one grid step (44-62 px) apart; lines are
  >= 7 px, markers >= 32 px, the ring 66-84 px wide.

## min_fps

Every question stores `params["min_fps"]`: the smallest of 1/2/3/4/6/8/12/24
fps whose sampling interval fits in the shortest gap the answer depends on
(between compared starts / finishes, a growth window, the stable window
around a clock moment, a ring ride or rest; the final hold for questions a
still frame answers; 0.25 s for clock-reading answers). Over 5000 scenes:
22.0% of questions need <= 1 fps, 75.5% <= 2 fps, 99.9% <= 4 fps (52 of
50000 need 6 fps). 72% of all questions, and 80% of L2-L5 questions, set
`requires_motion=True`.

## Diversity (5000 dry-run scenes, seeds 0-4999)

5000/5000 unique scene signatures; scenarios 966-1035 each; 4-6 stops,
4-8 routes, 0-17 corners and 4-67 grid steps per route, 19 grid steps,
clips 3.5-10.0 s; 6598 distinct (question, answer) pairs; answer letters
A/B/C/D 25.1 / 25.2 / 24.8 / 24.9 %. Generation retries a layout when
routing, stop placement, colours, timing or `sim.check` fail: about 8.3
rejected layouts per accepted scene (mostly a route that cannot be routed
without touching another); 1.4% of seeds drop one route after 150 failed
layouts. No seed fails.

## Closed loop

- **Questions** store `derivation_key` + `params` (route / stop indices,
  clock tenths, option sets, k) + `min_fps`.
- **Generation** (`sim.py`) works with grid nodes and frame windows.
- **Independent answers** (`answers.py`) never read those fields. They use
  `render.track(scene, i)` (the drawn polyline of every route and the ring
  centre on frame i), `render.stop_marks` and `render.timer_text`: start /
  finish / erase / gone frames are read off each route's drawn-length series,
  route ends and the growth end are the stop markers at the polyline's ends,
  corners and lengths are measured on the pixel polyline, the ring's stops
  and ridden routes come from its positions. Counterfactuals are re-simulated
  on that timeline.
- **`sim.check`** rejects scenes with out-of-range counts, duplicate shapes or
  colours, colours too close, paths that leave the grid, share a node, cross
  a park or a stop zone or come closer than 0.9 step in pixels, events closer
  than 2 frames or within 0.5 s of an end, clips over 10 s, overlap starts
  closer than 8 frames, ring rides that don't follow a route, and drawing
  that leaves the frame or reaches the clock.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`frame_check.py`** decodes the MP4s: reads the clock (template match) on
  every event frame, every question moment and every 12th frame; identifies
  each stop's shape by mask IoU; measures on every frame how far each route
  is drawn from its own colour along its path and fits start / finish /
  erase / gone frames (must be within 1 frame); tracks the ring (within 2 px);
  then re-derives every question with `answers.derive` on that measured
  timeline (clock answers within 0.1 s, everything else exact).

## Determinism

All randomness comes from the `random.Random(seed)` passed in; no global
RNG, time, uuid or set order decides anything. Same seed gives identical
scene + questions in separate processes with different `PYTHONHASHSEED`;
rendering is a pure function of the scene (raw-frame hashes compared across
processes); regenerating the preview seeds into a fresh folder reproduces
`annotations.jsonl` byte for byte (with `--no-render`; rendering only adds
timing metadata).

## Files and commands

```
sim.py                        scene generation, routing, timing, checks (pure, seeded)
questions.py                  the 10-question ladder (LADDER = "ten")
answers.py                    independent re-derivation from the rendered track
render.py                     PIL renderer (1280x720, 24 fps, piped to ffmpeg)
run.py                        CLI: generate / validate (common/video5.py)
frame_check.py                decoded-frame checks of rendered clips
test_colored_route_puzzle.py  pytest suite

python run.py generate --n 50 --seed 10018 --output-dir ../output_videos/colored_route_puzzle_preview
python run.py validate --dataset-dir ../output_videos/colored_route_puzzle_preview
python frame_check.py --dataset-dir ../output_videos/colored_route_puzzle_preview
python -m pytest test_colored_route_puzzle.py -q
```

`annotations.jsonl` is the final output (sbatch entry
`colored_route_puzzle:run.py::none:none`).
