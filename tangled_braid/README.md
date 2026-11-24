# tangled_braid

Video-QA sub-benchmark for **tracking a braid as it is woven** (after Jane
Street's *Tangled!* puzzle, Sep 2020). v2 is a rework on the shared PIL +
ffmpeg pipeline (`common/video5.py`) with the 10-question ladder
(`LADDER = "ten"` in `questions.py`). v1 had a 5-question ladder, 2-16
strands with repeated colours, move numbers printed next to every crossing
(which also printed the answer to "how many crossings"), and no clock.

## What each clip shows

- **Strands**: 4-8 strands come out of a **loom bar** side by side. Each
  has its own colour (10 colours, never repeated in a clip), a line style
  (solid, dashed, dotted, double, outlined, beaded, banded) and a thickness
  (7 / 10 / 13 px).
- **Crossings**: two neighbouring strands swap places, one passing over the
  other. The over-strand is drawn on top of a background halo, so the
  under-strand visibly breaks at the crossing.
- **Scrolling**: the braid is woven at the bar and scrolls away from it;
  everything farther than the viewing window (450-1130 px) fades out, so
  only the last 2-4 crossings are on screen. Early crossings, and so the
  over/under history, must be remembered.
- Clips are 3.75-10 s at 24 fps, 1280x720. The first crossing starts
  ≥ 14 frames (0.58 s) after the start; the last one ends ≥ 16 frames
  before the end.

### Scenarios (`scene["scenario"]`, about 20% each)

| scenario | setup |
|---|---|
| `classic` | vertical braid (woven at the bottom, scrolling up, or woven at the top, scrolling down), one crossing per step, 4-8 strands, 5-10 crossings, never an immediate undo of the previous crossing |
| `parallel` | vertical braid where steps cross 1-3 pairs **at the same moment** (pairs never touch); at least two steps have simultaneous crossings |
| `sideways` | the braid runs left-right (woven at the right or left edge); places are counted from the top; 4-6 strands |
| `pauses` | the loom stops completely for 0.6-1.25 s between at least two pairs of crossings (only the clock moves) |
| `pure` | the crossings bring every strand back to its starting place (a random tangle, then a random untangling) - order questions can't be guessed from "something moved" |

### Random knobs

Scenario; strand count; colours, styles and thicknesses; start order;
crossing places and which side goes over; number of crossings; crossing
duration (10-20 frames, per step), crossing length along the braid
(70-150 px) and so scroll speed (3.9-15 px/frame); still gaps (4-12 frames)
and pauses (14-30); intro (14-26 frames) and outro (16-30); orientation and
weaving direction (bottom / top / right / left); strand spacing (70-150 px);
braid position; bar position and window length; crossing curve (smoothstep,
cosine, smootherstep); halo width; background colour (8) and pattern (plain,
grid, dots, gradient, hatch) with pattern spacing; bar colour (4).

Over 5000 dry-run scenes: 100% unique scene signatures (sha256 of the
rounded scene), see the stats section below.

## On-screen text

- **Clock** (always, the only text): fixed dark panel top-left, "s.t s"
  (truncated to 0.1 s), 36 px (digit height 26 px ≥ 24 px), white on
  near-black. The braid, the bar and the fade band always stay clear of it
  (`sim.layout_errors`), and the frame check confirms no strand colour
  appears next to it.
- **No other text.** v1 printed a move number beside every crossing. They
  are not needed: crossings are named by their order ("the third
  crossing"), by the two strands and which time they cross, or by the clock;
  strands are named by colour; places are counted from the left (top). The
  move numbers also gave away the crossing count (the last number on
  screen), so they are gone. `scene["labels"]` is empty; the clock is
  `scene["timer"]` (box, size).

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total
10, `L5.ten_counts`); questions are spread over as many templates as
possible (each scenario's featured templates usually first), ordered by
level, numbered q01..q10. Every question has 4 options. 32 templates:

- **L1 observation:** number of strands; number of crossings in the whole
  clip (old ones have scrolled away); crossings a strand takes part in;
  which strand is leftmost / rightmost (top / bottom) at the bar at the
  start / at the end; which two strands cross first; which strand never
  crosses.
- **L2 relational:** which strand passes over at the k-th crossing (not in
  `parallel`); what happens the first time a strand crosses another (over or
  under which strand); which of four strands is first to cross; which strand
  is directly left/right of (above/below) X at the bar when the clock reads
  t; which strand X crosses first / next after Y; which pair crosses at the
  same moment as X and Y (`parallel`).
- **L3 measurement:** how many times X and Y cross; clock reading when X and
  Y meet; crossings between two clock readings; by how many places X moved
  from start to end; X's place at clock t; crossings X has taken part in by
  clock t.
- **L4 integration:** signed over/under total of a pair (+1 X over, -1 X
  under); how many times X passes over; strand with the most crossings /
  the most passes over; X's partners in order; how many strands end where
  they started; which strand ends where X started.
- **L5 counterfactual (re-simulated):** a crossing left out (the rest keep
  their places and side-on-top) → which strand ends at place p, or how many
  times X and Y would cross; one crossing flipped (the other strand on top)
  → the pair's signed total; every crossing flipped → how many times X
  passes over; two strands swap starting places → which strand ends at
  place p.

Distractors: strands near the asked one or involved in neighbouring
crossings, the under-strand for "who is on top", the real-video answer for
counterfactuals, the previous order for time questions, other crossings'
clock readings (≥ 0.4 s apart) for clock questions, near integers for
counts. All L2-L5 questions have `requires_motion=True`; only
`n_strands` and the start/end edge questions are single-frame.

## Margins (no near ties)

- Crossing meeting moments are ≥ 12 frames (0.5 s) apart; between two
  steps the strands are still for ≥ 4 frames; simultaneous crossings never
  share a strand or touch.
- Every crossing stays on screen ≥ 14 frames after its strands meet.
- Clock readings used in questions lie ≥ 2 frames inside a still stretch
  (no strand moving at the bar) and ≥ 0.5 s before the end.
- Clock options are ≥ 0.4 s apart; "first to cross" candidates cross in
  different steps; "most" questions need a strict maximum.
- Places are discrete; strand spacing ≥ 70 px; lines ≥ 4 px wide.

## min_fps

Every question stores `params["min_fps"]`, the smallest of
1/2/3/4/6/8/12/24 fps that puts a sample inside the shortest interval its
answer needs: how long each crossing involved stays on screen, the still
stretch holding a clock reading, the start / end stretch of a place, half
the gap to the nearest clock option. Over 5000 scenes: see stats below.

## Closed loop

- **Questions** store `derivation_key` + `params` (strand names, crossing
  index, place, clock tenths, option sets) + `min_fps`.
- **Independent answers** (`answers.py`) never read the generator's moves,
  log or sim helpers. They measure the clip from `render.track(scene, f)`,
  the renderer's own per-frame state: crossings are the frames at which two
  neighbouring strands' positions at the bar meet, the over strand is the
  one the renderer draws on top, places come from the measured order.
  Counterfactuals are re-played on the measured (place, side-on-top) list
  with a separate swap loop.
- **`sim.check`** rejects scenes with repeated colours or names, scenario
  rule violations, meetings closer than 12 frames or within 0.5 s of an
  edge, fewer than 4 still frames between steps, crossings visible under
  14 frames, clips over 10 s, layouts touching the timer or the frame edge.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`run.py framecheck`** decodes the MP4s and works from pixels only:
  reads the clock (template match) at every meeting frame and every frame
  showing a question's clock reading; reads the strand order on the bar
  scanline (nearest strand colour, voting over 11 rows) in every still
  stretch; finds each crossing's meeting frame (line fit of the over strand's
  centre to the midpoint) and the over strand (colour of the patch at the
  meeting point, where the halo hides the under strand); builds a
  measurement from those reads and re-derives **all 10 answers** of each
  clip from it.
- **`run.py tamper`** corrupts 3 stored answers; validate must flag
  exactly those 3.

## Determinism

All randomness comes from the `random.Random(seed)` passed in; no global
RNG, time, uuid or set order decides anything. Same seed → byte-identical
scene + questions + answers in separate processes with different
`PYTHONHASHSEED`; rendering is a pure function of the scene
(`run.py framehash --seed S` hashes the raw frames).

## Files and commands

```
sim.py                  scene generation, timing, rule and layout checks (pure, seeded)
questions.py            the 10-question ladder (LADDER = "ten")
answers.py              independent re-derivation from the rendered track
render.py               PIL renderer (1280x720, 24 fps, piped to ffmpeg)
run.py                  CLI: generate / validate (common/video5.py) + stats / tamper / framecheck / framehash
checks.py               stats, tamper test, decoded-frame checks, frame hashes
test_tangled_braid.py   pytest suite

python run.py generate --n 50 --seed 1050 --output-dir ../output_videos/tangled_braid_preview
python run.py validate --dataset-dir ../output_videos/tangled_braid_preview
python run.py framecheck --dataset-dir ../output_videos/tangled_braid_preview
python run.py stats --n 5000
python -m pytest test_tangled_braid.py -q
```

`annotations.jsonl` is the final output (sbatch entry
`tangled_braid:run.py::none:none`).
