# rainbow_bagel

Video-QA sub-benchmark adapted from **Disassembled Rainbow Bagel (Jane Street, May 2019)**: object permanence and exact area.

## What each clip shows

A bagel of 3-4 concentric rings is cut into quarters (the original setup, kept in every clip: 3-4 rings, quarters, two trays). One at a time or in bunches (see scenarios), the quarters lift off, turn to the same upright arch shape, glide among each other, pass **completely** behind an opaque screen and come to rest in the left or right tray below. All pieces of one ring look exactly alike (same colour, shape and pattern), so where a piece came from can only be known by following it. One piece, from the innermost or the outermost ring, turns **gold in mid-air**, at least 150 px from where it was cut and never earlier than 0.5 s after it lifts off. Its ring and slice have to be traced back through the look-alikes. In the original version the gold piece was gold from the first frame, so its origin could be read off frame 0 without tracking.

Ring *i* (ring 1 is the innermost) runs from radius r0+i-1 to r0+i units, where r0 is the hole radius. One unit is one tick on the ruler. A quarter of ring *i* has area π·((r0+i)² − (r0+i−1)²)/4.

Clips are 1280x720, 24 fps, at most 10 s (7-10 s). Every question-relevant event (cut, lift-offs, gold turn, arrivals) happens at least 0.5 s after the start and at least 0.5 s before the end.

## Scenarios (`scene["scenario"]`)

Every clip is 3-4 rings in quarters with two trays; the scenarios vary only things the original allows.

| scenario | what differs | rings |
|---|---|---|
| `wide_hole` | the hole has radius 2 units instead of 1 | 3 |
| `one_by_one` | pieces lift off one at a time, 0.25-0.33 s apart | 3-4 |
| `bunches` | pieces lift off in bunches of 3-4, 0.5-0.8 s apart | 3-4 |
| `long_hide` | every piece is completely behind the screen for >= 21 frames (~1 s); in the other scenarios 14-20 frames | 3-4 |

Across all four, `scene["gold_ring"]` picks whether the gold piece comes from the innermost or the outermost ring (drawn once per seed, roughly half and half). Each scenario is visible in the video: hole size against the ruler, how pieces lift off, how long they stay hidden.

## Random knobs

Bagel position (centre x 470-810 px) and size (outer radius 96-108 px; ring width ≥ 20 px). Ruler side, height and length (r0+k+1 or +2 ticks). Tray assignment (at least two pieces per tray). Lift-off order and bunch sizes and spacing (per scenario). Each piece's glide time, time fully behind the screen (per scenario), drift behind the screen (±40-240 px), exit point and settle time. Which piece turns gold, and when. Theme (7 backgrounds, 2 of them dark). Palette: a single colour from 7, or "rainbow" with a different colour per ring from 6 (pieces of one ring always look alike). Piece pattern: plain, sesame seeds, speckles or a band. Screen colour (5) and pattern (5). Colours are chosen to contrast with the background, the trays and the gold.

## On-screen text

The **clock** in the top-left corner (seconds to 0.1 s, 34 px bold, ≥ 24 px tall, on its own panel) is the only text. It never covers the action: `sim.check` tests every piece outline on every frame against the clock box.

There are no titles, captions, tray names or rule hints. The previous version's "3 rings; follow the gold piece" caption gave away the ring-count question and has been removed. The ruler is a measuring aid with **unnumbered** ticks one unit apart. Numbers aren't needed, because the questions say "its ticks are 1 unit apart", and an unnumbered ruler can't print an answer. Trays are named by position (left / middle / right) and pieces by where they sat in the whole bagel ("the top-left quarter of ring 2"), which is visible before the cut. The scene dict stores the text it shows: `scene["text"] = {"timer": {box, font_px, format}, "labels": []}`. The renderer draws the clock from it, and the tests check it against `render.layout`.

## The 10-question ladder (`LADDER = "ten"`)

Each clip has 10 multiple-choice questions (4 options each), ordered by level as q01..q10. The per-level counts are drawn per seed: L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3. No key+params pair repeats within a clip.

- **L1 observation**: `n_pieces`, `n_rings`, `tray_count`
- **L2 relational**: `tray_more_pieces` and `tray_more_outer` ("which tray, and by how many"), `ring_most` (which ring has the most pieces in a tray), `first_lift` (which of four pieces lifts off first / last)
- **L3 measurement**: `tray_area` and `gold_area` (with the ruler), `ruler_radius` (hole / outer radius), `land_time` (clock time when the first / last piece comes to rest in a tray), `gold_time` (clock time when a piece turns gold), `count_at_time` (pieces resting in a tray when the clock reads X)
- **L4 integration**: `marked_origin` (trace the gold piece back to its ring and slice), `piece_tray` (follow a named look-alike through the screen: which tray, and how many pieces are already there when it arrives), `first_land_origin` (the first piece to rest in a tray: which ring and slice), `gold_already`
- **L5 counterfactual** (re-simulated): `cf_move_ring` (a whole ring goes to the other tray: difference between the trays' areas), `cf_gold_other`, `cf_more_slices` (each ring cut into 2/3/6/8 pieces instead of quarters), `cf_count_move`, `cf_gold_delay` (the gold piece lifts off 1 s earlier / 1-2 s later), `cf_screen_slow` (every pass behind the screen takes 1-2 s longer)

Most L2-L5 questions need the motion (`requires_motion`). Templates that need it get higher pick weights.

**Margins.** Compared events are ≥ 8 frames (> 0.3 s) apart. A clock reading in a question is ≥ 0.5 s from any arrival or settle in that tray. Time options are 1.0 s apart. Area answers are never within 0.1 of a rounding boundary (1 decimal), and area options are ≥ 0.3 apart. Ties are an explicit option ("same number", "a tie for the most"), never an accident.

## Closed loop

Every question stores `derivation_key` + `params`. `answers.py` re-derives each answer **only from what the renderer draws**: every piece's outline and fill colour on every frame (`render.track`), plus the bagel centre, ruler tick spacing and tray/screen rectangles (`render.layout`). Rings and slices come from outline radii and angles on the first cut frame. Units come from radii divided by the ruler's tick spacing. Lift-off is the first frame a piece moves after the cut, rest is the first frame from which it never moves again, gold is the first frame it is filled gold, and trays are where pieces are drawn on the last frame. It never reads the plan's lift/land frames, trays, rings or the gold index.

Counterfactuals are re-simulated in `questions.py`: a changed copy of the scene is re-planned and its trays re-packed by `sim`. `answers.py` applies the same rule change to its own measurements.

`sim.check` rejects any scene where:

- a piece is fully behind the screen for < 13 frames, or hides twice;
- look-alikes behind the screen at the same time enter or exit within 8 frames *and* 90 px of each other;
- the setup leaves the original (not 3-4 rings, quarters, two trays, or the hole the scenario says), a one-by-one clip lifts two pieces together, or hide times break the scenario's rule;
- the gold piece is not from the ring `gold_ring` says, turns gold outside its glide or < 150 px from its slice, or can't be followed back at ≤ 8 fps;
- any outline leaves the frame or touches the clock or ruler;
- tray pieces touch or leave their tray;
- events fall within 0.5 s of either end;
- colours are too close.

`python run.py validate` fails a clip if an answer disagrees, if the seed rebuilds a different scene or questions, if `sim.check` fails, or if the ladder is malformed.

## Frame-sampling rules and `min_fps`

Every question stores `params["min_fps"]`, the sampling rate its answer needs:

- end-state questions: 1;
- clock readings: 2 (options 1 s apart);
- ordering of events g frames apart: the smallest rate with a sample between them (4 for the 8-frame minimum);
- following a look-alike: `sim.track_fps`, the smallest rate at which nearest-neighbour matching never confuses it with a same-ring piece for any sampling phase, combined with the rate needed to tell its screen entry/exit from a concurrent look-alike's. Tracking questions are only asked when this is ≤ 8 fps.
- the gold piece's time and origin: at least one sample must fall in the frames where it is already gold but not yet behind the screen (always ≥ 8 frames).

Every result persists (pieces stay where they rest, the gold stays gold). Pieces are ≥ 20 px wide, lines ≥ 3 px, and the clock and ruler stay readable at 640x360.

## Determinism

`sim.build(rng)` and `questions.build(scene, rng)` use only the seeded `random.Random`. No global random, numpy RNG, time, uuid or set/dict iteration order decides anything. The same seed gives byte-identical JSON across processes and `PYTHONHASHSEED` values, and identical raw frames when rendered twice. `build()` retries a seed's own rng stream until `check()` passes; the attempt count and reasons are kept in `scene["build_tries"]` / `scene["retry_reasons"]`.

## Files and commands

```
sim.py        scene plan, motion, tray packing, tracking-rate test, consistency check (pure, seeded)
questions.py  the 10-question ladder (templates, margins, min_fps, re-simulated counterfactuals)
answers.py    independent answers from the drawn outlines and colours (used by the validator)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)
test_rainbow_bagel.py   determinism, ladder, uniqueness, tamper, clock data, review fixes, original setup, hand checks

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/rainbow_bagel
python run.py validate --dataset-dir ../output_videos/rainbow_bagel
python -m pytest test_rainbow_bagel.py
```
