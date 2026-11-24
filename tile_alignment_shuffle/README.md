# tile_alignment_shuffle

Video-QA sub-benchmark for **following tiles through a shuffle and spotting
when a line of one kind forms**. v3 reworks v2 on the shared PIL + ffmpeg
pipeline (`common/video5.py`) with the **10-question ladder**
(`common/ladder5.py`, `LADDER = "ten"`), five scenarios, clips of 10 s or
less, and a clock.

What changed from v2:
- The "Shuffle k" counter is gone. It described what was happening and printed the
  answer to "how many shuffles". A clock (0.1 s) took its place.
- Five scenarios instead of one: different kinds of move, not just pair swaps.
- Clips are 4-10 s (v2: 15-27 s), with wider ranges for speed, layout, theme and tile style.
- Answer shortcuts were removed:
  - The line now forms on the last move in only ~1 clip in 4 (v2: usually), so "pick the last
    move" no longer wins.
  - What-if line counts and other counts are balanced so no single value dominates.
- Transient lines are gone. No board ever shows a complete line except the final one, from
  the move that completes it onward. v2 could show a line that formed and broke again, so
  "when does the first line appear" could point to a line missing from the final board.
- 10 questions per clip from 28 templates (v2: 5 from 15).

## What each clip shows

- **The grid:** 3×3 to 6×6 bordered tiles, row numbers on the left, column numbers on top.
  Each tile shows one solid shape or one coloured symbol.
  - There are N kinds of tile with N copies each, every kind a different shape or symbol in a
    different colour.
  - Shapes: circle, square, triangle, star, heart, hexagon, diamond, cross, crescent, pentagon.
  - Symbols: `@ & # ? % $ Ω Σ`.
  - Colours: red, blue, green, orange, purple, teal, pink, brown, yellow, black. Any two are at
    least ΔE 40 apart (CIE76), so they stay distinct after downscaling.
- **False tiles:** about 55% of clips have 1-2 of them.
  - A false tile has the shape or symbol of a real kind but a colour no other tile has.
  - About half of these clips end on a near miss: a line of one kind except for the false tile.
- **Moves:** one at a time, each a short glide followed by a rest. Every moving tile looks
  different from the tiles it moves with, so a move can be read from the boards before and
  after it.

| scenario | one move | grid | moves |
|---|---|---|---|
| `swap` | two tiles anywhere trade places, passing each other on bowed paths | 3-6 | 5-11 |
| `neighbour` | two side-by-side tiles trade places | 3-6 | 6-11 |
| `cycle` | three tiles rotate places at once (A→B→C→A) on straight paths | 3-6 | 4-9 |
| `slide` | a whole row or column slides one cell; the end tile slides under the board edge and comes back in at the other end | 3-6 | 4-8 |
| `double` | two separate pairs trade places at the same time | 4-6 | 4-9 |

- **Lines:** a complete line is a full row, column or diagonal of one kind.
  - **~73% of clips:** exactly one line appears, right after move f, and stays (later moves
    never touch its cells). f is the last move about a quarter of the time, otherwise 40-90% of
    the way through.
  - **The rest:** no board ever shows a line.
  - The start board never has one.
- **Paths stay on the board:**
  - Swapping tiles lift (they grow 5-10% and cast a shadow) and pass on opposite sides, at least
    0.85 cells apart centre to centre.
  - The path style is the first of four that keeps both tiles on the board: an early-opening bow
    first, then gentler sine bows. Every pair of cells on every grid size has a legal path.
  - Cycles use straight paths. Collinear triples are not used, and the three tiles stay at least
    0.85 cells apart throughout.
  - Slides are clipped to the board, so nothing ever moves over the numbers or the clock.

### Random knobs (per clip)

- **Content:** scenario, grid size, number of moves, the kinds (shape or symbol, colour), false
  tiles and near misses, which line forms and when, and the start board.
- **Timing:** move time 10-18 frames, pause 5-10 frames, intro 0.5-1.0 s, final hold 0.6-1.25 s.
- **Board placement:** size 360-600 px (cells 76-200 px) and position anywhere that keeps the
  numbers and clock clear.
- **Look:**
  - Background theme: paper, slate, sky, sage, sand, night or wood.
  - Card colour: ivory, white or cream.
  - Corner radius, tile gap, shape/symbol scale, border width and lift amount.

5000 dry-run scenes give 5000 unique scene signatures and 5000 distinct tile shuffles, even
ignoring look, timing and layout.

## On-screen text

| text | why it is needed |
|---|---|
| clock, top-left, dark box, `x.y s` | required in every clip; clock-time questions read it |
| row numbers 1..N (left of the board) and column numbers 1..N (above it) | questions name one particular tile by its cell, because copies of a kind look identical. Counting to the 5th or 6th row on a frame shrunk to 640×360 is error-prone; the numbers are identifiers only, never an answer |

Nothing else appears: no counter, title, legend, hint or status text.

- **Label style:**
  - Labels are 36 px bold (digits about 26 px tall) in a fixed spot per row and column, in the
    theme's ink colour (dark on light themes, light on dark ones).
  - They sit 34 px off the board edge and never move.
- **Stored data:** labels are stored in `scene["labels"]` (text, x, y, size, role), which the
  renderer, `sim.check` and `verify_frames.py` all use.
- **Guarantees:**
  - `sim.check` rejects any layout where a label or the clock box touches the board, the frame
    edge or another label.
  - `verify_frames.py` template-matches every number, measures its ink height, and confirms its
    pixels never change in any frame.

## The 10-question ladder

Per clip there are 1-3 / 2-3 / 2-3 / 1-3 / 1-3 questions on levels 1-5, with the counts drawn
per seed. The 10 questions are ordered by level as q01..q10.

- **Templates:** each is used at most once per clip before any repeats, and no
  `derivation_key + params` pair repeats.
- **Naming:** tiles are named by look ("blue star") or, for one particular tile, by the cell it
  starts or ends in. Moves are named by order ("the 3rd move"), and questions that count or
  order moves define a move.
- **Options:** every question has 4 distinct options, exactly one correct. Distractors are the
  tile's other cells along its path, neighbouring cells, other kinds on the board (or a shape and
  a colour from the board that never appear together), nearby counts, and the clock times of
  other moves.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `count_design` how many tiles of a kind; `n_false` tiles that match nothing; `n_kinds` different kinds; `n_moves` how many moves; `tile_at_start` / `tile_at_end` kind in a cell; `line_where` where the line is at the end |
| L2 relational | `partner` kind that moves into the cell a tile leaves the first time it moves; `first_mover` which of two tiles moves first; `more_moves` which of two tiles moves more often; `aligned_design` kind forming the line |
| L3 measurement | `moves_of` how many times a tile moves; `end_cell` where a tile ends; `cell_after` where a tile is right after move k; `first_rest_time` clock when a tile first comes to rest in a new cell; `line_after` after which move the line first appears; `line_time` clock when it appears |
| L4 integration | `most_moves` tile that moves most (unique); `start_of` where the tile ending in a cell started; `never_moved` tiles that never move; `back_home` tiles that move yet end where they began; `cells_visited` distinct cells a tile sits in; `line_origin` line tiles that started in that line; `partner_origin` (swap / neighbour / double) start cell of the tile a tile first trades with |
| L5 counterfactual | `cf_skip` where a tile would end if move k were skipped; `cf_skip_moves` how often it would move then; `cf_skip_lines` lines on the final board then; `cf_false_real` lines if a false tile were the real kind it imitates. "Skipped" means every other move still moves whatever tiles sit in the same cells, in the same way; all are re-simulated |

In 5000 clips, 95.7% of L2-L5 questions need the motion (`requires_motion`).

### Shortcut checks (5000 clips)

- **"After which move does the line first appear":** the answer is the last move in about 25% of
  cases, so "pick the largest option" is at chance.
- **`cf_skip_lines`:** the move k is picked so that 0 and 1 are about equally likely.
- **`n_false` and `back_home`:** asked less often when the answer would be 0, so 0 is not the
  usual answer.

Exact figures are in the rework report.

## Timing, frame sampling and `min_fps`

- **Frame format:** 1280×720 at 24 fps, at most 240 frames (10 s).
- **Event spacing:**
  - Every question-relevant event (a move ending) is at least 0.5 s from the start and the end,
    and at least 5 frames (the pause) from the next event.
  - Each move leaves a persistent change: the board stays as it is until the next move.
- **Clock-time questions:**
  - Asked only where the clock shows the same reading one frame before and after the event, so
    the answer never sits on a 0.1 s boundary.
  - In line clips, the intro length is chosen so that the moment the line completes always
    qualifies.
  - The wrong options are other moves' times, at least one move period (≥ 0.67 s) apart.
- **`params["min_fps"]`:** each question stores the sampling rate needed to see its answer.
  - **Tracking questions:** `24 / (move + pause)`, i.e. one sample per move. Any single
    sample, even mid-move, shows the empty slots and the flying tiles' looks, which pins down
    the move.
  - **Clock questions:** twice that rate, because the right option must be the nearest one to
    the first sample after the event.
  - **Start or end board questions:** `24 / intro` or `24 / hold`.
  - **Whole-clip counts:** `24 / n_frames`.
  - **Over 5000 clips:** 20% of questions work at 1 fps, 94% at 2 fps, 100% at 4 fps
    (max 3.0).
- **Sizes at 1280×720:**
  - Tiles are 69-190 px.
  - Borders are 3-4 px.
  - Digits are about 26 px tall, so still about 13 px at 640×360.

## Closed loop

- **Question records:** every question stores its `derivation_key` and `params`
  (incl. `min_fps`).
- **`answers.py`:** re-derives every answer without sim.py's bookkeeping.
  - It reads `render.state` for every frame and snaps each resting tile's pixel centre to a
    cell, giving the resting boards.
  - It recovers each move as the tiles that changed cell between boards.
  - It recomputes clock readings from frame numbers.
  - It does its own line scans, counts, paths and what-if replays.
- **`sim.check`:** rejects a scene in any of these cases:
  - the move count or grid size is out of range;
  - the line rule is broken (a line before move f, a second line, a move touching the line after
    it forms, or a line in a no-line clip);
  - two kinds share a colour, or a false tile is wrong;
  - a move can't be read from the boards (lookalike movers, or a slide that is invisible or
    direction-ambiguous);
  - moving tiles come within 0.85 cells of each other, or a path leaves the board;
  - the timing breaks the 10 s, 0.5 s or pause rules;
  - labels or the clock overlap the board, each other or the frame edge.
- **`run.py validate`:**
  - fails any mismatch, any scene or question that does not rebuild identically from its seed,
    any ladder error and any check failure;
  - with video, also checks the MP4 resolution, duration and motion.
- **`verify_frames.py` (pixel level, on rendered clips):**
  - decodes each MP4 and finds the rest frames;
  - classifies every cell against freshly drawn sprites, with a clear margin over the
    runner-up;
  - rebuilds the moves and re-answers every question from the pixels;
  - template-matches the clock at every move end and every row/column number;
  - checks that nothing ever covers the numbers or the clock.

## Determinism

- **Randomness:** everything comes from the `random.Random(seed)` passed in. Nothing uses the
  global RNG, time, uuid or set order.
- **Same seed:** gives byte-identical scene and question JSON, across processes with different
  `PYTHONHASHSEED`.
- **Frames:** rendering is deterministic. The raw-frame sha1 is the same in separate processes.
- **Re-rendering:** gives the same annotations except `render_meta.render_time_s`, which is a
  wall-clock time written by the shared pipeline.

## Files and commands

```
sim.py                 scene generation + rule checks (pure, seeded)
questions.py           the 10-question ladder (28 templates)
answers.py             independent answers from the renderer's per-frame geometry (validator only)
render.py              PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py                 CLI: generate / validate (shared pipeline in common/video5.py)
generate.py            same as `run.py generate` (kept for generate_all_datasets.sbatch)
flatten_annotations.py annotations.jsonl -> dataset_final.csv / .jsonl (one row per question)
verify_frames.py       pixel-level check of rendered clips (needs opencv-python)
test_tile_alignment_shuffle.py  determinism, ladder, uniqueness, labels, tamper, hand-checked clip

python run.py generate --n 50 --seed 10005 --output-dir ../output_videos/tile_alignment_shuffle_preview
python run.py validate --dataset-dir ../output_videos/tile_alignment_shuffle_preview
python verify_frames.py --dataset-dir ../output_videos/tile_alignment_shuffle_preview
python run.py generate --n 2000 --no-render --output-dir <dir> && python run.py validate --dataset-dir <dir> --skip-video
python -m pytest -q test_tile_alignment_shuffle.py
```
