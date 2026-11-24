# chess_dance_trace

Video-QA sub-benchmark for **tracking several chess pieces that move at the
same time**, checking each move against the piece's movement rules, and
reasoning about what would change if one move were different. v3 is a
rework on the shared PIL + ffmpeg pipeline (`common/video5.py`) with the
10-question ladder (`LADDER = "ten"` in `questions.py`).

## What each clip shows

- **Board:** 6×6, 7×7 or 8×8 squares; six muted board colour schemes, seven
  plain/gradient/dotted/striped backgrounds (light and dark), board size
  56-86 px per square and a random position in the frame.
- **Pieces:** 3-9 pieces, each a queen, king, rook, bishop or knight
  (python-chess artwork). Types repeat; every piece has its own colour from
  11 saturated colours (pairwise ≥ 90 RGB apart and far from every board and
  background colour), so "the teal knight" is always unique. Three piece
  styles: a coloured disc or a coloured rounded tile with the figure on it,
  or the figure itself filled with the colour.
- **Motion:** rounds of moves. In a round 1-3 pieces glide to their new
  squares at the same moment (smoothstep easing, knights hop in an arc), then
  the board rests. A captured piece disappears on the frame its capturer
  lands. Clips are 4.7-10 s at 24 fps; the first glide starts ≥ 0.58 s
  after the start and the last landing is ≥ 0.58 s before the end.

### Scenarios (`scene["scenario"]`, about 20% each)

| scenario | setup |
|---|---|
| `dance` | 4-9 pieces, 1-3 movers per round, 4-6 rounds, every move legal |
| `rogue` | like `dance`, but exactly one move in a middle round breaks the rules: the wrong pattern for its type (rook moving diagonally, king moving two squares, queen jumping like a knight, knight stepping like a king) or a slide through a piece |
| `relay` | 3-7 pieces, one piece per round, 6-9 faster rounds |
| `capture` | 5-9 pieces, 1-2 movers per round, 2-3 moves land on a piece that stands still that round; it disappears (captures are colour-blind) |
| `holes` | 3-9 squares are cut out of the board (dark hatched squares); nothing lands on a hole and slides cannot cross one, knights jump |

A move is legal when it follows the piece's pattern, lands on a square that
is free when the round begins (or, in `capture`, on a piece standing still)
and does not slide through a square occupied (or a hole) when the round
begins. Movers in one round never share a destination or cross paths, stay
≥ 0.95 squares apart on screen, and no glide (knight arcs included) passes
within 0.75 squares of a piece standing still, except the deliberate rogue
slide through a piece; a captured piece is the only one ever touched.

### Random knobs

Scenario, board size, piece count, types, colours, start squares, movers per
round, every destination, glide length (12-22 frames), pause (9-14 frames),
intro and closing hold, board scheme, background and pattern, square size,
board position, piece style and size (72-98% of a square), knight arc
height, rank labels left/right/both, drop shadows, hole layout.

## On-screen text

- **Clock** (always): fixed dark panel at the top-left, "s.t s", truncated
  to 0.1 s, 36 px bold (cap height 26 px). The board is always placed to its
  right, so nothing passes under it.
- **Board coordinates** (always): file letters A-H under the board and rank
  numbers 1-8 on the left, right or both sides, 34 px bold (cap height
  25 px), in a high-contrast ink chosen from the background. They are needed
  because squares are the answer of several questions (start/end square,
  square at a clock moment, never-stood-on square) and are named in
  counterfactuals ("if square D5 had been cut out"): every square of a board
  looks alike, so colour or shape cannot name one, and unnumbered grid lines
  would still leave the viewer counting from an edge with no agreed origin.
  They are identifiers only: they never move, never name a piece, a move or
  an answer. Uppercase file letters keep every glyph ≥ 24 px tall.
- The labels are part of the scene (`scene["labels"]`: text, centre, size)
  and the renderer draws exactly that list. `sim.check` rejects any scene
  whose label boxes overlap each other, the board, the clock panel or any
  box a piece sweeps (knight arcs included; labels are never above the
  board, where arcs go).
- Nothing else: no titles, round counters, move labels, trails or hints.

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3
(total 10, `L5.ten_counts`), questions are spread over as many templates as
possible (the scenario's own templates usually first), ordered by level and
numbered q01..q10. Every question has 4 options. Pieces are named by colour
and type, squares by coordinates, moments by the clock.

- **L1 observation:** pieces at the start; pieces of one type; a piece's
  start square; a piece's end square; colour of the only piece of a type;
  number of missing squares (holes); pieces left at the end (capture).
- **L2 relational:** which of four pieces moves first; which of four pieces
  moves at the same moment as X; which of two pieces moves more; which of two
  starts moving earlier; which piece captures X (capture).
- **L3 measurement:** squares travelled in X's k-th move; clock reading when
  X comes to rest after its k-th move; how many times X moves; X's square
  when the clock shows t; how many pieces are moving when the clock shows t;
  number of captures (capture).
- **L4 integration:** which piece (if any) makes an illegal move; clock
  reading when the illegal move ends (rogue); total moves; which piece moves
  most; which of four squares X never stands on; X's total distance; king
  steps between X's start and end squares; which piece captures most / is
  captured first (capture).
- **L5 counterfactual (re-simulated):** how many pieces X would attack after
  the round had it skipped its k-th move; had X never left its start square,
  which piece would first find a move blocked (or none); had square S been a
  hole, which piece would first be unable to move (holes); had X never
  moved, how many pieces would be left at the end (capture, full replay with
  captures); had the rogue piece stayed put, would its next move be legal or
  which rule would it break (rogue).

L2-L5 questions all set `requires_motion=True`.

## Margins (no near ties)

- Clock moments are chosen so every frame showing that reading lies ≥ 2
  frames inside a rest of the piece (or ≥ 2 frames inside a glide for
  "how many are moving"), and never inside the rogue glide.
- Compared first moves are in different rounds (≥ 0.87 s apart).
- Clock-reading answers are offered in 0.5 s steps around the true reading
  of the landing frame (frame-exact, since every event is on a frame).
- Counts and squares are exact; "which of four" questions have exactly one
  qualifying option (checked again by answers.py).

## min_fps

Every question stores `params["min_fps"]`, the smallest of 1/2/3/4/6/...
fps that puts at least one sample in the shortest interval the answer
needs: the rest of the piece (square questions), the pauses around the
relevant rounds (counts, orders, captures, counterfactuals), the glide
(moving count, rogue slides) or ±0.25 s around a landing (clock answers,
4 fps). Over 5000 scenes: 8% of questions need ≤ 1 fps, 63% ≤ 2 fps,
100% ≤ 4 fps.

## Closed loop

- **Questions** store `derivation_key` + `params` (piece ids, move number,
  clock tenths, option sets) + `min_fps`.
- **Generation** (`sim.py`) uses python-chess (`Board.attacks` with pawns
  standing in for pieces, holes and the squares outside a smaller board) as
  its geometric oracle.
- **Independent answers** (`answers.py`) never read the move list or any
  flag. They decode the clip from the renderer's per-frame track
  (`render.track`: every piece's on-screen centre and visibility in every
  frame): glides are runs of changing frames, rest squares come from pixel
  centres through the board geometry, rounds from shared start frames,
  captures from a piece vanishing on the frame another lands on its square,
  clock readings from frame numbers. Legality, attacks and every
  counterfactual replay use hand-written direction-stepping geometry (no
  python-chess).
- **`sim.check`** rejects scenes with out-of-range counts, duplicate or
  too-similar colours, events within 0.5 s of either end, label overlaps,
  movers too close on screen, legality/capture flags that do not match the
  geometry, or too few askable questions per level.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`frame_check.py`** decodes the MP4s: at the middle of every rest it
  finds each piece by its colour blob and compares the square with the
  replay (captured pieces must be gone); at every frame showing a clock
  moment a question names it template-matches the clock against all 100
  readings and re-reads the square / counts pieces away from their glide
  start; checks clock answers against the landing frame; template-matches
  every coordinate label against all 16 glyphs; checks hole squares; repeats
  the label-overlap geometry.

## Determinism

All randomness comes from the `random.Random(seed)` passed in; no global
RNG, time, uuid or set order decides anything (sets are sorted before use).
Same seed → identical scene + questions in separate processes with
different `PYTHONHASHSEED`; rendering the same scene twice gives identical
raw frames (tested by hashing frames).

## Files and commands

```
sim.py                    scene generation, timing, rule checks (pure, seeded)
questions.py              the 10-question ladder (LADDER = "ten")
answers.py                independent re-derivation from the rendered track
render.py                 PIL renderer (1280x720, 24 fps, piped to ffmpeg)
run.py                    CLI: generate / validate (common/video5.py)
frame_check.py            decoded-frame checks of rendered clips
test_chess_dance_trace.py pytest suite

python run.py generate --n 50 --seed 200 --output-dir ../output_videos/chess_dance_trace
python run.py validate --dataset-dir ../output_videos/chess_dance_trace
python frame_check.py --dataset-dir ../output_videos/chess_dance_trace
python -m pytest test_chess_dance_trace.py -q
```

`annotations.jsonl` is the final output (sbatch entry
`chess_dance_trace:run.py::none:none`).
