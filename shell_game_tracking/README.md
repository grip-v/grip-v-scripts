# shell_game_tracking

Video-QA sub-benchmark for **tracking hidden objects through permutations**
(the shell game). Identical cups stand on mats; at the start every cup is
lifted for a moment and the tokens under some of them are shown; then the
cups are lowered and moved around with the tokens hidden inside. Only by
following every move can one tell where a token ends up. v2 moves the
dataset to the shared PIL pipeline with the 10-question ladder
(`LADDER = "ten"` in `questions.py`), five scenarios, and an independent
decoder that follows the drawn cups frame by frame.

## What each clip shows

- 3-6 identical cups (one look per clip: 8 colours x 4 shapes - cup, dome,
  box, bell - x 4 patterns, 70-132 px wide, 3-4 px outline) standing on
  mats on a table. The cups are deliberately all alike, so a single frame
  never tells which cup is which; questions name positions, never cups.
- Tokens (1-3, each a unique colour - red, yellow, green, blue, purple,
  orange, pink, cyan - and one of ball / star / cube / diamond / triangle,
  32-50 px) are drawn only while the cup above them is lifted high enough:
  at the start reveal (all cups up for 16-26 frames, then lowered over 5-8)
  and at peeks. At no other moment is any pixel of a token colour drawn.
- A move: each moving cup steps out sideways of its straight path (a cup
  going right rises, one going left drops towards the viewer; in the grid,
  the left normal of the path), slides across, and steps back in, so the
  two cups of a swap pass each other well apart and a cup passing over
  others clears them. Moves last 12-53 frames (longer moves take longer:
  no cup travels more than 40 px per frame), with 6-16 frames between them.
- Clips 4-10 s (1280x720, 24 fps); 3-8 moves.

### Scenarios (`scene["scenario"]`, 20 % each)

| scenario | setup | moves |
|---|---|---|
| `classic` | 3-6 cups in a row, one token | pair swaps (up to 3 positions apart) |
| `multi` | 4-6 cups in a row, 2-3 different tokens under different cups | pair swaps |
| `rotate` | 4-6 cups in a row, one token | pair swaps, three cups turning at once (each to the next of three positions), or two pair swaps at the same time |
| `grid` | 4 or 6 cups on two rows (2x2 / 2x3), one token | pair swaps in any direction: sideways, front-back, diagonal |
| `peek` | 3-6 cups in a row, one token | pair swaps; 1-3 times between swaps a single cup is lifted for 10-14 frames, showing the token or nothing (half the time the token's cup) |

### Random knobs

Scenario; cup count; cup width, height, shape, pattern and colour; token
count, colours, shapes, size and starting cups; row pitch, row position,
grid spacing; step-out height factor; move count, kinds, positions and
durations; gaps; reveal length and lift; peek times, lengths and cups; end
pause; background (7 palettes x flat / gradient / vignette), table colour,
mat style (ellipse, square, ring), outline width. Colours are picked with a
minimum contrast: cup vs table and background (luminance), tokens vs table,
cup and background (RGB distance >= 110).

## On-screen text

- **Clock** (the only text): fixed dark panel at the top-left, "s.t s"
  truncated to 0.1 s, 36 px bold (glyphs >= 26 px), white on near-black;
  no cup comes within 14 px of it (checked on every frame).
- No labels or numbers: positions are named by where they are ("the
  leftmost position", "the third position from the left", "the
  top-right position"), tokens by colour and shape, moves by order, moments
  by the clock. Positions in a row of up to six (or a 2x3 grid) are easy to
  name without numbers, so no slot numbers are drawn. `scene["labels"]` is
  empty and the renderer draws none.

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total
10) among the combinations the clip can fill (a candidate pool per level
first). Different templates first; above L1 motion questions first, with at
most one single-frame question per level; no (derivation_key, params)
repeats; any question that would need more than 4 fps is dropped. 4 options
except yes/no, which of two, and 3-cup clips. 31 templates:

- **L1 observation:** how many cups / swaps (moves) / tokens / peeks; under
  which position a token starts; which two positions trade cups in the
  first swap; is the token under the cup lifted at a clock time; in which
  position that cup is.
- **L2 relational:** does the token's cup take part in the k-th move;
  does a token end further left, further right or in the same position;
  which token's cup moves first; which of two tokens ends further left;
  where the token's cup goes the first time it moves; which of two cups
  (named by start position) changes position more often; which of two
  positions a cup leaves first.
- **L3 measurement:** how many times the token's cup changes position; how
  many times a cup leaves a position; the clock time the token's cup first
  moves / the k-th move starts; moves starting between two clock readings;
  how many different positions a token occupies.
- **L4 integration:** where a token ends; where it is at a clock reading;
  does its cup come back to its start before the last move; where the cup
  that starts in a position ends; what is under a position at the end
  (multi); where the cup lifted at a peek started.
- **L5 counterfactual (re-played):** where a token would end if the k-th
  move had not happened, if two named positions traded once more after the
  end, if it had started under another cup, or if two consecutive moves had
  happened in the opposite order.

Over 5000 scenes 100 % of L2-L5 questions need the motion.

## Margins (no near ties)

- Every move start and end, peek and the end of the reveal is a whole frame,
  >= 2 frames from every other event and >= 12 frames (0.5 s) from both ends
  of the clip; moves are >= 12 frames long and >= 6 frames apart; compared
  first moves are >= 8 frames apart.
- Cup boxes stay >= 6 px apart on every frame (no overlap, ever); a cup
  moves visibly off its mat from the first frame of a move (sine step-out)
  and is back exactly on the last.
- Clock answers are the reading on the event frame, options 0.5 s apart;
  "where is the token at a clock reading" only when its cup rests from 7
  frames before to 7 after the frames showing it; "moves between two
  readings" only when no move is within 7 frames of either reading.
- Token colours are unique in a clip; a token is shown only when fully
  clear of every cup (checked).

## min_fps

Every question stores `params["min_fps"]`: the smallest of 1/2/3/4/6/8/12/24
fps that puts samples where the answer needs them. Counting moves needs a
sample in every move and in every gap (shortest move / gap); following a
cup needs two samples inside each of its moves (half the shortest such
move), since with identical cups one sample mid-swap cannot tell which cup
went where; what-ifs need every move followed; clock answers 4 fps; the
start reveal and peeks their visible frames. Over 5000 scenes: 6.3 % of
questions need 1 fps, 13.8 % <= 2 fps, 100 % <= 4 fps.

## Closed loop

- **Questions** store `derivation_key` + `params` (token by name, positions
  by name, move index, clock tenths) + `min_fps`; ground truth comes from
  `sim.play` (which cup is where after each move) and the scene's frames.
- **Independent answers** (`answers.py`) never read the moves, peeks or
  `sim.play`. They read `render.track`: on every frame the cups' bottom
  centres as an UNORDERED list (sorted by x), the tokens visible on that
  frame and the mats. The decoder follows every cup from frame to frame
  (constant-velocity prediction and nearest match; it refuses to answer if
  a match is ever unclear), classifies every stretch off a mat as a lift
  (straight up and back onto the same mat) or a move, reads each move as a
  permutation of mats, binds tokens to cups from the start reveal, checks
  every peek against its tracking, and re-plays what-ifs on its own list of
  permutations.
- **`sim.check`** rejects scenes with events too close to each other or the
  ends, short or too-fast moves, cups closer than 6 px, a cup off the frame
  or near the clock, a token covered while shown, or look-alike tokens.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`frame_check.py`** decodes the MP4s: the clock (template match on every
  6th frame, every event frame and every frame a question names); cups read
  from the pixels (everything that differs from the empty table, one blob
  per cup, centre of mass and solid bottom row) equal to the track on every
  frame; each shown token found at its spot, and no token colour anywhere
  on any other frame (no leaks); every move seen starting and ending on its
  exact frames; cup counts, clock answers and peek positions read off the
  frames equal to the answers. Shifting a clip by one frame, or painting a
  token colour onto a hidden frame, makes the check fail.

## Determinism

All randomness comes from the `random.Random(seed)` passed in; times are
whole frames; no global RNG, time, uuid or set order decides anything. Same
seed -> byte-identical annotations in separate processes with different
`PYTHONHASHSEED`; the same scene renders identical raw frames (hashed);
regenerating seeds into a fresh folder reproduces `annotations.jsonl`
exactly (apart from `render_meta` timing when rendering).

## Diversity (5000 dry-run seeds)

5000 / 5000 unique scenes; scenarios classic 1035 / peek 1014 / grid 1012 /
multi 973 / rotate 966; cups 3-6 (row) and 2x2 / 2x3 (grid), 16
scenario x cup-count combinations; 3-8 moves (1415 / 1764 / 1223 / 535 / 61
/ 2); 1-3 tokens (4027 / 664 / 309) from 40 colour x shape looks; 0-3 peeks;
move lengths 12-53 frames; cup widths 70-132 px; 8 cup colours x 4 shapes x
4 patterns, 3 mat styles, 7 backgrounds x 3 styles; clips 4-10 s (47 % over
9 s); 15172 distinct (question, answer) pairs in 50000 questions; answer
letters A/B/C/D 25.2 / 25.0 / 25.3 / 24.6 %; level counts L1 1/2/3 =
2323/1792/885, L2 2/3 = 2979/2021, L3 2926/2074, L4 1/2/3 = 2258/1861/881,
L5 2192/1896/912 (22 combinations); min_fps: 6.3 % of questions need 1 fps,
13.8 % <= 2 fps, 100 % <= 4 fps (most need 3-4 fps: following identical
cups needs two samples inside a move). Generation retries: 235 scenes
(4.7 %) needed more than one attempt (mean 1.05, max 4; too long 265, a cup
off the frame 10); no seed fails, no warnings.

## Files and commands

```
sim.py                      scene generation, moves, kinematics, checks (pure, seeded)
questions.py                the 10-question ladder (LADDER = "ten")
answers.py                  independent re-derivation by following the drawn cups
render.py                   PIL renderer (1280x720, 24 fps, piped to ffmpeg); render.track = what is drawn
run.py                      CLI: generate / validate (common/video5.py)
generate.py                 same as `run.py generate` (generate_all_datasets.sbatch)
flatten.py                  annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
frame_check.py              decoded-frame checks of rendered clips
test_shell_game_tracking.py pytest suite

python run.py generate --n 50 --seed 3 --output-dir ../output_videos/shell_game_tracking_preview
python run.py validate --dataset-dir ../output_videos/shell_game_tracking_preview
python frame_check.py --dataset-dir ../output_videos/shell_game_tracking_preview
python -m pytest test_shell_game_tracking.py -q
```
