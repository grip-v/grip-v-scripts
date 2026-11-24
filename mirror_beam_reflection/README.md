# mirror_beam_reflection

Closed-loop 10-question video-QA benchmark on light bouncing off mirrors. A pulse of light leaves an emitter on the
edge of a board of square cells and moves one cell per step, leaving a lit trail; mirrors sit on cell diagonals and
turn the pulse 90 degrees when it reaches the centre of their cell. The pipeline is PIL frames piped to ffmpeg
(`common/video5.py`) with the 10-question ladder in `common/ladder5.py` (`LADDER = "ten"`). Clips are 1280x720 at
24 fps and 4.5-10 s long (108-240 frames). Each clip stops while light is still on the board, with at least one
bounce, split or stop still to come, so the later questions need the reflection rule, not just the pixels.

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | setup | what is drawn so it is readable |
|---|---|---|
| `classic` | one beam, two-sided mirrors | each mirror is a line on the cell diagonal ('/' or '\\') |
| `one_sided` | some mirrors are silvered on one face only; a pulse hitting the other face is absorbed | a dark hatched backing on the absorbing face; a stopped pulse leaves a ring in its colour |
| `splitter` | 1-2 half-mirrors: the pulse goes on straight and a reflected copy is sent off | the half-mirror is drawn dashed |
| `flip` | 1-2 pivoting mirrors turn to the other diagonal at a fixed moment; a pulse uses the diagonal it meets | a round hub at the centre; the mirror visibly turns over 0.3 s just before its turn time |
| `two_beams` | two emitters, two beam colours, launched at different times | each beam has its own colour for emitter, trail and head |

Any scenario may add 0-3 absorber blocks, drawn as dark squares; a pulse reaching one stops there. Emitters are a
box or wedge in the beam colour inside the entry cell, pointing in; the trail stays lit and the head is a larger round
dot in the same colour.

## Random knobs (`sim._world`, `sim._route`, `sim._style`, `sim.layout`)

- board 5-10 columns by 4-8 rows; beam path first: from each emitter a random route of 3-7 legs is walked, with a
  turning mirror every 1-4 cells (75 % turning towards the side with more room), then 1 to max(2, cells // 8) decoy
  mirrors are placed off the route; 0-3 absorbers; mirror kind '/' or '\\' per mirror;
- one-sided: 2 to max(2, mirrors // 2) mirrors, route mirrors face the oncoming beam 80 % of the time; splitter: 1-2
  half-mirrors; flip: 1-2 pivots, only on mirrors the beam really meets;
- speed: 0.22-0.45 s per cell; launch time 0.5-1.0 s (first beam), 1.4-3.0 s (second beam); a pivot turns at a frame
  instant, either before the first hit (0.8 s up to two steps before it) or from two steps after it to 1.5 s later;
- beam colours (red, green, orange, magenta, blue, distinct per clip), emitter edge and index, emitter shape (box or
  wedge);
- theme (dark twice as likely as light), background, board colour and pattern (plain, checker, dots), grid colour and
  style (solid, dashed, dotted), mirror colour, mirror width 5-8 px, mirror length 0.62-0.82 of a diagonal, trail
  width 5-9 px, head size 22-30 px;
- board position and cell size (60 px up to 96 px, always leaving room for the labels).

`sim.build` redraws until the board passes `sim.check` and `sim._ladder_ok` (see Closed loop).

## On-screen text

Two kinds of text, nothing else.

- **The clock**: a box at the top-left with the time in seconds to 0.1 s, white 44 px text on a dark box. The value is
  truncated from the frame number (`tenths = frame * 10 // 24`).
- **Cell labels**: a column letter above each column (A, B, ...) and a row number left of each row (1, 2, ...), 34 px
  bold in the theme's ink colour, stored in `scene["labels"]` (`sim.labels`) and drawn into the static layer. A cell
  is named "C4" = column C, row 4.

Why the labels are needed: all mirrors look alike, and a board can hold many of them, so colour or shape cannot name
one. Questions therefore name mirrors by cell, and the options for "where does the beam end" are cells and exit
points (edge plus row number or column letter). Labels are identifiers, never answers: they are the same text every
frame, sit outside the board, and never overlap the board, each other, the emitters (which are inside the board) or
the clock (`sim.check` keeps the board and its label strips inside the play area; `frame_check.py` tests glyph boxes
for overlap with the board, each other and the clock, and requires glyphs at least 18 px tall).

## The 10-question ladder (`LADDER = "ten"`)

Per clip the count per level is drawn from L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3 (total exactly 10), ordered by level
as q01..q10. Each question has four distinct options. `questions.build` retries up to 20 times to find a level split
the candidates can fill.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `n_mirrors`, `n_slash` (mirrors leaning like '/' at the start), `n_absorbers`, `n_special` (one-sided, half or pivoting mirrors, by their drawing), `entry_edge` (edge a colour comes in from) |
| L2 relational | `first_mirror` (cell of a beam's first turn), `turn_dir_first` (direction after it), `head_cell_at_t` (cell of a beam head at a clock time), `n_turns_by_t`, `n_heads_at_t` (moving heads on the board), `split_cell` (splitter) |
| L3 measurement | `time_turn_at` (clock reading when light first reaches a mirror), `time_between_mirrors`, `time_between_turns`, `cells_between`, `launch_time` (second beam), `speed` (cells per second), `n_lit_cells` (cells the trail passes through in the last frame) |
| L4 integration | `exit_point` (where a beam ends if the clip went on), `exit_time`, `time_after_end`, `n_turns_total`, `n_exits` (splitter) |
| L5 counterfactual | `cf_exit_point` (remove, rotate or add a mirror; pivots frozen; mirrors two-sided; dark blocks removed), `cf_exit_time` (twice as fast; a beam launched 0.5 / 1.0 / 1.5 s later in `two_beams` and `flip`), `cf_n_exits` (splitter) |

Margin rules (`questions.py`, `sim.py`):

- `EV` = 0.3 s: events a question relies on stay this far from the clock time it names; question times use the 0.5 s
  grid (`Ctx.grid_t`).
- Seconds answers are at least 0.015 s from a 0.1 s rounding boundary (`safe1`); exit and end times need the event
  between 0.15 and 0.85 of a tenth.
- Clock readings of an event (`event_ok`): the event falls 0.25-0.75 of the way through a frame interval and both
  frames around it show the same tenth.
- Option spacing: durations at least max(0.3 s, 15 %) apart; clock readings at least 0.5 s apart; counts are
  distinct integers near the answer.
- `head_cell_at_t`: over the frames that show that tenth, the head stays in one cell and at least 0.2 cell from any
  cell border, with one pulse of that colour on the board; `n_lit_cells` is dropped if a head sits within 0.15 cell
  of a border.
- Pivot turns are at least 2 steps from any hit on that mirror (`sim.check`, and `_cf_ok` for what-ifs); a turn lies
  at least 0.5 s inside the clip.
- Candidate questions are picked with distinct keys first; with probability 70 % the templates that need the motion
  (`requires_motion`) are preferred (`_pick`).

## Closed loop

`answers.py` does not read `sim.py`'s events. Factual answers are measured from what `render.state` draws, frame by
frame:

- pulse heads of each colour are linked into tracks by matching each head to the predicted position (previous
  position plus the last step) within 0.35 cell; a head with no match starts a new track;
- a corner (turn) is where two straight runs of a track meet: the intersection of the incoming and outgoing lines,
  timed from the measured speed; a split shows up as a track that starts inside the board (`split_cell`);
- `first_reach` gives the earliest time a head is at a cell centre, measured on the incoming side only, because right
  after a split the outgoing frames can hold two overlapping heads;
- the speed is the median over tracks; lit cells come from the trail polylines of the last frame.

Predictions and what-ifs use a separate tracer in `answers.run`: complex-number directions, pulses processed in time
order from a heap, mirrors read from the stored cell, kind, type and face. It does not import `sim.trace`;
`test_tracers_agree` compares the two on 60 seeds. `sim._ladder_ok` rejects boards that cannot fill the ladder (a
trial `questions.build` with its own fixed rng). `sim.check` also rejects loops, pulses that never finish, heads
closer than one cell, shared cells, an emitter facing an object, and clips with fewer than 3 events shown or none left
after the end.

`frame_check.py` decodes the MP4s and checks: the clock box on every frame against templates of all readings 0.0-10.0 s;
every cell label against templates of all label texts on three frames; heads located in the pixels against
`render.state` (tolerance 3 px, same count); mirror pixels on the stated diagonal at the first and last frame; the
frame each beam launches; the lit-cell count of the last frame. It then re-derives the track-based answers
(`entry_edge`, `first_mirror`, `turn_dir_first`, `head_cell_at_t`, `n_turns_by_t`, `n_heads_at_t`, `split_cell`,
`time_turn_at`, `time_between_mirrors`, `speed`, `time_between_turns`, `cells_between`) from the decoded heads with a
jitter band of 2 px, and checks the clock at the frame of every question that names a time.

## Frame sampling and `min_fps`

Every question stores `params["min_fps"]`, the lowest sampling rate at which it is still answerable:

- static counts and identities (L1, `first_mirror`, `turn_dir_first`, `split_cell`, `cells_between`, `n_lit_cells`,
  `exit_point`, `n_turns_total`, `n_exits`, most `cf_exit_point` / `cf_n_exits`) need 1 fps;
- questions about a named clock time need 1 fps for whole seconds and 2 fps for half seconds (`align`), or
  `fps_for(gaps)` = ceil(1 / smallest gap in seconds to a launch, end or turn) if larger, at most 24;
- time measurements and clock readings (`time_turn_at`, `time_between_*`, `launch_time`, `exit_time`,
  `time_after_end`, `cf_exit_time`) have a floor of 4; `speed` has 2.

Share of questions by `min_fps` (5000 dry-run scenes, 50,000 questions):

| min_fps | share | cumulative |
|---|---:|---:|
| 1 | 52.8 % | 52.8 % |
| 2 | 16.4 % | 69.2 % |
| 3-4 | 30.8 % | 100 % |

Most 1 fps questions are answerable from any late frame because the trail persists; the timing questions
(clock readings, speed, durations, heads at a time) need 2-4 fps. 66.7 % of L2-L5 questions need the motion.

## Determinism

All randomness comes from the `random.Random` passed to `sim.build` and `questions.build`. Clip length is a whole
number of frames, pivot turn times are exact frame instants (`round(t * 24) / 24`), and other stored parameters are
rounded before anything is derived from them. Caches in `render.py` and `answers.py` are keyed by the full scene JSON.
The same seed gives byte-identical JSON across `PYTHONHASHSEED` values and identical raw frames
(`test_mirror_beam_reflection.py`).

## Files and commands

```
sim.py                          scenes, tracer, cell labels, consistency check
questions.py                    the 10-question ladder
answers.py                      answers from the drawn track, separate tracer for predictions and what-ifs
render.py                       PIL renderer (1280x720, 24 fps)
frame_check.py                  decodes MP4s, re-derives answers from pixels
preview.py                      renders the first N seeds of every scenario
stats.py                        diversity / balance report
run.py                          CLI (shared pipeline in common/video5.py)
test_mirror_beam_reflection.py  determinism, ladder, tamper, labels, tracers agree, hand-checked rules

python3 run.py generate --n 20 --seed 0 --output-dir out
python3 run.py validate --dataset-dir out [--skip-video]
python3 preview.py
python3 frame_check.py --dataset-dir ../output_videos/mirror_beam_reflection_preview
python3 stats.py out/annotations.jsonl
python3 test_mirror_beam_reflection.py
```

## Results

- 5000-scene dry run (`--no-render`, validate `--skip-video`, 4 chunks): 0 mismatches, 0 ladder errors, 0 check
  failures, no warnings (`python -W error`). 4101 of 5000 boards (82 %) were redrawn at least once (mean 8.4
  attempts, max 136; mostly routes too short or a clip window that leaves too few events); no seed failed.
- Diversity: 5000 / 5000 unique scene signatures; scenarios 19.3-20.7 % each; 14,168 distinct (question, answer)
  pairs; answer letters A 24.8 %, B 25.0 %, C 25.2 %, D 25.0 %.
- Tamper: 3 answers corrupted in each of 100 scenes -> exactly 300 questions flagged (plus one "rebuilt questions
  differ" line per tampered scene).
- Determinism: identical JSON across `PYTHONHASHSEED` values; identical raw frames on re-render; regenerating seeds
  0-49 into a fresh folder reproduces `annotations.jsonl` byte for byte.
- Previews (`preview.py`, 10 per scenario): `run.py validate` 0 mismatches; `frame_check.py` 50 clips, 6354 frames,
  clock template match 6354 / 6354, 713 labels template-matched at 3 frames each (all read as their scene text, none
  overlapping), 252 answers re-derived from decoded frames, max head error 2.47 px, 0 problems.
