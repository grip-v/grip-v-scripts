# hidden_mirrors

Closed-loop 10-question video-QA benchmark on inverse optics (after Jane Street *Hall of Mirrors*, Apr 2015,
hidden-mirror variant). A board of square cells hides two-sided diagonal mirrors ('/' or '\\') that are never drawn.
Light pulses, one colour each, are fired in from ports on the board edge; a pulse moves one cell per step, leaves a
trail in its colour, and turns 90 degrees when it reaches the centre of a mirror's cell. Where the mirrors are, and
which way they lean, has to be worked out from how the pulses bend. Each trail stays lit for 1.0 s after its pulse
ends, then fades out over 0.6 s, so the clip never shows every path at once and the integration questions need
memory across pulses.

The pipeline is PIL frames piped to ffmpeg (`common/video5.py`), with the 10-question ladder in `common/ladder5.py`
(`LADDER = "ten"`). Clips are 1280x720 at 24 fps and 4-10 s long (96-240 frames).

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | setup | what makes it readable from the video |
|---|---|---|
| `classic` | square board, 4-7 cells a side, 3-6 pulses fired one after another (a pulse may be fired before the previous one has left) | each pulse has its own colour for head, trail and lit ports |
| `rect` | rectangular board, 5-9 columns x 4-7 rows (never square) | same; more ports on the long sides |
| `volley` | square board, 5-7 cells a side, 4-7 pulses fired in pairs from two ports at the same moment | two heads in flight at once, always at least one cell apart, different colours |
| `blocks` | 1-3 dark blocks stop any pulse that reaches them; at least one pulse is stopped | blocks are drawn as dark squares; a stopped pulse leaves a ring on the block |
| `partial` | 1 to a third of the mirrors are drawn (as a line on their diagonal), the rest stay hidden; at least one pulse turns at a hidden one | drawn mirrors are a solid line in a mirror colour distinct from every pulse colour |

## Random knobs (`sim._board`, `sim._schedule`, `sim._style`, `sim.layout`)

- board size (above); mirror density 14-32 % of the cells (at least 3 mirrors), random cells and slants;
- speed 0.17-0.27 s per cell (3.7-5.9 cells per s); first launch 0.6-1.1 s; later launches 0.6 s up to "previous
  pulse's run + 0.6 s" after the previous one (65 %) or just after it has left (35 %); launches are frame instants;
- ports: drawn at random among those not yet used as an entry or exit; pulses longer than 3.4 s are redrawn, at most
  one pulse per clip goes straight through, and a pulse that would leave through its own entry port is kept 30 % of
  the time;
- pulse colours: a random subset of 8 per theme (dark: gold, blue, pink, green, purple, orange, red, white; light:
  white is replaced by black), every pair at least 100 apart in summed RGB, and at least 100 from mirrors, blocks,
  board, grid and background;
- theme (dark twice as likely as light), background, board colour and pattern (plain, checker, dots), grid colour
  and line style (solid, dashed, dotted), drawn-mirror colour and width 5-8 px, trail width 5-9 px, head size
  max(22, 2 x trail + 10)-30 px, head shape (disc, square, diamond), port-marker shape (wedge, bar, notch);
- camera framing: board position and cell size (58 px up to 108 px, as large as the board allows, always leaving
  room for the markers and labels);
- clip tail: 1.1-1.6 s after the last pulse ends (at least 1.0 s up to the last frame shown).

`sim.build` redraws until the scene passes `sim.check` and `sim._ladder_ok`.

## On-screen text

Two kinds of text, nothing else: no titles, captions, legends or status lines.

- **The clock**: a box at the top-left, outside the board area, with the time in seconds to 0.1 s (white 44 px on a
  dark box). The value is truncated from the frame number (`tenths = frame * 10 // 24`).
- **Cell labels**: a column letter above each column (A, B, ...) and a row number left of each row (1, 2, ...), 34 px
  bold (glyphs >= 24 px tall) in the theme's ink colour, stored in `scene["labels"]` (`sim.labels`), fixed for the
  clip and drawn into the static layer, outside the port markers.

Why the labels are needed: the task is to locate invisible mirrors, so answers are cells ("C4") and ports ("the top
of column C", "the left end of row 4"). Every cell and every port looks the same, so neither colour nor shape can
name them; unnumbered grid lines alone would force counting up to 9 columns at a reduced resolution. The labels are
identifiers only: they name positions, never a mirror, a path or an answer, and are the same every frame. Pulses are
named by colour. The original renderer's title, rule text, "Pulse 2 enters at port 7" status lines, port numbers and
13 px per-cell names are gone.

## The 10-question ladder (`LADDER = "ten"`)

Per clip the count per level is drawn from L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3 (total exactly 10), uniformly among
the splits the candidates can fill, ordered by level as q01..q10. Every question has four distinct options.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `n_pulses`, `entry_port` (where a colour is fired in), `first_colour`, `n_straight` (pulses that never turn), `n_blocks`, `n_shown` (drawn mirrors) |
| L2 relational | `exit_port` (where a colour leaves, or which block stops it), `first_turn_dir`, `first_turn_cell`, `launch_order`, `exit_order` (three colours each) |
| L3 measurement | `n_turns`, `n_cells` (cells crossed), `time_on_board`, `exit_clock`, `turn_clock` (clock readings), `launch_gap`, `speed` |
| L4 integration | `n_turn_cells` (different cells where some pulse turns), `total_turns`, `locate_mirror` (cell + slant of a colour's k-th hidden mirror), `must_mirror` / `certainly_empty` (which of four cells must hold / certainly lacks a mirror), `n_unvisited` (cells no pulse enters) |
| L5 counterfactual | `cf_reverse` (fired back in where a colour left), `cf_new_port` (an unused port), `cf_change` (a mirror flipped or removed, a block removed: where would a colour leave), `cf_change_turns` (same, how many turns), `cf_fast` (twice as fast: clock time of leaving) |

Distractors are plausible: wrong ends are what a flipped or missing mirror on the path, going straight on, a
neighbouring port or the entry port would give; wrong cells are the cells before and after on the path, other turn
cells and neighbours; wrong slants are the other diagonal; wrong orders swap a neighbouring pair first; counts are
nearby integers; times are spaced (below).

What-ifs are only asked when they can be worked out from the video: every cell the hypothetical pulse enters must have
been entered by a real pulse (a crossed-straight cell is empty, a turn cell holds a mirror of known slant), or be a
block or drawn mirror. The answer is re-simulated with the change applied, never guessed.

Margin rules (`questions.py`, `sim.py`):

- order questions: consecutive launches or exits at least 0.3 s (7 frames) apart;
- durations and `cf_fast`: at least 0.025 s from a 0.1 s rounding boundary (`safe1`); options at least
  max(0.3 s, 15 %) apart (`cf_fast`: 0.4 s); speed is asked to the nearest half cell per second, at least 0.1 cells/s
  from a rounding boundary, with options 0.5 apart;
- clock readings (`event_ok`): the event falls 0.3-0.7 of the way through a frame interval and both frames around it
  show the same tenth; options at least 0.5 s apart;
- every launch is at least 0.6 s into the clip and every pulse ends at least 1.0 s before the end; question events
  are therefore far more than 2 frames from the clip edges;
- trails on screen at the same time never share a half-cell segment, never share a port, and the heads in flight
  stay at least one cell apart (checked every frame); two objects never share a cell.

`requires_motion` is false for the static counts (`n_blocks`, `n_shown`), `entry_port`, and `n_turns` / `n_cells`
(a whole path is visible in one frame just after its pulse leaves); true otherwise (96.1 % of L2-L5 questions).

## Closed loop

`answers.py` never reads `sim.py`'s traces. It measures the per-frame heads that `render.state` draws (or that
`frame_check.py` finds in decoded MP4 frames): one head per colour, linked into a track; a turn is where two straight
runs meet (the intersection of the incoming and outgoing lines, timed from the measured speed), and a single frame off
the line at either end of a track marks a turn in the first or last cell; the entry port is where the track starts,
the exit is the edge the last head is heading into (a head that ends more than 0.39 cell short of it was stopped).
Counts, ports, orders, clock readings and durations come from these tracks. For what-ifs and the mirror-location
questions it builds the **mirror map the video reveals** (corner cells hold a mirror whose slant follows from the turn,
cells a track crosses straight are empty, blocks and drawn mirrors from the drawing) and re-simulates with its own
tracer (complex-number directions). If a hypothetical path enters a cell no pulse has visited it returns None, which
fails validation.

`sim.check` rejects clashing trails, heads closer than one cell, shared ports or cells, launches off frame instants,
events too close to the clip edges, objects too small, boards that leave their area, and scenarios whose special
feature never shows (no stopped pulse, no hidden turn, no simultaneous pair). `sim._ladder_ok` rejects scenes that
cannot fill the ladder.

`frame_check.py` decodes the MP4s and checks: the clock on every frame against templates of all readings 0.0-10.0 s;
every label against templates of all label texts on three frames, glyph height and overlaps with the board, the port
markers, each other and the clock; heads located in the pixels against `render.state` (3 px tolerance, same set of
colours; heads on the board edge are not position-checked); that hidden mirrors leave no mirror-coloured pixels and
drawn ones lie on their diagonal; the dark blocks; that entry and exit markers light up in the right colour two frames
after the event. It then re-derives every answer except `n_blocks` / `n_shown` from the decoded heads, and checks the
decoded clock at the frame of every clock-reading answer.

## Frame sampling and `min_fps`

Every question stores `params["min_fps"]`:

- 1 fps for counts, ports, cells, slants, directions and what-ifs: a trail is fully lit for 1.0 s after its pulse
  ends, so a 1 fps sampler always sees every bend and both lit ports;
- order questions: `ceil(1 / smallest gap)` between the events compared (1-4 fps);
- clock readings, durations, launch gaps, speed and `cf_fast`: 4 fps.

5000 dry-run scenes (50,000 questions): 69.5 % answerable at 1 fps, 74.7 % at 2 fps, 100 % at 4 fps. At 640x360 the
labels are 17 px, the clock 22 px, heads 11-15 px and trails 3-5 px.

## Determinism

All randomness comes from the `random.Random` passed to `sim.build` and `questions.build` (no global RNG, time, uuid
or set-order decisions; sets are sorted before use). Launch times are frame instants, clip length is a whole number of
frames, and the step is rounded to 3 decimals before anything is derived from it. Caches in `render.py` and
`answers.py` are keyed by the scene JSON. The same seed gives byte-identical JSON across `PYTHONHASHSEED` values and
identical raw frames (`test_hidden_mirrors.py`).

## Files and commands

```
sim.py                  scenes, tracer, labels, consistency check
questions.py            the 10-question ladder
answers.py              answers from the drawn tracks; own tracer on the revealed mirror map
render.py               PIL renderer (1280x720, 24 fps)
frame_check.py          decodes MP4s, checks clock / labels / heads / hidden mirrors / ports, re-derives answers
preview.py              renders the first N seeds of every scenario
stats.py                diversity / balance report
run.py                  CLI (shared pipeline in common/video5.py)
test_hidden_mirrors.py  determinism, ladder, tamper, labels, tracers agree, revealed map vs truth, hand checks

python3 run.py generate --n 20 --seed 0 --output-dir out
python3 run.py validate --dataset-dir out [--skip-video]
python3 preview.py --per 10
python3 frame_check.py --dataset-dir ../output_videos/hidden_mirrors_preview
python3 stats.py out/annotations.jsonl
python3 test_hidden_mirrors.py
```

## Results

- 5000-scene dry run (`--no-render`, validate `--skip-video`, 4 chunks, `python -W error`): 0 mismatches, 0 ladder
  errors, 0 check failures, no warnings or exceptions. 361 of 5000 seeds (7.2 %) redrew their board at least once
  (mean 1.10 attempts, max 7); no seed failed.
- Diversity: 5000 / 5000 unique scene signatures; scenarios 19.3-20.7 % each; 12,431 distinct (question, answer)
  pairs; answer letters A 25.12 %, B 24.72 %, C 24.94 %, D 25.22 %; 29 templates; all 22 level splits used.
- Tamper: 3 answers corrupted in each of 100 scenes -> exactly 300 questions flagged by `run.py validate` (plus one
  "rebuilt questions differ" line per tampered scene).
- Determinism: identical JSON across `PYTHONHASHSEED` values; identical raw frames on re-render; regenerating seeds
  0-49 into a fresh folder reproduces `annotations.jsonl` byte for byte.
- Previews (`preview.py`, 10 per scenario): `run.py validate` 0 mismatches; `frame_check.py` on the decoded MP4s: 50
  clips, 8526 frames, clock template match 8526 / 8526, 545 labels template-matched at 3 frames each (all read as their
  scene text, none overlapping), hidden mirrors invisible and drawn ones on their diagonal, blocks and lit ports
  as stored, 497 answers re-derived from decoded frames, max head error 1.85 px, 0 problems.
