# vessels_and_clocks

Closed-loop 10-question video-QA benchmark on fluid volume and rate laws (communicating vessels + hourglass vs
water clock, New Problem Types B4 + B32). 3-5 glass vessels of different widths stand in a row on a bench. Each is a
rectangular column 10 marks tall; all are equally deep, so the water in a vessel is level x width. Each vessel does one
thing:

- **tap**: water pours in at a constant rate, the level rises in a straight line and, once full, the vessel overflows
  (water runs down both outer walls and pools on the bench) and stays full;
- **steady drain** (hourglass law): constant outflow, the level falls in a straight line until empty;
- **leaky drain** (water-clock law): outflow k sqrt(level), the level falls fast at first and then slows;
- **joined pair**: two neighbours share a pipe at the bottom with a valve; when it opens, water flows from the higher
  level to the lower at a rate proportional to the difference and both settle at (w_a h_a + w_b h_b) / (w_a + w_b).

The pipeline is PIL frames piped to ffmpeg (`common/video5.py`), with the 10-question ladder in `common/ladder5.py`
(`LADDER = "ten"`). Clips are 1280x720 at 24 fps and 7-10 s long (168-240 frames).

## Scenarios (`scene["scenario"]`, drawn per seed)

| scenario | setup | readable from the video because |
|---|---|---|
| `taps_drains` | 3-5 vessels, each a tap, a steady drain or a leaky drain (at least one tap and one drain) | taps are drawn pouring, drains have a spout with a stream |
| `drain_race` | 3-5 drains, at least one of each law | the steady one falls in a straight line, the water clock visibly slows (its level at least halves over its fall) |
| `fill_race` | 3-5 taps with different widths, levels and flows | overflow is a persistent spill down both walls |
| `joined_pair` | one valve-joined pair plus 1-3 taps / drains | the valve handle turns from across the pipe to along it and the pipe fills with water |
| `two_pairs` | two joined pairs whose valves open at different times (plus a tap / drain with 5 vessels) | each pair has its own pipe and valve |

At least two overflow / empty events in the first three scenarios, at least one in `joined_pair`.

## Random knobs (`sim._draw`, `sim._solo`, `sim._pair`, `sim._style`, `sim._layout`)

- vessel count 3-5 (4-5 in `two_pairs`); inner widths 60-220 px (narrower with more vessels), any two at least 12 %
  apart; start levels 0.5-9.5 marks; clip length 7-10 s;
- taps rise 0.35-3 marks/s, steady drains fall 0.35-3 marks/s, water clocks start at 0.5-4 marks/s; 70-80 % of
  taps / drains have their event inside the clip, the rest end clearly short of full / empty (>= 0.3 mark);
- joined pairs start 2.5+ marks apart (which side is higher is random), settle with a time constant of 0.3-0.7 s,
  valve opening time anywhere that still lets them settle before the end;
- vessel identity colours: a random subset of 8 per theme (no blues), unique per clip; water tint 4 per theme;
- theme (light / dark), background colour and pattern (plain, gradient, stripes, dots), gridline colour and style
  (solid, dashed, dotted), glass colour, wall thickness 3-6 px, rim style (band, lip), plinth height and pattern
  (plain, stripes, dots), bench colour, height and pattern (plain, grain, tiles), metal colour of taps / spouts /
  valves, tick side and length, drain spout side;
- camera framing: 34-44 px per mark, bench height, gaps 70-150 px, row position and ruler side (left / right).

`sim.build` redraws until the scene passes `sim.check` and can fill the ladder.

## On-screen text

- **The clock**: a box at the top-left with the time in seconds to 0.1 s (white 44 px on a dark box), the value is
  `frame * 10 // 24` tenths. Nothing is drawn above y = 96 or in the clock box.
- **Ruler numbers** 0, 2, 4, 6, 8, 10: one shared ruler at one end of the row, 36 px bold (digits >= 24 px tall), in the
  theme's ink colour, beside the ruler, vertically centred on their marks; stored in `scene["labels"]` (`sim.labels`),
  fixed for the clip.

Why the numbers are needed: level questions ask for a reading on a 0-10 scale ("at which mark do they settle", "the
level when the clock shows 4.0 s"). Unnumbered gridlines alone would force counting up to ten lines at reduced
resolution; numbering every second mark and drawing an unnumbered gridline at every mark (plus tick marks on each
vessel) keeps the text to six short numbers. They are a measuring aid only: they never name a vessel, an event or an
answer. Vessels are named by colour (a coloured rim and plinth, unique per clip), so they need no letters. There are
no titles, captions, legends, rule texts or status lines; the old renderer's vessel letters and per-vessel scale
numbers are gone.

## The 10-question ladder (`LADDER = "ten"`)

Per clip the count per level is drawn from L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3 (total exactly 10), uniformly among
the splits the candidates can fill, ordered by level as q01..q10. Distinct templates are used before a template is
repeated with other params; every question has four distinct options.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `n_overflow`, `n_empty_end`, `n_taps`, `n_rising` (vessels whose level rises at some point), `first_event` (first to overflow or run empty), `widest`, `highest_start` |
| L2 relational | `steady_vs_leaky` (which drain falls at a steady rate), `more_water_end` (level x width), `higher_end`, `faster_rise` (two taps), `earlier_event` (two events incl. valve openings), `valve_flow` (whose level rises when the valve opens) |
| L3 measurement | `level_end`, `drop_marks`, `level_at_clock` (level when the clock shows s.0), `rise_rate` (marks per second, nearest half), `event_clock`, `valve_clock` (clock readings) |
| L4 integration | `event_order` (three events), `most_water_end`, `shared_level` (settle mark of a pair), `time_between` (clock difference of two events), `most_lost` (water lost, level x width) |
| L5 counterfactual | `cf_wider_time` / `cf_fuller_time` (clock time of overflow / emptying if twice as wide / started 1-2 marks higher or lower, or "it would not ... before the clip ends"), `cf_wider_level` (end level if twice as wide), `cf_shared_extra` (settle mark if one joined vessel had started 1-3 marks higher / lower), `cf_valve_never` (most water if the valve never opened), `cf_faster_tap` (first event if a tap poured twice as fast) |

Distractors are plausible: wrong vessels are those with events / the next-widest / the highest level; binary questions
add "both the same" / "neither" options that the margins make wrong; settle levels include the naive average of the two
start levels; what-if times include the real event time and "it would not ... before the clip ends" (or a clock time
when that is the answer); orders swap a neighbouring pair first; counts and readings are nearby integers.

Margins (`questions.py`, `sim.check`):

- level readings at least 0.2 mark from a .5 rounding boundary (the midpoint between two gridlines); `level_at_clock`
  adds the drift over +-0.25 s; compared levels differ by >= 0.6 mark (>= 20 px), amounts of water by >= 15 % (and the
  larger is at least 0.5 mark deep), rates by >= 25 %, rise rates at least 0.1 mark/s from a half-step boundary;
- any two in-clip events (overflow, run empty, valve opens) are >= 0.3 s (7 frames) apart and >= 0.8 s from both ends
  of the clip; each sits 0.2-0.4 or 0.6-0.8 of the way through a frame interval, so the first frame showing it is
  unambiguous (and twice the time, for the "twice as wide" what-if, is too); events after the clip are >= 0.3 s past it;
- clock-time options are >= 0.6 s apart; what-if times must also sit >= 0.15 frame from a frame instant and >= 0.3 s
  from the clip end;
- a water clock's level at least halves over what the clip shows of its fall (the slowing is visible); joined vessels
  are within 0.01 mark of each other at the end; vessels that don't overflow / empty end >= 0.3 mark short of it.

`requires_motion` is false for `n_taps`, `widest` and `highest_start` (one frame shows them), true otherwise (all
L2-L5 questions).

## Closed loop

Every question stores its `derivation_key` and `params`. `answers.py` never reads the flow parameters (start levels,
q, k, K, valve times) or `sim.py`'s closed forms. It works on a per-frame **track** of what is drawn (`render.state`,
or in `frame_check.py` the decoded MP4 pixels): each vessel's water height in px, whether it overflows, whether a tap
pours into it, its drain stream, and each valve handle. From the static drawing it takes only the vessel pixel widths,
colours, which vessels a pipe joins, and the ruler (the "0" and "10" numbers give the bottom and the px per mark).
Levels are px / px-per-mark; event frames are the first frames showing a spill, no water, or an open valve; clock
readings are `frame * 10 // 24`. The drain law is classified from the track (does the second half of the fall go
slower than the first). What-ifs fit the law to the track (rise / fall rate, slope of sqrt(level), the pair's exchange
rate from the decay of the level difference) and integrate it numerically (RK4 at 1/2400 s, or Euler until the pair
settles) under the change.

`sim.check` rejects events too close to each other, to a frame instant or to the clip ends, invisible slowing,
unsettled pairs, near-full / near-empty end states, widths too alike, shared colours, and any overlap between vessels,
spouts, spills, faucets, the ruler, its numbers and the clock.

`frame_check.py` decodes the MP4s and checks: the clock on every frame (template match against all readings
0.0-10.0 s); each ruler number by template matching against every ruler text drawn on the label-free background (3
frames), glyph height, label overlaps, and that no water ever enters a label box; the vessel plinths found in the pixels
(count, extent and centre within 2 px) and the water widths; per frame and vessel the water height (within 2 px of
`render.state`), overflow, tap stream, drain stream and valve state (exact on every frame, so event frames are
frame-accurate); it then rebuilds the track from the pixels, re-derives every answer from it with `answers.derive`, and
reads the decoded clock at the frame of every clock-reading answer.

## Frame sampling and `min_fps`

Every question stores `params["min_fps"]`:

- 1 fps for counts, end states, widths, start levels, laws, rates, settle levels and most what-ifs: every
  question-relevant change leaves a persistent result (a spill keeps running, an empty vessel stays empty, an open valve
  stays open, levels stay put after an event);
- order and "first" questions: `ceil(1 / smallest gap)` between the events compared (1-4 fps);
- clock readings, time differences, `level_at_clock` and what-if clock times: 4 fps (options are >= 0.6 s apart).

At 640x360 the ruler digits are ~13 px, the clock 22 px, vessels >= 30 px wide, walls and gridlines >= 1.5 px, streams
2.5-9 px, the valve handle 17 x 3 px.

## Determinism

All randomness comes from the `random.Random` passed to `sim.build` and `questions.build` (no global RNG, time, uuid
or set-order decisions; palette names are sorted before sampling). Widths are whole pixels (`w` = px / 100). Caches in
`render.py` and `answers.py` are keyed by the scene JSON. The same seed gives byte-identical JSON across
`PYTHONHASHSEED` values and identical raw frames (`test_vessels_and_clocks.py`).

## Files and commands

```
sim.py                      scenes, closed-form flow laws, labels, layout, consistency check
questions.py                the 10-question ladder
answers.py                  answers from the drawn per-frame track; fitted laws + numerical re-simulation
render.py                   PIL renderer (1280x720, 24 fps) and render.state
frame_check.py              decodes MP4s, checks clock / ruler numbers / vessels / water / events, re-derives answers
preview.py                  renders the first N seeds of every scenario
stats.py                    diversity / balance report
run.py                      CLI (shared pipeline in common/video5.py)
test_vessels_and_clocks.py  determinism, ladder, tamper, labels, drawn levels, hand-checked scene, pinned seeds

python3 run.py generate --n 20 --seed 0 --output-dir out [--no-render]
python3 run.py validate --dataset-dir out [--skip-video]
python3 preview.py --per 10
python3 frame_check.py --dataset-dir ../output_videos/vessels_and_clocks_preview
python3 stats.py out/annotations.jsonl
python3 test_vessels_and_clocks.py
```

## Results

- 5000-scene dry run (`--no-render`, validate `--skip-video`, 4 chunks, `python -W error`): 0 mismatches, 0 ladder
  errors, 0 check failures, no warnings or exceptions. 2264 of 5000 seeds (45.3 %) redrew their scene at least once
  (mean 1.90 attempts, max 15); no seed failed.
- Diversity: 5000 / 5000 unique scene signatures; scenarios 19.3-20.7 % each; 12,424 distinct (question, answer)
  pairs; answer letters A 24.94 %, B 24.69 %, C 24.82 %, D 25.55 %; 30 templates; all 22 level splits used (193-269
  clips each).
- min_fps: 69.9 % of questions answerable at 1 fps, 73.0 % at 2 fps, 100 % at 4 fps; requires_motion on 100 % of
  L2-L5 questions.
- Tamper: 3 answers corrupted in each of 100 scenes -> exactly those 300 questions flagged (plus one "rebuilding from
  the seed gives different questions" line per tampered scene, 400 lines in all).
- Determinism: identical JSON across `PYTHONHASHSEED` values; identical raw-frame hashes across two processes;
  regenerating seeds 0-49 into fresh folders reproduces `annotations.jsonl` byte for byte (and matches the 5000 dry run).
- Previews (`preview.py`, 10 per scenario): `run.py validate` 0 mismatches; `frame_check.py` on the decoded MP4s: 50
  clips, 10,362 frames, clock template match 10,362 / 10,362, 300 ruler numbers template-matched at 3 frames each (all
  read as their scene text, no overlaps, no water in any label box), vessel plinths and water widths where the layout
  says, water heights within 2 px of `render.state` on every frame, overflow / tap / drain / valve state exact on every
  frame, 500 answers re-derived from the decoded frames (495 exact, 5 what-if clock times within 0.1 s from fitting
  px-quantised levels), decoded clock at every clock-answer event frame matches, 0 problems.
- Shared code: `common/ladder5.py` / `common/video5.py` unchanged (the 10-question mode already existed); 300-scene
  dry-run validate on `motion_parallax` and `moving_points` (5-question ladder): 0 mismatches each.
