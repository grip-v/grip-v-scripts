# spring_chain_waves

Video-QA sub-benchmark for **wave pulses on a chain of beads**: pulses run
along the chain, bounce off its ends (upside down off a clamp or a held
paddle, the same way up off a ring sliding on a pole), split where the
beads change thickness, and pass through each other. v3 moves the dataset
to the 10-question ladder (`LADDER = "ten"` in `questions.py`), adds four
scenarios, removes every caption and keeps clips to 10 s.

## What each clip shows

- A horizontal chain from ruler mark 0 to mark L (L = 8, 10 or 12),
  820-1050 px wide, placed anywhere that keeps it clear of the clock;
  beads every 1/2-1/4 mark (circles, squares or diamonds, 5-8 px; thick
  beads 1.6x bigger, another colour, a line twice as wide).
- Ends: a paddle (flicks pulses in; being held, it sends arriving pulses
  back upside down), a clamp (back upside down) or a ring on a pole (back
  the same way up, swinging to twice the height as a pulse arrives).
- Pulses: one fixed shape per clip (round, peaked or smooth), 3-8 frames
  long, bumps up or dips down, big (1) or small (0.6); 1.5-4 marks/s on thin
  beads (quarter steps), half or a third of that on thick beads.
- Where the thickness changes a pulse splits by the impedance-step rule:
  back (c2 - c1)/(c2 + c1), on 2 c2/(c1 + c2) of the arriving height (into
  slower beads: a third or half comes back inverted, two thirds or half goes
  on; into faster beads: a third or half comes back upright, 4/3 or 3/2 goes
  on). Parts under 0.3 are not followed (and not drawn).
- The clip is an exact list of travelling pulses (fractions, method of
  images), so a clamped end never moves and a ring swings to double height.
- 4-10 s at 24 fps (median 8.6 s); six background palettes, flat / gradient /
  vignette.

### Scenarios (`scene["scenario"]`, 25 % each)

| scenario | what happens |
|---|---|
| `reflect` | uniform chain; a paddle sends one pulse; the far end is a clamp or a ring; the pulse runs back and forth |
| `junction` | thin and thick beads: a step at one mark, or a thick band in the middle; one pulse that splits at every change |
| `collide` | a paddle at each end; each sends a pulse (one sometimes sends two), timed so the first two meet at a ruler mark; they pass through each other and bounce between the paddles |
| `train` | uniform chain; a paddle sends 2-4 bumps and dips of two sizes, 0.5-1.25 s apart; they bounce off a clamp or a ring and come back through the ones still going out |

### Random knobs

Scenario, chain length, width, left edge and height on screen, amplitude
(58-92 px per unit), thin speed, thick-bead ratio (2 or 3), step or band and
its marks, which side is thick, which end has the paddle, far-end hardware,
number / sign / size / timing of pulses, pulse shape and duration, clip
length, background, chain colours, bead shape / size / spacing, line width,
paddle and metal colours, ring size.

## On-screen text

- **Clock** (always): fixed dark panel at the top-left, "s.t s" truncated to
  0.1 s, 36 px bold (glyphs ≥ 26 px), white on near-black; the moving chain
  and the end hardware stay at least 26 px below it.
- **Ruler numbers 0..L** under the chain (the only other text). Positions
  and speeds are asked in ruler marks ("at which mark do they meet", "how many
  marks per second"), and unnumbered ticks would mean counting up to 12 ticks
  on a shrunk frame. They are a measuring aid, never an answer by
  themselves: no number marks an event. Fixed style: 36 px bold in the
  clip's ink colour, centred under each whole-mark tick (half marks have
  short unnumbered ticks), ≥ 26 px glyphs, ≥ 18 px apart, all below the
  chain's lowest reach; `scene["labels"]` holds text, position and size and
  the renderer draws exactly those.
- No titles, captions, legends or hints. Ends are named by side and
  hardware, parts of the chain by bead thickness or ruler marks, pulses by
  launch order and shape, moments by the clock.

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total
10) among the combinations the clip can fill (a candidate pool per level
first). Different templates first, motion questions first, at most one
single-frame question per level while others remain; no (derivation_key,
params) repeats; 4 options (3 for the three-way "which paddle" / "which
part" questions).

- **L1 observation:** what holds the far end; what the first pulse looks
  like; how many pulses are sent; where the beads change (marks); which
  paddle sends first; bumps on the chain at the end.
- **L2 relational:** how a pulse comes back off the far end; how the first
  pulse looks when it is back at the paddle; what happens at the first
  thickness change; where pulses are slowest; which comes first after the
  split (back at the paddle or on at the far end); taller or shorter after
  the change; the order of bumps and dips coming back; which paddle's pulse
  reaches the other paddle first; where the first two meet (left / right /
  middle); which first pulse is taller; which way the pulse moves at a clock
  reading.
- **L3 measurement:** speed in marks per second; clock time of the first far
  arrival / of the return to the paddle / of a pulse reaching the other
  paddle; meeting mark; height after the change and of the part bounced
  back (1/3 ... 3/2); thick-bead speed as a fraction; bumps on the chain at a
  clock reading; time between the first two launches; spacing of the first
  two pulses in marks.
- **L4 integration:** arrivals at the far end / at a paddle; the k-th pulse
  to reach the far end; clock time of the k-th far arrival; how many times
  pulses pass through each other; mark of the second meeting; pulses moving
  toward one end at a clock reading; clock time at which the part bounced at
  the change reaches the paddle; how the pulse looks on its second return.
- **L5 counterfactual (re-traced):** far end switched (clamp ↔ ring); twice
  as fast (far arrivals); all thin beads (return time); thin and thick
  swapped; chain ending at mark m; half as fast (first far arrival); one
  paddle 0.5 / 1 s later (meeting mark); a paddle replaced by a clamp or a
  ring; the last pulse never sent; twice as fast (meetings).

All L2-L5 questions in the 5000-scene run need the motion
(`requires_motion=True`); the single-frame ones (bumps at a clock reading,
spacing) exist but are picked only when nothing else is left.

## Margins (no near ties)

- Launches on whole frames; every arrival / launch at one point ≥ 12 frames
  (0.5 s) from the next; the first visible motion ≥ 12 frames after the
  start; no event in the last 0.5 s or the first 0.4 s after the clip.
- Every pulse that starts ≥ 0.5 s before the end is seen on its own (whole
  bump inside its stretch, nothing else within reach) for ≥ 8 frames.
- Clock answers are the reading on the event frame, options 0.5 s apart.
- Meeting marks within 0.15 of a whole mark; "nearer the left / right"
  only when ≥ 0.8 marks from the middle; "which first" only with ≥ 8 frames
  (0.33 s) between the two; spacing within 0.1 of a half mark.
- Pulses ≥ 0.3 high (≥ 17 px), ≥ 12 px wide; what-if counts only when no
  what-if arrival lands within 8 frames of the end.

## min_fps

Every question stores `params["min_fps"]`: the smallest of 1/2/3/4/6/8/12/24
fps that puts a sample into the shortest interval the answer needs: the
time a pulse needs to cross the chain (for shapes, bounces, splits), half
of it (for speeds), the gap between compared arrivals or launches, the
stable window around a clock reading, and ±0.25 s for clock answers
(4 fps). Over 5000 scenes: 64 % of questions need 1 fps, 83 % ≤ 2 fps, 100 %
≤ 4 fps.

## Closed loop

- **Questions** store `derivation_key` + `params` (side, k, clock tenths,
  mark, delay, hardware) + `min_fps`; ground truth comes from the exact
  trace in `sim.py`.
- **Independent answers** (`answers.py`) never read the pulse list, the
  launches, the speeds or the sections. They read `render.track` (the
  chain's drawn height every 2 px on every frame, which samples have thick
  beads, the end hardware, the ruler labels): bumps on each frame (single
  peak, symmetric, clear of ends and changes), linked into tracks with a
  robust line fit; speeds from the tracks (length-weighted, quarter-mark
  grid; thick = half or a third); launches as tracks leaving a paddle that
  no arrival explains; a separate depth-first re-trace from those readings,
  checked against every track (and every traced pulse must be seen);
  answers from the re-trace, counts at a moment counted as bumps on the
  drawn frame, what-ifs re-traced.
- **`sim.check`** rejects scenes whose stored pulses don't re-trace, too-small
  pulses, a chain reaching the clock or the ruler, overlapping labels, events
  too close together or to the ends, pulses never seen on their own, and
  meetings too near a paddle or each other.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`frame_check.py`** decodes the MP4s: the clock (template match, every
  6th frame, every event frame, every frame a question names); every ruler
  number read back by template match and never covered; the drawn chain
  where the track puts it, column by column; on calm frames the bumps found in
  the decoded picture match the traced pulses in number, sign and position
  (0.12 marks); clock answers read on the event frame; counts at a clock
  reading counted on the decoded frames; meeting marks from the two decoded
  bumps just before they meet.

## Determinism

All randomness comes from the `random.Random(seed)` passed in; times are
exact fractions; no global RNG, time, uuid or set order decides anything.
Same seed → byte-identical annotations in separate processes with different
`PYTHONHASHSEED`; the same scene renders identical raw frames (hashed);
regenerating seeds into a fresh folder reproduces `annotations.jsonl`
exactly (apart from `render_meta` timing when rendering).

## Diversity (5000 dry-run seeds)

5000 / 5000 unique scenes; scenarios train 1283 / reflect 1273 / collide
1234 / junction 1210; chain lengths 8 / 10 / 12 at 1586 / 1688 / 1726;
11 thin speeds 1.5-4 marks/s; 1-4 pulses (2483 / 1200 / 885 / 432); 3-12
traced pulse segments; clips 96-240 frames; 2665 distinct (question,
answer) pairs; answer letters A/B/C/D 25.6 / 25.2 / 25.6 / 23.7 %; mean 2.8
attempts per scene (no failures).

## Files and commands

```
sim.py                     scene generation, exact pulse trace, isolation, checks (pure, seeded)
questions.py               the 10-question ladder (LADDER = "ten")
answers.py                 independent re-derivation from the drawn chain
render.py                  PIL renderer (1280x720, 24 fps, piped to ffmpeg); render.track = what is drawn
run.py                     CLI: generate / validate (common/video5.py)
generate.py                same as `run.py generate` (generate_all_datasets.sbatch)
flatten.py                 annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
frame_check.py             decoded-frame checks of rendered clips
test_spring_chain_waves.py pytest suite

python run.py generate --n 50 --seed 0 --output-dir ../output_videos/spring_chain_waves_preview
python run.py validate --dataset-dir ../output_videos/spring_chain_waves_preview
python frame_check.py --dataset-dir ../output_videos/spring_chain_waves_preview
python -m pytest test_spring_chain_waves.py -q
```
