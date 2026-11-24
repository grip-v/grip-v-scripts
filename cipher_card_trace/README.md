# cipher_card_trace

Video-QA sub-benchmark for **decoding a message whose key changes while it
is being sent**. v3 is a rework on the shared PIL + ffmpeg pipeline
(`common/video5.py`) with the 10-question ladder (`LADDER = "ten"` in
`questions.py`). v2 showed only one card at a time but still allowed many
questions to be answered from a single frame, and its options sometimes
named cards that were never dealt (so "the one I saw" was always right).
v3 makes the answers depend on order and timing: cards cover earlier cards,
a ring turns mid-clip, and every card option is a card that really is dealt.

## What each clip shows

- **Cipher wheel** on the left or right half: an outer ring of the 26
  letters (clockwise, random starting rotation) and an inner ring of the 26
  card symbols (13 red ranks, 13 black ranks; order: red/black blocks,
  alternating, second block reversed, or shuffled; random starting rotation).
- **Pile(s)** on the other half: every card slides in from off screen and
  lands exactly on the pile, covering what is there. Only the top card of a
  pile is ever visible, and the pile does not grow thicker, so the final
  frame does not reveal how many cards were dealt.
- **Decoding:** a card stands for the outer letter its colour + rank sits
  under at the moment it lands. The cards spell a random code of 4-6
  distinct letters (not a word, so the answer can't be guessed from English).
- **Key changes:** a turn rotates one ring by 1-5 positions (smoothstep, a
  few frames per position) a moment after a joker (or trigger card) lands.
- Clips are 4.75-10 s at 24 fps, 1280×720. The first slide starts
  ≥ 13 frames (0.54 s) after the start and the last event ends ≥ 13 frames
  before the end.

### Scenarios (`scene["scenario"]`, about 20% each)

| scenario | setup |
|---|---|
| `clockwise` | 1-3 jokers; each turns the inner ring clockwise |
| `both_ways` | 2-3 jokers; each turns the inner ring clockwise or counter-clockwise, both directions occur (the direction is seen only in motion) |
| `letter_ring` | 1-3 jokers; the outer letter ring turns instead, either way |
| `trigger` | no jokers; 1-3 ordinary cards turn the inner ring clockwise a moment after they land (the card itself uses the key in force when it landed); 5-6 cards |
| `two_piles` | 1-2 jokers, inner ring clockwise; cards and jokers land on a left or a right pile (each gets ≥ 2); the code is read in dealing order across both piles |

### Random knobs

Scenario, code length and letters, ring order style, both starting
rotations, number / position / size / direction of turns, suits, pile of
each item, wheel side, wheel size (outer radius 298-336 px) and band widths,
wheel position, card size (148-192 px wide), pile positions, slide-in edge
(far side, top, bottom, two diagonals), slide length (7-11 frames), per-card
rest (9-19 frames), turn speed (3-6 frames per position), pauses before
and after turns, intro and hold, felt colour (8), background pattern
(plain/gradient/dots/weave), wheel palette (6), joker colour (5), card edge
colour, shadows, centre pip size.

## On-screen text

- **Clock** (always): fixed dark panel top-left, "s.t s" (truncated to
  0.1 s), 36 px bold (cap height 26 px). `sim.check` rejects any wheel or
  card slide that comes within 8 px of it.
- **Ring letters A-Z** (outer ring) and **card ranks A, 2-10, J, Q, K**
  (inner ring, red or black): these are the cipher itself. Colour or shape
  cannot replace them: the task is a letter cipher keyed by card symbols,
  and every decode / counterfactual question asks for a letter or a symbol.
  They are identifiers, never answers: no decoded letter, running decode or
  message is ever drawn. 34 px bold (cap height 25 px ≥ 24 px), dark ink on
  light ring bands. Each label is upright and sits at a fixed radius at
  angle −90° + (index + ring rotation) × 360/26, so it moves with its ring.
  The labels are part of the scene (`scene["labels"]`: ring, index, text,
  ink, radius, size); the renderer draws exactly that list. `sim.check`
  rejects scenes where a label is under 24 px, leaves its band, or touches a
  neighbour at any of 8 sub-positions of a turn; the wheel never overlaps
  the timer or a card's path.
- **Card faces**: the rank in two corners plus suit pips; jokers carry only
  stars (no "JOKER" text). These are the objects themselves.
- Nothing else: no titles, counters, arrows, hints or legends.

## The 10-question ladder

Per clip the level counts are drawn from 1-3 / 2-3 / 2-3 / 1-3 / 1-3 (total
10, `L5.ten_counts`); questions are spread over as many templates as
possible (the scenario's own templates usually first), ordered by level,
numbered q01..q10, and every question has 4 options. 34 templates:

- **L1 observation:** cards dealt (covered cards make this a video count);
  jokers dealt; red / black cards; cards of a suit; first card; how many
  times the ring turns (trigger); items on the left/right pile (two_piles).
- **L2 relational:** which of four cards is dealt first / last (the last
  card dealt is never among the "last" options); card right after / before
  a card; card right after / before the k-th joker; card on top when the
  k-th turn starts (trigger); which of four cards lands with the wheel in
  the same position as X; next card to cover X; which of four cards lands on
  the left/right pile (two_piles).
- **L3 measurement:** size (and direction, in both_ways / letter_ring) of
  the k-th turn; the letter card X stands for; clock reading when X lands;
  top card when the clock shows t; letter a given inner symbol sits under
  when the clock shows t.
- **L4 integration:** the whole code; which card stands for letter L; the
  total / net turn; how many cards would still decode the same with the end
  wheel; card after the biggest turn; letters of one pile (two_piles).
- **L5 counterfactual (re-simulated):** the code if the k-th turn had not
  happened; one card's letter without the k-th turn; the code had the ring
  never turned; the code had the k-th turn gone the other way
  (both_ways / letter_ring); the symbol the last card would need with no
  turns; a card's letter had two cards swapped places in the deal.

Distractors: card options are always other dealt cards (nearest in the
deal first); letters come from the same card under the other keys of the
clip, then ±1; codes are the decodes from plausible mistakes (one key for
all cards, the start or end key, a missed / reversed / off-by-one turn).
All questions set `requires_motion=True`.

## Margins (no near ties)

- Every event (slide start, landing, turn start/end) is ≥ 2 frames from the
  next one, and in practice ≥ 5; each card lies uncovered for ≥ 9 frames,
  every ring rest is ≥ 6 frames.
- Clock moments sit ≥ 2 frames inside a top-card span or a ring rest on
  every frame showing that reading. Landing-time options are other cards'
  landing readings, ≥ 0.7 s apart.
- At every landing the ring is at rest on an integer position, so the
  symbol is centred under its letter (0° misalignment, sectors 13.8° wide).
- Turns are 1-5 positions, so the direction of the shortest rotation is
  never ambiguous; turns of all sizes are drawn independently.

## min_fps

Every question stores `params["min_fps"]`, the smallest of 1/2/3/4/6/...
fps that puts a sample inside the shortest interval its answer needs: the
uncovered span of each card it involves, the ring rest around a landing or
a turn, half the gap between landing-time options. Over 5000 scenes: 20.8%
of questions need ≤ 1 fps, 70.9% ≤ 2 fps, 100% ≤ 4 fps.

## Closed loop

- **Questions** store `derivation_key` + `params` (deal indices, joker
  number, clock tenths, symbol, option sets) + `min_fps`.
- **Generation** (`sim.py`) works with integer ring offsets.
- **Independent answers** (`answers.py`) never read the offsets, the code
  or any sim helper. They decode the clip from `render.track(scene, i)`,
  the renderer's per-frame state (ring rotations in degrees, cards drawn and
  where): landings are the first frame a card sits on a pile centre, turns
  are runs of frames where a rotation changes, and each card is decoded by
  comparing the angle of its symbol with the angle of every letter at its
  landing frame. Counterfactuals are replayed on those measured angles.
- **`sim.check`** rejects scenes whose cards don't spell the code, with
  duplicate cards, jokers at an end or side by side, scenario-rule
  violations, events closer than 2 frames or within 0.5 s of an edge, over
  10 s, layout overlaps (wheel/timer/slides/piles/labels), or counterfactual
  decodes that equal the real code.
- **`run.py validate`** rebuilds every sample from its seed, runs
  `sim.check`, the ladder check and answers.py on every question, and (with
  videos) checks resolution, duration and motion.
- **`frame_check.py`** decodes the MP4s: identifies the top card of each
  pile in the middle of every uncovered span by template matching against
  all 52 faces + joker (sequence must equal the deal); two frames after
  every landing reads all 52 ring labels by glyph matching (must equal the
  scene labels) and reads the landed card's letter off the wheel (must
  equal the code); reads the clock at every landing, every question moment
  and every clock answer; re-reads pile / wheel at question moments.

## Determinism

All randomness comes from the `random.Random(seed)` passed in; no global
RNG, time, uuid or set order decides anything. Same seed → identical scene +
questions in separate processes with different `PYTHONHASHSEED`; rendering
is a pure function of the scene (frame hashes checked across processes).

## Files and commands

```
sim.py                     scene generation, timing, rule and layout checks (pure, seeded)
questions.py               the 10-question ladder (LADDER = "ten")
answers.py                 independent re-derivation from the rendered track
render.py                  PIL renderer (1280x720, 24 fps, piped to ffmpeg)
run.py                     CLI: generate / validate (common/video5.py)
frame_check.py             decoded-frame checks of rendered clips
test_cipher_card_trace.py  pytest suite

python run.py generate --n 50 --seed 2000 --output-dir ../output_videos/cipher_card_trace
python run.py validate --dataset-dir ../output_videos/cipher_card_trace
python frame_check.py --dataset-dir ../output_videos/cipher_card_trace
python -m pytest test_cipher_card_trace.py -q
```

`annotations.jsonl` is the final output (sbatch entry
`cipher_card_trace:run.py::none:none`).
