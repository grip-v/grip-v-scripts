# random_walk_home

Video-VQA sub-benchmark testing **multi-agent state tracking, discrete
event counting under a stochastic process, causal/collision reasoning,
wind-direction reasoning, and symbol-to-location matching** — 1-5 ants,
each starting AWAY from its own home, take a random walk on a shared hex
maze board until each is either home or dead, with five independent
optional hazards along the way: a home that relocates mid-walk, board
tiles that swap open/closed, hazard holes that send an ant back to its own
start, roaming predators that end an ant's walk for good, and wind that
blows in gusts, pushing every ant downwind at once.

This dataset started as a single-ant "walk on a fixed graph until you
step back onto home" mechanic, then was rebuilt into its current
multi-agent, dynamic-environment form directly from user feedback ("the
only thing here is coming back home... i need more complexity") — the
original single-ant version has been retired in favor of this one. The
wind layer and the five-level question ladder were added later, also from
user feedback (an earlier "homing pull" idea was dropped as unfair).

## 1. What each clip shows

- **The board**: a fixed-size hex maze (`ARENA_RADIUS = 4`, 61 cells),
  ~25% randomly blocked (black, impassable) at the start, the rest
  walkable (white).
- **1-5 ants**, each a distinct glyph (♠ spade / ♥ heart / ♦ diamond / ♣
  club / ★ star) drawn in that ant's own fixed color, each assigned:
  - a **home cell** — a distinct white cell, tinted and ringed in that
    ant's own color, so you find your home by matching the ring's color
    to your own ant's color (once an ant is home, its glyph sits inside
    the ring);
  - a **start cell** — a *different* white cell, so every ant genuinely
    has to search, not just confirm it's already there.
  - All ants move simultaneously, one hex-hop per tick, choosing
    uniformly among their current cell's currently-walkable neighbors
    (unless a gust blows them — see wind below).
- **Five independent, per-clip-randomized hazards** (not all-or-nothing —
  most clips have 1-3 active, mirroring `gear_train`'s stall/reversal
  convention):
  - **home relocation** (`HOME_MOVE_PROBABILITY = 0.25`) — one still-
    searching ant's home jumps to a new cell partway through (never onto
    that ant's own start cell, where a hole would bounce it);
  - **tile swaps** (`TILE_SWAP_PROBABILITY = 0.3`) — a blocked/open pair
    of cells trades state, opening a new route or closing an old one
    (only ever applied if it provably keeps every still-relevant cell —
    every active ant's position, home and, when the board has holes, start
    cell, plus every predator — mutually reachable; otherwise skipped that
    tick);
  - **hazard holes** (`HOLE_PROBABILITY = 0.3`, 1-2 grey cells, fixed for
    the whole clip) — stepping on one bounces that ant back to its own
    **start** cell (not home) — a setback, not a death. On screen the ant
    walks onto the grey cell, shrinks into it, and reappears on its start;
  - **predators** (`PREDATOR_PROBABILITY = 0.3`, a red dot wandering the
    board) — an ant sharing a predator's cell after a tick **dies**
    (vanishes for the rest of the clip, no reset — permanent, unlike a
    hole). Predators hop cell to cell in step with the ants, so a catch
    shows the predator landing on the ant as it vanishes;
  - **wind** (`WIND_PROBABILITY = 0.62`; after rejection sampling about
    42% of clips are calm) — a **light** or **strong** wind toward one of
    the 6 hex directions (up, up-right, down-right, down, down-left,
    up-left). It blows in **gusts**: each tick is a gust or a lull
    (`WIND_GUST`: light 0.20 start / 0.50 continue, about 27% of ticks;
    strong 0.30 / 0.65, about 42%). During a gust **every** walking ant
    whose next cell downwind is on the board and open is blown one cell
    downwind, all at once (a grey hole there still swallows it); ants
    whose downwind cell is blocked, and every ant during a lull, take
    their normal random step. Predators are never blown. In 40% of windy
    clips the wind shifts once to a new direction at the start of a tick
    between 8 and 20 (`WIND_SHIFT_*`), and there is never a gust on that
    tick. The same wind helps ants whose home lies downwind and works
    against the others.
- **Wind on screen**: a gauge beside the board — an arrow pointing the way
  the wind blows, one chevron for light and two for strong, or **CALM** —
  a **GUST** tag that fades in and out with each gust, and light-blue
  streaks that drift slowly in a lull and race (longer, brighter) in a
  gust. The arrow visibly turns when the wind shifts.
- **Clip length and pacing**: every clip lasts **10-15 s**
  (`MIN_VIDEO_DURATION_S` / `MAX_VIDEO_DURATION_S`), and every tick of a
  clip plays at the same pace: the nominal `TICK_DURATION_RANGE` when that
  fits, otherwise evenly slower for a short walk (at most ~0.78 s per tick)
  or evenly quicker for a long one (at least ~0.18 s per tick).
- The clip ends once every ant is resolved — home or dead — with a flash
  on each home an ant reaches, then holds the final frame.

## 2. Why this needs video, not a single frame

The final frame shows only where each surviving ant ended up (or that it
vanished) — never *when* a home moved, *which* tile flipped, *whether* an
ant fell into a hole and had to restart, *which* predator caught which
ant, *how many* gusts blew, or *which way* the wind blew before it
shifted. Every non-trivial question here is a property of the full event
timeline, not the held frame.

## 3. The 5-question ladder

One question per level per clip, always in this order. Each level picks
one of its variants per clip (wind variants about half the time when they
apply); a variant is only offered when its answer is unique and visible.

| qid | level | difficulty | asks |
|---|---|---:|---|
| q01 | Direct Observation | 1 | how many ants started; how many reached home; was there any wind; which way it blew at the start / at the end (after a shift); light or strong |
| q02 | Relational Reasoning | 2 | did this ant reach its own home (only when some ant didn't); which ant was caught; whose home jumped; was this ant's home higher or lower on screen; was it downwind or upwind; would one step with the wind bring it closer to home; which ant would the wind help |
| q03 | Measurement | 3 | how many catches (predator clips), hole falls (clips with holes), tile swaps; an ant's starting hex distance to home; how many of an ant's moves went with the wind; on which move the wind shifted; how many gusts |
| q04 | Multi-step Integration | 4 | total event count (home moves + tile swaps + hole falls + catches + wind shifts, each counted once); ants that fell in a hole and still got home; first ant home; longest walk; ants the wind helped / hindered that still got home; total gust pushes across all ants |
| q05 | Counterfactual Reasoning | 5 | (same moves held fixed) if holes were deadly, if the home never jumped, if the home were the cell after move N; if the ant walked a shortest open route; if every move were a gust (how many moves before it stops / would it reach home) |

Answer-balance rules: yes/no and two-way questions pick the asked-about
ant so both answers are about equally likely; count questions whose answer
would almost always be 0 are thinned out (see above); counted quantities
stay small enough to follow by eye (move numbers at most 12, gust pushes
at most 12).

## 4. Ground truth / simulation

`simulate.py` is pure Python/networkx, zero rendering imports — the
entire multi-ant scenario (every tick, every event) is simulated before
Manim ever runs. Per tick: any scheduled tile-swap or home-move event is
attempted first (each independently skipped if no *safe* candidate exists
— see below), then a scheduled wind shift is applied and the tick's
gust/lull state is drawn, then every still-searching ant picks uniformly
among its current neighbors that are walkable *at that tick* (replaced by
one cell downwind in a gust, when that cell is open), then predators move
uniformly the same way, then collisions are resolved. The scenario ends
once every ant is either `home` or `dead`. Every hole fall is recorded
(`hole_hit_history`) so the scene can show it; this draws no random
numbers.

**Safety of tile swaps.** A swap-to-white candidate is always chosen
adjacent to the current walkable region (so it can only ever *expand*
reachability). A swap-to-black candidate is accepted only if, after both
flips, every ant/home/predator cell still relevant to the simulation —
including, when the board has holes, the start cell of every walking ant,
since a hole sends it back there — remains in one connected component of
the new walkable graph (checked via `networkx.node_connected_component`)
— otherwise that attempt is abandoned and the swap simply doesn't happen
that tick. No ant is ever silently walled in or bounced onto a black cell.

**Rejection sampling for scenario length.** A candidate scenario that
hasn't fully resolved (every ant home or dead) within `MAX_TICKS = 70`, or
that resolves in under `MIN_TICKS = 10` ticks (too short to fill a 10 s
clip without crawling), is discarded and a fresh one is drawn from the
same rng stream — never a mid-walk truncation. This board/hazard
combination rejects fairly often (measured: about 4.4 candidate draws per
accepted scenario, median 3, over 1000 seeds) — cheap
in wall-clock time (all pure Python, no rendering), but real, and
documented here rather than hidden: it means the accepted sample is
implicitly conditioned on "resolves within 70 ticks," which very lightly
favors scenarios where ants get eaten quickly over scenarios where every
ant wanders for a long time before finding home.

## 5. Files

```
config.py       sampling ranges, colors, board/hazard/wind/timing constants
arena.py        hex-board geometry: neighbor lookup, pixel conversion, initial blocked-cell layout
simulate.py     full multi-ant tick-by-tick scenario simulation (incl. wind + gusts) + every derived ground-truth statistic
questions.py    builds the 5-level question ladder from a ScenarioResult
scene.py        Manim scene (reads params from $MAG_PARAMS_FILE); pure playback of the precomputed tick schedule
generate.py     params -> annotations.jsonl (+ manim render + final-frame PNG extraction)
flatten_annotations.py   annotations.jsonl -> dataset_final.csv / dataset_final.jsonl
validate_random_walk_dataset.py   independent re-simulation + statistic/question re-derivation + video checks
annotations.jsonl, videos/, frames/, dataset_final.{csv,jsonl}, validation_report.txt
```

## 6. Generation and reproducibility

Python 3, NetworkX, Manim, OpenCV (validator only). Sample `i` uses
`random.Random(i)` inside `simulate.simulate_scenario` for the entire
scenario, and a second, independent `random.Random(i)` stream in
`generate.py` for question-variant selection (drawn first) and tick
timing (drawn second) — `validate_random_walk_dataset.py` replays both
streams exactly.

```
python generate.py --n 20 --output-dir . --seed 0 --quality low
python flatten_annotations.py --dataset-dir .
python validate_random_walk_dataset.py --dataset-dir . --report validation_report.txt
```

`--sample` for 5 dev examples at `-qh`, `--no-render` for a param-only
dry run, `--append` to resume a batched/interrupted run.

## 7. Validation

`validate_random_walk_dataset.py` re-simulates every scenario from its
stored seed and diffs, exactly: every ant's outcome/steps_taken/
reset_count, and every scenario-level counter (`home_moved`,
`tile_swap_count`, `hole_reset_count`, `death_count`, `total_events`,
`total_ticks`, wind strength, wind shift tick, gust ticks); confirms
`n_ants_home + n_ants_dead == n_ants` and that every ant's outcome is
`home` or `dead` (never left unresolved — a scenario that hits
`MAX_TICKS` unresolved is rejected at generation time and never stored);
re-derives all 5 questions and diffs `groundtruth`/`answer`/`options`; and
decodes every rendered MP4 with OpenCV, checking its frame count against
the stored timing schedule.

The wind and ladder update was additionally checked with independent
tools (kept outside this folder): every answer re-derived by code that
does not import `questions.py` (2000 clips, 10,000 questions, 0 errors);
the gust rule (in a gust, every ant with an open downwind cell was blown:
13,037 of 13,037; in lulls ants and predators moved downwind only at the
random rate); and pixel checks on 10 rendered clips of the gauge arrow
direction, chevrons, GUST tag, CALM, streak direction, ant positions,
predator hops and catches, and hole falls (all pass; deliberately wrong
inputs fail them). Rendered clips were also inspected by eye: home rings
stay identifiable by color across the whole clip, a moving home is easy
to follow (a discrete jump, not a drift), tile swaps visibly open/close
specific hexes, hole cells (grey) and predators (red dots) are
immediately distinguishable from the maze, and a caught ant's glyph
vanishes as the predator lands on it.

Ends with a machine-greppable `RESULT: <n_samples> samples, <n_questions>
questions, <n_mismatches> mismatches` line. Current status: **20 samples,
100 questions, 0 mismatches**.

## 8. Known limitations

- **Predators are quite lethal on this board size.** Empirically, when a
  predator is present, roughly 55-60% of that clip's ants end up caught
  rather than reaching home (measured again after the wind update: 58%
  over 1000 simulated scenarios) — a consequence of a single
  random-walking predator having a high meeting-probability with a
  random-walking ant over tens of ticks on a ~46-walkable-cell board, not
  a bug in the collision rule. `N_PREDATORS_RANGE` is deliberately capped
  at exactly 1 (not 1-2, as first considered) to avoid making it worse.
- **Rejection-sampling selection bias** (see §4): scenarios where an ant
  dies quickly resolve faster than scenarios where every ant has to
  genuinely search, so the `MAX_TICKS = 70` cutoff mildly over-represents
  predator-active, ants-die-fast scenarios in the accepted sample
  relative to the nominal `PREDATOR_PROBABILITY = 0.3` draw rate. It also
  lowers the share of windy clips from the nominal 62% to about 58%, and
  `MIN_TICKS = 10` leaves out walks that would end in under 10 ticks.
- **Board size is fixed, not walk-bounded** — always the same
  `ARENA_RADIUS = 4` board, so up to 5 simultaneous homes/starts/hazards
  all fit legibly in frame; a longer walk just means more hops within
  that same fixed board, not a bigger rendered area.
- **Tile-swap connectivity checking is per-attempt, not globally
  optimal.** A swap is accepted the first time a safe candidate is found
  (bounded to 12 candidate attempts), not the *best* available swap — if
  no safe candidate turns up in those 12 tries, that tick's swap is
  skipped entirely rather than exhaustively searching every remaining
  candidate.
- The Level-4 event-count question treats a home move, a tile swap, a
  hole fall, a death and a wind shift as equally-weighted "one event"
  each, even though they're qualitatively different in significance — a
  deliberate simplification to keep the question's ground truth an
  unambiguous single integer.
- **Long walks play fast.** The longest walks run at ~0.18 s per tick
  (about 5.5 frames), so a hole fall there is quick (still drawn in full);
  the shortest walks run at up to ~0.78 s per tick.
- **Lulls are subtle.** Between gusts the streaks are faint by design;
  the wind is read from the gauge and from the gusts.
- Two rare questions lean: "would the ant have stepped onto its original
  home" is mostly *no*; "would a constant gale carry it home" is asked in
  only about 1 clip in 80.

## 9. Reasoning skills tested

- **Multi-agent state tracking** — following several independently-moving
  agents at once, not just one.
- **Discrete event counting under a stochastic process**, compounded
  across five distinct event types (home move / tile swap / hole fall /
  death / wind shift), plus gust counting.
- **Causal/collision reasoning** — recognizing when and why a specific
  ant's walk ended (caught vs. arrived).
- **Wind-direction reasoning** — reading the wind's direction and
  strength, relating it to each ant's position and home (downwind or
  upwind, helped or hindered), and seeing every ant pushed together in a
  gust.
- **Symbol-to-location matching** — matching each ant to its home by its
  color, with each ant's distinct glyph keeping the ants apart.
- **Comparative reasoning across agents** — first home, longest walk,
  among simultaneously-moving walks.
- **Persistence vs. permanence reasoning** — a hole is a *setback*
  (walk continues from start), a predator is *permanent* (walk ends) —
  telling these apart requires having watched what happened, not just
  noting an ant is no longer visible.
- **Counterfactual reasoning** — what would have happened under a stated
  change (deadly holes, an unmoved home, a constant gale), with
  everything else held fixed.
