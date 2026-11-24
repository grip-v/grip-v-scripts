# mirror_beam_reflection_v2

Harder, longer version of `mirror_beam_reflection`. A beam enters a labelled grid at any non-corner edge cell, moving inward, and bounces off diagonal mirrors ('/' and '\'). The clip shows a long prefix of the beam's path (at least 4 bounces, at most about 75% of the path, 0.3 s per cell, 1.2 s hold at the end, roughly 5-10 s total). It stops with the beam head on an empty (mirror-free) cell, after a visible bounce and before the exit, and the unseen remainder always contains at least one more bounce, so the exit questions need the reflection rule.

## What a clip shows

- 1280x720, rows/cols labelled, mirrors (cyan) drawn over the beam (yellow), a bright head, a small dot on the entry cell and a stub from the grid edge showing the entry side.
- Tier (`scene['tier']`, sampled ~evenly by seed):

| tier   | grid | mirrors |
|--------|------|---------|
| easy   | 6x6  | 8-10    |
| medium | 7x7  | 10-13   |
| hard   | 8x8  | 13-16   |

- Layouts are searched with retries (favouring long paths) and are loop-free: the beam always exits.

## The 5-rung question ladder

Exactly five multiple-choice questions per clip, one per rung (`common/ladder5.py`); every question stores `derivation_key` and `params`. Each rung has several variants:

- L1 observation: `mirror_count`, `slash_mirror_count`, `backslash_mirror_count`, `entry_side`
- L2 relational: `bounces_shown`, `first_mirror_hit`, `third_mirror_hit`
- L3 measurement: `cells_shown` (distinct cells crossed), `slash_hits_shown`, `backslash_hits_shown`
- L4 integration: `exit_edge`, `exit_cell` (of the unseen remainder)
- L5 counterfactual: `cf_insert_exit_edge`, `cf_insert_exit_cell` (add a mirror on a cell the beam never visited in the video), `cf_remove_exit_edge`, `cf_flip_exit_edge`, `cf_flip_exit_cell` (remove/flip a mirror the beam only meets after the clip ends). The changed layout always still exits.

## Closed loop

`answers.py` re-derives every answer from the stored scene only (a separate complex-number trace; counterfactuals re-trace the modified layout) and does not import `sim.py` or `questions.py`. `python run.py validate` fails a clip on any disagreement, seed-rebuild mismatch, `sim.check` error (loop, stop on a mirror, <4 bounces shown, no unseen bounce, >75% shown) or bad MP4 duration, resolution or motion.

## Files and commands

```
sim.py        scene generation (tiers, layout search, truncation) + check
questions.py  the 5-rung ladder
answers.py    independent answer re-derivation (validator only)
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (shared pipeline in common/video5.py)

python run.py generate --n 10 --seed 0 --output-dir ../output_videos/mirror_beam_reflection_v2
python run.py validate --dataset-dir ../output_videos/mirror_beam_reflection_v2
```
