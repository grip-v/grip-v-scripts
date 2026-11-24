# bouncing_ball_restitution

Video-QA sub-benchmark: **Bouncing Ball**, about restitution, energy loss and counting bounces. Objects fall under gravity and bounce on a floor. Each contact keeps a fraction *e* of the impact speed (the coefficient of restitution), so each rebound rises to *e*² of the previous height. Clips are rendered with the shared PIL + ffmpeg pipeline (`common/video5.py`) at 1280x720, 24 fps, 4-10 s long. Each clip has **10 multiple-choice questions** on the 5-rung ladder (`common/ladder5.py`, `LADDER = "ten"`).

This replaces the earlier standalone Manim/pymunk pipeline (`generate_bouncing_ball_dataset.py`, `config.py`, `physics.py`, `pymunk_sim.py`, `bounce_scene.py`, `validate_bouncing_ball_dataset.py`, `flatten_annotations.py`, presets, `ROADMAP.md` and related modules), which has been removed. The physics ideas carry over (restitution, bounce decay, counting, extrapolation), but they are now rebuilt as exact closed-form motion on the shared pipeline.

## Physics

- Units are pixels and seconds. Gravity g is 650-1500 px/s² (380-700 in `throw`) and is fixed within a clip.
- Between contacts the motion is pure free fall. At each floor contact, v_out = e · v_in.
- **Stop rule:** if a rebound would rise less than 30 px, the object stops on the floor instead. This means every bounce shown is at least 30 px high.
- In `throw`, the side walls reverse the sideways velocity with no loss. After an object stops bouncing it keeps sliding.
- `sim.bounce` gives the floor-contact times, rebound speeds and apex heights in closed form. `sim.center_at` gives where each object is drawn at time t.

## Scenarios (`scene["scenario"]`, chosen uniformly per seed)

| scenario | setup |
|---|---|
| `drop` | 3-5 objects in side-by-side lanes, all released at t=0 from different heights, each with its own e |
| `staggered` | 3-5 objects hang in clamps under a rail; the clamps open one at a time (0.6-6.6 s, at least 0.5 s apart) and stay open |
| `pads` | 3-5 objects fall onto coloured floor pads (solid, hatched or dotted). The **pad** sets e, so the questions ask which pad is bounciest, etc. |
| `throw` | 2-3 stacked shelves between two walls. Each object slides sideways (110-420 px/s, either direction) while bouncing on its own shelf |

## Random knobs

- Object count; shape (ball, block, hexagon, diamond); colour (10 colours, unique within a clip); pattern (solid, striped, dotted, ringed, two-tone, crossed); radius 17-40 px (20-40 px for diamonds, which cover half a square's area); drop height; e from √0.14 to √0.82.
- Gravity; release times; sideways speed and direction.
- Background (7 themes, 4 light and 3 dark, with gradients); overlay (none, grid, dots, height ruler, lane guides); floor style (solid, hatched, tiles, stripe) and floor tint; object outline (none, 2 px, 3 px).
- Framing: floor height, arena width and position, lane order and jitter, shelf heights.
- Clip length: either everything settles and then the clip waits 0.8-2 s, or the clip ends at a random 5.5-10 s while objects are still bouncing.

Objects are named by pattern + colour + shape ("the striped red ball"). Pads are named by colour ("the green pad"). Colours are unique within a clip, and pad colours never match object colours. Every colour is at least 120 RGB units away from the background.

## Frame content and sampling rules

- The only text in the frame is the clock: a fixed panel at the top left showing seconds truncated to 0.1 s ("3.4 s", 34 px font). Nothing else is drawn in that corner, or above y=100.
- Objects are at least 34 px across (40 px for diamonds). Pattern accents, the lower half of two-tone objects and pad markings use a neutral white or near-black, never a shade that could pass for another colour after compression. Floor, shelf, clamp and ruler lines are at least 3 px wide. Nothing flashes: every event leaves a lasting result (a rise after a bounce, an open clamp, a reversed direction).
- **Event timing (`sim.check`):** every event (release, floor contact, wall bounce) happens at least 0.5 s after the start. No event falls between 0.5 s before the end and 0.3 s after it. Events of one object are at least 3 frames apart. Rebounds are never within 12 % of the 30 px stop threshold, so there are no borderline stops.
- **Margins for compared quantities:** times ≥ 0.3 s apart; heights ≥ 15 px and ≥ 10 % apart; e² ≥ 0.1 apart; sizes ≥ 15 % apart. Counting windows stay ≥ 0.3 s from every contact.
- **Clock answers:** only asked when both frames bracketing the event show the same reading, and the event is 0.03 s or more from a 0.1 s boundary. Options are 0.5 s apart.
- **`params["min_fps"]`** is the sampling rate needed to answer, rounded up to 1/2/3/4/6/8/12/24:
  - counting bounces: 2 / (shortest flight between the object's contacts)
  - ordering events: 1 / (gap between the compared events)
  - clock readings: 4, since the options are 0.5 s apart
  - static questions: 1

## The 10-question ladder

Per-level counts are drawn with `L5.ten_counts` (L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3, total 10). Within a level, as many different templates as possible are used, and derivation_key + params never repeat. All questions have 4 options; decoys are objects that are not in the clip.

- **L1 observation:** number of objects; how many times X hits the floor; largest / smallest object; which starts highest; how many objects lie still at the end; which is released first (`staggered`); colour of the pad under X (`pads`); how many wall bounces X makes (`throw`).
- **L2 relational:** which of these hits the floor first; which rises highest after its first bounce; which stops bouncing first; which of two keeps more height per bounce; which of two lands more times; which is released right after X (`staggered`); which of these bounces off a wall first (`throw`).
- **L3 measurement:** clock time of X's first hit, last landing, release or first wall bounce; time between X's first and second hits; rebound height as a share of the drop (about 20/35/50/65/80 %); hits between two clock times; hits by a clock time.
- **L4 integration:** total hits across all objects; most / least bouncy object or pad; which object hits the most times; hits by the others before X's first hit; how many objects are still moving at time T; which comes to rest last; total wall bounces (`throw`).
- **L5 counterfactual (re-simulated):** hits in N more seconds; hits if X started twice or half as high; hits if X were as bouncy as Y, or fell onto another pad; total landings if X had stuck at its first landing; which would land first if all were released together (`staggered`); wall bounces at double or half the sliding speed (`throw`).

## Closed loop

- Every question stores `derivation_key` + `params`.
- `answers.py` never reads the generator's bounce lists. It works from the motion the renderer draws:
  - It samples `render.object_centers` at 480 Hz.
  - Floor hits are local minima of the gap to the floor. Hit times are refined by fitting the free-fall parabola through the two samples before contact.
  - It also detects rests, wall bounces (sign flips of the sideways motion), releases (first movement) and rebound heights from this track.
- Counterfactuals are re-simulated with a separate fixed-step integrator (dt = 1/2000 s, with the crossing solved inside the step).
- `python run.py validate` fails a clip if any of these hold:
  - rebuilding from the seed gives a different scene or questions
  - `sim.check` finds a problem
  - the ladder is malformed (wrong counts, order, a repeated key + params, bad options, missing min_fps)
  - an independent answer disagrees with the stored one
  - the MP4's duration, resolution or motion is wrong

## Determinism

- All randomness comes from the `random.Random(seed)` passed in. The scene and the questions are built in sequence from one stream.
- The code uses no global RNG, no time and no uuid, and no set or dict iteration order decides anything.
- Rejected attempts (failed `sim.check`, or too few askable questions) are retried from the same stream, so they stay deterministic.
- Renders are pure functions of the scene. The caches are keyed on the full scene JSON or the sprite parameters.

`test_bouncing_ball_restitution.py` covers:

- determinism across processes with different `PYTHONHASHSEED`, and frame-hash determinism
- ladder counts and min_fps
- independent-answer agreement
- uniqueness of scenes and objects
- hand-checked physics, wall reflection and clock cases
- a hand-checked seed
- the tamper test

## Files and commands

```
sim.py        scenes, closed-form bounce physics, sim.check
questions.py  10-question ladder (LADDER = "ten")
answers.py    independent re-derivation from the rendered track + fixed-step re-simulation
render.py     PIL frame renderer (1280x720, 24 fps, piped to ffmpeg)
run.py        CLI: generate / validate (common/video5.py)
test_bouncing_ball_restitution.py

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/bouncing_ball_restitution
python run.py validate --dataset-dir ../output_videos/bouncing_ball_restitution
python -m pytest test_bouncing_ball_restitution.py -q
```

`annotations.jsonl` is the final output; there is no flatten step. The `generate_all_datasets.sbatch` entry is `bouncing_ball_restitution:run.py::none:none`.
