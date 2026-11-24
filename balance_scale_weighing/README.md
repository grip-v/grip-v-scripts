# balance_scale_weighing

Video-QA dataset about a balance scale. Weights land on the scale, are lifted off it or jump from one pan to the other, and the beam tips each time. Every weight is a coloured shape, and its mass is the number of white dots printed on it (one dot = one unit of mass). Clips are 1280x720, 24 fps, at most 10 s long, rendered with PIL on the shared pipeline (`common/video5.py`). Each clip has 10 questions on the 5-rung ladder (`common/ladder5.py`, `LADDER = "ten"`).

## Physics

- **Torque:** right minus left. On a pan every weight has arm 1 (where it sits on the pan doesn't matter). On the lever the arm is the peg number, 1-4 counting outward from the pivot.
- **Beam angle:** `deg_per_unit x torque`, clamped at `+-max_deg`. Positive means the right side is lower. When the target changes, the beam eases to it over `trans_f` frames (smoothstep, 10-14 frames). If a new change arrives mid-swing, the beam eases from wherever it is.
- **Resting positions:** left side lower, right side lower, or level. Torques are integers, so "level" is exact, and every tilted rest is at least 4 degrees.

## Scenarios (`scene["scenario"]`, about 20% each)

| scenario | what happens |
|---|---|
| `drop` | 4-7 weights fall one at a time onto the left or right pan |
| `add_remove` | 2-4 weights start on the pans; 2-4 new ones fall on and 2-4 are lifted off (they rise out of the top of the frame) |
| `transfer` | 3-6 weights start on the pans; weights jump over to the other pan 3-6 times (a weight can jump twice), with up to 2 extra drops |
| `lever` | a plain beam on a triangular fulcrum with 4 pegs per side (ticks under the beam keep them visible); 4-6 weights fall onto empty pegs, with 0-2 already there at the start. Weights tilt with the beam. |
| `mystery` | like `drop`, but one weight (never the last to land) has no dots. Its mass (1-6) is chosen so the beam comes to rest exactly level at some moment while it is on the scale, which makes the mass solvable. |

## Random knobs (all drawn per seed)

- **Weights:** 3-8 per clip. Each one gets a unique colour (from 9), a shape (circle, square, hexagon, triangle, pentagon), a pattern (plain, diagonal stripes, inner ring, darker lower half), a mass of 1-5 (1-4 on the lever) and a size of 58-80 px (54-62 px on the lever).
- **Physics and timing:** `deg_per_unit` is 4-6 for pans and 2.6-4.2 for the lever, and `max_deg` is 12-16. Each clip also draws its own transition time, fall time, lift time, hop time, gaps between events, time of the first event and how long the end is held.
- **Look:** 8 base colour themes with per-channel jitter, 5 background patterns, 3 floor patterns, 3 stand styles and 3 pan styles.
- **Camera:** zoom 0.90-1.06, horizontal offset +-40 px, pivot height 390-445 px.

Over 5000 seeds, every scene signature is unique.

## Frame rules

- **Text:** the only text in the frame is the timer in a fixed box in the bottom-left corner, away from the action. It shows `floor(t*10)/10` seconds and stays readable at 640x360. Questions name weights by colour and shape ("the red hexagon"). Colours are unique within a clip.
- **Event timing:** every event a question depends on (a landing, a lift-off, a landing after a jump) happens at least 0.5 s after the start and at least 0.5 s before the end. It lands on a frame whose exact time is within 0.045 s above a tenth, so the timer reading and the true time agree.
- **Rests and gaps:** every rest state lasts at least 11 frames (0.46 s), and the final one at least 22 frames. Events are always at least 2 frames apart.
- **Persistent results:** every event leaves a lasting result: a weight stays on its pan or peg, or is gone for good. There are no single-frame flashes.
- **Minimum sizes:** weights are at least 50 px, lines at least 3 px and dots at least about 9 px across.

## The 10-question ladder

Per-level counts come from `L5.ten_counts`: L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3, 10 questions in total, ordered q01..q10 by level. The template is drawn first and its params second. Every template is used once before any template repeats with different params, and no `derivation_key` + params pair repeats within a clip. All questions have 4 options, and the correct letter is uniform across A-D. Each tilt question's 4th option is "it keeps rocking and never comes to rest".

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `final_count`, `side_count`, `end_dir`, `dots`, `shape`, `pattern`, `peg` (lever), `first_arrival` |
| L2 relational | `dir_after` (which side is lower once the beam rests after an event), `first_of` / `last_of` (which of 4 weights lands first or last), `heaviest_of`, `where_end` (add_remove, transfer) |
| L3 measurement | `event_time` (timer reading at a landing or lift), `gap` (time between two events), `side_mass_end`, `landings`, `lifts`, `hops`, `hops_to`, `max_side_count` |
| L4 integration | `changes` (number of resting-position changes), `level_count`, `longest_dir`, `first_turn` (which weight first made a side the lower one), `mass_after` (mass on a side right after an event), `max_side_mass`, `torque_end` (lever: sum of mass x peg), `mystery` (mass of the dotless weight) |
| L5 counterfactual | `cf_remove`, `cf_other_side`, `cf_heavier` (+1 or +2 units), `cf_extra` (one more weight with k dots, on a pan or an empty peg), `cf_changes_remove`, `cf_skip_hop`, `cf_no_lift`, `cf_shift_peg` |

Every counterfactual is re-simulated: `sim.timeline` runs with the edit (object removed, side flipped, mass changed, event skipped, peg shifted). It is never guessed. About 87% of L2-L5 questions have `requires_motion=True`.

### min_fps

Every question stores `params["min_fps"]`: the sampling rate needed so that at least one sampled frame falls in each window the answer depends on.

- **Rest-state and order questions:** `1 / shortest relevant window`, where the window is a rest state or the gap between the two events compared.
- **Event-time questions:** `2 / (gap to the nearest wrong option)`. Wrong options are at least 0.8 s from the answer.
- **Time-gap questions:** `4 / 1.2 s`. Options are 1.2 s apart, and both endpoints carry sampling error.

Over 5000 clips, 59% of questions are answerable at 1 fps, 91% at 2 fps and 100% at 4 fps (max 3.34).

## Closed loop

- **Stored derivation:** every question stores `derivation_key` + `params`.
- **Independent answers:** `answers.py` never calls `sim.py`. It replays the renderer's per-frame geometry (`render.layout`, the same data `render.frame` draws) and works from that:
  - **What is on the scale:** which weights sit still in each frame, and on which side of the pivot.
  - **Torque:** taken from the drawn geometry (mass x pan-post distance, or mass x distance along the drawn beam).
  - **Rest runs:** runs of at least 3 frames with an unchanged beam angle. Each run's torque sign is cross-checked against the sign of the drawn angle.
  - **Event frames:** found from rest/non-rest transitions. Times come from the timer string drawn on that frame.
  - **Counterfactuals:** the scene is edited and the geometry re-run.
- **Generator checks:** `sim.check` rejects a scene if any of these hold:
  - a tilt is under 4 degrees;
  - a rest window is too short, or the beam comes to rest while a weight is mid-jump;
  - an event frame isn't timer-aligned, or is within 0.5 s of either end;
  - two events are less than 2 frames apart;
  - pans hold more than 4 weights, or two weights share a peg;
  - colours repeat;
  - a mystery weight can't be solved;
  - the resting position changes fewer than 2 times.
- **Question checks:** question builders also refuse near-ties. `heaviest_of` needs a unique maximum. `longest_dir` needs a lead of at least 0.75 s. Time options must be at least 0.3 s apart.
- **Validator:** `python run.py validate` rebuilds every clip from its seed and diffs scene and questions. It also runs `sim.check` and the 10-question ladder check, which includes `min_fps`, re-derives every answer independently and, without `--skip-video`, checks the MP4.
- **Frame check:** `python verify_frames.py --dataset-dir <dir>` decodes the clips and checks:
  - **Timer:** the timer reading at every event frame, by template match against freshly drawn readings.
  - **Event frames:** every landing is at its resting spot on its event frame and not 3 frames earlier, and every lift-off has left its spot 3 frames after.
  - **Positions:** every 6th frame, each resting weight's base-colour pixels are where its sprite should be (coverage >= 60%, centroid within 12 px), and weights not in the frame leave no blob of their colour.
  - **Final count:** the weight count in the last frame.

## Determinism

All randomness comes from the `random.Random(seed)` passed in. The code uses no global RNG, time, uuid or set/dict-order decisions. The scene is plain JSON, and a sample is byte-identical across processes with different `PYTHONHASHSEED`. Rendering the same seed twice gives identical raw frames. `test_balance_scale_weighing.py` covers determinism, the ladder, uniqueness, the closed loop, frame bounds and overlap, a tamper test and hand-checked edge cases.

## Files and commands

```
sim.py            scene generation, physics timeline, checks (pure, seeded)
questions.py      the 10-question ladder
answers.py        independent answer re-derivation from the renderer's geometry
render.py         PIL renderer + per-frame layout (1280x720, 24 fps, piped to ffmpeg)
run.py            CLI: generate / validate (shared pipeline in common/video5.py)
verify_frames.py  decoded-frame checks of timer, positions and counts
test_balance_scale_weighing.py

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/balance_scale_weighing
python run.py validate --dataset-dir ../output_videos/balance_scale_weighing
python -m pytest -q test_balance_scale_weighing.py
```
