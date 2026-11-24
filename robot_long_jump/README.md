# robot_long_jump

Video-QA dataset adapted from **Robot Long Jump (Jane Street, Mar 2023)**.

Each robot hops forward along its own lane in random steps. Once it stands at or past its own hidden take-off threshold, it makes one long jump into the sand pit. Its score is take-off position + jump length, which is the spot where it lands. If a hop carries a robot past the red foul line at 1.0 before it has jumped, the robot fouls: it topples over and scores 0.

Clips are 1280x720, 24 fps and at most 10 s long, rendered with PIL on the shared pipeline (`common/video5.py`). Each clip has 10 questions on the 5-rung ladder (`common/ladder5.py`, `LADDER = "ten"`).

## What is on screen

- **Track:**
  - **Lanes:** one lane per robot. A runway from 0.0 to the red foul line at 1.0, then a sand pit up to 2.0.
  - **Gridlines:** vertical gridlines every 0.1 (darker every 0.5) run from the ruler up through all lanes.
  - **Ruler:** at the bottom.
- **Positions on gridlines:** all positions are whole tenths, so every footprint, take-off spot and landing spot sits exactly on a gridline. A hop never lands exactly on the foul line.
- **Footprints:** every hop leaves a footprint that stays for the rest of the clip, white on the runway and dark on the sand. The take-off spot is the robot's last footprint on the runway.
- **Landings:** a landing leaves a dent in the sand, and the robot stays standing on it.
- **Fouls:** a robot that fouls lands its hop past the line, topples onto its side and stays there.
- **Robots:** each is a coloured shape on two legs, with an eye on its front.

## Scenarios (`scene["scenario"]`, about 25% each)

| scenario | what happens |
|---|---|
| `turns` | 3-4 robots go one after another (random order); hops are 0.1-0.4 long |
| `together` | 3-5 robots hop at the same time on a shared beat; each jumps on the beat after it reaches its threshold |
| `staggered` | 3-4 robots go one after another, each starting from its own start block at 0.0-0.3 (white blocks) |
| `steady` | 3-5 robots hop together on a beat, each with one fixed hop length (0.1-0.4) |

## Random knobs (all drawn per seed)

- **Robots:**
  - **Look:** a unique colour from 7 (orange, yellow, green, teal, blue, purple, pink; never red), a shape (circle, square, triangle, hexagon, diamond), a pattern (plain, striped, ringed, two-tone) and a body radius of 22-28 px.
  - **Placement:** a random lane.
  - **Rule:** a hidden threshold of 0.5-0.9, hops of 0.1-0.4 and a jump of 0.2-1.0 that lands in the pit (1.1-1.9).
- **Timing:**
  - **Motion:** hop time 5-7 frames, rest 2-4 frames (the beat is at least 9 frames), jump time 12-16 frames.
  - **Clip structure:** gap between turns 3-6 frames, intro 12-20 frames, final hold 12-22 frames.
- **Look and framing:**
  - **Colours:** 6 base themes (background, runway, sand) with per-channel jitter.
  - **Style:** footprint style (oval or bar), eye style (round or visor), gridline strength.
  - **Framing:** track origin x 150-186 px, scale 500-550 px per unit, lane pitch 101-150 px, vertical offset +-12 px.

Diversity over 5000 seeds (`--no-render`):

- **Uniqueness:** every scene signature is unique (5000/5000), and so is every robot line-up (colour/shape/pattern/start/hops/jump/lane) (5000/5000).
- **Scenario mix:** 1273 turns, 1210 together, 1234 staggered, 1283 steady.
- **Questions:** 3620 distinct (question, answer) pairs.
- **Answer letters:** A-D within 0.45 points of 25%.

## On-screen text

Besides the timer, the only text is the five ruler numbers **0.0, 0.5, 1.0, 1.5, 2.0**.

- **Why they are needed:** the measurement questions ask for positions and lengths in ruler units: take-off spot, jump length, score, hop length and start mark. Unnumbered gridlines alone give no absolute scale. Five anchors are enough, and the gridlines every 0.1 do the rest.
- **They never give an answer:** they never print a value a question asks for, and nothing labels what is happening. There is no "FOUL" text, no names and no title.
- **Style:**
  - **Position:** the labels sit in a fixed white ruler band at the bottom, below every lane. No robot ever enters that band.
  - **Look:** a single font, size (34 px, digits over 24 px tall) and colour.
  - **Storage:** the labels are stored in `scene["labels"]` (text, x, y, size). The renderer draws them from there, `answers.py` converts pixels to ruler units through them, and `verify_frames.py` reads them back off the frames.
- **Timer:** fixed in a dark box in the top-left corner, away from the action. It shows `floor(t*10)/10` s, at 36 px.
- **Naming in questions:** robots are named by colour ("the teal robot"). Positions are given in ruler units ("0.7").

## The 10-question ladder

- **Counts:** per-level counts come from `L5.ten_counts`: L1 1-3, L2 2-3, L3 2-3, L4 1-3, L5 1-3, 10 in total, ordered q01..q10 by level.
- **Selection:** the template is drawn first (motion templates weighted 4:1), then its params. Every template is used once before any template repeats with new params, and no `derivation_key` + params pair repeats within a clip.
- **Options:** every question has 4 options with exactly one correct. Pair questions add "they ... the same" / "neither of them ..." options, which are never correct.
- **Rule text:** questions about scores state the rule in the question.

| level | templates (`derivation_key`) |
|---|---|
| L1 observation | `hops_jump` (hops before the long jump, jumpers only), `hops_foul` (hops including the one past the foul line, foulers only), `n_fouls`, `shape`, `who_fouls` |
| L2 relational | `farther_landing`, `farther_takeoff`, `longer_jump`, `first_land` (pair), `hop_diff`, `top_scorer`, `first_to_jump`, `last_to_land` |
| L3 measurement | `takeoff`, `jump_length`, `longest_hop`, `start_mark` (staggered), `land_time`, `takeoff_time` (timer readings), `landed_by` (robots landed when the clock first reads t) |
| L4 integration | `score`, `place`, `win_margin`, `ranking` (3 robots, at most one foul), `leader_after`, `lead_changes`, `landed_before` |
| L5 counterfactual | `cf_jump_earlier`, `cf_threshold` (jump as soon as at p or beyond), `cf_foul_line` (line at 0.6-0.9), `cf_swap` (jump as far as another robot), `cf_longer_place`, `cf_lead_changes` (a jumper had fouled), `cf_leader_after`, `cf_wait_beat` (first lander waits one beat) |

- **Counterfactuals:** every counterfactual is re-simulated, never guessed. `sim.run` replays the hops under the new foul line, `sim.leaders` / `sim.landing_frames` redo the landing order, and `answers.py` re-runs the rule on the drawn footprints.
- **Motion share:** over 5000 clips, 54% of L2-L5 questions have `requires_motion=True`. These are order, timing and leader questions.
- **Static answers:** positions and counts persist as footprints, dents and the robots' final poses, so those questions are honestly marked static.

### The two fixes from review

1. **Ranking ties:**
   - **Old bug:** the generator only kept 1st and 2nd at least 0.06 apart, so in 6 of 191 ranking questions 2nd and 3rd tied exactly (seed 81: two robots on the same spot). Another 26 were under 0.06 apart.
   - **Fix:** `sim.check` now rejects any scene where two jumping robots' scores are less than 0.1 apart. That is one full gridline (52+ px), and it applies to every pair.
   - **Test:** covered in `test_robot_long_jump.py`, including a hand-made tie that must be rejected.
2. **Step-count wording:**
   - **Old bug:** "How many forward steps does X take before jumping or fouling?" was also asked of robots that fouled, and it was unclear whether the step over the line counted.
   - **Fix:** it is now two templates:
     - `hops_jump`: "How many short hops does X make before its long jump into the sand (not counting the jump itself)?" (jumpers only).
     - `hops_foul`: "How many hops does X make in total, counting the hop that carries it past the red foul line?" (foulers only).

### min_fps

Every question stores `params["min_fps"]`, the sampling rate needed so that at least one sampled frame falls in each window the answer depends on.

- **Static answers** (footprints, dents, final poses): `1 / (time from the end of the relevant attempts to the end of the clip)`.
- **Order and leader questions:** `1 / (smallest gap between the landings compared)`. These gaps are at least 8 frames (0.33 s).
- **Timer readings:** `2 / 0.7 s`. Wrong options are 0.7 s apart.

Over 5000 clips, 35% of questions are answerable at 1 fps, 79% at 2 fps and 100% at 4 fps (max 3.0).

## Frame rules

- **Duration and edges:** at most 240 frames (10 s). The first hop starts at least 12 frames (0.5 s) in, and the last landing or topple ends at least 12 frames before the end.
- **Timer alignment:** in `turns` / `staggered`, every sand landing is on a timer-aligned frame (exact time within 0.045 s above a tenth). `land_time` / `takeoff_time` questions are asked only on aligned frames. For take-offs, the frame after must show the same reading as well.
- **Event spacing:** any two landings compared in time are at least 8 frames apart, and so is a take-off from any landing it is compared with.
- **Persistent results:** footprints, dents and toppled robots stay, so there are no single-frame flashes.
- **No overlaps:** jump arcs are capped by lane pitch, and a toppled robot comes to rest lying on top of its strip. So no two robots ever overlap, not even a jumper passing under a toppled robot in the lane above (checked on every frame of 1000 scenes and in the tests). No robot ever touches the timer box, a label or the ruler band.
- **Minimum sizes:** robots are at least 44 px, footprints 22 px wide, and lines at least 3 px.

## Closed loop

- **Stored derivation:** every question stores `derivation_key` + `params`.
- **Independent answers:** `answers.py` never calls `sim.py` and never reads the generator's per-robot facts (threshold, hops, jump, foul flag, frames). It replays `render.layout` (the geometry `render.frame` draws) for every frame and reads:
  - footprints as they appear, converted to ruler units through the ruler labels (each must sit on a gridline);
  - the start spot from frame 0;
  - the take-off frame (first frame in the jump);
  - the landing frame and dent;
  - toppled robots;
  - timer strings.

  Counterfactuals re-run the rule on those footprints.
- **Generator checks:** `sim.check` rejects a scene if any of these hold:
  - hops don't follow the rule;
  - a footprint is on the foul line;
  - a landing is outside the pit;
  - two scores are under 0.1 apart;
  - fewer than two robots jump;
  - an event is within 0.5 s of either end;
  - a landing in a turn-based scenario isn't timer-aligned;
  - attempts overlap in turn-based scenarios;
  - the clip is over 10 s.

  Scenes are redrawn until they pass: 0.95 retries per scene on average; 46% of seeds needed at least one.
- **Validator:** `python run.py validate` rebuilds every clip from its seed and diffs scene and questions. It also runs `sim.check` and the 10-question ladder check, which includes `min_fps`, re-derives every answer independently and, without `--skip-video`, checks the MP4.
- **Frame check:** `python verify_frames.py --dataset-dir <dir>` decodes the clips and reads the answers back off the pixels:
  - **Ruler labels:** template-matched against "0.0".."2.0", at most 3 px from their scene position and at least 24 px tall. They give the pixel-to-ruler mapping.
  - **Robots:** tracked by body colour, giving landing frames, take-off frames and toppled robots.
  - **Footprints:** counted and located in the last frame.
  - **Timer:** template-matched at every take-off and landing frame.
  - **Answers:** hops, fouls, take-off, jump length, score, longest hop, start mark, land/take-off times and first-to-land are compared with the stored answers.
  - **Overlaps:** no robot overlaps a label, the timer box or the ruler band in any frame.

## Determinism

- **Randomness:** all of it comes from the `random.Random(seed)` passed in. The code uses no global RNG, time, uuid or set/dict-order decisions. The scene is plain JSON with integer positions.
- **Same JSON:** a sample is byte-identical across processes with different `PYTHONHASHSEED`, and regenerating into a fresh folder reproduces `annotations.jsonl` byte for byte (render metadata aside, since `render_time_s` is a wall-clock time).
- **Same frames:** rendering the same seed twice gives identical raw frames (hash of the RGB bytes).
- **Tests:** `test_robot_long_jump.py` covers determinism, the ladder, uniqueness, the closed loop, labels vs renderer, a tamper test and hand-checked edge cases.

## Files and commands

```
sim.py            scene generation, the take-off / foul rule, checks (pure, seeded)
questions.py      the 10-question ladder
answers.py        independent answer re-derivation from the renderer's geometry
render.py         PIL renderer + per-frame layout (1280x720, 24 fps, piped to ffmpeg)
run.py            CLI: generate / validate (shared pipeline in common/video5.py)
verify_frames.py  decoded-frame checks of labels, timer, robots, footprints and answers
test_robot_long_jump.py

python run.py generate --n 20 --seed 0 --output-dir ../output_videos/robot_long_jump
python run.py validate --dataset-dir ../output_videos/robot_long_jump
python verify_frames.py --dataset-dir ../output_videos/robot_long_jump
python -m pytest -q test_robot_long_jump.py
```
