# GRIP-V generators

Generation and validation code for **GRIP-V**, from *Watch, Don't Glance: A Procedurally Generated Video Benchmark for Temporal Geometric and Physical Reasoning*.

[Project page](https://grip-v.github.io/) · [Dataset on Hugging Face](https://huggingface.co/datasets/hamzahanif/GRIP-V) · [GRIP (image benchmark)](https://arxiv.org/abs/2610.16170)

GRIP-V is the video version of GRIP. It has 40 sub-benchmarks in seven reasoning families, with 2,500 clips per sub-benchmark and ten multiple-choice questions per clip: 100,000 clips and one million questions. Every sub-benchmark here is a seeded generator. It simulates a scene, renders the video from the simulated state, and computes each answer from the same state. A separate validator rebuilds every sample and re-derives every answer.

## How a sub-benchmark works

- **Simulate.** `sim.py` samples one of the sub-benchmark's scenario variants from the seed and computes the scene state over time: positions, collisions, flows, flashes, moves.
- **Render.** `render.py` draws every frame from that state (1280×720, 24 fps, at most 10 s) and encodes it as H.264 with `ffmpeg`. The only on-screen text is a clock and, where a quantity must be read, scale labels.
- **Ask.** `questions.py` writes ten questions per clip on a five-rung ladder: L1 observation, L2 relations, L3 measurement, L4 multi-step integration, L5 counterfactual. Each question stores the parameters used to derive its answer, and a `requires_motion` flag marks questions that cannot be answered from any single frame.
- **Validate.** `answers.py` re-derives every answer from the stored scene through a separate code path. `run.py validate` compares the result with the stored answers and checks the ladder, the simulation invariants, and the video files.

Shared code lives in [`common/`](common/): the generate/validate pipeline (`video5.py`), the question ladder (`ladder5.py`), the annotation schema, and shape and font helpers. [`SHARED_CONTRACT.md`](SHARED_CONTRACT.md) lists the rules every generator follows.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ffmpeg` must be on your `PATH`. The code is tested with Python 3.13.

## Usage

Every sub-benchmark folder has a `run.py` with two commands:

```bash
# Render 20 clips (seeds 0..19) and write annotations.jsonl
python tangled_braid/run.py generate --n 20 --seed 0 --output-dir out/tangled_braid

# Re-derive every answer and check the clips
python tangled_braid/run.py validate --dataset-dir out/tangled_braid
```

`generate --no-render` writes the questions without rendering video, and `validate --skip-video` checks only the annotations. Seeds are deterministic: clip *k* of the release is seed *k*.

Most sub-benchmarks also have a `frame_check.py`, which decodes the rendered clips and compares the objects, clock, and labels in the frames with the scene state:

```bash
python chess_dance_trace/frame_check.py --dataset-dir out/chess_dance_trace
```

Each clip has one entry in `annotations.jsonl` with its seed, scenario parameters, video path, and ten questions. Each question has its options, answer letter, answer text, rung (`difficulty_level`), and `requires_motion` flag.

## The 40 sub-benchmarks

| Family | Sub-benchmarks |
|---|---|
| Mechanics & motion | [bouncing_ball_restitution](bouncing_ball_restitution/), [shape_collision](shape_collision/), [gear_train](gear_train/), [simple_harmonic_motion](simple_harmonic_motion/), [spring_chain_waves](spring_chain_waves/), [balance_scale_weighing](balance_scale_weighing/), [rolling_races](rolling_races/), [inertia_race](inertia_race/), [stacking_collapse](stacking_collapse/) |
| Fluids & optics | [buoyancy_tank](buoyancy_tank/), [vessels_and_clocks](vessels_and_clocks/), [glasses_fill](glasses_fill/), [well_flood_trace](well_flood_trace/), [mirror_beam_reflection](mirror_beam_reflection/), [hidden_mirrors](hidden_mirrors/) |
| Spatial & 3D geometry | [spinning_polyhedra](spinning_polyhedra/), [turntable_voxels](turntable_voxels/), [moving_shadows](moving_shadows/), [symmetry_pattern_video](symmetry_pattern_video/) |
| Occlusion & permanence | [occluded_pattern_counting](occluded_pattern_counting/), [translucent_occlusion](translucent_occlusion/), [overlapping_shapes](overlapping_shapes/), [shell_game_tracking](shell_game_tracking/), [object_tracking_mot_v2](object_tracking_mot_v2/), [rainbow_bagel](rainbow_bagel/) |
| Temporal events | [temporal_flash_counting](temporal_flash_counting/), [nested_shapes](nested_shapes/), [blinking_networks](blinking_networks/), [sandpile_cascade](sandpile_cascade/), [cellular_automata](cellular_automata/) |
| Agents & navigation | [multi_agent_chase](multi_agent_chase/), [colored_route_puzzle](colored_route_puzzle/), [traffic_light_queue](traffic_light_queue/), [swing_time](swing_time/), [robot_long_jump](robot_long_jump/) |
| Rules & algorithms | [chess_dance_trace](chess_dance_trace/), [tangled_braid](tangled_braid/), [tile_alignment_shuffle](tile_alignment_shuffle/), [conveyor_belt_sort](conveyor_belt_sort/), [robot_archery](robot_archery/) |

Three sub-benchmarks inject rule violations into some clips: `well_flood_trace`, `symmetry_pattern_video`, and `chess_dance_trace`. Six adapt published Jane Street puzzles: `tangled_braid`, `hidden_mirrors`, `rainbow_bagel`, `swing_time`, `robot_long_jump`, and `robot_archery`. Each folder's README describes its scenario variants and explains why its questions need the video.

### Other generators

The remaining folders are generators that are not part of the GRIP-V release: earlier versions (`mirror_beam_reflection_v2`, `shell_game_tracking_v2`, `object_tracking_mot`), prototypes (`cipher_card_trace`, `domino_chains`, `fold_punch_unfold`, `motion_parallax`, `moving_points`, `minesweeper_reveal_trace`, `phantom_traffic_jams`, `robot_capture_flag`, `robot_road_trip`, `robot_swimming_trials`, `rotating_rays`, `theory_of_mind_belief`), and generators with their own renderers (`die_roll_trace`, `hooks_grid_trace`, `laser_mirror_trace`, `random_walk_home`). Some of these need [Manim](https://www.manim.community/). `concept_demos/` and `tools/` hold demos and maintenance scripts.

## Citation

```bibtex
@article{hanif2026gripv,
  title  = {Watch, Don't Glance: A Procedurally Generated Video Benchmark
            for Temporal Geometric and Physical Reasoning},
  author = {Hanif, Hamza and Khan, Muhammad Bilal and Rehman, Inam ur and
            Muneer, Amgad and Jawaid, Bilal and Uddin, Muhammad Maaz and
            Zafar, Anas and Abbas, Shahzaib and Qamar, Ahmad Abdullah and
            Zafar, Aizan and Chadha, Aman and Yan, Hong and Qureshi, Rizwan},
  year   = {2026}
}
```
