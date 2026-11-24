"""laser_mirror_trace generator: params -> annotations.jsonl (+ raster render).

Usage:
    python generate.py --n 20 --output-dir . --seed 0
    python generate.py --n 5 --sample --output-dir .   # 5 dev examples
    python generate.py --n 3000 --no-render             # dry run
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import config as C
import puzzle as P
import questions as Q
import render as R

HERE = Path(__file__).resolve().parent


def _mirror_layout_out(scene: P.SceneData):
    return {f"{r},{c}": o for (r, c), o in scene.mirror_layout.items()}


def build_one(seed: int) -> dict:
    rng = random.Random(seed)
    scene = P.build_scene(rng)
    qs = Q.build_questions(scene, rng)
    canvas_px = rng.randrange(C.CANVAS_PX_RANGE[0], C.CANVAS_PX_RANGE[1] + 1)
    fps = rng.randrange(C.FPS_RANGE[0], C.FPS_RANGE[1] + 1)
    step_duration_s = round(rng.uniform(*C.STEP_DURATION_S_RANGE), 3)

    n_steps = len(scene.displayed_path)
    step_frames = max(1, round(step_duration_s * fps))
    move_frames = step_frames * n_steps
    hold_frames = max(1, round(move_frames * C.FINAL_HOLD_FRAC / (1 - C.FINAL_HOLD_FRAC)))
    total_frames = move_frames + hold_frames

    params = {
        "seed": seed, "grid_size": scene.grid_size, "mirror_layout": _mirror_layout_out(scene),
        "goal_positions": scene.goal_positions, "entry_position": scene.entry_position,
        "entry_direction": scene.entry_direction, "entry_edge": scene.entry_edge,
        "true_path": scene.true_path, "displayed_path": scene.displayed_path,
        "content_mode": scene.content_mode, "outcome_type": scene.outcome_type,
        "reached_goal_position": scene.reached_goal_position,
        "total_cells_traversed": scene.total_cells_traversed, "total_reflections": scene.total_reflections,
        "direction_change_count": scene.total_reflections,
        "broken_reflection_index": scene.broken_reflection_index,
        "broken_correct_direction": scene.broken_correct_direction,
        "broken_displayed_direction": scene.broken_displayed_direction,
        "loop_cycle_length": scene.loop_cycle_length,
        # the true POST-reflection direction on the exit step, since each displayed_path
        # entry's own "direction_of_travel" is the INCOMING direction -- when the beam
        # reflects on its very last cell before exiting, those differ, and no path entry
        # alone records the actual exit direction
        "exit_direction": scene.displayed_result.get("exit_direction"),
        "true_exit_direction": scene.true_result.get("exit_direction"),
        "canvas_px": canvas_px, "fps": fps, "step_duration_s": step_duration_s,
        "duration_seconds": round(total_frames / fps, 3), "frame_count": total_frames,
    }
    return {
        "sample_id": f"laser_trace_{seed:04d}", "dataset": "laser_mirror_trace",
        "modality": "video", "seed": seed, "params": params, "questions": qs,
    }


def render_and_finish(sample: dict, out_dir: Path):
    seed = sample["seed"]
    rng = random.Random(seed)
    scene = P.build_scene(rng)   # rebuild in-process, deterministic
    Q.build_questions(scene, rng)   # consumed here too, keeps rng in lockstep with build_one
    p = sample["params"]

    video_path = out_dir / "videos" / f"{sample['sample_id']}.mp4"
    render_meta = R.render_clip(scene, p["canvas_px"], p["fps"], p["step_duration_s"], video_path)
    p["duration_seconds"] = render_meta["duration_seconds"]
    p["frame_count"] = render_meta["frame_count"]

    frame_path = out_dir / "frames" / f"{sample['sample_id']}_final.png"
    R.extract_final_frame(scene, p["canvas_px"], frame_path)

    sample["video_path"] = f"videos/{sample['sample_id']}.mp4"
    sample["final_frame_path"] = f"frames/{sample['sample_id']}_final.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--output-dir", default=str(HERE.parent / "output_videos" / "laser_mirror_trace"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    n, seeds = (5, range(args.seed, args.seed + 5)) if args.sample else (args.n, range(args.seed, args.seed + args.n))

    ann_path = out_dir / "annotations.jsonl"
    mode = "a" if (args.append and ann_path.exists()) else "w"
    with open(ann_path, mode, encoding="utf8") as f:
        for i, seed in enumerate(seeds):
            sample = build_one(seed)
            p = sample["params"]
            print(f"[{i + 1}/{n}] {sample['sample_id']} (grid={p['grid_size']}, mode={p['content_mode']}, "
                  f"outcome={p['outcome_type']}, cells={p['total_cells_traversed']})")
            if not args.no_render:
                render_and_finish(sample, out_dir)
            f.write(json.dumps(sample) + "\n")

    print(f"done: {n} samples written -> {ann_path}")


if __name__ == "__main__":
    main()
