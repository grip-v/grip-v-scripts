"""die_roll_trace generator: params -> annotations.jsonl (+ raster render).

Usage:
    python generate.py --n 20 --output-dir . --seed 0
    python generate.py --n 5 --sample --output-dir .   # 5 dev examples
    python generate.py --n 3000 --no-render             # dry run
    (CHANGE: --workers is new)
    python generate.py --n 3000 --workers 8             # parallel render (default: all cores)
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp   # CHANGE: new import, for --workers
import os                      # CHANGE: new import, for --workers
import random
from pathlib import Path

import config as C
import puzzle as P
import questions as Q
import render as R

HERE = Path(__file__).resolve().parent


def build_one(seed: int) -> dict:
    rng = random.Random(seed)
    scene = P.build_scene(rng)
    qs = Q.build_questions(scene, rng)
    fps = rng.randrange(C.FPS_RANGE[0], C.FPS_RANGE[1] + 1)
    step_duration_s = round(rng.uniform(*C.STEP_DURATION_S_RANGE), 3)

    n_steps = scene.total_rolls
    step_frames = max(2, round(step_duration_s * fps))
    move_frames = step_frames * max(n_steps, 1)
    hold_frames = max(1, round(move_frames * C.FINAL_HOLD_FRAC / (1 - C.FINAL_HOLD_FRAC)))
    total_frames = move_frames + hold_frames

    params = {
        "seed": seed, "N": scene.N, "canvas_px": scene.canvas_px,
        "grid_values": scene.grid_values, "grid_color_idx": scene.grid_color_idx,
        "die_net_layout": scene.die_net_layout, "die_face_adjacency": scene.die_face_adjacency,
        "initial_orientation": scene.initial_orientation,
        "start_cell": scene.start_cell, "path_cells": scene.path_cells,
        "true_roll_sequence": scene.true_roll_sequence,
        "displayed_roll_sequence": scene.displayed_roll_sequence,
        "content_mode": scene.content_mode, "broken_step_index": scene.broken_step_index,
        "correct_orientation_at_break": scene.correct_orientation_at_break,
        "displayed_wrong_orientation_at_break": scene.displayed_wrong_orientation_at_break,
        "total_rolls": scene.total_rolls, "total_cells_visited": scene.total_cells_visited,
        "unique_cells_visited": scene.unique_cells_visited,
        "top_face_value_history": scene.top_face_value_history,
        "fps": fps, "step_duration_s": step_duration_s,
        "duration_seconds": round(total_frames / fps, 3), "frame_count": total_frames,
    }
    return {
        "sample_id": f"die_roll_trace_{seed:04d}", "dataset": "die_roll_trace",
        "modality": "video", "seed": seed, "params": params, "questions": qs,
    }


def render_and_finish(sample: dict, out_dir: Path):
    seed = sample["seed"]
    rng = random.Random(seed)
    scene = P.build_scene(rng)   # rebuild in-process, deterministic
    Q.build_questions(scene, rng)   # consumed here too, keeps rng in lockstep with build_one
    p = sample["params"]

    video_path = out_dir / "videos" / f"{sample['sample_id']}.mp4"
    render_meta = R.render_clip(scene, p["fps"], p["step_duration_s"], video_path)
    p["duration_seconds"] = render_meta["duration_seconds"]
    p["frame_count"] = render_meta["frame_count"]

    frame_path = out_dir / "frames" / f"{sample['sample_id']}_final.png"
    R.extract_final_frame(scene, frame_path)

    sample["video_path"] = f"videos/{sample['sample_id']}.mp4"
    sample["final_frame_path"] = f"frames/{sample['sample_id']}_final.png"


# CHANGE: new function, sets up each parallel render process.
def _worker_init():
    import cv2
    cv2.setNumThreads(1)   # one process per core already; avoid oversubscription


# CHANGE: new function, one unit of parallel work.
def _build_job(job):
    """Top-level (picklable) unit of work: one seed -> finished sample dict.
    Each seed is fully independent and deterministic, so running these in
    any process/order yields byte-identical samples to the serial path."""
    seed, out_dir, render = job
    sample = build_one(seed)
    if render:
        render_and_finish(sample, Path(out_dir))
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--output-dir", default=str(HERE.parent / "output_videos" / "die_roll_trace"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--append", action="store_true")
    # CHANGE: new option, render several videos at once.
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="parallel render processes (1 = serial)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    n, seeds = (5, range(args.seed, args.seed + 5)) if args.sample else (args.n, range(args.seed, args.seed + args.n))

    ann_path = out_dir / "annotations.jsonl"
    mode = "a" if (args.append and ann_path.exists()) else "w"
    # CHANGE: the loop below replaces the old one-video-at-a-time loop. It runs
    # the same build_one() + render_and_finish() per seed, in parallel, and
    # writes samples in the same seed order as before.
    jobs = [(seed, str(out_dir), not args.no_render) for seed in seeds]
    workers = max(1, min(args.workers, len(jobs)))
    pool = mp.Pool(workers, initializer=_worker_init) if workers > 1 else None
    try:
        results = pool.imap(_build_job, jobs) if pool else map(_build_job, jobs)
        with open(ann_path, mode, encoding="utf8") as f:
            for i, sample in enumerate(results):   # imap keeps seed order
                p = sample["params"]
                print(f"[{i + 1}/{n}] {sample['sample_id']} (N={p['N']}, rolls={p['total_rolls']}, "
                      f"mode={p['content_mode']}, break={p['broken_step_index']})", flush=True)
                f.write(json.dumps(sample) + "\n")
    finally:
        if pool:
            pool.close()
            pool.join()

    print(f"done: {n} samples written -> {ann_path}")


if __name__ == "__main__":
    main()
