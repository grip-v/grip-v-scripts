"""hooks_grid_trace generator: params -> annotations.jsonl (+ manim render).

Usage:
    python generate.py --n 50 --output-dir . --seed 0 --quality low
    python generate.py --n 5 --sample --output-dir .   # 5 dev examples, rendered at -qh
    python generate.py --n 3000 --no-render             # dry run
"""
from __future__ import annotations
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import config as C
import hooks as H
import questions as Q

HERE = Path(__file__).resolve().parent


def _running_sums(scene: H.SceneData):
    """{hook_index: {row_or_col_index: cumulative_sum_so_far}} -- only for
    rows/cols that hook actually touches, computed strictly in fill_order
    so it matches what the animation has actually shown by that point."""
    n = scene.grid_size
    cum_row = [0] * n
    cum_col = [0] * n
    row_running, col_running = {}, {}
    for k in scene.fill_order:
        cells = scene.hooks[k]
        for (r, c) in cells:
            val = scene.final_grid[r][c]
            cum_row[r] += val
            cum_col[c] += val
        touched_rows = sorted({r for r, c in cells})
        touched_cols = sorted({c for r, c in cells})
        row_running[k] = {r: cum_row[r] for r in touched_rows}
        col_running[k] = {c: cum_col[c] for c in touched_cols}
    return row_running, col_running


def _timing(rng: random.Random, grid_size: int, running_sums_displayed: bool):
    """Solves for (step_duration_s, total_duration_s) such that scene.py's
    ACTUAL elapsed time -- hook fills + the running-sums overlay's fixed
    0.5s/hook overhead (if shown) -- plus a hold matching FINAL_HOLD_FRAC
    of the total, lands exactly on the returned total_duration_s. Without
    accounting for the overlay overhead here, the real render runs longer
    than the stored duration/frame_count -- caught by the validator's
    frame-count check the first time this was written."""
    target_total = rng.uniform(*C.DURATION_S_RANGE)
    overlay_overhead = grid_size * 0.5 if running_sums_displayed else 0.0
    min_step = 0.18

    desired_move_budget = target_total * (1 - C.FINAL_HOLD_FRAC) - overlay_overhead
    step_s = max(min_step, desired_move_budget / grid_size)
    elapsed_before_hold = step_s * grid_size + overlay_overhead
    total = elapsed_before_hold / (1 - C.FINAL_HOLD_FRAC)
    return round(step_s, 4), round(total, 3)


def build_one(seed: int) -> dict:
    rng = random.Random(seed)
    scene = H.build_scene(rng)
    questions = Q.build_questions(scene, rng)
    step_s, total_duration = _timing(random.Random(seed + 1), scene.grid_size, scene.running_sums_displayed)

    hooks_out = [{"hook_index": k, "digit_value": k, "cell_positions": [list(p) for p in scene.hooks[k]],
                  "fill_step_order": scene.fill_order.index(k)} for k in range(1, scene.grid_size + 1)]

    params = {
        "seed": seed, "grid_size": scene.grid_size, "fill_direction": scene.fill_direction,
        "running_sums_displayed": scene.running_sums_displayed, "hooks": hooks_out,
        "fill_order": scene.fill_order, "row_targets": scene.row_targets, "column_targets": scene.column_targets,
        "shaded_cells": [list(c) for c in scene.shaded_cells], "shaded_cells_sum": scene.shaded_cells_sum,
        "is_broken": scene.is_broken,
        "error_cell_position": list(scene.error_cell_position) if scene.error_cell_position else None,
        "error_hook_index": scene.error_hook_index, "correct_digit": scene.correct_digit,
        "placed_digit": scene.placed_digit, "affected_row_index": scene.affected_row_index,
        "affected_column_index": scene.affected_column_index, "error_fill_step": scene.error_fill_step,
        "final_grid": scene.final_grid, "final_row_sums": scene.final_row_sums,
        "final_column_sums": scene.final_column_sums, "row_sum_matches": scene.row_sum_matches,
        "column_sum_matches": scene.column_sum_matches,
        "duration_seconds": total_duration, "fps": C.FPS,
        "frame_count": int(round(total_duration * C.FPS)),
    }
    return {
        "sample_id": f"hooks_grid_{seed:04d}", "dataset": "hooks_grid_trace",
        "modality": "video", "seed": seed, "params": params, "questions": questions,
    }


def render_clip(seed: int, out_path: Path, quality: str) -> dict:
    rng = random.Random(seed)
    scene = H.build_scene(rng)   # rebuild in-process, deterministic
    Q.build_questions(scene, rng)   # consumed here too, keeps rng in lockstep with build_one
    step_s, total_duration = _timing(random.Random(seed + 1), scene.grid_size, scene.running_sums_displayed)

    row_running, col_running = _running_sums(scene)
    n = scene.grid_size
    cell_size = C.cell_size_for(n)

    render_params = {
        "background_hex": C.BACKGROUND_HEX, "grid_line_hex": C.GRID_LINE_HEX, "hook_line_hex": C.HOOK_LINE_HEX,
        "target_hex": C.TARGET_TEXT_HEX, "digit_hex": C.DIGIT_TEXT_HEX, "shade_hex": C.SHADE_TINT_HEX,
        "grid_size": n, "cell_size": cell_size,
        "hooks": {str(k): [list(p) for p in scene.hooks[k]] for k in range(1, n + 1)},
        "fill_order": scene.fill_order, "final_grid": scene.final_grid,
        "row_targets": scene.row_targets, "column_targets": scene.column_targets,
        "shaded_cells": [list(c) for c in scene.shaded_cells],
        "error_cell_position": list(scene.error_cell_position) if scene.error_cell_position else None,
        "running_sums_displayed": scene.running_sums_displayed,
        "row_running_sums": {str(k): {str(r): v for r, v in d.items()} for k, d in row_running.items()},
        "col_running_sums": {str(k): {str(c): v for c, v in d.items()} for k, d in col_running.items()},
        "final_row_sums": scene.final_row_sums, "final_column_sums": scene.final_column_sums,
        "row_sum_matches": scene.row_sum_matches, "column_sum_matches": scene.column_sum_matches,
        "step_duration_s": step_s, "total_duration_s": total_duration,
    }
    params_path = out_path.with_suffix(".params.json")
    params_path.write_text(json.dumps(render_params))

    env = dict(os.environ)
    env["HGT_PARAMS_FILE"] = str(params_path)

    qflag = {"low": "-ql", "medium": "-qm", "high": "-qh"}[quality]
    with tempfile.TemporaryDirectory(prefix="manim_media_") as media_dir:
        t0 = time.time()
        cmd = [sys.executable, "-m", "manim", qflag, "--fps", str(C.FPS),
               "--media_dir", media_dir, "--output_file", out_path.stem,
               str(HERE / "scene.py"), "HooksGridScene"]
        subprocess.run(cmd, cwd=str(HERE), env=env, check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        render_time = time.time() - t0
        rendered = list(Path(media_dir).rglob(f"{out_path.stem}.mp4"))
        if not rendered:
            raise RuntimeError(f"manim did not produce {out_path.stem}.mp4")
        # shutil.move (not Path.replace/os.rename) -- manim's media_dir may be on a
        # different filesystem than out_path (e.g. local /tmp vs. a netscratch mount),
        # and a same-filesystem-only rename raises OSError: Invalid cross-device link.
        shutil.move(str(rendered[0]), str(out_path))
    params_path.unlink(missing_ok=True)
    return {"renderer": "manim", "render_time_s": round(render_time, 2), "bytes": out_path.stat().st_size}


def _extract_final_frame(video_path: Path, out_png: Path):
    cmd = ["ffmpeg", "-y", "-sseof", "-1", "-i", str(video_path), "-update", "1", "-q:v", "2",
           str(out_png), "-loglevel", "error"]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--output-dir", default=str(HERE.parent / "output_videos" / "hooks_grid_trace"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--quality", default="medium", choices=["low", "medium", "high"],
                     help="manim quality preset; default 'medium' renders at 1280x720 30fps")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    if args.sample:
        n, seeds, quality = 5, range(args.seed, args.seed + 5), "high"
    else:
        n, seeds, quality = args.n, range(args.seed, args.seed + args.n), args.quality

    ann_path = out_dir / "annotations.jsonl"
    mode = "a" if (args.append and ann_path.exists()) else "w"
    with open(ann_path, mode, encoding="utf8") as f:
        for i, seed in enumerate(seeds):
            sample = build_one(seed)
            p = sample["params"]
            rel_video = f"videos/{sample['sample_id']}.mp4"
            print(f"[{i + 1}/{n}] {sample['sample_id']} (grid={p['grid_size']}, dir={p['fill_direction']}, "
                  f"broken={p['is_broken']}, running_sums={p['running_sums_displayed']})")

            if not args.no_render:
                video_path = out_dir / rel_video
                render_meta = render_clip(seed, video_path, quality)
                sample["render_meta"] = render_meta
                frame_path = out_dir / "frames" / f"{sample['sample_id']}_final.png"
                _extract_final_frame(video_path, frame_path)
                sample["video_path"] = rel_video
                sample["final_frame_path"] = f"frames/{sample['sample_id']}_final.png"
                time.sleep(0.3)

            f.write(json.dumps(sample) + "\n")

    print(f"done: {n} samples written -> {ann_path}")


if __name__ == "__main__":
    main()
