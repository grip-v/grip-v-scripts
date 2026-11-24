"""random_walk_home generator: params -> annotations.jsonl (+ manim render).

Usage:
    python generate.py --n 50 --output-dir . --seed 0 --quality low
    python generate.py --n 5 --sample --output-dir .   # 5 dev examples, rendered at -qh
    python generate.py --n 500 --no-render               # dry run
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

import arena as A
import config as C
import questions as Q
import simulate as S

HERE = Path(__file__).resolve().parent


def _tick_schedule(total_ticks: int, rng: random.Random):
    nominal = rng.uniform(*C.TICK_DURATION_RANGE)
    hold = C.MAX_VIDEO_DURATION_S * C.FINAL_HOLD_FRAC
    move_budget_capped = C.MAX_VIDEO_DURATION_S - hold - C.HOME_FLASH_DURATION_S

    # CHANGE: one steady pace for the whole clip, chosen so the clip lasts MIN..MAX_VIDEO_DURATION_S (10-15 s). A walk
    # too long for the nominal pace plays every move evenly faster (>= ~0.18 s at MAX_TICKS); a walk too short plays
    # every move evenly slower (<= ~0.78 s at MIN_TICKS). Before: a ~6x speed-up after 9 s, and clips as short as 3 s.
    move_budget_min = C.MIN_VIDEO_DURATION_S - hold - C.HOME_FLASH_DURATION_S + 0.001   # +1 ms: rounding margin
    sped_up = total_ticks * nominal > move_budget_capped
    pace = nominal
    if sped_up:
        pace = move_budget_capped / total_ticks
    elif total_ticks * nominal < move_budget_min:
        pace = move_budget_min / total_ticks
    durs = [pace] * total_ticks

    cum, running = [], 0.0
    for d in durs:
        running += d
        cum.append(round(running, 4))
    move_duration = cum[-1] if cum else 0.0
    total_duration = round(move_duration + C.HOME_FLASH_DURATION_S + hold, 4)
    return cum, move_duration, total_duration, sped_up


def build_one(seed: int) -> dict:
    scenario = S.simulate_scenario(seed)
    aux_rng = random.Random(seed)
    questions = Q.build_questions(scenario, aux_rng)
    cum_times, move_duration, total_duration, sped_up = _tick_schedule(scenario.total_ticks, aux_rng)

    n_home = sum(1 for a in scenario.ants if a.outcome == "home")
    params = {
        "seed": seed, "n_ants": scenario.n_ants,
        "ants": [{"symbol": a.symbol, "start": list(a.start), "home_initial": list(a.home_initial),
                  "home_final": list(a.home_final), "outcome": a.outcome, "steps_taken": a.steps_taken,
                  "reset_count": a.reset_count, "died_tick": a.died_tick, "died_by_predator": a.died_by_predator}
                 for a in scenario.ants],
        "n_ants_home": n_home, "n_ants_dead": scenario.n_ants - n_home,
        "n_predators": scenario.n_predators, "n_holes": scenario.n_holes,
        # CHANGE: wind
        "wind_strength": scenario.wind_strength,
        "wind_dir_start": None if scenario.wind_history[0] is None else C.WIND_DIR_NAMES[scenario.wind_history[0]],
        "wind_dir_end": None if scenario.wind_history[-1] is None else C.WIND_DIR_NAMES[scenario.wind_history[-1]],
        "wind_shift_tick": scenario.wind_shift_tick,
        "gust_ticks": [k for k, g in enumerate(scenario.gust_history) if g],
        "n_gusts": sum(1 for k in range(1, len(scenario.gust_history))
                       if scenario.gust_history[k] and not scenario.gust_history[k - 1]),
        "home_moved": scenario.home_moved, "home_moved_ant_symbol": scenario.home_moved_ant_symbol,
        "tile_swap_count": scenario.tile_swap_count, "hole_reset_count": scenario.hole_reset_count,
        "death_count": scenario.death_count, "total_events": scenario.total_events,
        "total_ticks": scenario.total_ticks, "playback_sped_up": sped_up,
        "duration_seconds": total_duration, "fps": C.FPS,
        "frame_count": int(round(total_duration * C.FPS)),
    }
    return {
        "sample_id": f"random_walk_home_{seed:04d}", "dataset": "random_walk_home",
        "modality": "video", "seed": seed, "params": params, "questions": questions,
    }


def render_clip(seed: int, out_path: Path, quality: str) -> dict:
    scenario = S.simulate_scenario(seed)   # rebuild in-process, deterministic
    aux_rng = random.Random(seed)
    Q.build_questions(scenario, aux_rng)   # consumed here too, keeps aux_rng in lockstep with build_one
    cum_times, move_duration, total_duration, sped_up = _tick_schedule(scenario.total_ticks, aux_rng)

    cells = [{"q": q, "r": r, "pos": list(A.axial_to_pixel(q, r, C.CELL_SIZE)), "size": C.CELL_SIZE}
             for q, r in scenario.board]
    blocked_by_tick = [[bh[(c["q"], c["r"])] for c in cells] for bh in scenario.blocked_history]
    holes_set = set(scenario.holes)
    is_hole = [(c["q"], c["r"]) in holes_set for c in cells]

    symbol_color = {sym: C.HOME_COLOR_PALETTE[i] for i, sym in enumerate(C.SYMBOLS)}
    ants_render = []
    for a in scenario.ants:
        positions = [A.axial_to_pixel(*ph[a.symbol], C.CELL_SIZE) for ph in scenario.positions_history]
        homes = [A.axial_to_pixel(*hh[a.symbol], C.CELL_SIZE) for hh in scenario.home_history]
        ants_render.append({
            "symbol": a.symbol, "glyph": C.SYMBOL_GLYPH[a.symbol], "color": symbol_color[a.symbol],
            "positions": [list(p) for p in positions], "homes": [list(h) for h in homes],
            "outcome": a.outcome, "resolved_tick": a.steps_taken,
            # CHANGE: per tick, pixel position of the grey hole the ant stepped into (None if it didn't)
            "hole_hits": [None if hh[a.symbol] is None else list(A.axial_to_pixel(*hh[a.symbol], C.CELL_SIZE))
                          for hh in scenario.hole_hit_history],
        })

    n_pred = scenario.n_predators
    predator_positions = [[list(A.axial_to_pixel(*p, C.CELL_SIZE)) for p in ph]
                           for ph in scenario.predator_history]

    render_params = {
        "background_hex": C.BACKGROUND_HEX, "fps": C.FPS,
        "cells": [{"pos": c["pos"], "size": c["size"]} for c in cells],
        "blocked_by_tick": blocked_by_tick, "is_hole": is_hole,
        "ants": ants_render, "n_predators": n_pred, "predator_positions": predator_positions,
        "cum_times": cum_times, "total_duration_s": total_duration,
        # CHANGE: wind gauge + streaks; wind_dirs[k] = direction index during move k (None = calm)
        "wind_strength": scenario.wind_strength, "wind_dirs": scenario.wind_history,
        "gusts": scenario.gust_history,
        "wind_vectors": [list(A.axial_to_pixel(dq, dr, 1.0)) for dq, dr in A.HEX_NEIGHBOR_OFFSETS],
    }
    params_path = out_path.with_suffix(".params.json")
    params_path.write_text(json.dumps(render_params))

    env = dict(os.environ)
    env["MAG_PARAMS_FILE"] = str(params_path)

    qflag = {"low": "-ql", "medium": "-qm", "high": "-qh"}[quality]
    with tempfile.TemporaryDirectory(prefix="manim_media_") as media_dir:
        t0 = time.time()
        cmd = [sys.executable, "-m", "manim", qflag, "--fps", str(C.FPS),
               "-r", f"{C.RESOLUTION[0]},{C.RESOLUTION[1]}",
               "--media_dir", media_dir, "--output_file", out_path.stem,
               str(HERE / "scene.py"), "MultiAgentWalkScene"]
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
    ap.add_argument("--output-dir", default=str(HERE.parent / "output_videos" / "random_walk_home"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--quality", default="low", choices=["low", "medium", "high"])
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
            rel_video = f"videos/{sample['sample_id']}.mp4"
            p = sample["params"]
            print(f"[{i + 1}/{n}] {sample['sample_id']} (n_ants={p['n_ants']}, home={p['n_ants_home']}, "
                  f"dead={p['n_ants_dead']}, ticks={p['total_ticks']}, holes={p['n_holes']}, "
                  f"predators={p['n_predators']}, home_moved={p['home_moved']}, swaps={p['tile_swap_count']}, "
                  f"wind={p['wind_strength']}:{p['wind_dir_start']}->{p['wind_dir_end']})")

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
