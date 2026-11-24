"""Render the preview set: the first PER seeds of every scenario (seed order), 1280x720 @ 24 fps.

    python preview.py [--per 13] [--total 50] [--output-dir ../output_videos/occluded_pattern_counting_preview]
                      [--poster-dir <dir>]

4 scenarios x 13 = 52, trimmed to 50 (largest groups first, every scenario keeps >= 8). Writes annotations.jsonl in
the same format as `run.py generate`, so `run.py validate` and `frame_check.py` run on it unchanged. With
--poster-dir, also writes one small mid-clip JPEG (<= 40 KB) per clip for pages.
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import video5 as V  # noqa: E402
import sim, questions, render  # noqa: E402

DS = "occluded_pattern_counting"


def poster(scene, path):
    img = render.frame_at(scene, scene["holds"][0][1] // 2).resize((640, 360))
    for q in (80, 70, 60, 50, 40, 30):
        img.save(path, "JPEG", quality=q, optimize=True)
        if path.stat().st_size <= 40_000:
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=13)
    ap.add_argument("--total", type=int, default=50)
    ap.add_argument("--output-dir", default=str(HERE.parent / "output_videos" / f"{DS}_preview"))
    ap.add_argument("--poster-dir", default=None)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    (out / "videos").mkdir(parents=True, exist_ok=True)
    if args.poster_dir:
        Path(args.poster_dir).mkdir(parents=True, exist_ok=True)
    picked, seed = {k: [] for k in sim.SCENARIOS}, 0
    while any(len(v) < args.per for v in picked.values()):
        sc = sim.build(random.Random(seed))["scenario"]
        if len(picked[sc]) < args.per:
            picked[sc].append(seed)
        seed += 1
    seeds = sorted(s for v in picked.values() for s in v)
    while len(seeds) > args.total:                   # trim the largest groups, keeping every scenario >= 8
        sc = max(picked, key=lambda k: len(picked[k]))
        if len(picked[sc]) <= 8:
            break
        seeds.remove(picked[sc].pop())
    with open(out / "annotations.jsonl", "w", encoding="utf8") as f:
        for i, seed in enumerate(seeds):
            s = V._roundtrip(V.build_sample(DS, sim, questions, seed))
            s["media_path"] = f"videos/{s['sample_id']}.mp4"
            s["media_meta"] = {"fps": V.FPS, "resolution": list(V.SIZE),
                               "duration_s": round(render.duration(s["params"]), 3)}
            if not args.no_render:
                s["render_meta"] = V.encode(render, s["params"], out / s["media_path"])
                if args.poster_dir:
                    poster(s["params"], Path(args.poster_dir) / f"{s['sample_id']}.jpg")
            f.write(json.dumps(s) + "\n")
            print(f"[{i + 1}/{len(seeds)}] {s['sample_id']} {s['params']['scenario']}", flush=True)
